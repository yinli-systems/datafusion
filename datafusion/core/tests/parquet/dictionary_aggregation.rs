// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fs::File;
use std::sync::Arc;

use arrow::array::{Int64Array, StringViewArray};
use arrow::datatypes::DataType;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_common::Result;
use datafusion_datasource::file_scan_config::FileScanConfig;
use parquet::basic::Encoding;
use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use tempfile::tempdir;

const VALUE: &str = "dictionary-value-dictionary-value";

fn write_dictionary_file(path: &std::path::Path) -> Result<()> {
    let schema = Arc::new(parse_message_type(
        "message schema { REQUIRED BYTE_ARRAY key (UTF8); }",
    )?);
    let properties = Arc::new(
        WriterProperties::builder()
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .build(),
    );
    let mut writer = SerializedFileWriter::new(File::create(path)?, schema, properties)?;
    let mut row_group = writer.next_row_group()?;
    let mut column = row_group
        .next_column()?
        .expect("test schema contains one column");
    let value = ByteArray::from(VALUE);
    let values = vec![value.clone(); 10_000];
    column
        .typed::<ByteArrayType>()
        .write_batch_with_statistics(
            &values,
            None,
            None,
            Some(&value),
            Some(&value),
            Some(1),
        )?;
    column.close()?;
    row_group.close()?;
    writer.close()?;
    Ok(())
}

fn scan_file_type(plan: &Arc<dyn ExecutionPlan>) -> Option<DataType> {
    if let Some(scan) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = scan.data_source().downcast_ref::<FileScanConfig>()
    {
        return Some(config.file_schema().field(0).data_type().clone());
    }
    plan.children().into_iter().find_map(scan_file_type)
}

#[tokio::test]
async fn preserves_native_dictionary_for_profitable_group_by() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("dictionary.parquet");
    write_dictionary_file(&path)?;

    // Verify that the generated file carries every independent fact used by
    // the optimizer. This keeps the test from passing due to a writer-default
    // change that silently makes the optimization ineligible.
    let reader = SerializedFileReader::new(File::open(&path)?)?;
    let column = reader.metadata().row_group(0).column(0);
    assert!(column.dictionary_page_offset().is_some());
    assert!(column.page_encoding_stats_mask().is_some_and(|mask| {
        mask.is_only(Encoding::PLAIN_DICTIONARY) || mask.is_only(Encoding::RLE_DICTIONARY)
    }));
    assert_eq!(
        column.unencoded_byte_array_data_bytes(),
        Some((VALUE.len() * 10_000) as i64)
    );
    assert_eq!(
        column
            .statistics()
            .and_then(|statistics| statistics.distinct_count_opt()),
        Some(1)
    );
    drop(reader);

    let mut baseline_config = SessionConfig::new();
    baseline_config.options_mut().execution.target_partitions = 1;
    let baseline_ctx = SessionContext::new_with_config(baseline_config);
    baseline_ctx
        .register_parquet(
            "t",
            path.to_str().expect("temporary path should be UTF-8"),
            ParquetReadOptions::default(),
        )
        .await?;
    let baseline_plan = baseline_ctx
        .sql("SELECT key, COUNT(*) AS count FROM t GROUP BY key")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(scan_file_type(&baseline_plan), Some(DataType::Utf8View));
    let baseline_batches = collect(baseline_plan, baseline_ctx.task_ctx()).await?;

    let mut config = SessionConfig::new();
    config.options_mut().optimizer.enable_dictionary_aggregation = true;
    config.options_mut().execution.target_partitions = 1;
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "t",
        path.to_str().expect("temporary path should be UTF-8"),
        ParquetReadOptions::default(),
    )
    .await?;

    let dataframe = ctx
        .sql("SELECT key, COUNT(*) AS count FROM t GROUP BY key")
        .await?;
    let plan = dataframe.create_physical_plan().await?;

    // The dictionary is an internal physical choice. SQL-visible output stays
    // Utf8View, while the actual Parquet scan emits Dictionary<Int32, Utf8>.
    assert_eq!(plan.schema().field(0).data_type(), &DataType::Utf8View);
    assert_eq!(
        scan_file_type(&plan),
        Some(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        ))
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches, baseline_batches);
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
    let key = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("output should retain Utf8View");
    let count = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT output should be Int64");
    assert_eq!(key.value(0), VALUE);
    assert_eq!(count.value(0), 10_000);

    // A second plan is built from cached file metadata. Physical encoding
    // facts must survive that path just like logical column statistics do.
    let cached_plan = ctx
        .sql("SELECT key, COUNT(*) AS count FROM t GROUP BY key")
        .await?
        .create_physical_plan()
        .await?;
    assert!(matches!(
        scan_file_type(&cached_plan),
        Some(DataType::Dictionary(_, _))
    ));
    Ok(())
}
