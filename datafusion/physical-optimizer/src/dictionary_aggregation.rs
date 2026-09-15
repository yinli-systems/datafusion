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

//! Preserve profitable file-native dictionary encodings through hash aggregation.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{PhysicalFileStatistics, Result};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{CastExpr, Column};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::projection::ProjectionExec;

use crate::PhysicalOptimizerRule;

// Utf8View stores values up to 12 bytes inline. The benchmark for this rule
// shows that dictionary keys become advantageous only after that fast path no
// longer applies. Leave additional headroom for length variance and metadata.
const MIN_AVERAGE_VALUE_BYTES: usize = 16;
// The current native dictionary path pays a per-dictionary setup cost. Require
// enough repeated rows to amortize it. This deliberately starts conservative;
// configurable cost thresholds are tracked separately in #24113.
const MIN_ROWS_PER_DISTINCT_VALUE: usize = 10_000;

/// Rewrites eligible string GROUP BY columns to use dictionaries emitted
/// natively by their file source.
#[derive(Debug, Default)]
pub struct DictionaryAggregation {}

#[derive(Debug)]
struct GroupColumn {
    group_index: usize,
    scan_output_index: usize,
    table_index: usize,
}

impl DictionaryAggregation {
    /// Create a new [`DictionaryAggregation`] optimizer rule.
    pub fn new() -> Self {
        Self {}
    }

    fn try_rewrite(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(final_agg) = plan.downcast_ref::<AggregateExec>() else {
            return Ok(None);
        };

        match final_agg.mode() {
            AggregateMode::Single | AggregateMode::SinglePartitioned => {
                let Some(scan) = final_agg.input().downcast_ref::<DataSourceExec>()
                else {
                    return Ok(None);
                };
                Self::rewrite_aggregate_chain(final_agg, None, scan)
            }
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                let Some(partial_agg) = final_agg.input().downcast_ref::<AggregateExec>()
                else {
                    return Ok(None);
                };
                if partial_agg.mode() != &AggregateMode::Partial {
                    return Ok(None);
                }
                let Some(scan) = partial_agg.input().downcast_ref::<DataSourceExec>()
                else {
                    return Ok(None);
                };
                Self::rewrite_aggregate_chain(final_agg, Some(partial_agg), scan)
            }
            AggregateMode::Partial | AggregateMode::PartialReduce => Ok(None),
        }
    }

    fn rewrite_aggregate_chain(
        final_agg: &AggregateExec,
        partial_agg: Option<&AggregateExec>,
        scan: &DataSourceExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let raw_agg = partial_agg.unwrap_or(final_agg);
        if raw_agg.group_expr().is_empty()
            || raw_agg.group_expr().has_grouping_set()
            || !raw_agg.group_expr().null_expr().is_empty()
        {
            return Ok(None);
        }

        let Some(config) = scan.data_source().downcast_ref::<FileScanConfig>() else {
            return Ok(None);
        };
        // A pushed filter or scan limit invalidates the table-wide row count,
        // NDV and byte-width relationship used by the cost gate.
        if config.file_source.filter().is_some() || config.limit.is_some() {
            return Ok(None);
        }

        let group_columns = Self::group_columns(raw_agg, config)?;
        if group_columns.is_empty() {
            return Ok(None);
        }

        let statistics = config.statistics();
        let group_columns: Vec<_> = group_columns
            .into_iter()
            .filter(|column| Self::is_profitable(config, &statistics, column.table_index))
            .collect();
        if group_columns.is_empty() {
            return Ok(None);
        }

        let encoded_scan_indices: HashSet<_> = group_columns
            .iter()
            .map(|column| column.scan_output_index)
            .collect();
        if Self::aggregate_references_columns(raw_agg, &encoded_scan_indices) {
            return Ok(None);
        }

        if let Some(partial_agg) = partial_agg {
            if final_agg.group_expr().expr().len() != raw_agg.group_expr().expr().len()
                || !final_agg.group_expr().expr().iter().enumerate().all(
                    |(index, (expr, _))| {
                        expr.downcast_ref::<Column>()
                            .is_some_and(|column| column.index() == index)
                    },
                )
            {
                return Ok(None);
            }
            let final_group_indices: HashSet<_> = group_columns
                .iter()
                .map(|column| column.group_index)
                .collect();
            if Self::aggregate_references_columns(final_agg, &final_group_indices)
                || partial_agg.group_expr().expr().len()
                    != final_agg.group_expr().expr().len()
            {
                return Ok(None);
            }
        }

        let table_indices: Vec<_> = group_columns
            .iter()
            .map(|column| column.table_index)
            .collect();

        let Some(file_source) = config
            .file_source
            .try_pushdown_dictionary_encoding(&table_indices)?
        else {
            return Ok(None);
        };
        let new_config = FileScanConfigBuilder::from(config.clone())
            .with_source(file_source)
            .build();
        let new_scan: Arc<dyn ExecutionPlan> =
            DataSourceExec::from_data_source(new_config);
        let raw_input_schema = new_scan.schema();

        let new_raw_agg = Arc::new(
            AggregateExec::try_new(
                *raw_agg.mode(),
                raw_agg.group_expr().clone(),
                raw_agg.aggr_expr().to_vec(),
                raw_agg.filter_expr().to_vec(),
                new_scan,
                Arc::clone(&raw_input_schema),
            )?
            .with_limit_options(raw_agg.limit_options()),
        );

        let new_final_agg: Arc<dyn ExecutionPlan> = if partial_agg.is_some() {
            Arc::new(
                AggregateExec::try_new(
                    *final_agg.mode(),
                    final_agg.group_expr().clone(),
                    final_agg.aggr_expr().to_vec(),
                    final_agg.filter_expr().to_vec(),
                    new_raw_agg,
                    raw_input_schema,
                )?
                .with_limit_options(final_agg.limit_options()),
            )
        } else {
            new_raw_agg
        };

        Self::restore_output_schema(&final_agg.schema(), new_final_agg, &group_columns)
            .map(Some)
    }

    fn group_columns(
        aggregate: &AggregateExec,
        config: &FileScanConfig,
    ) -> Result<Vec<GroupColumn>> {
        let projected_schema = config.projected_schema()?;
        // Re-typing a source column beneath an arbitrary pushed projection can
        // invalidate that expression's cached input fields. Ordinary file scans
        // use a column-only projection, so keep this first implementation to
        // that well-defined shape.
        if config.file_source.projection().is_some_and(|projection| {
            projection
                .as_ref()
                .iter()
                .any(|expr| expr.expr.downcast_ref::<Column>().is_none())
        }) {
            return Ok(vec![]);
        }

        let mut columns = vec![];
        for (group_index, (expr, _)) in aggregate.group_expr().expr().iter().enumerate() {
            let Some(column) = expr.downcast_ref::<Column>() else {
                // A grouping expression could itself depend on a candidate
                // column and therefore cannot safely retain its old input type.
                return Ok(vec![]);
            };
            if projected_schema
                .fields()
                .get(column.index())
                .is_none_or(|field| field.data_type() != &DataType::Utf8View)
            {
                continue;
            }
            let table_index = match config.file_source.projection() {
                Some(projection) => projection.as_ref()[column.index()]
                    .expr
                    .downcast_ref::<Column>()
                    .expect("column-only projection checked above")
                    .index(),
                None => column.index(),
            };
            columns.push(GroupColumn {
                group_index,
                scan_output_index: column.index(),
                table_index,
            });
        }
        Ok(columns)
    }

    fn aggregate_references_columns(
        aggregate: &AggregateExec,
        column_indices: &HashSet<usize>,
    ) -> bool {
        let references_candidate = |expr: &Arc<dyn PhysicalExpr>| {
            collect_columns(expr)
                .iter()
                .any(|column| column_indices.contains(&column.index()))
        };
        aggregate.aggr_expr().iter().any(|aggregate| {
            aggregate.expressions().iter().any(references_candidate)
                || aggregate
                    .order_bys()
                    .iter()
                    .any(|order| references_candidate(&order.expr))
        }) || aggregate
            .filter_expr()
            .iter()
            .flatten()
            .any(references_candidate)
    }

    fn is_profitable(
        config: &FileScanConfig,
        statistics: &datafusion_common::Statistics,
        table_index: usize,
    ) -> bool {
        let Precision::Exact(num_rows) = statistics.num_rows else {
            return false;
        };
        let Some(column_statistics) = statistics.column_statistics.get(table_index)
        else {
            return false;
        };
        let (Precision::Exact(null_count), Precision::Exact(distinct_count)) = (
            column_statistics.null_count,
            column_statistics.distinct_count,
        ) else {
            return false;
        };
        let Some(non_null_rows) = num_rows.checked_sub(null_count) else {
            return false;
        };
        if distinct_count == 0
            || non_null_rows / distinct_count < MIN_ROWS_PER_DISTINCT_VALUE
        {
            return false;
        }

        let mut seen_files = HashSet::new();
        let mut unencoded_value_bytes = 0usize;
        for file in config.file_groups.iter().flat_map(|group| group.iter()) {
            if file.range.is_some() {
                return false;
            }
            let identity = (
                file.object_meta.location.to_string(),
                file.object_meta.e_tag.clone(),
                file.object_meta.version.clone(),
            );
            if !seen_files.insert(identity) {
                continue;
            }
            let Some(physical) = file.extension::<PhysicalFileStatistics>() else {
                return false;
            };
            let Some(column) = physical.column_statistics.get(table_index) else {
                return false;
            };
            if !column.all_dictionary_encoded {
                return false;
            }
            let Some(bytes) = column.unencoded_value_bytes else {
                return false;
            };
            let Some(total) = unencoded_value_bytes.checked_add(bytes) else {
                return false;
            };
            unencoded_value_bytes = total;
        }
        !seen_files.is_empty()
            && non_null_rows != 0
            && unencoded_value_bytes / non_null_rows >= MIN_AVERAGE_VALUE_BYTES
    }

    fn restore_output_schema(
        output_schema: &SchemaRef,
        input: Arc<dyn ExecutionPlan>,
        group_columns: &[GroupColumn],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let encoded_groups: HashSet<_> = group_columns
            .iter()
            .map(|column| column.group_index)
            .collect();
        let expressions =
            output_schema
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let column = Arc::new(Column::new(field.name(), index))
                        as Arc<dyn PhysicalExpr>;
                    let expr = if encoded_groups.contains(&index) {
                        Arc::new(CastExpr::new_with_target_field(
                            column,
                            Arc::clone(field),
                            None,
                        )) as Arc<dyn PhysicalExpr>
                    } else {
                        column
                    };
                    (expr, field.name().clone())
                });
        Ok(Arc::new(ProjectionExec::try_new_with_schema_metadata(
            expressions,
            input,
            output_schema.as_ref(),
        )?))
    }
}

impl PhysicalOptimizerRule for DictionaryAggregation {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_dictionary_aggregation {
            return Ok(plan);
        }
        plan.transform_down(|plan| {
            Ok(match Self::try_rewrite(&plan)? {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(plan),
            })
        })
        .data()
    }

    fn name(&self) -> &str {
        "DictionaryAggregation"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};
    use datafusion_common::{ColumnStatistics, PhysicalColumnStatistics, Statistics};
    use datafusion_datasource::PartitionedFile;
    use datafusion_datasource_parquet::source::ParquetSource;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_plan::aggregates::PhysicalGroupBy;

    fn aggregate_plan(
        value_bytes: &[usize],
        distinct_counts: &[Precision<usize>],
        all_dictionary_encoded: bool,
        include_physical_statistics: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fields = value_bytes
            .iter()
            .enumerate()
            .map(|(index, _)| {
                Field::new(format!("key_{index}"), DataType::Utf8View, false)
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));
        let physical_statistics = PhysicalFileStatistics {
            column_statistics: value_bytes
                .iter()
                .map(|bytes| PhysicalColumnStatistics {
                    all_dictionary_encoded,
                    unencoded_value_bytes: Some(*bytes),
                })
                .collect(),
        };
        let mut file = PartitionedFile::new("data.parquet", 1024);
        if include_physical_statistics {
            file = file.with_extension(physical_statistics);
        }
        let statistics = Statistics {
            num_rows: Precision::Exact(10_000_000),
            total_byte_size: Precision::Absent,
            column_statistics: distinct_counts
                .iter()
                .map(|distinct_count| ColumnStatistics {
                    null_count: Precision::Exact(0),
                    distinct_count: *distinct_count,
                    ..Default::default()
                })
                .collect(),
        };
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(ParquetSource::new(Arc::clone(&schema))),
        )
        .with_file(file)
        .with_statistics(statistics)
        .build();
        let scan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
        let raw_input_schema = scan.schema();
        let raw_group_by = PhysicalGroupBy::new_single(
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    (
                        Arc::new(Column::new(field.name(), index))
                            as Arc<dyn PhysicalExpr>,
                        field.name().clone(),
                    )
                })
                .collect(),
        );
        let partial = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            raw_group_by,
            vec![],
            vec![],
            scan,
            Arc::clone(&raw_input_schema),
        )?);
        let final_group_by = PhysicalGroupBy::new_single(
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    (
                        Arc::new(Column::new(field.name(), index))
                            as Arc<dyn PhysicalExpr>,
                        field.name().clone(),
                    )
                })
                .collect(),
        );
        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            final_group_by,
            vec![],
            vec![],
            partial,
            raw_input_schema,
        )?))
    }

    fn optimize(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let mut config = ConfigOptions::new();
        config.optimizer.enable_dictionary_aggregation = true;
        DictionaryAggregation::new().optimize(plan, &config)
    }

    fn rewritten_scan(plan: &Arc<dyn ExecutionPlan>) -> &FileScanConfig {
        let projection = plan
            .downcast_ref::<ProjectionExec>()
            .expect("eligible aggregation should have a schema-restoring projection");
        let final_agg = projection
            .input()
            .downcast_ref::<AggregateExec>()
            .expect("projection input should be final aggregate");
        let partial_agg = final_agg
            .input()
            .downcast_ref::<AggregateExec>()
            .expect("final aggregate input should be partial aggregate");
        let scan = partial_agg
            .input()
            .downcast_ref::<DataSourceExec>()
            .expect("partial aggregate input should be scan");
        scan.data_source()
            .downcast_ref::<FileScanConfig>()
            .expect("scan should use FileScanConfig")
    }

    #[test]
    fn rewrites_profitable_group_key_and_preserves_output_schema() -> Result<()> {
        let plan =
            aggregate_plan(&[240_000_000], &[Precision::Exact(1_000)], true, true)?;
        let original_schema = plan.schema();
        let optimized = optimize(plan)?;

        assert_eq!(optimized.schema(), original_schema);
        let scan = rewritten_scan(&optimized);
        assert_eq!(
            scan.file_schema().field(0).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
        let optimized_again = optimize(Arc::clone(&optimized))?;
        assert!(
            Arc::ptr_eq(&optimized_again, &optimized),
            "dictionary rewrite should be idempotent"
        );
        Ok(())
    }

    #[test]
    fn rewrites_only_columns_that_pass_the_cost_gate() -> Result<()> {
        let plan = aggregate_plan(
            &[240_000_000, 80_000_000],
            &[Precision::Exact(1_000), Precision::Exact(1_000)],
            true,
            true,
        )?;
        let original_schema = plan.schema();
        let optimized = optimize(plan)?;

        assert_eq!(optimized.schema(), original_schema);
        let scan = rewritten_scan(&optimized);
        assert!(matches!(
            scan.file_schema().field(0).data_type(),
            DataType::Dictionary(_, _)
        ));
        assert_eq!(scan.file_schema().field(1).data_type(), &DataType::Utf8View);
        Ok(())
    }

    #[test]
    fn skips_when_cost_or_storage_evidence_is_incomplete() -> Result<()> {
        let cases = [
            // Utf8View's short-value inline representation is faster.
            (80_000_000, Precision::Exact(1_000), true, true),
            // Too few rows repeat each distinct value.
            (240_000_000, Precision::Exact(10_000), true, true),
            // Approximate NDV is not sufficient for a cost decision.
            (240_000_000, Precision::Inexact(1_000), true, true),
            // At least one data page is not dictionary encoded.
            (240_000_000, Precision::Exact(1_000), false, true),
            // The file format supplied no physical encoding metadata.
            (240_000_000, Precision::Exact(1_000), true, false),
        ];
        for (bytes, distinct_count, encoded, include_physical) in cases {
            let plan =
                aggregate_plan(&[bytes], &[distinct_count], encoded, include_physical)?;
            let optimized = optimize(Arc::clone(&plan))?;
            assert!(
                Arc::ptr_eq(&optimized, &plan),
                "ineligible plan should be returned unchanged"
            );
        }
        Ok(())
    }

    #[test]
    fn disabled_rule_returns_plan_unchanged() -> Result<()> {
        let plan =
            aggregate_plan(&[240_000_000], &[Precision::Exact(1_000)], true, true)?;
        let config = ConfigOptions::new();
        let optimized =
            DictionaryAggregation::new().optimize(Arc::clone(&plan), &config)?;
        assert!(Arc::ptr_eq(&optimized, &plan));
        Ok(())
    }
}
