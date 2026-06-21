#include "analyzer.h"
#include "../../core/duckdb_connector.h"
#include "../../core/data_registry.h"
#include "../../core/arrow_dataset.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

Analyzer::Analyzer()
    : analysis_running_(false)
    , analysis_progress_(0.0f)
    , duckdb_(std::make_unique<DuckDBConnector>())
{
    spdlog::info("[Data Studio] Analyzer initialized");
}

Analyzer::~Analyzer() = default;

void Analyzer::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Analyzer set active dataset: {}", dataset_name);

    // Auto-run analysis on dataset change
    RunAnalysis();
}

bool Analyzer::RunAnalysis() {
    if (current_dataset_.empty()) {
        last_error_ = "No dataset selected";
        return false;
    }

    spdlog::info("[Data Studio] Running analysis on: {}", current_dataset_);

    analysis_running_ = true;
    analysis_progress_ = 0.0f;
    last_error_ = "";

    try {
        // Get Arrow dataset from DataRegistry
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.GetArrowDataset(current_dataset_);

        if (!arrow_dataset) {
            last_error_ = "Failed to load dataset from registry";
            analysis_running_ = false;
            return false;
        }

        auto arrow_table = arrow_dataset->GetArrowTable();
        if (!arrow_table) {
            last_error_ = "Dataset has no Arrow table";
            analysis_running_ = false;
            return false;
        }

        analysis_progress_ = 0.2f;

        // Register table with DuckDB
        std::string table_name = "analysis_table";
        if (!duckdb_->RegisterTable(table_name, arrow_table)) {
            last_error_ = "Failed to register table with DuckDB: " + duckdb_->GetLastError();
            analysis_running_ = false;
            return false;
        }

        analysis_progress_ = 0.4f;

        // Get table schema
        auto schema = duckdb_->GetTableSchema(table_name);
        current_report_.total_columns = schema.size();

        // Get row count
        current_report_.total_rows = duckdb_->GetRowCount(table_name);

        analysis_progress_ = 0.6f;

        // Compute statistics for all numeric columns
        auto stats = duckdb_->ComputeStats(table_name);

        current_report_.column_stats.clear();
        size_t total_nulls = 0;
        size_t total_cells = 0;
        int numeric_count = 0;
        int categorical_count = 0;

        for (const auto& col_schema : schema) {
            ColumnStatistics col_stat;
            col_stat.name = col_schema.name;
            col_stat.data_type = col_schema.type;

            // Find corresponding stats
            auto stats_it = std::find_if(stats.begin(), stats.end(),
                                        [&](const DuckDBConnector::ColumnStats& s) {
                                            return s.column_name == col_schema.name;
                                        });

            if (stats_it != stats.end()) {
                // Numeric column with statistics
                col_stat.non_null_count = stats_it->count;
                col_stat.null_count = stats_it->null_count;
                col_stat.mean = stats_it->mean;
                col_stat.std = stats_it->std_dev;
                col_stat.min = stats_it->min;
                col_stat.max = stats_it->max;
                col_stat.q25 = stats_it->percentile_25;
                col_stat.q50 = stats_it->median;
                col_stat.q75 = stats_it->percentile_75;

                total_nulls += stats_it->null_count;
                total_cells += stats_it->count + stats_it->null_count;
                numeric_count++;
            } else {
                // Categorical/text column - get count only
                auto count_result = duckdb_->Query(
                    "SELECT COUNT(*) as cnt, COUNT(" + col_schema.name + ") as non_null FROM " + table_name
                );

                if (count_result && count_result->num_rows() > 0) {
                    auto cnt_array = count_result->column(0)->chunk(0);
                    auto non_null_array = count_result->column(1)->chunk(0);

                    int64_t total = std::static_pointer_cast<arrow::Int64Array>(cnt_array)->Value(0);
                    int64_t non_null = std::static_pointer_cast<arrow::Int64Array>(non_null_array)->Value(0);

                    col_stat.non_null_count = non_null;
                    col_stat.null_count = total - non_null;

                    total_nulls += col_stat.null_count;
                    total_cells += total;
                }

                categorical_count++;
            }

            current_report_.column_stats.push_back(col_stat);
        }

        current_report_.numeric_columns = numeric_count;
        current_report_.categorical_columns = categorical_count;

        // Calculate missing percentage
        if (total_cells > 0) {
            current_report_.missing_percentage = (100.0 * total_nulls) / total_cells;
        } else {
            current_report_.missing_percentage = 0.0;
        }

        analysis_progress_ = 0.8f;

        // Calculate data quality score (simple heuristic)
        // - Start at 100
        // - Subtract 2 points per 1% missing data
        // - Cap at 0
        double quality = 100.0 - (current_report_.missing_percentage * 2.0);
        current_report_.data_quality_score = std::max(0.0, quality);

        analysis_progress_ = 1.0f;

        // Unregister table
        duckdb_->UnregisterTable(table_name);

        spdlog::info("[Data Studio] Analysis complete: {} rows, {} columns, {:.1f}% quality",
                     current_report_.total_rows, current_report_.total_columns,
                     current_report_.data_quality_score);

    } catch (const std::exception& e) {
        last_error_ = std::string("Analysis failed: ") + e.what();
        spdlog::error("[Data Studio] {}", last_error_);
        analysis_running_ = false;
        return false;
    }

    analysis_running_ = false;
    return true;
}

void Analyzer::ComputeStatistics() {
    // TODO: Phase 1 Week 2 - Implement via DuckDB SQL queries
}

void Analyzer::ComputeCorrelations() {
    // TODO: Phase 1 Week 2 - Compute correlation matrix
}

void Analyzer::DetectOutliers() {
    // TODO: Phase 1 Week 2 - IQR-based outlier detection
}

void Analyzer::CalculateDataQuality() {
    // TODO: Phase 1 Week 2 - Composite quality score
}

} // namespace cyxwiz
