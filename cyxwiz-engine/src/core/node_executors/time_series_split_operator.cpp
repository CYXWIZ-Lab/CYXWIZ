#include "time_series_split_operator.h"
#include "materialization_memory_preflight.h"
#include "../profiler_trace.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    float progress,
                    uint64_t processed_items = 0,
                    uint64_t total_items = 0,
                    uint64_t estimated_memory_bytes = 0) {
    if (!callback) return;
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.progress = progress;
    event.processed_items = processed_items;
    event.total_items = total_items;
    event.estimated_memory_bytes = estimated_memory_bytes;
    callback(event);
}

bool ReadFloatParam(
    const std::map<std::string, std::string>& params,
    const char* key,
    float default_value,
    float& out,
    std::string& error) {

    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = default_value;
        return true;
    }
    try {
        out = std::stof(it->second);
    } catch (...) {
        error = std::string("TimeSeriesSplit: '") + key +
                "' is not a valid float: " + it->second;
        return false;
    }
    return true;
}

bool ReadInt64Param(
    const std::map<std::string, std::string>& params,
    const char* key,
    int64_t default_value,
    int64_t& out,
    std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = default_value;
        return true;
    }
    try {
        size_t consumed = 0;
        out = std::stoll(it->second, &consumed);
        if (consumed != it->second.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (...) {
        error = std::string("TimeSeriesSplit: '") + key +
                "' is not a valid integer: " + it->second;
        return false;
    }
    return true;
}

arrow::Result<std::vector<int64_t>> ReadRequiredInt64Column(
    const std::shared_ptr<arrow::Table>& input,
    const char* name) {
    auto column = input->GetColumnByName(name);
    if (!column) {
        return arrow::Status::Invalid(
            "TimeSeriesSplit: boundary_policy=targets_within_partition "
            "requires TimeSeriesWindow metadata column '", name,
            "'. Re-run the upstream window node or use boundary_policy=window_rows "
            "only for a legacy workflow.");
    }
    if (column->type()->id() != arrow::Type::INT64) {
        return arrow::Status::TypeError(
            "TimeSeriesSplit: metadata column '", name,
            "' must be int64 (got ", column->type()->ToString(), ")");
    }

    std::vector<int64_t> values;
    values.reserve(static_cast<size_t>(column->length()));
    for (const auto& chunk : column->chunks()) {
        auto array = std::static_pointer_cast<arrow::Int64Array>(chunk);
        for (int64_t row = 0; row < array->length(); ++row) {
            if (array->IsNull(row)) {
                return arrow::Status::Invalid(
                    "TimeSeriesSplit: metadata column '", name,
                    "' contains a null value");
            }
            values.push_back(array->Value(row));
        }
    }
    return values;
}

} // namespace

bool TimeSeriesSplitOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    train_ratio_ = 0.8f;
    val_ratio_ = 0.1f;
    test_ratio_ = 0.1f;
    boundary_policy_ = "window_rows";
    train_end_source_row_ = -1;
    val_end_source_row_ = -1;

    if (!ReadFloatParam(params, "train_ratio", 0.8f, train_ratio_, error)) return false;
    if (!ReadFloatParam(params, "val_ratio",   0.1f, val_ratio_,   error)) return false;
    if (!ReadFloatParam(params, "test_ratio",  0.1f, test_ratio_,  error)) return false;

    auto policy_it = params.find("boundary_policy");
    if (policy_it != params.end() && !policy_it->second.empty()) {
        boundary_policy_ = policy_it->second;
    }
    if (boundary_policy_ != "window_rows" &&
        boundary_policy_ != "targets_within_partition") {
        error = "TimeSeriesSplit: boundary_policy must be 'window_rows' or "
                "'targets_within_partition' (got '" + boundary_policy_ + "')";
        return false;
    }

    if (!ReadInt64Param(params, "train_end_source_row", -1,
                        train_end_source_row_, error)) return false;
    if (!ReadInt64Param(params, "val_end_source_row", -1,
                        val_end_source_row_, error)) return false;
    const bool has_train_end = train_end_source_row_ >= 0;
    const bool has_val_end = val_end_source_row_ >= 0;
    if (has_train_end != has_val_end) {
        error = "TimeSeriesSplit: train_end_source_row and val_end_source_row "
                "must be provided together";
        return false;
    }
    if (has_train_end &&
        (train_end_source_row_ <= 0 ||
         val_end_source_row_ <= train_end_source_row_)) {
        error = "TimeSeriesSplit: explicit source boundaries require "
                "0 < train_end_source_row < val_end_source_row";
        return false;
    }

    if (train_ratio_ < 0.0f || val_ratio_ < 0.0f || test_ratio_ < 0.0f) {
        error = "TimeSeriesSplit: all ratios must be >= 0";
        return false;
    }
    if (train_ratio_ <= 0.0f) {
        error = "TimeSeriesSplit: train_ratio must be > 0";
        return false;
    }

    const float sum = train_ratio_ + val_ratio_ + test_ratio_;
    if (std::fabs(sum - 1.0f) > 0.01f) {
        error = "TimeSeriesSplit: ratios must sum to 1.0 (got " +
                std::to_string(sum) + ")";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>>
TimeSeriesSplitOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {

    if (!input_schema) {
        return arrow::Status::Invalid("TimeSeriesSplit: input schema is null");
    }
    auto fields = input_schema->fields();
    fields.push_back(arrow::field(kPartitionColumnName, arrow::int8()));
    return arrow::schema(fields);
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesSplitOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesSplit Materializer");
    if (!input) {
        return arrow::Status::Invalid("TimeSeriesSplit: input table is null");
    }

    const int64_t n = input->num_rows();
    if (n == 0) {
        return arrow::Status::Invalid(
            "TimeSeriesSplit: input table has zero rows - upstream TimeSeriesWindow "
            "probably produced no windows");
    }

    // The leakage-safe path can simultaneously own two int64 metadata copies,
    // an int8 assignment vector, and the Arrow partition output. Three int64
    // row units conservatively cover both split policies.
    const auto split_estimate = EstimateDenseMaterializationMemory(
        static_cast<uint64_t>(n), 3, static_cast<uint64_t>(sizeof(int64_t)));
    ARROW_ASSIGN_OR_RAISE(
        auto preflight_estimate,
        EmitMaterializationMemoryPreflight(
            split_estimate,
            "TimeSeriesSplit",
            "planned_row_buffers",
            "Reduce input windows or split a smaller materialized dataset.",
            GetMaterializationMemoryContext(),
            progress_callback_,
            SaturatingMaterializationItemCount(static_cast<uint64_t>(n), 3),
            0.10f));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    if (boundary_policy_ == "targets_within_partition") {
        ReportProgress(progress_callback_, "plan_split",
                       "Planning target-contained chronological split", 0.30f,
                       0, static_cast<uint64_t>(n),
                       preflight_estimate.estimated_peak_bytes);

        ARROW_ASSIGN_OR_RAISE(
            auto target_starts,
            ReadRequiredInt64Column(input, "__target_start_index"));
        ARROW_ASSIGN_OR_RAISE(
            auto target_ends,
            ReadRequiredInt64Column(input, "__target_end_index"));
        if (target_starts.size() != static_cast<size_t>(n) ||
            target_ends.size() != static_cast<size_t>(n)) {
            return arrow::Status::Invalid(
                "TimeSeriesSplit: target-bound metadata row count does not "
                "match the window table");
        }

        const int64_t source_rows = target_ends.back() + 1;
        if (source_rows <= 0) {
            return arrow::Status::Invalid(
                "TimeSeriesSplit: invalid source row count derived from "
                "__target_end_index");
        }

        const bool has_explicit_boundaries = train_end_source_row_ >= 0;
        const int64_t train_end = has_explicit_boundaries
            ? train_end_source_row_
            : static_cast<int64_t>(std::floor(source_rows * train_ratio_));
        const int64_t val_end = has_explicit_boundaries
            ? val_end_source_row_
            : train_end +
                static_cast<int64_t>(std::floor(source_rows * val_ratio_));
        if (train_end <= 0 || val_end <= train_end || val_end > source_rows) {
            return arrow::Status::Invalid(
                "TimeSeriesSplit: source-row boundaries must satisfy 0 < train_end (",
                train_end, ") < val_end (", val_end,
                ") <= source rows (", source_rows, ")");
        }

        int64_t train_count = 0;
        int64_t val_count = 0;
        int64_t test_count = 0;
        int64_t purged_count = 0;
        int64_t previous_start = -1;
        std::vector<int8_t> assignments;
        assignments.reserve(static_cast<size_t>(n));
        for (int64_t row = 0; row < n; ++row) {
            if ((row & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
            const int64_t target_start =
                target_starts[static_cast<size_t>(row)];
            const int64_t target_end = target_ends[static_cast<size_t>(row)];
            if (target_start < 0 || target_end < target_start ||
                target_end >= source_rows || target_start < previous_start) {
                return arrow::Status::Invalid(
                    "TimeSeriesSplit: invalid or non-monotonic target bounds at "
                    "window row ", row, " (start=", target_start,
                    ", end=", target_end, ")");
            }
            previous_start = target_start;

            int8_t partition = -1;
            if (target_end < train_end) {
                partition = 0;
                ++train_count;
            } else if (target_start >= train_end && target_end < val_end) {
                partition = 1;
                ++val_count;
            } else if (target_start >= val_end) {
                partition = 2;
                ++test_count;
            } else {
                ++purged_count;
            }
            assignments.push_back(partition);
        }

        arrow::Int8Builder builder(arrow::default_memory_pool());
        ARROW_RETURN_NOT_OK(builder.Reserve(n));
        ReportProgress(progress_callback_, "write_partitions",
                       "Writing target-contained partition assignments", 0.60f,
                       0, static_cast<uint64_t>(n),
                       preflight_estimate.estimated_peak_bytes);
        for (size_t row = 0; row < assignments.size(); ++row) {
            if ((row & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
            ARROW_RETURN_NOT_OK(builder.Append(assignments[row]));
        }
        std::shared_ptr<arrow::Array> partition_array;
        ARROW_RETURN_NOT_OK(builder.Finish(&partition_array));
        auto partition_chunked =
            std::make_shared<arrow::ChunkedArray>(partition_array);
        auto partition_field = arrow::field(kPartitionColumnName, arrow::int8());
        ARROW_ASSIGN_OR_RAISE(
            auto out_table,
            input->AddColumn(input->num_columns(), partition_field,
                             partition_chunked));

        spdlog::info(
            "TimeSeriesSplit: {} windows -> train={}, val={}, test={}, purged={} "
            "(policy=targets_within_partition, train_end={}, val_end={}, "
            "source_rows={}, explicit={})",
            n, train_count, val_count, test_count, purged_count,
            train_end, val_end, source_rows, has_explicit_boundaries);
        ReportProgress(progress_callback_, "complete",
                       "Leakage-safe time-series split complete", 1.0f,
                       static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                        preflight_estimate.estimated_peak_bytes);
        return out_table;
    }

    int64_t train_count = static_cast<int64_t>(std::floor(n * train_ratio_));
    int64_t val_count   = static_cast<int64_t>(std::floor(n * val_ratio_));
    // Give any remainder from floor rounding to the train split so train
    // always gets the largest slice. Test takes whatever is left over so
    // train_count + val_count + test_count == n exactly.
    if (train_count + val_count > n) val_count = n - train_count;
    int64_t test_count = n - train_count - val_count;

    ReportProgress(progress_callback_, "plan_split",
                   "Planning chronological train/val/test split", 0.30f, 0,
                   static_cast<uint64_t>(n),
                   preflight_estimate.estimated_peak_bytes);

    // Guard: if val or test is zero because n is too small for the
    // requested ratio, warn and steal a single row from train. This
    // keeps the smoke test viable on tiny datasets (airline passengers
    // has 144 rows; with input_width=12 + shift=1 that's 132 windows,
    // so 80/10/10 gives train=105 val=13 test=14 — fine. But a user
    // testing with 20 rows + input_width=16 would hit 4 windows and
    // get train=3 val=0 test=1 without this guard).
    if (val_count == 0 && train_count > 1) {
        --train_count;
        ++val_count;
        spdlog::warn("TimeSeriesSplit: val_count rounded to 0 - stealing 1 row from train");
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::Int8Builder builder(pool);
    ARROW_RETURN_NOT_OK(builder.Reserve(n));

    ReportProgress(progress_callback_, "write_partitions",
                   "Writing partition assignments", 0.60f, 0,
                   static_cast<uint64_t>(n),
                   preflight_estimate.estimated_peak_bytes);
    for (int64_t i = 0; i < train_count; ++i) {
        if ((i & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        ARROW_RETURN_NOT_OK(builder.Append(0));
    }
    for (int64_t i = 0; i < val_count; ++i) {
        if ((i & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        ARROW_RETURN_NOT_OK(builder.Append(1));
    }
    for (int64_t i = 0; i < test_count; ++i) {
        if ((i & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        ARROW_RETURN_NOT_OK(builder.Append(2));
    }

    std::shared_ptr<arrow::Array> partition_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&partition_array));

    // Append the new column to the existing schema. Arrow's AddColumn
    // takes the column index to insert at (we append at the end) and a
    // ChunkedArray wrapping our single-chunk builder result.
    auto partition_chunked = std::make_shared<arrow::ChunkedArray>(partition_array);
    auto partition_field = arrow::field(kPartitionColumnName, arrow::int8());

    ReportProgress(progress_callback_, "append_output",
                   "Appending partition column", 0.90f,
                   static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                   preflight_estimate.estimated_peak_bytes);
    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        input->AddColumn(input->num_columns(), partition_field, partition_chunked));

    spdlog::info("TimeSeriesSplit: {} rows -> train={}, val={}, test={} "
                 "(ratios {:.2f}/{:.2f}/{:.2f})",
                 n, train_count, val_count, test_count,
                 train_ratio_, val_ratio_, test_ratio_);

    ReportProgress(progress_callback_, "complete",
                   "Time-series split complete", 1.0f,
                   static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                   preflight_estimate.estimated_peak_bytes);
    return out_table;
}

} // namespace cyxwiz
