#include "differencing_operator.h"
#include "../profiler_trace.h"
#include "ts_column_utils.h"

#include <spdlog/spdlog.h>
#include <cstdint>
#include <utility>

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

} // namespace

bool DifferencingOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    value_col_.clear();
    lag_ = 1;
    order_ = 1;

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "Differencing: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;

    auto read_int = [&](const char* key, int default_value, int& out) -> bool {
        auto p = params.find(key);
        if (p == params.end() || p->second.empty()) {
            out = default_value;
            return true;
        }
        try { out = std::stoi(p->second); }
        catch (...) {
            error = std::string("Differencing: '") + key +
                    "' is not a valid integer: " + p->second;
            return false;
        }
        return true;
    };

    if (!read_int("lag", 1, lag_)) return false;
    if (!read_int("order", 1, order_)) return false;

    if (lag_ < 1) {
        error = "Differencing: lag must be >= 1 (got " +
                std::to_string(lag_) + ")";
        return false;
    }
    if (order_ < 1 || order_ > 3) {
        error = "Differencing: order must be 1-3 (got " +
                std::to_string(order_) + ")";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
DifferencingOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz Differencing Materializer");
    if (!input) {
        return arrow::Status::Invalid("Differencing: input table is null");
    }

    ReportProgress(progress_callback_, "read_column",
                   "Reading column for differencing", 0.10f, 0,
                   static_cast<uint64_t>(input->num_rows()));
    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "Differencing: column '" + value_col_ + "' not found");
    }

    int col_idx = input->schema()->GetFieldIndex(value_col_);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "Differencing: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    const int64_t original_n = static_cast<int64_t>(values.size());

    ReportProgress(progress_callback_, "difference",
                   "Applying time-series differencing", 0.45f, 0,
                   static_cast<uint64_t>(original_n),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    // Apply differencing `order_` times. Each pass drops `lag_` values
    // from the front of the series.
    for (int d = 0; d < order_; ++d) {
        const size_t n = values.size();
        if (static_cast<int>(n) <= lag_) {
            return arrow::Status::Invalid(
                "Differencing: not enough rows after pass " +
                std::to_string(d + 1) + " (have " + std::to_string(n) +
                ", need > " + std::to_string(lag_) + ")");
        }
        std::vector<float> diffed;
        diffed.reserve(n - lag_);
        for (size_t i = lag_; i < n; ++i) {
            diffed.push_back(values[i] - values[i - lag_]);
        }
        values = std::move(diffed);
    }

    const int64_t rows_dropped = original_n - static_cast<int64_t>(values.size());

    // Slice the entire table to drop the first `rows_dropped` rows
    // (keeping all columns aligned), then replace the value column
    // with the differenced float values.
    auto sliced = input->Slice(rows_dropped);

    ReportProgress(progress_callback_, "write_output",
                   "Writing differenced output", 0.90f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(original_n),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        ReplaceColumnWithFloat(sliced, col_idx, values,
                               static_cast<int64_t>(values.size())));

    spdlog::info("Differencing: lag={}, order={}, {} -> {} rows (dropped {}), col='{}'",
                 lag_, order_, original_n, values.size(), rows_dropped, value_col_);
    ReportProgress(progress_callback_, "complete",
                   "Differencing complete", 1.0f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(original_n),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    return out_table;
}

} // namespace cyxwiz
