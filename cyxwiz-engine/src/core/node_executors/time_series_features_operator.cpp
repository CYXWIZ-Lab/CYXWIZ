#include "time_series_features_operator.h"
#include "../profiler_trace.h"
#include "ts_column_utils.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Supported rolling-aggregation kinds. Kept as a typed enum so the
// hot loop doesn't string-compare on every window step.
enum class RollingAgg { Mean, Std, Min, Max, Median };

bool ParseAggKind(const std::string& s, RollingAgg& out,
                  std::string& error) {
    if (s == "mean")   { out = RollingAgg::Mean;   return true; }
    if (s == "std")    { out = RollingAgg::Std;    return true; }
    if (s == "min")    { out = RollingAgg::Min;    return true; }
    if (s == "max")    { out = RollingAgg::Max;    return true; }
    if (s == "median") { out = RollingAgg::Median; return true; }
    error = "TimeSeriesFeatures: rolling_aggregations '" + s +
            "' is not one of mean/std/min/max/median";
    return false;
}

float ComputeRolling(const std::vector<float>& values, int64_t i,
                     int w, RollingAgg agg, std::vector<float>& scratch) {
    // Window is [i - w + 1, i], inclusive.
    const int64_t start = i - w + 1;
    scratch.assign(values.begin() + start, values.begin() + i + 1);
    switch (agg) {
        case RollingAgg::Mean: {
            double sum = std::accumulate(scratch.begin(), scratch.end(), 0.0);
            return static_cast<float>(sum / w);
        }
        case RollingAgg::Std: {
            double sum = std::accumulate(scratch.begin(), scratch.end(), 0.0);
            double mean = sum / w;
            double sq = 0.0;
            for (float v : scratch) sq += (v - mean) * (v - mean);
            return static_cast<float>(std::sqrt(sq / w));
        }
        case RollingAgg::Min: {
            return *std::min_element(scratch.begin(), scratch.end());
        }
        case RollingAgg::Max: {
            return *std::max_element(scratch.begin(), scratch.end());
        }
        case RollingAgg::Median: {
            std::sort(scratch.begin(), scratch.end());
            const size_t n = scratch.size();
            return (n % 2 == 0)
                ? 0.5f * (scratch[n / 2 - 1] + scratch[n / 2])
                : scratch[n / 2];
        }
    }
    return 0.0f;  // unreachable
}

// Parse a comma-separated integer list. Empty string -> empty vector.
// Non-numeric tokens or negative values populate `error` and return false.
bool ParseIntList(const std::string& s, std::vector<int>& out,
                  const char* key, std::string& error) {
    out.clear();
    if (s.empty()) return true;

    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);
        if (token.empty()) continue;

        int v;
        try { v = std::stoi(token); }
        catch (...) {
            error = std::string("TimeSeriesFeatures: '") + key +
                    "' has invalid integer token '" + token + "'";
            return false;
        }
        if (v < 1) {
            error = std::string("TimeSeriesFeatures: '") + key +
                    "' values must be >= 1 (got " + std::to_string(v) + ")";
            return false;
        }
        out.push_back(v);
    }
    return true;
}

} // namespace

bool TimeSeriesFeaturesOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    value_col_.clear();
    lags_.clear();
    rolling_windows_.clear();
    rolling_aggregations_.clear();

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "TimeSeriesFeatures: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;

    auto lag_it = params.find("lag_values");
    const std::string lag_str = (lag_it != params.end()) ? lag_it->second : "";
    if (!ParseIntList(lag_str, lags_, "lag_values", error)) return false;

    auto roll_it = params.find("rolling_windows");
    const std::string roll_str = (roll_it != params.end()) ? roll_it->second : "";
    if (!ParseIntList(roll_str, rolling_windows_, "rolling_windows", error)) return false;

    // Parse rolling aggregations. Default keeps the old mean-only
    // behavior so existing graphs continue to emit `{col}_roll_{w}_mean`.
    auto agg_it = params.find("rolling_aggregations");
    const std::string agg_str = (agg_it != params.end()) ? agg_it->second : "";
    if (agg_str.empty()) {
        rolling_aggregations_.push_back("mean");
    } else {
        std::stringstream ss(agg_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start == std::string::npos) continue;
            token = token.substr(start, end - start + 1);
            if (token.empty()) continue;
            // Validate each kind at Configure time so bad tokens surface
            // at compile-gate instead of mid-pipeline.
            RollingAgg k;
            if (!ParseAggKind(token, k, error)) return false;
            rolling_aggregations_.push_back(token);
        }
        if (rolling_aggregations_.empty()) {
            rolling_aggregations_.push_back("mean");
        }
    }

    if (lags_.empty() && rolling_windows_.empty()) {
        error = "TimeSeriesFeatures: at least one of 'lag_values' or "
                "'rolling_windows' must be non-empty";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesFeaturesOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesFeatures Materializer");

    if (!input) {
        return arrow::Status::Invalid("TimeSeriesFeatures: input table is null");
    }

    auto report_progress = [&](std::string stage,
                               std::string message,
                               double progress,
                               uint64_t processed = 0,
                               uint64_t total = 0,
                               uint64_t memory = 0) {
        if (!progress_callback_) {
            return;
        }
        PipelineOperatorProgress event;
        event.stage = std::move(stage);
        event.message = std::move(message);
        event.progress = static_cast<float>(progress);
        event.processed_items = processed;
        event.total_items = total;
        event.estimated_memory_bytes = memory;
        progress_callback_(event);
    };

    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "TimeSeriesFeatures: column '" + value_col_ + "' not found");
    }

    report_progress("Reading value column",
                    "Reading time-series feature source column '" +
                    value_col_ + "'",
                    0.05);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "TimeSeriesFeatures: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    const int64_t n = static_cast<int64_t>(values.size());
    report_progress("Value column ready",
                    "Read " + std::to_string(n) +
                    " source rows for feature engineering",
                    0.15,
                    static_cast<uint64_t>(n),
                    static_cast<uint64_t>(n));

    // Rows to drop = max needed history.
    // lag_k needs index i - k >= 0, so i >= k    → drops first `max_lag` rows.
    // roll_w needs indices [i - w + 1, i] >= 0, so i >= w - 1 → drops first `max_window - 1` rows.
    int max_lag = lags_.empty() ? 0 : *std::max_element(lags_.begin(), lags_.end());
    int max_window = rolling_windows_.empty()
        ? 0 : *std::max_element(rolling_windows_.begin(), rolling_windows_.end());
    int64_t rows_to_drop = std::max(
        static_cast<int64_t>(max_lag),
        static_cast<int64_t>(max_window - 1));
    const int64_t out_rows = n - rows_to_drop;

    if (out_rows <= 0) {
        return arrow::Status::Invalid(
            "TimeSeriesFeatures: not enough rows (" + std::to_string(n) +
            ") for max_lag=" + std::to_string(max_lag) +
            " / max_window=" + std::to_string(max_window));
    }

    const size_t lag_columns = lags_.size();
    const size_t rolling_columns =
        rolling_windows_.size() * rolling_aggregations_.size();
    const size_t engineered_columns = lag_columns + rolling_columns;
    const uint64_t estimated_engineered_bytes =
        static_cast<uint64_t>(out_rows) *
        static_cast<uint64_t>(engineered_columns) *
        sizeof(float);
    report_progress("Planning features",
                    "Planning " + std::to_string(engineered_columns) +
                    " engineered columns over " +
                    std::to_string(out_rows) + " output rows",
                    0.25,
                    0,
                    static_cast<uint64_t>(engineered_columns),
                    estimated_engineered_bytes);

    // Slice the existing table to drop the first `rows_to_drop` rows.
    auto sliced = input->Slice(rows_to_drop);

    // Build new columns per lag and rolling window.
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::Field>> new_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_chunks;

    size_t completed_columns = 0;
    for (int lag : lags_) {
        report_progress("Building lag features",
                        "Computing lag feature " + std::to_string(lag),
                        0.30 + (0.55 * static_cast<double>(completed_columns) /
                                static_cast<double>(engineered_columns)),
                        static_cast<uint64_t>(completed_columns),
                        static_cast<uint64_t>(engineered_columns),
                        estimated_engineered_bytes);
        arrow::FloatBuilder builder(pool);
        ARROW_RETURN_NOT_OK(builder.Reserve(out_rows));
        for (int64_t i = rows_to_drop; i < n; ++i) {
            ARROW_RETURN_NOT_OK(builder.Append(values[i - lag]));
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        const std::string col_name = value_col_ + "_lag_" + std::to_string(lag);
        new_fields.push_back(arrow::field(col_name, arrow::float32()));
        new_chunks.push_back(std::make_shared<arrow::ChunkedArray>(arr));
        ++completed_columns;
    }

    // For each window × aggregation combination, emit one new column.
    std::vector<float> scratch;
    for (int w : rolling_windows_) {
        for (const auto& agg_name : rolling_aggregations_) {
            RollingAgg agg;
            std::string unused;
            ParseAggKind(agg_name, agg, unused);  // already validated in Configure
            report_progress("Building rolling features",
                            "Computing rolling " + agg_name +
                            " feature for window " + std::to_string(w),
                            0.30 + (0.55 * static_cast<double>(completed_columns) /
                                    static_cast<double>(engineered_columns)),
                            static_cast<uint64_t>(completed_columns),
                            static_cast<uint64_t>(engineered_columns),
                            estimated_engineered_bytes);
            arrow::FloatBuilder builder(pool);
            ARROW_RETURN_NOT_OK(builder.Reserve(out_rows));
            for (int64_t i = rows_to_drop; i < n; ++i) {
                ARROW_RETURN_NOT_OK(
                    builder.Append(ComputeRolling(values, i, w, agg, scratch)));
            }
            std::shared_ptr<arrow::Array> arr;
            ARROW_RETURN_NOT_OK(builder.Finish(&arr));
            const std::string col_name = value_col_ + "_roll_" +
                std::to_string(w) + "_" + agg_name;
            new_fields.push_back(arrow::field(col_name, arrow::float32()));
            new_chunks.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            ++completed_columns;
        }
    }

    // Append all new columns via AddColumn, one at a time.
    report_progress("Appending columns",
                    "Appending engineered feature columns to Arrow table",
                    0.90,
                    0,
                    static_cast<uint64_t>(new_fields.size()),
                    estimated_engineered_bytes);
    auto out_table = sliced;
    for (size_t i = 0; i < new_fields.size(); ++i) {
        ARROW_ASSIGN_OR_RAISE(
            out_table,
            out_table->AddColumn(out_table->num_columns(),
                                 new_fields[i], new_chunks[i]));
        report_progress("Appending columns",
                        "Appending engineered feature columns to Arrow table",
                        0.90 + (0.08 * static_cast<double>(i + 1) /
                                static_cast<double>(new_fields.size())),
                        static_cast<uint64_t>(i + 1),
                        static_cast<uint64_t>(new_fields.size()),
                        estimated_engineered_bytes);
    }

    spdlog::info("TimeSeriesFeatures: {} rows -> {} rows (dropped {}), "
                 "added {} lag + {} rolling cols ({} windows x {} aggs) from '{}'",
                 n, out_rows, rows_to_drop,
                 lags_.size(),
                 rolling_windows_.size() * rolling_aggregations_.size(),
                 rolling_windows_.size(), rolling_aggregations_.size(),
                 value_col_);
    report_progress("Complete",
                    "TimeSeriesFeatures materialization complete",
                    1.0,
                    static_cast<uint64_t>(out_rows),
                    static_cast<uint64_t>(out_rows),
                    estimated_engineered_bytes);
    return out_table;
}

} // namespace cyxwiz
