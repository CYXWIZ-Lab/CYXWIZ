#include "time_series_features_operator.h"
#include "ts_column_utils.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

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

    if (lags_.empty() && rolling_windows_.empty()) {
        error = "TimeSeriesFeatures: at least one of 'lag_values' or "
                "'rolling_windows' must be non-empty";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesFeaturesOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("TimeSeriesFeatures: input table is null");
    }

    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "TimeSeriesFeatures: column '" + value_col_ + "' not found");
    }

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "TimeSeriesFeatures: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    const int64_t n = static_cast<int64_t>(values.size());

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

    // Slice the existing table to drop the first `rows_to_drop` rows.
    auto sliced = input->Slice(rows_to_drop);

    // Build new columns per lag and rolling window.
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::Field>> new_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_chunks;

    for (int lag : lags_) {
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
    }

    for (int w : rolling_windows_) {
        arrow::FloatBuilder builder(pool);
        ARROW_RETURN_NOT_OK(builder.Reserve(out_rows));
        for (int64_t i = rows_to_drop; i < n; ++i) {
            double sum = 0.0;
            for (int k = 0; k < w; ++k) {
                sum += values[i - k];
            }
            ARROW_RETURN_NOT_OK(builder.Append(static_cast<float>(sum / w)));
        }
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builder.Finish(&arr));
        const std::string col_name = value_col_ + "_roll_" + std::to_string(w) + "_mean";
        new_fields.push_back(arrow::field(col_name, arrow::float32()));
        new_chunks.push_back(std::make_shared<arrow::ChunkedArray>(arr));
    }

    // Append all new columns via AddColumn, one at a time.
    auto out_table = sliced;
    for (size_t i = 0; i < new_fields.size(); ++i) {
        ARROW_ASSIGN_OR_RAISE(
            out_table,
            out_table->AddColumn(out_table->num_columns(),
                                 new_fields[i], new_chunks[i]));
    }

    spdlog::info("TimeSeriesFeatures: {} rows -> {} rows (dropped {}), "
                 "added {} lag + {} rolling cols from '{}'",
                 n, out_rows, rows_to_drop,
                 lags_.size(), rolling_windows_.size(), value_col_);
    return out_table;
}

} // namespace cyxwiz
