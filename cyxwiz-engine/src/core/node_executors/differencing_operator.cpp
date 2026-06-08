#include "differencing_operator.h"
#include "ts_column_utils.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

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
    if (!input) {
        return arrow::Status::Invalid("Differencing: input table is null");
    }

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

    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        ReplaceColumnWithFloat(sliced, col_idx, values,
                               static_cast<int64_t>(values.size())));

    spdlog::info("Differencing: lag={}, order={}, {} -> {} rows (dropped {}), col='{}'",
                 lag_, order_, original_n, values.size(), rows_dropped, value_col_);
    return out_table;
}

} // namespace cyxwiz
