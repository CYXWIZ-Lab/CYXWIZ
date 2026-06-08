#include "log_transform_operator.h"
#include "ts_column_utils.h"

#include <spdlog/spdlog.h>
#include <cmath>

namespace cyxwiz {

bool LogTransformOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    value_col_.clear();

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "LogTransform: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
LogTransformOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("LogTransform: input table is null");
    }

    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "LogTransform: column '" + value_col_ + "' not found");
    }

    int col_idx = input->schema()->GetFieldIndex(value_col_);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "LogTransform: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    for (float& v : values) {
        v = std::log1p(v);
    }

    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        ReplaceColumnWithFloat(input, col_idx, values,
                               static_cast<int64_t>(values.size())));

    spdlog::info("LogTransform: applied log1p to '{}' ({} values)",
                 value_col_, values.size());
    return out_table;
}

} // namespace cyxwiz
