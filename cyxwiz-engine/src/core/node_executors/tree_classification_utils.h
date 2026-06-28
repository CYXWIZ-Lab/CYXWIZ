#pragma once

#include <arrow/api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

std::string TrimAscii(const std::string& value);
std::string ToLowerAscii(std::string value);

bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   int& out,
                   const std::string& op_name,
                   std::string& error);

arrow::Status ReadClassificationLabels(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& target_col,
    const std::string& op_name,
    std::vector<int>& labels,
    std::vector<std::string>& class_labels,
    bool& numeric_labels);

arrow::Result<std::shared_ptr<arrow::Table>> AppendClassificationPredictions(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& prediction_col,
    const std::vector<std::string>& class_labels,
    bool numeric_labels,
    const std::vector<int>& predicted_classes,
    const std::string& op_name);

} // namespace cyxwiz
