#include "tree_classification_utils.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace cyxwiz {

std::string TrimAscii(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   int& out,
                   const std::string& op_name,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return true;
    }
    try {
        size_t parsed = 0;
        const int value = std::stoi(it->second, &parsed);
        if (parsed != it->second.size()) {
            throw std::runtime_error("trailing characters");
        }
        out = value;
        return true;
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid integer: " +
                it->second;
        return false;
    }
}

namespace {

bool IsNumericType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) return false;
    switch (type->id()) {
        case arrow::Type::DOUBLE:
        case arrow::Type::FLOAT:
        case arrow::Type::INT64:
        case arrow::Type::INT32:
        case arrow::Type::INT16:
        case arrow::Type::INT8:
        case arrow::Type::UINT64:
        case arrow::Type::UINT32:
        case arrow::Type::UINT16:
        case arrow::Type::UINT8:
            return true;
        default:
            return false;
    }
}

bool IsStringType(const std::shared_ptr<arrow::DataType>& type) {
    return type && (type->id() == arrow::Type::STRING ||
                    type->id() == arrow::Type::LARGE_STRING);
}

} // namespace

arrow::Status ReadClassificationLabels(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& target_col,
    const std::string& op_name,
    std::vector<int>& labels,
    std::vector<std::string>& class_labels,
    bool& numeric_labels) {
    auto column = table->GetColumnByName(target_col);
    if (!column) {
        return arrow::Status::KeyError(
            op_name + ": target column '" + target_col + "' not found");
    }
    if (column->num_chunks() == 0) {
        return arrow::Status::Invalid(op_name + ": target column has no chunks");
    }
    const auto type = column->type();
    numeric_labels = IsNumericType(type);
    if (!numeric_labels && !IsStringType(type)) {
        return arrow::Status::TypeError(
            op_name + ": target column must be numeric or string");
    }

    std::map<std::string, int> label_to_id;
    labels.clear();
    class_labels.clear();
    labels.reserve(static_cast<size_t>(table->num_rows()));

    const auto record_label = [&](const std::string& key) {
        auto it = label_to_id.find(key);
        if (it == label_to_id.end()) {
            const int id = static_cast<int>(class_labels.size());
            it = label_to_id.emplace(key, id).first;
            class_labels.push_back(key);
        }
        labels.push_back(it->second);
    };

    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto chunk = column->chunk(chunk_index);
        if (chunk->type_id() == arrow::Type::STRING) {
            const auto strings =
                std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t row = 0; row < strings->length(); ++row) {
                if (strings->IsNull(row)) {
                    return arrow::Status::Invalid(
                        op_name + ": target column contains null labels");
                }
                record_label(strings->GetString(row));
            }
            continue;
        }
        if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            const auto strings =
                std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t row = 0; row < strings->length(); ++row) {
                if (strings->IsNull(row)) {
                    return arrow::Status::Invalid(
                        op_name + ": target column contains null labels");
                }
                record_label(strings->GetString(row));
            }
            continue;
        }
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) {
                return arrow::Status::Invalid(
                    op_name + ": target column contains null labels");
            }
            ARROW_ASSIGN_OR_RAISE(auto scalar, chunk->GetScalar(row));
            record_label(scalar->ToString());
        }
    }

    if (class_labels.size() < 2) {
        return arrow::Status::Invalid(
            op_name + ": target column must contain at least two classes");
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendClassificationPredictions(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& prediction_col,
    const std::vector<std::string>& class_labels,
    bool numeric_labels,
    const std::vector<int>& predicted_classes,
    const std::string& op_name) {
    if (input->num_rows() != static_cast<int64_t>(predicted_classes.size())) {
        return arrow::Status::Invalid(
            op_name + ": prediction count does not match row count");
    }

    if (numeric_labels) {
        arrow::DoubleBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(input->num_rows()));
        for (int cls : predicted_classes) {
            const auto& raw = class_labels[static_cast<size_t>(cls)];
            try {
                ARROW_RETURN_NOT_OK(builder.Append(std::stod(raw)));
            } catch (...) {
                return arrow::Status::Invalid(
                    op_name + ": numeric label could not be restored: " + raw);
            }
        }
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return input->AddColumn(
            input->num_columns(),
            arrow::field(prediction_col, arrow::float64()),
            std::make_shared<arrow::ChunkedArray>(array));
    }

    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(input->num_rows()));
    for (int cls : predicted_classes) {
        ARROW_RETURN_NOT_OK(builder.Append(class_labels[static_cast<size_t>(cls)]));
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return input->AddColumn(
        input->num_columns(),
        arrow::field(prediction_col, arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(array));
}

} // namespace cyxwiz
