#pragma once

#include <arrow/type.h>

#include <memory>
#include <cstddef>
#include <string>
#include <string_view>

namespace cyxwiz {

inline bool IsCommonLabelColumnName(std::string_view name) {
    return name == "label" || name == "Label" || name == "LABEL" ||
           name == "class" || name == "Class" || name == "CLASS" ||
           name == "target" || name == "Target" || name == "TARGET" ||
           name == "y" || name == "Y" || name == "digit" ||
           name == "category";
}

inline int FindCommonLabelColumnIndex(
    const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) return -1;
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (IsCommonLabelColumnName(schema->field(i)->name())) {
            return i;
        }
    }
    return -1;
}

inline int ResolveLabelColumnIndex(
    const std::shared_ptr<arrow::Schema>& schema,
    std::string_view requested_label = {}) {
    if (!schema) return -1;
    if (!requested_label.empty()) {
        return schema->GetFieldIndex(std::string(requested_label));
    }
    return FindCommonLabelColumnIndex(schema);
}

inline std::string ResolveLabelColumnName(
    const std::shared_ptr<arrow::Schema>& schema,
    std::string_view requested_label = {}) {
    const int index = ResolveLabelColumnIndex(schema, requested_label);
    return index >= 0 ? schema->field(index)->name() : std::string{};
}

inline bool IsNumericBatchFeatureType(
    const std::shared_ptr<arrow::DataType>& type) {
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

inline size_t CountNumericBatchFeatureColumns(
    const std::shared_ptr<arrow::Schema>& schema,
    int label_column_index = -1,
    std::string_view partition_column = {}) {
    if (!schema) return 0;
    size_t count = 0;
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (i == label_column_index) continue;
        const auto& field = schema->field(i);
        if (!field) continue;
        const std::string& name = field->name();
        if (name.rfind("__", 0) == 0 ||
            (!partition_column.empty() && name == partition_column)) {
            continue;
        }
        if (IsNumericBatchFeatureType(field->type())) {
            ++count;
        }
    }
    return count;
}

} // namespace cyxwiz
