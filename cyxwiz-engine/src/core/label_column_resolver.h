#pragma once

#include <arrow/type.h>

#include <memory>
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

} // namespace cyxwiz
