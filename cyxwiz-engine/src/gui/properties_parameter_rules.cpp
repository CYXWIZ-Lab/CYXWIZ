#include "properties_parameter_rules.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>

namespace gui::properties_rules {

bool TryParseDoubleStrict(const std::string& text, double& value) {
    if (text.empty()) {
        return false;
    }

    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && end && *end == '\0' && std::isfinite(value);
}

bool TryParseIntStrict(const std::string& text, int& value) {
    if (text.empty()) {
        return false;
    }

    char* end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || !end || *end != '\0') {
        return false;
    }
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

NumericRange ParseNumericRange(const std::string& validation) {
    NumericRange range;
    if (validation.empty()) {
        return range;
    }

    const size_t dash_pos = validation.find('-', 1);
    if (dash_pos == std::string::npos || dash_pos + 1 >= validation.size()) {
        return range;
    }

    double min_value = 0.0;
    double max_value = 0.0;
    if (!TryParseDoubleStrict(validation.substr(0, dash_pos), min_value) ||
        !TryParseDoubleStrict(validation.substr(dash_pos + 1), max_value) ||
        min_value > max_value) {
        return range;
    }

    range.min_value = min_value;
    range.max_value = max_value;
    range.has_range = true;
    return range;
}

namespace {

bool IsDialogOwnedNode(NodeType type) {
    switch (type) {
        case NodeType::DataInput:
        case NodeType::DataOutput:
            return true;
        default:
            return false;
    }
}

bool IsInternalParameterName(const std::string& name) {
    if (!name.empty() && name[0] == '_') {
        return true;
    }

    static const std::set<std::string> internal_params = {
        "configured",
        "data_loaded",
        "loaded_backend",
        "loaded_cols",
        "loaded_memory_bytes",
        "loaded_rows",
        "audit_errors",
        "audit_warnings",
        "audit_status",
        "file_category",
        "source_type"
    };
    return internal_params.count(name) > 0;
}

} // namespace

bool ShouldHideGenericParameter(NodeType type, const cyxwiz::ParameterDefinition& param) {
    if (IsInternalParameterName(param.name)) {
        return true;
    }

    if (IsDialogOwnedNode(type) &&
        (param.name == "dataset_name" ||
         param.name == "file_path" ||
         param.name == "folder_path" ||
         param.type == "file")) {
        return true;
    }

    if ((type == NodeType::LSTM ||
         type == NodeType::GRU ||
         type == NodeType::RNN) &&
        param.name == "input_size") {
        return true;
    }

    return false;
}

bool ValidateParameter(
    const std::string& value,
    const cyxwiz::ParameterDefinition& param,
    std::string& error) {
    if (value.empty()) {
        if (param.required) {
            error = "Required";
            return false;
        }
        return true;
    }

    if (param.type == "int") {
        int v = 0;
        if (!TryParseIntStrict(value, v)) {
            error = "Invalid integer";
            return false;
        }

        NumericRange range = ParseNumericRange(param.validation);
        if (range.has_range) {
            if (v < static_cast<int>(range.min_value)) {
                error = "Value must be >= " + std::to_string(static_cast<int>(range.min_value));
                return false;
            }
            if (v > static_cast<int>(range.max_value)) {
                error = "Value must be <= " + std::to_string(static_cast<int>(range.max_value));
                return false;
            }
        }
    } else if (param.type == "float") {
        double v = 0.0;
        if (!TryParseDoubleStrict(value, v)) {
            error = "Invalid number";
            return false;
        }

        NumericRange range = ParseNumericRange(param.validation);
        if (range.has_range) {
            if (v < range.min_value) {
                error = "Value must be >= " + std::to_string(range.min_value);
                return false;
            }
            if (v > range.max_value) {
                error = "Value must be <= " + std::to_string(range.max_value);
                return false;
            }
        }
    } else if ((param.type == "enum" || param.type == "dropdown") && !param.enum_values.empty()) {
        bool found = false;
        for (const auto& ev : param.enum_values) {
            if (ev == value) {
                found = true;
                break;
            }
        }
        if (!found) {
            error = "Invalid option";
            return false;
        }
    }

    return true;
}

} // namespace gui::properties_rules
