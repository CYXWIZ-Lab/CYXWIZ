#pragma once

#include "../core/node_metadata.h"
#include "node_editor.h"
#include <string>

namespace gui::properties_rules {

struct NumericRange {
    double min_value = 0.0;
    double max_value = 0.0;
    bool has_range = false;
};

bool TryParseDoubleStrict(const std::string& text, double& value);
bool TryParseIntStrict(const std::string& text, int& value);
NumericRange ParseNumericRange(const std::string& validation);
bool ShouldHideGenericParameter(NodeType type, const cyxwiz::ParameterDefinition& param);
bool ValidateParameter(
    const std::string& value,
    const cyxwiz::ParameterDefinition& param,
    std::string& error);

} // namespace gui::properties_rules
