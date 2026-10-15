#pragma once

#include "../core/node_metadata.h"
#include "node_editor.h"
#include <string>
#include <utility>
#include <vector>

namespace gui::properties_rules {

struct NumericRange {
    double min_value = 0.0;
    double max_value = 0.0;
    bool has_range = false;
};

bool TryParseDoubleStrict(const std::string& text, double& value);
bool TryParseIntStrict(const std::string& text, int& value);
NumericRange ParseNumericRange(const std::string& validation);
bool ShouldHideGenericParameter(const MLNode& node,
                                const cyxwiz::ParameterDefinition& param);
bool ShouldUseSaveFileDialog(const MLNode& node,
                             const cyxwiz::ParameterDefinition& param);
// Open-dialog filters ({name, "ext1,ext2"}) for a file parameter: its
// validation patterns ("*.csv;*.parquet") as "Supported files", then All Files.
std::vector<std::pair<std::string, std::string>> FileParameterFilters(
    const cyxwiz::ParameterDefinition& param);
bool ValidateParameter(
    const std::string& value,
    const cyxwiz::ParameterDefinition& param,
    std::string& error);

} // namespace gui::properties_rules
