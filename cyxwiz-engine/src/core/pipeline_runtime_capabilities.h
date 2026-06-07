#pragma once

#include "../gui/node_editor.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct PipelineOperatorRuntimeCapability {
    const char* legacy_type_name;
    gui::NodeType node_type;
};

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities();

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name);

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name);

} // namespace cyxwiz
