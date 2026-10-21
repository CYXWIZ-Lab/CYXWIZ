#pragma once

#include "node_editor.h"

#include <optional>
#include <span>
#include <string_view>

namespace gui {

struct NodeTypeImportName {
    std::string_view name;
    NodeType node_type = NodeType::Unknown;
    bool legacy_import_compatibility_only = false;
};

std::span<const NodeTypeImportName> GetNodeTypeImportNames();

std::optional<NodeType> ResolveNodeTypeImportName(std::string_view name);

} // namespace gui
