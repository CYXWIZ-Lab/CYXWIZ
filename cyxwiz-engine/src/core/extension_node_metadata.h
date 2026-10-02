#pragma once

// One way to get the metadata of a node on the canvas (TOFIX125). The
// catalog (NodeMetadataRegistry) is keyed by NodeType, so every extension
// node would resolve to the generic PluginCustom entry there.

#include "graph_model.h"
#include "node_metadata.h"

#include <optional>

namespace cyxwiz {

// Built-in node: its catalog entry. Extension node: its descriptor's
// metadata. Extension node whose extension is not registered: a placeholder
// built from the node's own pins, badged "Not installed". Empty only for a
// built-in type the catalog does not know.
std::optional<NodeMetadata> ResolveNodeMetadata(const gui::MLNode& node);

}  // namespace cyxwiz
