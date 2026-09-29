#pragma once

// The node factory shared by the editor and headless hosts (TOFIX118 P2):
// each node type's pins and default parameters, so a saved graph loads into
// the same nodes, pins and links everywhere.

#include "graph_model.h"

#include <string>
#include <vector>

namespace gui {

NodeCategory GetNodeCategoryForType(NodeType type);

// A node with its pins; ids come from (and advance) the two cursors. For
// NodeType::PluginCustom, `name` is the extension type id.
MLNode CreateGraphNode(NodeType type, const std::string& name, int& next_node_id, int& next_pin_id);

// An extension node (TOFIX125) by its type id "<provider_id>:<TypeName>".
// Pins and default parameters come from ExtensionNodeRegistry. When the
// extension is not registered the node is marked extension_missing and gets
// one input and one output.
MLNode CreateExtensionGraphNode(const std::string& type_id, int& next_node_id, int& next_pin_id);

// Data Input / Data Split / Data Loader pins: the dataset.v2 contract, or the
// preserved legacy pins for an unmigrated graph.
void RebuildDataBoundaryPins(MLNode& node, bool legacy_contract, int& next_pin_id);

// A SQL step's named table inputs, from its parameters; returns removed pin ids.
std::vector<int> SyncSqlStepInputPins(MLNode& node, int& next_pin_id);

}  // namespace gui
