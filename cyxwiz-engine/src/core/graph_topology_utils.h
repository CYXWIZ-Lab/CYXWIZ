#pragma once

#include "../gui/node_editor.h"

#include <unordered_set>
#include <vector>

namespace cyxwiz {

std::unordered_set<int> CollectReachableNodeIds(
    int source_node_id,
    const std::vector<gui::NodeLink>& links);

std::unordered_set<int> CollectAncestorNodeIds(
    int sink_node_id,
    const std::vector<gui::NodeLink>& links);

bool HasOutgoingLink(
    int node_id,
    const std::vector<gui::NodeLink>& links);

const gui::MLNode* FindFirstReachableNodeOfType(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& reachable_node_ids,
    gui::NodeType type);

} // namespace cyxwiz
