#pragma once

#include "graph_model.h"

#include <functional>
#include <string>
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

// Data Input nodes Train loads before it compiles (a reopened project keeps
// the settings, not the loaded data): they name a file or folder, feed a
// Data Loader, and `is_loaded` says their dataset_name is not loaded (an
// empty name is not loaded). HDF5 sources are left out; Apply does not load
// them.
std::vector<int> CollectDataInputsToLoad(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::function<bool(const std::string& dataset_name)>& is_loaded);

} // namespace cyxwiz
