#pragma once

#include "../gui/node_editor.h"

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

/**
 * Pin-preserving node record for the selected training path.
 *
 * This is intentionally passive in the first graph-runtime slice: existing
 * training still consumes TrainingConfiguration::layers, while future
 * GraphExecutableModel work can consume this plan without re-walking GUI nodes
 * or guessing pin roles.
 */
struct CompiledGraphNode {
    gui::NodeType type;
    int node_id = -1;
    std::string name;
    std::map<std::string, std::string> parameters;
    std::vector<int> input_pin_ids;
    std::vector<int> output_pin_ids;
};

struct CompiledGraphEdge {
    int from_node_id = -1;
    int from_pin_id = -1;
    int to_node_id = -1;
    int to_pin_id = -1;
};

struct CompiledGraphPlan {
    bool available = false;
    std::vector<CompiledGraphNode> nodes;
    std::vector<CompiledGraphEdge> edges;

    int data_node_id = -1;
    int data_pin_id = -1;
    int label_pin_id = -1;
    int prediction_pin_id = -1;
    int label_target_pin_id = -1;
    int loss_node_id = -1;
    int loss_output_pin_id = -1;
    int optimizer_node_id = -1;
};

CompiledGraphPlan BuildCompiledGraphPlan(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<int>& sorted_node_ids,
    const std::unordered_set<int>& training_path_ids,
    int data_node_id,
    int loss_node_id,
    int optimizer_node_id);

} // namespace cyxwiz
