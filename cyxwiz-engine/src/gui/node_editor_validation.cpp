// ========================================
// Graph Validation Module
// ========================================
// This file contains graph validation functions for the NodeEditor.
// Includes cycle detection, reachability analysis, and graph integrity checks.

#include "node_editor.h"
#include "node_editor_shape_inference.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <spdlog/spdlog.h>
#include <set>
#include <queue>
#include <functional>
#include <map>

namespace gui {

bool NodeEditor::ValidateGraph(std::string& error_message) {
    if (nodes_.empty()) {
        error_message = "Graph is empty. Add nodes first.";
        return false;
    }

    // Check for Input node
    if (!HasInputNode()) {
        error_message = "Graph must have at least one Input node.";
        return false;
    }

    // Check for Output node
    if (!HasOutputNode()) {
        error_message = "Graph must have at least one Output node.";
        return false;
    }

    // Check for cycles
    if (HasCycle()) {
        error_message = "Graph contains cycles. Neural networks must be acyclic (DAG).";
        return false;
    }

    // Check that all nodes are reachable from input
    if (!AllNodesReachable()) {
        error_message = "Some nodes are not connected to the network. All nodes must be reachable from input nodes.";
        return false;
    }

    // Check variadic pin requirements
    for (const auto& node : nodes_) {
        for (const auto& pin : node.inputs) {
            int conn_count = GetConnectionCount(pin.id);

            // Check minimum connections for variadic pins
            if (pin.is_variadic && conn_count < pin.min_connections) {
                error_message = "Node '" + node.name + "' requires at least " +
                    std::to_string(pin.min_connections) + " inputs on pin '" +
                    pin.name + "' (has " + std::to_string(conn_count) + ").";
                return false;
            }

            // Check required pins have at least one connection
            if (pin.is_required && !pin.is_variadic && conn_count == 0) {
                error_message = "Node '" + node.name + "' has required input '" +
                    pin.name + "' that is not connected.";
                return false;
            }
        }
    }

    // Compute shape validation warnings (non-blocking)
    validation_warnings_ = ValidateShapes();

    return true;
}

std::vector<ValidationWarning> NodeEditor::ValidateShapes() {
    std::vector<ValidationWarning> warnings;

    // First, compute shapes for all nodes
    if (shape_inference_) {
        shape_inference_->ComputeAllShapes(nodes_, links_);
    } else {
        spdlog::error("ValidateShapes: shape_inference_ is null");
        return warnings;
    }

    // Check each link for shape mismatches
    for (const auto& link : links_) {
        // Find source and target nodes
        MLNode* from_node = nullptr;
        MLNode* to_node = nullptr;

        for (auto& node : nodes_) {
            for (const auto& pin : node.outputs) {
                if (pin.id == link.from_pin) {
                    from_node = &node;
                    break;
                }
            }
            for (const auto& pin : node.inputs) {
                if (pin.id == link.to_pin) {
                    to_node = &node;
                    break;
                }
            }
        }

        if (!from_node || !to_node) continue;

        // Check for Conv2D/MaxPool2D (4D) → Dense (2D) without Flatten
        if (Is4DOutputNode(from_node->type) && Expects2DInput(to_node->type)) {
            // Check if there's already a Flatten node between them
            bool has_flatten_between = false;

            // Simple check: see if target node has Flatten as immediate predecessor
            // (More sophisticated: check full path, but this is good enough for now)
            for (const auto& other_link : links_) {
                if (other_link.to_node == to_node->id) {
                    for (const auto& node : nodes_) {
                        if (node.id == other_link.from_node && node.type == NodeType::Flatten) {
                            has_flatten_between = true;
                            break;
                        }
                    }
                }
            }

            if (!has_flatten_between) {
                warnings.push_back({
                    .node_id = to_node->id,
                    .severity = ValidationSeverity::Warning,
                    .message = "Dense layer expects 2D input but receiving 4D from " + from_node->name + ". Insert Flatten node.",
                    .suggested_fix = "Auto-insert Flatten",
                    .has_auto_fix = true,
                    .from_node_id = from_node->id,
                    .to_node_id = to_node->id
                });
            }
        }
    }

    return warnings;
}

bool NodeEditor::Is4DOutputNode(NodeType type) const {
    // Nodes that output 4D tensors: [batch, height, width, channels]
    return type == NodeType::Conv2D ||
           type == NodeType::Conv1D ||
           type == NodeType::Conv3D ||
           type == NodeType::DepthwiseConv2D ||
           type == NodeType::MaxPool2D ||
           type == NodeType::AvgPool2D;
}

bool NodeEditor::Expects2DInput(NodeType type) const {
    // Nodes that expect 2D tensors: [batch, features]
    return type == NodeType::Dense;
}

bool NodeEditor::IsGraphValid() const {
    // Quick check for training readiness
    // Need: DatasetInput node, at least one model layer, and a loss node
    if (nodes_.empty()) return false;

    bool has_dataset_input = false;
    bool has_loss = false;
    bool has_model_layer = false;

    for (const auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) has_dataset_input = true;
        if (node.type == NodeType::CrossEntropyLoss || node.type == NodeType::MSELoss) has_loss = true;
        if (node.type == NodeType::Dense || node.type == NodeType::Conv2D) has_model_layer = true;
    }

    // For training we need: dataset input, model layers, and loss
    return has_dataset_input && has_model_layer && has_loss;
}

void NodeEditor::ResolveDynamicPins(int node_id) {
    // Find the node
    MLNode* node = nullptr;
    for (auto& n : nodes_) {
        if (n.id == node_id) { node = &n; break; }
    }
    if (!node || !node->has_dynamic_pins || node->plugin_qualified_name.empty()) return;

    // Check if trigger value actually changed
    const std::string& trigger = node->dynamic_pin_trigger;
    std::string trigger_value;
    if (!trigger.empty()) {
        auto it = node->parameters.find(trigger);
        if (it != node->parameters.end()) trigger_value = it->second;
    }
    // Skip if trigger value hasn't changed, UNLESS both are empty and pins
    // haven't been resolved yet (plugin may have a loaded model to use as fallback)
    bool already_resolved = !node->resolved_config.empty();
    if (trigger_value == node->resolved_config && already_resolved) return;  // No change

    SaveUndoState();

    // Call plugin to resolve new pins
    auto result = cyxwiz::plugin::PluginNodeRegistry::Instance().ResolveDynamicPins(
        node->plugin_qualified_name, node->parameters);

    if (result.pins.empty()) {
        spdlog::warn("ResolveDynamicPins: plugin returned empty pins for {}", node->plugin_qualified_name);
        return;
    }

    // Save existing connections by pin name so we can restore matching ones
    struct SavedLink {
        std::string pin_name;
        bool is_input;
        int other_node;
        int other_pin;
        LinkType type;
    };
    std::vector<SavedLink> saved;

    // Collect all pin IDs belonging to this node
    std::map<int, std::string> pin_id_to_name;
    for (const auto& p : node->inputs) pin_id_to_name[p.id] = p.name;
    for (const auto& p : node->outputs) pin_id_to_name[p.id] = p.name;

    // Save links connected to this node's pins
    for (const auto& link : links_) {
        auto from_it = pin_id_to_name.find(link.from_pin);
        if (from_it != pin_id_to_name.end()) {
            saved.push_back({from_it->second, false, link.to_node, link.to_pin, link.type});
        }
        auto to_it = pin_id_to_name.find(link.to_pin);
        if (to_it != pin_id_to_name.end()) {
            saved.push_back({to_it->second, true, link.from_node, link.from_pin, link.type});
        }
    }

    // Remove all links connected to this node
    links_.erase(std::remove_if(links_.begin(), links_.end(), [&](const NodeLink& l) {
        return pin_id_to_name.count(l.from_pin) || pin_id_to_name.count(l.to_pin);
    }), links_.end());

    // Rebuild pins from plugin result
    node->inputs.clear();
    node->outputs.clear();

    std::map<std::string, int> new_pin_ids;  // pin name -> new pin id
    for (const auto& pin_info : result.pins) {
        NodePin p;
        p.id = next_pin_id_++;
        p.type = PinType::Tensor;
        p.name = pin_info.name;
        p.is_input = pin_info.is_input;
        if (p.is_input) node->inputs.push_back(p);
        else node->outputs.push_back(p);
        new_pin_ids[pin_info.name] = p.id;
    }

    // Store metadata in parameters
    for (const auto& [k, v] : result.metadata) {
        node->parameters["_meta_" + k] = v;
    }

    // Restore connections where pin names match
    for (const auto& s : saved) {
        auto it = new_pin_ids.find(s.pin_name);
        if (it == new_pin_ids.end()) continue;
        int new_pin = it->second;

        NodeLink link;
        link.id = next_link_id_++;
        link.type = s.type;
        if (s.is_input) {
            link.from_node = s.other_node;
            link.from_pin = s.other_pin;
            link.to_node = node_id;
            link.to_pin = new_pin;
        } else {
            link.from_node = node_id;
            link.from_pin = new_pin;
            link.to_node = s.other_node;
            link.to_pin = s.other_pin;
        }
        links_.push_back(link);
    }

    // Use the actual resolved path (may come from plugin fallback) for change detection
    auto meta_path_it = node->parameters.find("_meta_loaded_path");
    if (trigger_value.empty() && meta_path_it != node->parameters.end() && !meta_path_it->second.empty()) {
        node->resolved_config = "__env_lib:" + meta_path_it->second;
    } else {
        node->resolved_config = trigger_value;
    }

    spdlog::info("ResolveDynamicPins: node {} rebuilt with {} inputs, {} outputs",
                 node->name, node->inputs.size(), node->outputs.size());
}

void NodeEditor::UpdateDatasetNodeName(const std::string& dataset_name) {
    // Find the first DatasetInput node and update its name
    for (auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            // Use dataset name if provided, otherwise default to "DataInput"
            if (dataset_name.empty()) {
                node.name = "DataInput";
            } else {
                node.name = dataset_name;
            }
            node.parameters["dataset_name"] = dataset_name;
            spdlog::info("Updated DatasetInput node name to: {}", node.name);
            break;
        }
    }
}

bool NodeEditor::HasCycle() {
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // Track visited nodes and recursion stack for DFS
    std::set<int> visited;
    std::set<int> rec_stack;

    // DFS function to detect cycle
    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        visited.insert(node_id);
        rec_stack.insert(node_id);

        // Visit all neighbors
        if (adj.find(node_id) != adj.end()) {
            for (int neighbor : adj[node_id]) {
                if (!visited.count(neighbor)) {
                    // Recursively visit unvisited neighbors
                    if (dfs(neighbor)) {
                        return true;  // Cycle found in subtree
                    }
                } else if (rec_stack.count(neighbor)) {
                    // Found a back edge (cycle detected)
                    spdlog::warn("Cycle detected: node {} -> node {}", node_id, neighbor);
                    return true;
                }
            }
        }

        // Remove from recursion stack before returning
        rec_stack.erase(node_id);
        return false;
    };

    // Check each unvisited node (handles disconnected components)
    for (const auto& node : nodes_) {
        if (!visited.count(node.id)) {
            if (dfs(node.id)) {
                return true;  // Cycle found
            }
        }
    }

    return false;  // No cycles found
}

bool NodeEditor::AllNodesReachable() {
    if (nodes_.empty()) return true;

    // Find all DatasetInput nodes
    std::vector<int> input_nodes;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            input_nodes.push_back(node.id);
        }
    }

    if (input_nodes.empty()) return false;

    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // BFS from all input nodes to find reachable nodes
    std::set<int> reachable;
    std::queue<int> queue;

    // Start from all input nodes
    for (int input_id : input_nodes) {
        queue.push(input_id);
        reachable.insert(input_id);
    }

    // Perform BFS
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        // Visit all neighbors
        if (adj.find(current) != adj.end()) {
            for (int neighbor : adj[current]) {
                if (!reachable.count(neighbor)) {
                    reachable.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
    }

    // Check if all nodes are reachable
    bool all_reachable = (reachable.size() == nodes_.size());

    if (!all_reachable) {
        // Log which nodes are unreachable for debugging
        for (const auto& node : nodes_) {
            if (!reachable.count(node.id)) {
                spdlog::warn("Node {} ('{}') is not reachable from input nodes", node.id, node.name);
            }
        }
    }

    return all_reachable;
}

bool NodeEditor::HasInputNode() {
    for (const auto& node : nodes_) {
        // DatasetInput is the valid input source for the graph
        if (node.type == NodeType::DatasetInput) {
            return true;
        }
    }
    return false;
}

bool NodeEditor::HasOutputNode() {
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Output) {
            return true;
        }
    }
    return false;
}

}  // namespace gui
