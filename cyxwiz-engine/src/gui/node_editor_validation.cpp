// ========================================
// Graph Validation Module
// ========================================
// This file contains graph validation functions for the NodeEditor.
// Includes cycle detection, reachability analysis, and graph integrity checks.

#include "node_editor.h"
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

    return true;
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
