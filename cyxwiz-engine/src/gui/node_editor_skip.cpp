// node_editor_skip.cpp
// Skip/Residual Connection Module for NodeEditor
//
// This module contains helper functions for creating and managing skip connections
// (residual connections) in the visual node editor. Skip connections are essential
// for architectures like ResNet, DenseNet, U-Net, and Transformers.

#include "node_editor.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <set>
#include <queue>

namespace gui {

// ========== Merge Node Detection ==========

bool NodeEditor::IsMergeNode(NodeType type) const {
    switch (type) {
        case NodeType::Add:
        case NodeType::Concatenate:
        case NodeType::Multiply:
        case NodeType::Average:
            return true;
        default:
            return false;
    }
}

// ========== Link Color Functions ==========

ImU32 NodeEditor::GetLinkColor(LinkType type) const {
    switch (type) {
        case LinkType::TensorFlow:
            return IM_COL32(200, 200, 200, 255);  // Light gray - standard
        case LinkType::ResidualSkip:
            return IM_COL32(255, 165, 0, 255);    // Orange - residual/skip
        case LinkType::DenseSkip:
            return IM_COL32(0, 191, 255, 255);    // Deep sky blue - dense skip
        case LinkType::AttentionFlow:
            return IM_COL32(186, 85, 211, 255);   // Medium orchid - attention
        case LinkType::GradientFlow:
            return IM_COL32(255, 99, 71, 255);    // Tomato red - gradients
        case LinkType::ParameterFlow:
            return IM_COL32(144, 238, 144, 255);  // Light green - parameters
        case LinkType::LossFlow:
            return IM_COL32(220, 20, 60, 255);    // Crimson - loss
        default:
            return IM_COL32(200, 200, 200, 255);
    }
}

ImU32 NodeEditor::GetLinkHoverColor(LinkType type) const {
    // Return brighter version of the link color
    ImU32 base = GetLinkColor(type);
    int r = (base & 0xFF);
    int g = ((base >> 8) & 0xFF);
    int b = ((base >> 16) & 0xFF);

    // Brighten by 20%
    r = std::min(255, r + 50);
    g = std::min(255, g + 50);
    b = std::min(255, b + 50);

    return IM_COL32(r, g, b, 255);
}

// ========== Skip Connection Detection ==========

bool NodeEditor::IsSkipConnection(const NodeLink& link) const {
    // A skip connection bypasses at least one intermediate layer
    // We detect this by checking if there's a longer path between the same nodes

    const MLNode* from_node = FindNodeById(link.from_node);
    const MLNode* to_node = FindNodeById(link.to_node);
    if (!from_node || !to_node) return false;

    // Check if this link goes to a merge node (Add, Concatenate, etc.)
    bool is_merge_target = IsMergeNode(to_node->type);
    if (!is_merge_target) return false;

    // Count how many inputs the merge node has
    int input_count = GetConnectionCount(link.to_pin);
    if (input_count < 2) return false;

    // If there are multiple inputs to a merge node, one of them is likely a skip
    // The "skip" is typically the shorter path (directly from an earlier layer)

    // Build adjacency list for BFS
    std::map<int, std::vector<int>> adj;
    for (const auto& l : links_) {
        if (l.id != link.id) {  // Exclude the link we're checking
            adj[l.from_node].push_back(l.to_node);
        }
    }

    // BFS to find if there's an alternative path
    std::queue<int> queue;
    std::set<int> visited;
    queue.push(link.from_node);
    visited.insert(link.from_node);

    int path_length = 0;
    while (!queue.empty() && path_length < 10) {  // Limit search depth
        int level_size = queue.size();
        for (int i = 0; i < level_size; ++i) {
            int current = queue.front();
            queue.pop();

            if (current == link.to_node && path_length > 1) {
                // Found alternative path that's longer than direct connection
                return true;
            }

            if (adj.find(current) != adj.end()) {
                for (int neighbor : adj[current]) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }
        path_length++;
    }

    return false;
}

void NodeEditor::DetectSkipConnections() {
    // Auto-detect and mark skip connections based on topology
    for (auto& link : links_) {
        if (link.type == LinkType::TensorFlow) {
            // Check if this should be marked as a skip connection
            if (IsSkipConnection(link)) {
                // Determine if it's residual (to Add) or dense (to Concatenate)
                const MLNode* to_node = FindNodeById(link.to_node);
                if (to_node) {
                    if (to_node->type == NodeType::Add || to_node->type == NodeType::Average) {
                        link.type = LinkType::ResidualSkip;
                    } else if (to_node->type == NodeType::Concatenate) {
                        link.type = LinkType::DenseSkip;
                    }
                }
                spdlog::debug("Detected skip connection: link {} marked as {}",
                              link.id, (link.type == LinkType::ResidualSkip) ? "Residual" : "Dense");
            }
        }
    }
}

std::vector<NodeLink> NodeEditor::GetSkipConnections() const {
    std::vector<NodeLink> skip_connections;
    for (const auto& link : links_) {
        if (link.type == LinkType::ResidualSkip || link.type == LinkType::DenseSkip) {
            skip_connections.push_back(link);
        }
    }
    return skip_connections;
}

// ========== Skip Connection Creation Helpers ==========

int NodeEditor::AddResidualConnection(int from_node_id, int to_node_id) {
    SaveUndoState();

    const MLNode* from_node = FindNodeById(from_node_id);
    const MLNode* to_node = FindNodeById(to_node_id);

    if (!from_node || !to_node) {
        spdlog::warn("AddResidualConnection: Invalid node IDs");
        return -1;
    }

    // Check if to_node is already an Add node
    if (to_node->type == NodeType::Add) {
        // Check if Add node has available input
        if (!to_node->inputs.empty()) {
            // Find first available input
            for (const auto& pin : to_node->inputs) {
                if (!IsPinFull(pin.id)) {
                    // Create skip link
                    if (!from_node->outputs.empty()) {
                        NodeLink link;
                        link.id = next_link_id_++;
                        link.from_node = from_node_id;
                        link.from_pin = from_node->outputs[0].id;
                        link.to_node = to_node_id;
                        link.to_pin = pin.id;
                        link.type = LinkType::ResidualSkip;
                        links_.push_back(link);

                        spdlog::info("Added residual skip connection from node {} to Add node {}",
                                     from_node_id, to_node_id);
                        return to_node_id;
                    }
                }
            }
        }
        spdlog::warn("Add node {} has no available inputs", to_node_id);
        return -1;
    }

    // to_node is not an Add node - insert one
    // Find the link going INTO to_node
    NodeLink* incoming_link = nullptr;
    for (auto& link : links_) {
        if (link.to_node == to_node_id) {
            incoming_link = &link;
            break;
        }
    }

    if (!incoming_link) {
        spdlog::warn("No incoming link to node {}", to_node_id);
        return -1;
    }

    // Create new Add node
    MLNode add_node = CreateNode(NodeType::Add, "Residual Add");
    nodes_.push_back(add_node);

    // Position Add node between the incoming node and to_node
    ImVec2 from_pos = ImNodes::GetNodeGridSpacePos(incoming_link->from_node);
    ImVec2 to_pos = ImNodes::GetNodeGridSpacePos(to_node_id);
    ImVec2 add_pos = ImVec2((from_pos.x + to_pos.x) / 2 + 50, (from_pos.y + to_pos.y) / 2);
    pending_positions_[add_node.id] = add_pos;
    pending_positions_frames_ = 3;

    // Rewire: incoming -> Add (Input 1)
    incoming_link->to_node = add_node.id;
    incoming_link->to_pin = add_node.inputs[0].id;

    // Add link: Add -> to_node
    NodeLink add_to_target;
    add_to_target.id = next_link_id_++;
    add_to_target.from_node = add_node.id;
    add_to_target.from_pin = add_node.outputs[0].id;
    add_to_target.to_node = to_node_id;
    add_to_target.to_pin = to_node->inputs[0].id;
    add_to_target.type = LinkType::TensorFlow;
    links_.push_back(add_to_target);

    // Add skip connection: from_node -> Add (Input 2)
    NodeLink skip_link;
    skip_link.id = next_link_id_++;
    skip_link.from_node = from_node_id;
    skip_link.from_pin = from_node->outputs[0].id;
    skip_link.to_node = add_node.id;
    skip_link.to_pin = add_node.inputs[1].id;
    skip_link.type = LinkType::ResidualSkip;
    links_.push_back(skip_link);

    spdlog::info("Created Add node {} for residual connection from node {} to node {}",
                 add_node.id, from_node_id, to_node_id);

    return add_node.id;
}

int NodeEditor::AddDenseConnection(int from_node_id, int to_node_id) {
    SaveUndoState();

    const MLNode* from_node = FindNodeById(from_node_id);
    const MLNode* to_node = FindNodeById(to_node_id);

    if (!from_node || !to_node) {
        spdlog::warn("AddDenseConnection: Invalid node IDs");
        return -1;
    }

    // Check if to_node is already a Concatenate node
    if (to_node->type == NodeType::Concatenate) {
        // Check if Concatenate node has available input
        if (!to_node->inputs.empty()) {
            for (const auto& pin : to_node->inputs) {
                if (!IsPinFull(pin.id)) {
                    if (!from_node->outputs.empty()) {
                        NodeLink link;
                        link.id = next_link_id_++;
                        link.from_node = from_node_id;
                        link.from_pin = from_node->outputs[0].id;
                        link.to_node = to_node_id;
                        link.to_pin = pin.id;
                        link.type = LinkType::DenseSkip;
                        links_.push_back(link);

                        spdlog::info("Added dense skip connection from node {} to Concat node {}",
                                     from_node_id, to_node_id);
                        return to_node_id;
                    }
                }
            }
        }
        spdlog::warn("Concatenate node {} has no available inputs", to_node_id);
        return -1;
    }

    // to_node is not a Concatenate node - insert one
    NodeLink* incoming_link = nullptr;
    for (auto& link : links_) {
        if (link.to_node == to_node_id) {
            incoming_link = &link;
            break;
        }
    }

    if (!incoming_link) {
        spdlog::warn("No incoming link to node {}", to_node_id);
        return -1;
    }

    // Create new Concatenate node
    MLNode concat_node = CreateNode(NodeType::Concatenate, "Dense Concat");
    nodes_.push_back(concat_node);

    // Position Concat node
    ImVec2 from_pos = ImNodes::GetNodeGridSpacePos(incoming_link->from_node);
    ImVec2 to_pos = ImNodes::GetNodeGridSpacePos(to_node_id);
    ImVec2 concat_pos = ImVec2((from_pos.x + to_pos.x) / 2 + 50, (from_pos.y + to_pos.y) / 2);
    pending_positions_[concat_node.id] = concat_pos;
    pending_positions_frames_ = 3;

    // Rewire
    incoming_link->to_node = concat_node.id;
    incoming_link->to_pin = concat_node.inputs[0].id;

    // Add link: Concat -> to_node
    NodeLink concat_to_target;
    concat_to_target.id = next_link_id_++;
    concat_to_target.from_node = concat_node.id;
    concat_to_target.from_pin = concat_node.outputs[0].id;
    concat_to_target.to_node = to_node_id;
    concat_to_target.to_pin = to_node->inputs[0].id;
    concat_to_target.type = LinkType::TensorFlow;
    links_.push_back(concat_to_target);

    // Add skip connection
    NodeLink skip_link;
    skip_link.id = next_link_id_++;
    skip_link.from_node = from_node_id;
    skip_link.from_pin = from_node->outputs[0].id;
    skip_link.to_node = concat_node.id;
    skip_link.to_pin = concat_node.inputs[1].id;
    skip_link.type = LinkType::DenseSkip;
    links_.push_back(skip_link);

    spdlog::info("Created Concat node {} for dense connection from node {} to node {}",
                 concat_node.id, from_node_id, to_node_id);

    return concat_node.id;
}

void NodeEditor::WrapSelectionWithResidual() {
    if (selected_node_ids_.empty()) {
        spdlog::warn("WrapSelectionWithResidual: No nodes selected");
        return;
    }

    SaveUndoState();

    // Find the entry and exit nodes of the selection
    // Entry: node with incoming link from outside selection
    // Exit: node with outgoing link to outside selection

    std::set<int> selection_set(selected_node_ids_.begin(), selected_node_ids_.end());
    int entry_node_id = -1;
    int exit_node_id = -1;
    int entry_from_node = -1;  // Node before entry
    int exit_to_node = -1;     // Node after exit
    int entry_pin = -1;
    int exit_pin = -1;

    for (const auto& link : links_) {
        bool from_in = selection_set.count(link.from_node) > 0;
        bool to_in = selection_set.count(link.to_node) > 0;

        if (!from_in && to_in) {
            // Incoming link to selection
            entry_node_id = link.to_node;
            entry_from_node = link.from_node;
            entry_pin = link.from_pin;
        }
        if (from_in && !to_in) {
            // Outgoing link from selection
            exit_node_id = link.from_node;
            exit_to_node = link.to_node;
            exit_pin = link.to_pin;
        }
    }

    if (entry_node_id == -1 || exit_node_id == -1 || entry_from_node == -1 || exit_to_node == -1) {
        spdlog::warn("WrapSelectionWithResidual: Could not find entry/exit nodes");
        return;
    }

    // Create Add node after the exit
    MLNode add_node = CreateNode(NodeType::Add, "Residual Add");
    nodes_.push_back(add_node);

    // Position Add node after exit
    ImVec2 exit_pos = ImNodes::GetNodeGridSpacePos(exit_node_id);
    pending_positions_[add_node.id] = ImVec2(exit_pos.x + 200, exit_pos.y);
    pending_positions_frames_ = 3;

    // Find and modify the exit link
    for (auto& link : links_) {
        if (link.from_node == exit_node_id && link.to_node == exit_to_node) {
            // Redirect: exit -> Add (Input 1)
            link.to_node = add_node.id;
            link.to_pin = add_node.inputs[0].id;
            break;
        }
    }

    // Add link: Add -> original target
    const MLNode* exit_to = FindNodeById(exit_to_node);
    if (exit_to && !exit_to->inputs.empty()) {
        NodeLink add_to_target;
        add_to_target.id = next_link_id_++;
        add_to_target.from_node = add_node.id;
        add_to_target.from_pin = add_node.outputs[0].id;
        add_to_target.to_node = exit_to_node;
        add_to_target.to_pin = exit_pin;
        add_to_target.type = LinkType::TensorFlow;
        links_.push_back(add_to_target);
    }

    // Add skip connection: entry_from -> Add (Input 2)
    const MLNode* entry_from = FindNodeById(entry_from_node);
    if (entry_from) {
        NodeLink skip_link;
        skip_link.id = next_link_id_++;
        skip_link.from_node = entry_from_node;
        skip_link.from_pin = entry_pin;
        skip_link.to_node = add_node.id;
        skip_link.to_pin = add_node.inputs[1].id;
        skip_link.type = LinkType::ResidualSkip;
        links_.push_back(skip_link);
    }

    spdlog::info("Wrapped selection ({} nodes) with residual connection", selected_node_ids_.size());
}

}  // namespace gui
