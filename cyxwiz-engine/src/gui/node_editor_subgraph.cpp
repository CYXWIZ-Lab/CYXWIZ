#include "node_editor.h"
#include "subgraph_document.h"
#include "properties.h"
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <set>

namespace gui {
// ===== Subgraph Encapsulation =====

void NodeEditor::CreateSubgraphFromSelection(const std::string& name) {
    if (selected_node_ids_.size() < 2) {
        spdlog::warn("Need at least 2 selected nodes to create subgraph");
        return;
    }

    if (!CanReplaceGraph("create a subgraph")) return;
    for (int id : selected_node_ids_) {
        if (IsSubgraphNode(id) || IsSubgraphMember(id)) {
            spdlog::warn("Nested or overlapping subgraphs are not supported yet.");
            return;
        }
    }
    SaveUndoState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();

    // Collect selected nodes and their internal links
    std::vector<MLNode> internal_nodes;
    std::vector<NodeLink> internal_links;
    std::set<int> selected_set(selected_node_ids_.begin(), selected_node_ids_.end());

    // Copy selected nodes to internal storage
    for (const auto& node : nodes_) {
        if (selected_set.count(node.id)) {
            internal_nodes.push_back(node);
        }
    }

    // Find internal links (both endpoints in selection)
    for (const auto& link : links_) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            internal_links.push_back(link);
        }
    }

    // Find boundary pins - inputs are pins with external sources, outputs have external destinations
    std::vector<std::pair<int, int>> input_pins;   // (node_id, pin_id) pairs
    std::vector<std::pair<int, int>> output_pins;  // (node_id, pin_id) pairs

    for (const auto& link : links_) {
        // Link from outside to inside -> input boundary
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            input_pins.push_back({link.to_node, link.to_pin});
        }
        // Link from inside to outside -> output boundary
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            output_pins.push_back({link.from_node, link.from_pin});
        }
    }

    // Calculate center position of selected nodes
    float center_x = 0, center_y = 0;
    int count = 0;
    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it != cached_node_positions_.end()) {
            center_x += pos_it->second.x;
            center_y += pos_it->second.y;
            count++;
        }
    }
    if (count > 0) {
        center_x /= count;
        center_y /= count;
    }

    // Create the subgraph node
    MLNode subgraph_node{};
    subgraph_node.category = GetCategoryForNodeType(NodeType::Subgraph);
    subgraph_node.id = next_node_id_++;
    subgraph_node.type = NodeType::Subgraph;
    subgraph_node.name = name.empty() ? "Subgraph" : name;
    subgraph_node.parameters["node_count"] = std::to_string(internal_nodes.size());

    // Create input pins for boundary inputs
    for (size_t i = 0; i < input_pins.size(); ++i) {
        NodePin pin{};
        pin.id = next_pin_id_++;
        const auto* owner = FindNodeById(input_pins[i].first);
        for (const auto& original : owner->inputs) {
            if (original.id == input_pins[i].second) {
                const int id = pin.id;
                pin = original;
                pin.id = id;
                break;
            }
        }
        pin.is_input = true;
        subgraph_node.inputs.push_back(pin);
    }

    // Create output pins for boundary outputs
    for (size_t i = 0; i < output_pins.size(); ++i) {
        NodePin pin{};
        pin.id = next_pin_id_++;
        const auto* owner = FindNodeById(output_pins[i].first);
        for (const auto& original : owner->outputs) {
            if (original.id == output_pins[i].second) {
                const int id = pin.id;
                pin = original;
                pin.id = id;
                break;
            }
        }
        pin.is_input = false;
        subgraph_node.outputs.push_back(pin);
    }

    // Store subgraph data
    SubgraphData data;
    data.subgraph_node_id = subgraph_node.id;
    for (auto& internal : internal_nodes) {
        const auto pos = cached_node_positions_.find(internal.id);
        if (pos != cached_node_positions_.end()) {
            internal.initial_pos_x = pos->second.x;
            internal.initial_pos_y = pos->second.y;
            internal.has_initial_position = true;
        }
    }
    data.internal_nodes = std::move(internal_nodes);
    data.internal_links = std::move(internal_links);
    data.expanded = false;

    // Store pin mappings
    for (const auto& [node_id, pin_id] : input_pins) {
        data.input_pin_mappings.push_back(pin_id);
    }
    for (const auto& [node_id, pin_id] : output_pins) {
        data.output_pin_mappings.push_back(pin_id);
    }

    subgraphs_.push_back(std::move(data));

    // Rewire external connections to the subgraph node
    std::vector<NodeLink> links_to_add;
    std::vector<int> links_to_remove;

    for (size_t i = 0; i < links_.size(); ++i) {
        const auto& link = links_[i];

        // Link from outside to inside -> connect to subgraph input
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            // Find which input pin this maps to
            for (size_t j = 0; j < input_pins.size(); ++j) {
                if (input_pins[j].second == link.to_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = link.from_node;
                    new_link.from_pin = link.from_pin;
                    new_link.to_node = subgraph_node.id;
                    new_link.to_pin = subgraph_node.inputs[j].id;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Link from inside to outside -> connect from subgraph output
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            // Find which output pin this maps to
            for (size_t j = 0; j < output_pins.size(); ++j) {
                if (output_pins[j].second == link.from_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = subgraph_node.id;
                    new_link.from_pin = subgraph_node.outputs[j].id;
                    new_link.to_node = link.to_node;
                    new_link.to_pin = link.to_pin;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Internal links are removed from main graph
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            links_to_remove.push_back(static_cast<int>(i));
        }
    }

    // Remove old links (in reverse order to maintain indices)
    std::sort(links_to_remove.begin(), links_to_remove.end(), std::greater<int>());
    for (int idx : links_to_remove) {
        links_.erase(links_.begin() + idx);
    }

    // Add new links
    for (auto& link : links_to_add) {
        links_.push_back(link);
    }

    // Remove selected nodes from main graph
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&selected_set](const MLNode& n) { return selected_set.count(n.id); }),
        nodes_.end()
    );

    // Add subgraph node
    nodes_.push_back(subgraph_node);

    // Position the subgraph node at the center of removed nodes
    pending_positions_[subgraph_node.id] = ImVec2(center_x, center_y);
    pending_positions_frames_ = 3;

    // Clear selection and select the new subgraph
    selected_node_ids_.clear();
    selected_node_ids_.push_back(subgraph_node.id);

    ImNodes::ClearNodeSelection();
    ImNodes::SelectNode(subgraph_node.id);
    RebuildPinLookup();
    ClearValidationState();

    spdlog::info("Created subgraph '{}' with {} internal nodes",
                 subgraph_node.name, subgraphs_.back().internal_nodes.size());
}

void NodeEditor::ExpandSubgraph(int node_id) {
    if (!CanReplaceGraph("expand a subgraph")) return;
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) {
        spdlog::warn("Node {} is not a subgraph", node_id);
        return;
    }

    if (data->expanded) return;

    SaveUndoState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();
    data->expanded = true;

    // Get position of subgraph node
    ImVec2 base_pos = ImVec2(0, 0);
    auto pos_it = cached_node_positions_.find(node_id);
    if (pos_it != cached_node_positions_.end()) {
        base_pos = pos_it->second;
    }

    // Add internal nodes back to the main graph
    float offset_x = 0, offset_y = 50;
    for (auto& internal_node : data->internal_nodes) {
        // Offset position relative to subgraph node
        pending_positions_[internal_node.id] = internal_node.has_initial_position
            ? ImVec2(internal_node.initial_pos_x, internal_node.initial_pos_y)
            : ImVec2(base_pos.x + offset_x, base_pos.y + offset_y);
        nodes_.push_back(internal_node);
        offset_x += 180;
        if (offset_x > 500) {
            offset_x = 0;
            offset_y += 120;
        }
    }

    // Add internal links back
    for (const auto& link : data->internal_links) {
        links_.push_back(link);
    }

    pending_positions_frames_ = 3;
    RebuildPinLookup();
    ClearValidationState();
    spdlog::info("Expanded subgraph {} ({} internal nodes, {} internal links; container remains visible)", node_id, data->internal_nodes.size(), data->internal_links.size());
}

void NodeEditor::CollapseSubgraph(int node_id) {
    if (!CanReplaceGraph("collapse a subgraph")) return;
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (!data->expanded) return;

    try {
        auto captured = *data;
        detail::CaptureExpandedSubgraph(captured, nodes_, links_);
        for (auto& internal : captured.internal_nodes) {
            const auto pos = cached_node_positions_.find(internal.id);
            if (pos != cached_node_positions_.end()) {
                internal.initial_pos_x = pos->second.x;
                internal.initial_pos_y = pos->second.y;
                internal.has_initial_position = true;
            }
        }
        SaveUndoState();
        *data = std::move(captured);
    } catch (const std::exception& error) {
        spdlog::error("Cannot collapse subgraph: {}", error.what());
        return;
    }
    if (properties_panel_) properties_panel_->ClearNodeReferences();
    data->expanded = false;

    // Remove internal nodes from main graph
    std::set<int> internal_ids;
    for (const auto& node : data->internal_nodes) {
        internal_ids.insert(node.id);
    }

    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&internal_ids](const MLNode& n) { return internal_ids.count(n.id); }),
        nodes_.end()
    );

    // Remove internal links
    links_.erase(
        std::remove_if(links_.begin(), links_.end(),
            [&internal_ids](const NodeLink& l) {
                return internal_ids.count(l.from_node) && internal_ids.count(l.to_node);
            }),
        links_.end()
    );

    selected_node_ids_.clear();
    selected_node_id_ = -1;
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    RebuildPinLookup();
    ClearValidationState();
    spdlog::info("Collapsed subgraph {}", node_id);
}

void NodeEditor::ToggleSubgraphExpansion(int node_id) {
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (data->expanded) {
        CollapseSubgraph(node_id);
    } else {
        ExpandSubgraph(node_id);
    }
}

bool NodeEditor::IsSubgraphNode(int node_id) const {
    const MLNode* node = FindNodeById(node_id);
    return node && node->type == NodeType::Subgraph;
}

SubgraphData* NodeEditor::GetSubgraphData(int node_id) {
    for (auto& data : subgraphs_) {
        if (data.subgraph_node_id == node_id) {
            return &data;
        }
    }
    return nullptr;
}

bool NodeEditor::IsSubgraphMember(int node_id) const {
    for (const auto& data : subgraphs_)
        for (const auto& node : data.internal_nodes)
            if (node.id == node_id) return true;
    return false;
}

}
