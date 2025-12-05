#include "node_editor.h"
#include "properties.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <set>

namespace gui {

void NodeEditor::SaveUndoState() {
    // Create snapshot of current state
    GraphSnapshot snapshot;
    snapshot.nodes = nodes_;
    snapshot.links = links_;
    snapshot.next_node_id = next_node_id_;
    snapshot.next_pin_id = next_pin_id_;
    snapshot.next_link_id = next_link_id_;

    // Push to undo stack
    undo_stack_.push_back(snapshot);

    // Limit stack size
    if (undo_stack_.size() > MAX_UNDO_LEVELS) {
        undo_stack_.erase(undo_stack_.begin());
    }

    // Clear redo stack when new action is performed
    redo_stack_.clear();

    spdlog::debug("Saved undo state (stack size: {})", undo_stack_.size());
}

void NodeEditor::Undo() {
    if (!CanUndo()) {
        spdlog::debug("Nothing to undo");
        return;
    }

    // Save current state to redo stack
    GraphSnapshot current;
    current.nodes = nodes_;
    current.links = links_;
    current.next_node_id = next_node_id_;
    current.next_pin_id = next_pin_id_;
    current.next_link_id = next_link_id_;
    redo_stack_.push_back(current);

    // Restore previous state from undo stack
    GraphSnapshot previous = undo_stack_.back();
    undo_stack_.pop_back();

    nodes_ = previous.nodes;
    links_ = previous.links;
    next_node_id_ = previous.next_node_id;
    next_pin_id_ = previous.next_pin_id;
    next_link_id_ = previous.next_link_id;

    // Clear selection
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_id_ = -1;

    spdlog::info("Undo performed (undo stack: {}, redo stack: {})",
                 undo_stack_.size(), redo_stack_.size());
}

void NodeEditor::Redo() {
    if (!CanRedo()) {
        spdlog::debug("Nothing to redo");
        return;
    }

    // Save current state to undo stack
    GraphSnapshot current;
    current.nodes = nodes_;
    current.links = links_;
    current.next_node_id = next_node_id_;
    current.next_pin_id = next_pin_id_;
    current.next_link_id = next_link_id_;
    undo_stack_.push_back(current);

    // Restore next state from redo stack
    GraphSnapshot next = redo_stack_.back();
    redo_stack_.pop_back();

    nodes_ = next.nodes;
    links_ = next.links;
    next_node_id_ = next.next_node_id;
    next_pin_id_ = next.next_pin_id;
    next_link_id_ = next.next_link_id;

    // Clear selection
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_id_ = -1;

    spdlog::info("Redo performed (undo stack: {}, redo stack: {})",
                 undo_stack_.size(), redo_stack_.size());
}

// ===== Clipboard Support =====

void NodeEditor::SelectAll() {
    ImNodes::ClearNodeSelection();
    selected_node_ids_.clear();

    for (const auto& node : nodes_) {
        ImNodes::SelectNode(node.id);
        selected_node_ids_.push_back(node.id);
    }

    spdlog::info("Selected all {} nodes", nodes_.size());
}

void NodeEditor::ClearSelection() {
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_ids_.clear();
    selected_node_id_ = -1;

    if (properties_panel_) {
        properties_panel_->ClearSelection();
    }

    spdlog::debug("Cleared selection");
}

void NodeEditor::DeleteSelected() {
    // Delete selected nodes without copying to clipboard
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        spdlog::debug("No nodes selected to delete");
        return;
    }

    SaveUndoState();

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    for (int node_id : selected_ids) {
        DeleteNode(node_id);
    }

    ClearSelection();
    spdlog::info("Deleted {} selected nodes", num_selected);
}

ImVec2 NodeEditor::FindEmptyPosition() {
    // Find a position that doesn't overlap with existing nodes
    // Start at a reasonable default position and search for empty space

    const float NODE_WIDTH = 200.0f;
    const float NODE_HEIGHT = 120.0f;
    const float SPACING = 50.0f;

    // Get current view panning to place node in visible area
    ImVec2 panning = ImNodes::EditorContextGetPanning();

    // Start position - relative to current view
    float start_x = -panning.x + 100.0f;
    float start_y = -panning.y + 100.0f;

    // If no nodes exist, return a simple position
    if (nodes_.empty()) {
        return ImVec2(start_x, start_y);
    }

    // Collect all existing node positions
    std::vector<ImVec2> node_positions;
    for (const auto& node : nodes_) {
        auto it = cached_node_positions_.find(node.id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        node_positions.push_back(pos);
    }

    // Search for empty position using grid search
    for (int row = 0; row < 20; ++row) {
        for (int col = 0; col < 20; ++col) {
            float test_x = start_x + col * (NODE_WIDTH + SPACING);
            float test_y = start_y + row * (NODE_HEIGHT + SPACING);

            bool overlaps = false;
            for (const auto& pos : node_positions) {
                // Check if rectangles overlap
                if (test_x < pos.x + NODE_WIDTH + SPACING &&
                    test_x + NODE_WIDTH + SPACING > pos.x &&
                    test_y < pos.y + NODE_HEIGHT + SPACING &&
                    test_y + NODE_HEIGHT + SPACING > pos.y) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                return ImVec2(test_x, test_y);
            }
        }
    }

    // Fallback: place below the lowest node
    float max_y = 0.0f;
    for (const auto& pos : node_positions) {
        if (pos.y > max_y) {
            max_y = pos.y;
        }
    }
    return ImVec2(start_x, max_y + NODE_HEIGHT + SPACING);
}

void NodeEditor::CopySelection() {
    // Get selected nodes from ImNodes
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        spdlog::debug("No nodes selected to copy");
        return;
    }

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    // Build set for quick lookup
    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());

    // Copy selected nodes
    clipboard_.nodes.clear();
    clipboard_.links.clear();

    for (int node_id : selected_ids) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
            [node_id](const MLNode& node) { return node.id == node_id; });
        if (it != nodes_.end()) {
            clipboard_.nodes.push_back(*it);
        }
    }

    // Copy internal links (links between selected nodes only)
    for (const auto& link : links_) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            clipboard_.links.push_back(link);
        }
    }

    clipboard_.valid = true;
    spdlog::info("Copied {} nodes and {} internal links to clipboard",
                 clipboard_.nodes.size(), clipboard_.links.size());
}

void NodeEditor::CutSelection() {
    CopySelection();

    if (!clipboard_.valid) {
        return;
    }

    // Delete the selected nodes
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected > 0) {
        SaveUndoState();

        std::vector<int> selected_ids(num_selected);
        ImNodes::GetSelectedNodes(selected_ids.data());

        for (int node_id : selected_ids) {
            DeleteNode(node_id);
        }

        ClearSelection();
        spdlog::info("Cut {} nodes", num_selected);
    }
}

void NodeEditor::PasteClipboard() {
    if (!clipboard_.valid || clipboard_.nodes.empty()) {
        spdlog::debug("Nothing to paste");
        return;
    }

    SaveUndoState();

    // Map from old node IDs to new node IDs
    std::map<int, int> node_id_map;
    // Map from old pin IDs to new pin IDs
    std::map<int, int> pin_id_map;

    // Clear selection before pasting
    ImNodes::ClearNodeSelection();

    // Create new nodes with new IDs
    for (const auto& old_node : clipboard_.nodes) {
        MLNode new_node = old_node;
        int old_id = new_node.id;
        new_node.id = next_node_id_++;

        // Assign new pin IDs
        for (auto& pin : new_node.inputs) {
            int old_pin_id = pin.id;
            pin.id = next_pin_id_++;
            pin_id_map[old_pin_id] = pin.id;
        }
        for (auto& pin : new_node.outputs) {
            int old_pin_id = pin.id;
            pin.id = next_pin_id_++;
            pin_id_map[old_pin_id] = pin.id;
        }

        node_id_map[old_id] = new_node.id;
        nodes_.push_back(new_node);

        // Position the new node with offset
        ImVec2 old_pos = ImNodes::GetNodeGridSpacePos(old_id);
        // If we can't get old position (node doesn't exist), use a default
        ImVec2 new_pos(old_pos.x + paste_offset_.x, old_pos.y + paste_offset_.y);

        // For pasted nodes, we need to set position after they're added
        // ImNodes requires the node to exist first, so we'll set it in the next frame
        // For now, use a simple offset from screen center
        ImVec2 canvas_origin = ImNodes::GetNodeEditorSpacePos(0);
        new_pos = ImVec2(paste_offset_.x + nodes_.size() * 10.0f,
                         paste_offset_.y + nodes_.size() * 10.0f);

        ImNodes::SetNodeGridSpacePos(new_node.id, new_pos);
        ImNodes::SelectNode(new_node.id);
    }

    // Recreate internal links with new IDs
    for (const auto& old_link : clipboard_.links) {
        NodeLink new_link;
        new_link.id = next_link_id_++;
        new_link.from_node = node_id_map[old_link.from_node];
        new_link.to_node = node_id_map[old_link.to_node];
        new_link.from_pin = pin_id_map[old_link.from_pin];
        new_link.to_pin = pin_id_map[old_link.to_pin];

        links_.push_back(new_link);
    }

    // Increase paste offset for next paste
    paste_offset_.x += 50.0f;
    paste_offset_.y += 50.0f;

    // Reset paste offset if it gets too large
    if (paste_offset_.x > 300.0f) {
        paste_offset_ = ImVec2(50.0f, 50.0f);
    }

    spdlog::info("Pasted {} nodes and {} links",
                 clipboard_.nodes.size(), clipboard_.links.size());
}

void NodeEditor::DuplicateSelection() {
    CopySelection();
    PasteClipboard();
}

// ===== Keyboard Shortcuts =====

void NodeEditor::HandleKeyboardShortcuts() {
    // Only process shortcuts when node editor window is focused
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        return;
    }

    const bool ctrl = ImGui::GetIO().KeyCtrl;
    const bool shift = ImGui::GetIO().KeyShift;

    // Ctrl+Z - Undo
    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Z)) {
        Undo();
    }

    // Ctrl+Y or Ctrl+Shift+Z - Redo
    if ((ctrl && ImGui::IsKeyPressed(ImGuiKey_Y)) ||
        (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_Z))) {
        Redo();
    }

    // Ctrl+C - Copy
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_C)) {
        CopySelection();
    }

    // Ctrl+X - Cut
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_X)) {
        CutSelection();
    }

    // Ctrl+V - Paste
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_V)) {
        PasteClipboard();
    }

    // Ctrl+D - Duplicate
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D)) {
        DuplicateSelection();
    }

    // Ctrl+A - Select All
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_A)) {
        SelectAll();
    }

    // Escape - Close search bar, context menu, or clear selection
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        if (search_state_.search_visible) {
            search_state_.search_visible = false;
            search_state_.matching_node_ids.clear();
            search_state_.current_match_index = -1;
        } else if (show_context_menu_) {
            show_context_menu_ = false;
        } else {
            ClearSelection();
        }
    }

    // Ctrl+F - Toggle search bar
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_F)) {
        search_state_.search_visible = !search_state_.search_visible;
        if (search_state_.search_visible) {
            // Focus the search input when opened
            ImGui::SetKeyboardFocusHere();
        } else {
            // Clear search when closed
            search_state_.matching_node_ids.clear();
            search_state_.current_match_index = -1;
        }
    }

    // F3 - Navigate to next search match
    if (ImGui::IsKeyPressed(ImGuiKey_F3) && search_state_.search_visible) {
        if (shift) {
            NavigateToMatch(-1);  // Previous
        } else {
            NavigateToMatch(1);   // Next
        }
    }

    // Ctrl+G - Create group from selection
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_G) && !selected_node_ids_.empty()) {
        CreateGroupFromSelection("");  // Creates with default name
    }

    // Ctrl+Shift+G - Ungroup selection
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_G) && !selected_node_ids_.empty()) {
        UngroupSelection();
    }

    // Ctrl+Shift+S - Create subgraph from selection
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_S) && selected_node_ids_.size() >= 2) {
        CreateSubgraphFromSelection("");
    }

    // M - Toggle minimap
    if (ImGui::IsKeyPressed(ImGuiKey_M) && !ctrl) {
        show_minimap_ = !show_minimap_;
        spdlog::info("Minimap {}", show_minimap_ ? "enabled" : "disabled");
    }

    // F - Frame selected or frame all
    if (ImGui::IsKeyPressed(ImGuiKey_F) && !ctrl) {
        if (ImNodes::NumSelectedNodes() > 0) {
            FrameSelected();
        } else {
            FrameAll();
        }
    }
}

void NodeEditor::FrameSelected() {
    // Get bounding box of selected nodes and center view on them
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        return;
    }

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    // Calculate bounding box
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (int node_id : selected_ids) {
        auto it = cached_node_positions_.find(node_id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        ImVec2 dims = ImNodes::GetNodeDimensions(node_id);

        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + dims.x);
        max_y = std::max(max_y, pos.y + dims.y);
    }

    // Calculate center and pan to it
    ImVec2 center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
    ImNodes::EditorContextResetPanning(ImVec2(-center.x + 400.0f, -center.y + 300.0f));

    spdlog::debug("Framed {} selected nodes", num_selected);
}

void NodeEditor::FrameAll() {
    if (nodes_.empty()) {
        return;
    }

    // Calculate bounding box of all nodes
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (const auto& node : nodes_) {
        auto it = cached_node_positions_.find(node.id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        ImVec2 dims = ImNodes::GetNodeDimensions(node.id);

        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + dims.x);
        max_y = std::max(max_y, pos.y + dims.y);
    }

    // Calculate center and pan to it
    ImVec2 center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
    ImNodes::EditorContextResetPanning(ImVec2(-center.x + 400.0f, -center.y + 300.0f));

    spdlog::debug("Framed all {} nodes", nodes_.size());
}

void NodeEditor::CreateLink(int from_pin, int to_pin, int from_node, int to_node, LinkType type) {
    NodeLink link;
    link.id = next_link_id_++;
    link.from_pin = from_pin;
    link.to_pin = to_pin;
    link.from_node = from_node;
    link.to_node = to_node;
    link.type = type;
    links_.push_back(link);
}


} // namespace gui
