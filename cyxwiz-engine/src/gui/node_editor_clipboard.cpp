#include "node_editor.h"
#include "../core/studio_toolbar_presentation.h"
#include "subgraph_presentation.h"
#include "properties.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <set>

namespace gui {

GraphSnapshot NodeEditor::CaptureSnapshot() const {
    GraphSnapshot snapshot;
    snapshot.nodes = nodes_;
    snapshot.subgraphs = subgraphs_;
    snapshot.positions = cached_node_positions_;
    for (const auto& [id, pos] : pending_positions_) snapshot.positions[id] = pos;
    snapshot.links = links_;
    snapshot.next_node_id = next_node_id_;
    snapshot.next_pin_id = next_pin_id_;
    snapshot.next_link_id = next_link_id_;
    snapshot.annotations = annotations_;
    snapshot.next_annotation_id = next_annotation_id_;
    return snapshot;
}

void NodeEditor::SaveUndoState() { PushUndoSnapshot(CaptureSnapshot()); }

void NodeEditor::PushUndoSnapshot(GraphSnapshot snapshot) {
    undo_stack_.push_back(std::move(snapshot));

    // Limit stack size
    if (undo_stack_.size() > MAX_UNDO_LEVELS) {
        undo_stack_.erase(undo_stack_.begin());
    }

    // Clear redo stack when new action is performed
    redo_stack_.clear();

    spdlog::debug("Saved undo state (stack size: {})", undo_stack_.size());
}

void NodeEditor::Undo() {
    if (!CanReplaceGraph("undo graph edits")) return;
    if (!CanUndo()) {
        spdlog::debug("Nothing to undo");
        return;
    }

    // Save current state to redo stack
    redo_stack_.push_back(CaptureSnapshot());

    // Restore previous state from undo stack
    GraphSnapshot previous = undo_stack_.back();
    undo_stack_.pop_back();

    if (properties_panel_) properties_panel_->ClearNodeReferences();
    std::vector<MLNode> before = nodes_;
    for (const auto& data : subgraphs_)
        if (!data.expanded) before.insert(before.end(), data.internal_nodes.begin(), data.internal_nodes.end());
    nodes_ = previous.nodes;
    subgraphs_ = previous.subgraphs;
    cached_node_positions_ = previous.positions;
    pending_positions_ = previous.positions;
    pending_positions_frames_ = 3;
    pending_context_reset_ = true;
    pending_context_keep_view_ = true;
    selected_node_ids_.clear();
    ClearValidationState();
    links_ = previous.links;
    next_node_id_ = previous.next_node_id;
    next_pin_id_ = previous.next_pin_id;
    next_link_id_ = previous.next_link_id;
    annotations_ = previous.annotations;
    next_annotation_id_ = previous.next_annotation_id;
    selected_annotation_id_ = -1;
    dragging_annotation_id_ = -1;
    editing_annotation_ = false;

    // Rebuild pin lookup after restoring state
    RebuildPinLookup();
    SyncDataAfterSwap(before);

    // Clear selection
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_id_ = -1;

    spdlog::info("Undo performed (undo stack: {}, redo stack: {})",
                 undo_stack_.size(), redo_stack_.size());
}

void NodeEditor::Redo() {
    if (!CanReplaceGraph("redo graph edits")) return;
    if (!CanRedo()) {
        spdlog::debug("Nothing to redo");
        return;
    }

    // Save current state to undo stack
    undo_stack_.push_back(CaptureSnapshot());

    // Restore next state from redo stack
    GraphSnapshot next = redo_stack_.back();
    redo_stack_.pop_back();

    if (properties_panel_) properties_panel_->ClearNodeReferences();
    std::vector<MLNode> before = nodes_;
    for (const auto& data : subgraphs_)
        if (!data.expanded) before.insert(before.end(), data.internal_nodes.begin(), data.internal_nodes.end());
    nodes_ = next.nodes;
    subgraphs_ = next.subgraphs;
    cached_node_positions_ = next.positions;
    pending_positions_ = next.positions;
    pending_positions_frames_ = 3;
    pending_context_reset_ = true;
    pending_context_keep_view_ = true;
    selected_node_ids_.clear();
    ClearValidationState();
    links_ = next.links;
    next_node_id_ = next.next_node_id;
    next_pin_id_ = next.next_pin_id;
    next_link_id_ = next.next_link_id;
    annotations_ = next.annotations;
    next_annotation_id_ = next.next_annotation_id;
    selected_annotation_id_ = -1;
    dragging_annotation_id_ = -1;
    editing_annotation_ = false;

    // Rebuild pin lookup after restoring state
    RebuildPinLookup();
    SyncDataAfterSwap(before);

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
        if (detail::IsExpandedSubgraph(node.id, subgraphs_)) continue;
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

ImVec2 NodeEditor::FindEmptyPosition(ImVec2 size) {
    // The free spot nearest the middle of the view (owner 2026-11-14: a node
    // added from the menu, the Asset Browser or the Nodes panel, and a new
    // note, landed on the view centre over whatever was there). Nodes, nodes
    // still waiting for their place, and notes count as taken.
    const float kWidth = size.x;  // what is placed (grid units)
    const float kHeight = size.y;
    constexpr float kGap = 30.0f;
    const ImVec2 view_centre = canvas_size_.x > 0.0f && canvas_size_.y > 0.0f
        ? GridAt(ImVec2(canvas_origin_.x + canvas_size_.x * 0.5f, canvas_origin_.y + canvas_size_.y * 0.5f))
        : ImVec2(400.0f - ImNodes::EditorContextGetPanning().x, 300.0f - ImNodes::EditorContextGetPanning().y);
    const ImVec2 preferred(view_centre.x - kWidth * 0.5f, view_centre.y - kHeight * 0.5f);

    struct Rect { float x0, y0, x1, y1; };
    std::vector<Rect> taken;
    for (const auto& node : nodes_) {
        if (detail::IsExpandedSubgraph(node.id, subgraphs_)) continue;
        const auto at = cached_node_positions_.find(node.id);
        const auto waiting = pending_positions_.find(node.id);
        if (at == cached_node_positions_.end() && waiting == pending_positions_.end()) continue;
        const ImVec2 pos = waiting != pending_positions_.end() ? waiting->second : at->second;
        taken.push_back({pos.x, pos.y, pos.x + kNodeFootprint.x, pos.y + kNodeFootprint.y});
    }
    for (const auto& note : annotations_)
        taken.push_back({note.position.x, note.position.y, note.position.x + note.size.x, note.position.y + note.size.y});
    const auto free_at = [&taken, kWidth, kHeight](const ImVec2& p) {
        for (const Rect& r : taken)
            if (p.x < r.x1 + kGap && p.x + kWidth + kGap > r.x0 && p.y < r.y1 + kGap && p.y + kHeight + kGap > r.y0)
                return false;
        return true;
    };
    if (free_at(preferred)) return preferred;

    // Rings of candidate spots around the preferred one, nearest first.
    const float step_x = kWidth + kGap, step_y = kHeight + kGap;
    for (int ring = 1; ring <= 40; ++ring) {
        bool found = false;
        ImVec2 best;
        float best_distance = 0.0f;
        for (int dy = -ring; dy <= ring; ++dy) {
            for (int dx = -ring; dx <= ring; ++dx) {
                if (std::abs(dx) != ring && std::abs(dy) != ring) continue;  // the ring only
                const ImVec2 p(preferred.x + static_cast<float>(dx) * step_x, preferred.y + static_cast<float>(dy) * step_y);
                if (!free_at(p)) continue;
                const float distance = (p.x - preferred.x) * (p.x - preferred.x) + (p.y - preferred.y) * (p.y - preferred.y);
                if (!found || distance < best_distance) {
                    found = true;
                    best = p;
                    best_distance = distance;
                }
            }
        }
        if (found) return best;
    }
    // A canvas full in every direction: below everything.
    float bottom = preferred.y;
    for (const Rect& r : taken) bottom = std::max(bottom, r.y1);
    return ImVec2(preferred.x, bottom + kGap);
}

void NodeEditor::CopySelection() {
    clipboard_.valid = false;
    // Get selected nodes from ImNodes
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        spdlog::debug("No nodes selected to copy");
        return;
    }

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());
    CopyNodes(selected_ids);
}

bool NodeEditor::CopyNodes(const std::vector<int>& selected_ids) {
    clipboard_.valid = false;
    for (int id : selected_ids) {
        if (IsSubgraphNode(id) || IsSubgraphMember(id)) {
            spdlog::warn("Subgraph copy/cut/duplicate is not supported yet; save and load the complete graph instead.");
            return false;
        }
    }

    // Build set for quick lookup
    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());

    // Copy the nodes and where they sit
    clipboard_.nodes.clear();
    clipboard_.links.clear();
    clipboard_.positions.clear();
    clipboard_.frame.reset();

    for (int node_id : selected_ids) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
            [node_id](const MLNode& node) { return node.id == node_id; });
        if (it != nodes_.end()) {
            clipboard_.nodes.push_back(*it);
            const auto pos = cached_node_positions_.find(node_id);
            if (pos != cached_node_positions_.end()) clipboard_.positions[node_id] = pos->second;
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
    return true;
}

std::vector<int> NodeEditor::NodesInsideFrame(const CanvasFrame& frame) const {
    std::vector<int> inside;
    for (const auto& node : nodes_) {
        const auto pos = cached_node_positions_.find(node.id);
        if (pos == cached_node_positions_.end()) continue;
        if (pos->second.x >= frame.position.x && pos->second.x <= frame.position.x + frame.size.x &&
            pos->second.y >= frame.position.y && pos->second.y <= frame.position.y + frame.size.y)
            inside.push_back(node.id);
    }
    return inside;
}

void NodeEditor::CopyFrame(int frame_id) {
    const auto frame = std::find_if(frames_.begin(), frames_.end(),
                                    [frame_id](const CanvasFrame& f) { return f.id == frame_id; });
    if (frame == frames_.end()) return;
    if (!CopyNodes(NodesInsideFrame(*frame))) return;
    clipboard_.frame = *frame;
    clipboard_.frame->is_selected = false;
    spdlog::info("Copied frame '{}' with {} nodes", frame->title, clipboard_.nodes.size());
}

void NodeEditor::CutFrame(int frame_id) {
    CopyFrame(frame_id);
    if (!clipboard_.valid || !clipboard_.frame) return;
    SaveUndoState();
    for (const auto& node : clipboard_.nodes) DeleteNode(node.id);
    DeleteFrame(frame_id);
    ClearSelection();
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
    if (!clipboard_.valid || (clipboard_.nodes.empty() && !clipboard_.frame)) {
        spdlog::debug("Nothing to paste");
        return;
    }

    SaveUndoState();

    // Clear selection before pasting
    ImNodes::ClearNodeSelection();

    // The copied layout's top-left (the frame's, when one was copied) lands
    // at the paste offset; everything keeps its place relative to it.
    ImVec2 origin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    for (const auto& [id, pos] : clipboard_.positions) {
        origin.x = std::min(origin.x, pos.x);
        origin.y = std::min(origin.y, pos.y);
    }
    if (clipboard_.frame) {
        origin.x = std::min(origin.x, clipboard_.frame->position.x);
        origin.y = std::min(origin.y, clipboard_.frame->position.y);
    }
    if (origin.x == std::numeric_limits<float>::max()) origin = ImVec2(0.0f, 0.0f);
    if (clipboard_.frame) {
        CanvasFrame frame = *clipboard_.frame;
        frame.id = next_frame_id_++;
        frame.position = ImVec2(paste_offset_.x + frame.position.x - origin.x,
                                paste_offset_.y + frame.position.y - origin.y);
        frames_.push_back(frame);
    }

    // New ids for the nodes, their pins and the links between them.
    std::map<int, ImVec2> positions;
    for (size_t i = 0; i < clipboard_.nodes.size(); ++i) {
        const int old_id = clipboard_.nodes[i].id;
        const auto copied_at = clipboard_.positions.find(old_id);
        const float step = static_cast<float>(nodes_.size() + i) * 10.0f;
        positions[old_id] = copied_at != clipboard_.positions.end()
            ? ImVec2(paste_offset_.x + copied_at->second.x - origin.x, paste_offset_.y + copied_at->second.y - origin.y)
            : ImVec2(paste_offset_.x + step, paste_offset_.y + step);
    }
    for (const auto& [old_id, new_id] : AddWithFreshIds(clipboard_.nodes, clipboard_.links, positions)) {
        ImNodes::SetNodeGridSpacePos(new_id, positions[old_id]);
        ImNodes::SelectNode(new_id);
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

// Copies the selection beside itself through the clipboard, then puts back
// what the user had copied: Duplicate is not Copy.
void NodeEditor::DuplicateSelection() {
    const auto kept = clipboard_;
    CopySelection();
    if (clipboard_.valid) PasteClipboard();
    clipboard_ = kept;
}

// ===== Keyboard Shortcuts =====

void NodeEditor::HandleKeyboardShortcuts() {
    // Only process shortcuts when node editor window is focused
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        return;
    }
    // A text field on a node has the keys: Ctrl+A, M, F ... are its typing.
    if (ImGui::GetIO().WantTextInput) return;

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
            CloseFindBar();
        } else if (show_context_menu_) {
            show_context_menu_ = false;
        } else {
            ClearSelection();
        }
    }

    // Ctrl+F - Open the find bar (or close it); it takes the keyboard
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_F)) {
        if (search_state_.search_visible) CloseFindBar();
        else OpenFindBar();
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
    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_G) && !selected_node_ids_.empty()) {
        CreateGroupFromSelection("");  // Creates with default name
    }

    // Ctrl+Shift+G - Ungroup selection
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_G) && !selected_node_ids_.empty()) {
        UngroupSelection();
    }

    // Ctrl+S - Save the graph to its file (a name is asked the first time);
    // Ctrl+Shift+S - Save As a new file (the open one stays as it was).
    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_S, false)) {
        SaveFromStudio();
    }
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_S, false)) {
        SaveAsFromStudio();
    }

    // Ctrl+Shift+U - Create subgraph from selection (Ctrl+Shift+S is Save As)
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_U) && selected_node_ids_.size() >= 2) {
        CreateSubgraphFromSelection("");
    }

    // + and - zoom one step around the canvas centre (as View's - / +)
    if (!ctrl && (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) &&
        cyxwiz::studio_toolbar::ZoomIn(zoom_).enabled)
        ZoomAroundCentre(1);
    if (!ctrl && (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) &&
        cyxwiz::studio_toolbar::ZoomOut(zoom_).enabled)
        ZoomAroundCentre(-1);

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

void NodeEditor::CenterViewOn(ImVec2 grid_point) {
    // screen = grid * zoom + panning (relative to the canvas origin)
    ImNodes::EditorContextResetPanning(ImVec2(canvas_size_.x * 0.5f - grid_point.x * zoom_,
                                              canvas_size_.y * 0.5f - grid_point.y * zoom_));
}

void NodeEditor::RevealIfOffScreen(ImVec2 grid_min, ImVec2 size) {
    if (canvas_size_.x <= 0.0f || canvas_size_.y <= 0.0f) return;
    const ImVec2 view_min = GridAt(canvas_origin_);
    const ImVec2 view_max = GridAt(ImVec2(canvas_origin_.x + canvas_size_.x, canvas_origin_.y + canvas_size_.y));
    const bool inside = grid_min.x >= view_min.x && grid_min.y >= view_min.y &&
                        grid_min.x + size.x <= view_max.x && grid_min.y + size.y <= view_max.y;
    if (!inside) CenterViewOn(ImVec2(grid_min.x + size.x * 0.5f, grid_min.y + size.y * 0.5f));
}

void NodeEditor::FrameNodes(const std::vector<int>& node_ids) {
    if (node_ids.empty()) {
        return;
    }
    // Bounding box in grid space; node dimensions are on screen, at this zoom.
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;
    for (int node_id : node_ids) {
        auto it = cached_node_positions_.find(node_id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0, 0);
        ImVec2 dims = ImNodes::GetNodeDimensions(node_id);
        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + dims.x / zoom_);
        max_y = std::max(max_y, pos.y + dims.y / zoom_);
    }
    // Zoom out until the box fits with a margin (never in past 100%, never
    // below the wheel's minimum), then center it.
    constexpr float kMargin = 48.0f;
    const float fit_x = (canvas_size_.x - 2.0f * kMargin) / std::max(1.0f, max_x - min_x);
    const float fit_y = (canvas_size_.y - 2.0f * kMargin) / std::max(1.0f, max_y - min_y);
    zoom_ = std::clamp(std::min({fit_x, fit_y, 1.0f}), cyxwiz::studio_toolbar::kZoomMin, cyxwiz::studio_toolbar::kZoomMax);
    CenterViewOn(ImVec2((min_x + max_x) * 0.5f, (min_y + max_y) * 0.5f));
}

void NodeEditor::FrameSelected() {
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        return;
    }
    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());
    FrameNodes(selected_ids);
    spdlog::debug("Framed {} selected nodes", num_selected);
}

void NodeEditor::FrameAll() {
    std::vector<int> ids;
    for (const auto& node : nodes_) {
        if (detail::IsExpandedSubgraph(node.id, subgraphs_)) continue;
        ids.push_back(node.id);
    }
    FrameNodes(ids);
    spdlog::debug("Framed all {} nodes", ids.size());
}

void NodeEditor::CreateLink(int from_pin, int to_pin, int from_node, int to_node, LinkType type) {
    ClearValidationState();  // Graph changed — stale compile results

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
