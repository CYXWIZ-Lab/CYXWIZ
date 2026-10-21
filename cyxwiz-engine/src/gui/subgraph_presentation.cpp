#include "subgraph_presentation.h"
#include "ui_buttons.h"
#include <imnodes.h>
#include <algorithm>
#include <cfloat>

namespace gui::detail {
namespace ui = cyxwiz::ui;
namespace {
const MLNode* FindNode(const std::vector<MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(),
        [id](const MLNode& node) { return node.id == id; });
    return it == nodes.end() ? nullptr : &*it;
}
int ExpandedOwner(int id, const std::vector<SubgraphData>& subgraphs) {
    for (const auto& data : subgraphs) {
        if (!data.expanded) continue;
        if (data.subgraph_node_id == id) return id;
        for (const auto& member : data.internal_nodes)
            if (member.id == id) return data.subgraph_node_id;
    }
    return -1;
}
// Frame colours come from the theme's ImNodes style (Theme::ApplyImNodesStyle),
// so this file needs no token tables and its test links ImGui and ImNodes only.
ImU32 NodeStyleColour(ImNodesCol colour, float alpha) {
    ImVec4 value = ImGui::ColorConvertU32ToFloat4(ImNodes::GetStyle().Colors[colour]);
    value.w = alpha;
    return ImGui::ColorConvertFloat4ToU32(value);
}
bool ProjectEndpoint(int& node_id, int& pin_id, bool input,
    const std::vector<MLNode>& nodes, const std::vector<SubgraphData>& subgraphs) {
    for (const auto& data : subgraphs) {
        if (!data.expanded || data.subgraph_node_id != node_id) continue;
        const auto* wrapper = FindNode(nodes, node_id);
        if (!wrapper) return false;
        const auto& pins = input ? wrapper->inputs : wrapper->outputs;
        const auto& mappings = input ? data.input_pin_mappings : data.output_pin_mappings;
        for (size_t i = 0; i < pins.size() && i < mappings.size(); ++i) {
            if (pins[i].id != pin_id) continue;
            for (const auto& member : data.internal_nodes) {
                const auto* live = FindNode(nodes, member.id);
                if (!live) continue;
                const auto& candidates = input ? live->inputs : live->outputs;
                for (const auto& pin : candidates) if (pin.id == mappings[i]) {
                    node_id = live->id;
                    pin_id = pin.id;
                    return true;
                }
            }
        }
        return false;
    }
    return true;
}
}

bool IsExpandedSubgraph(int id, const std::vector<SubgraphData>& subgraphs) {
    return std::any_of(subgraphs.begin(), subgraphs.end(), [id](const SubgraphData& data) {
        return data.subgraph_node_id == id && data.expanded;
    });
}

std::optional<NodeLink> DisplaySubgraphLink(const NodeLink& link, const std::vector<MLNode>& nodes,
                             const std::vector<SubgraphData>& subgraphs) {
    NodeLink display = link;
    if (!ProjectEndpoint(display.from_node, display.from_pin, false, nodes, subgraphs) ||
        !ProjectEndpoint(display.to_node, display.to_pin, true, nodes, subgraphs)) return std::nullopt;
    return display;
}

bool CrossesExpandedSubgraphBoundary(int from, int to, const std::vector<SubgraphData>& subgraphs) {
    return ExpandedOwner(from, subgraphs) != ExpandedOwner(to, subgraphs);
}

SubgraphFrameActions DrawExpandedSubgraphFrames(const std::vector<MLNode>& nodes,
    const std::vector<SubgraphData>& subgraphs, float zoom, const std::string& busy_reason, int selected) {
    SubgraphFrameActions actions;
    auto* draw = ImGui::GetWindowDrawList();
    const int original_channel = draw->_Splitter._Current;
    const ImVec2 original_cursor = ImGui::GetCursorScreenPos();
    // ImNodes reserves channel zero for canvas backgrounds and links. A frame
    // must not become an ImNodes node: its hit rectangle would cover its children.
    draw->ChannelsSetCurrent(0);
    for (const auto& data : subgraphs) {
        if (!data.expanded) continue;
        const auto* wrapper = FindNode(nodes, data.subgraph_node_id);
        if (!wrapper) continue;
        ImVec2 minimum(FLT_MAX, FLT_MAX), maximum(-FLT_MAX, -FLT_MAX);
        bool any = false;
        for (const auto& member : data.internal_nodes) {
            const auto* node = FindNode(nodes, member.id);
            if (!node) continue;
            const auto pos = ImNodes::GetNodeScreenSpacePos(node->id);
            const auto size = ImNodes::GetNodeDimensions(node->id);
            const auto label = ImGui::CalcTextSize(node->name.c_str());
            const float label_width = std::max(size.x, label.x);
            const float left = pos.x - (label_width - size.x) * 0.5f;
            float right = left + label_width;
            float bottom = pos.y + size.y + 18.0f * zoom;
            if (!node->description.empty()) {
                const float width = std::max(size.x * 1.5f, 220.0f * zoom);
                const auto text = ImGui::GetFont()->CalcTextSizeA(14.0f * zoom,
                    FLT_MAX, width - 8.0f * zoom, node->description.c_str());
                right = std::max(right, pos.x + width);
                bottom = std::max(bottom, pos.y + size.y + text.y + 16.0f * zoom);
            }
            minimum.x = std::min(minimum.x, left);
            minimum.y = std::min(minimum.y, pos.y - label.y - 4.0f * zoom);
            maximum.x = std::max(maximum.x, right);
            maximum.y = std::max(maximum.y, bottom);
            any = true;
        }
        if (!any) continue;
        const float padding = 18.0f * zoom;
        const float header = ImGui::GetFrameHeight() + 12.0f * zoom;
        minimum.x -= padding; minimum.y -= padding + header;
        maximum.x += padding; maximum.y += padding;
        const std::string title = wrapper->name + "  (" + std::to_string(data.internal_nodes.size()) + " nodes)";
        const float button_width = ImGui::CalcTextSize("Collapse").x + 24.0f * zoom;
        maximum.x = std::max(maximum.x, minimum.x + ImGui::CalcTextSize(title.c_str()).x + button_width + 42.0f * zoom);
        const ImVec2 header_end(maximum.x, minimum.y + header);
        draw->AddRectFilled(minimum, maximum, NodeStyleColour(ImNodesCol_TitleBar, 30.0f / 255.0f), 7.0f * zoom);
        const bool picked = selected == data.subgraph_node_id;
        draw->AddRectFilled(minimum, header_end,
                            NodeStyleColour(picked ? ImNodesCol_TitleBarSelected : ImNodesCol_TitleBar, 235.0f / 255.0f),
                            7.0f * zoom, ImDrawFlags_RoundCornersTop);
        draw->AddRect(minimum, maximum, NodeStyleColour(ImNodesCol_TitleBarSelected, 210.0f / 255.0f), 7.0f * zoom, 0, 1.5f * zoom);
        draw->AddText(ImVec2(minimum.x + 10.0f * zoom, minimum.y + 6.0f * zoom),
                      ImGui::GetColorU32(ImGuiCol_Text), title.c_str());
        ImGui::PushID(data.subgraph_node_id);
        // The header, left of the button, selects the subgraph: its Properties
        // view (role, steps, Collapse, the role switch) while it is expanded.
        const float header_w = maximum.x - minimum.x - button_width - 12.0f * zoom;
        ImGui::SetCursorScreenPos(minimum);
        if (header_w > 0.0f && ImGui::InvisibleButton("##frame_header", ImVec2(header_w, header))) {
            actions.select = data.subgraph_node_id;
        }
        if (ImGui::IsItemHovered()) {
            actions.header_hovered = true;
            ImGui::SetTooltip("Click to show this subgraph in Properties");
        }
        ImGui::SetCursorScreenPos(ImVec2(maximum.x - button_width - 6.0f * zoom, minimum.y + 6.0f * zoom));
        const char* collapse_tip = busy_reason.empty()
            ? "Show this group as one node. Collapse first to change external connections."
            : busy_reason.c_str();
        if (ui::SecondaryButton("Collapse", busy_reason.empty(), busy_reason.c_str(), ui::ButtonSize::Small, button_width)) {
            actions.collapse = data.subgraph_node_id;
        }
        if (busy_reason.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", collapse_tip);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(original_cursor);
    draw->ChannelsSetCurrent(original_channel);
    return actions;
}

ImVec4 EstimateExpandedFrame(const std::vector<ImVec2>& steps) {
    // As DrawExpandedSubgraphFrames draws it at zoom 1: a step's label above
    // it, its name below, the frame's padding and header around them all.
    ImVec4 frame(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const ImVec2& p : steps) {
        frame.x = std::min(frame.x, p.x - 60.0f);
        frame.y = std::min(frame.y, p.y - 30.0f);
        frame.z = std::max(frame.z, p.x + 220.0f);
        frame.w = std::max(frame.w, p.y + 110.0f);
    }
    if (steps.empty()) return ImVec4(0, 0, 0, 0);
    frame.x -= 18.0f;
    frame.y -= 18.0f + 30.0f;
    frame.z += 18.0f;
    frame.w += 18.0f;
    return frame;
}

std::map<int, ImVec2> MakeRoomForFrame(const ImVec4& frame, const std::vector<PlacedNode>& outside, float gap) {
    std::map<int, ImVec2> moved;
    const auto covered = [&](const PlacedNode& n) {
        return n.pos.x < frame.z && n.pos.x + n.size.x > frame.x && n.pos.y < frame.w && n.pos.y + n.size.y > frame.y;
    };
    float leftmost = FLT_MAX;
    for (const auto& n : outside)
        if (covered(n)) leftmost = std::min(leftmost, n.pos.x);
    if (leftmost == FLT_MAX) return moved;
    const float shift = frame.z + gap - leftmost;
    for (const auto& n : outside) {
        // The frame's band: the rows it spans, so far-off rows stay put.
        const bool in_band = n.pos.y < frame.w && n.pos.y + n.size.y > frame.y;
        if (covered(n) || (in_band && n.pos.x >= leftmost)) moved[n.id] = ImVec2(n.pos.x + shift, n.pos.y);
    }
    return moved;
}
}
