#include "subgraph_presentation.h"
#include <imnodes.h>
#include <algorithm>
#include <cfloat>

namespace gui::detail {
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

int DrawExpandedSubgraphFrames(const std::vector<MLNode>& nodes,
    const std::vector<SubgraphData>& subgraphs, float zoom, const std::string& busy_reason) {
    int collapse = -1;
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
        draw->AddRectFilled(minimum, maximum, IM_COL32(25, 85, 95, 30), 7.0f * zoom);
        draw->AddRectFilled(minimum, header_end, IM_COL32(25, 85, 95, 235), 7.0f * zoom,
                            ImDrawFlags_RoundCornersTop);
        draw->AddRect(minimum, maximum, IM_COL32(65, 170, 185, 210), 7.0f * zoom, 0, 1.5f * zoom);
        draw->AddText(ImVec2(minimum.x + 10.0f * zoom, minimum.y + 6.0f * zoom),
                      IM_COL32(235, 245, 250, 255), title.c_str());
        ImGui::PushID(data.subgraph_node_id);
        ImGui::SetCursorScreenPos(ImVec2(maximum.x - button_width - 6.0f * zoom, minimum.y + 6.0f * zoom));
        ImGui::BeginDisabled(!busy_reason.empty());
        if (ImGui::Button("Collapse", ImVec2(button_width, ImGui::GetFrameHeight()))) collapse = data.subgraph_node_id;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", busy_reason.empty()
                ? "Show this group as one node. Collapse first to change external connections."
                : busy_reason.c_str());
        }
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(original_cursor);
    draw->ChannelsSetCurrent(original_channel);
    return collapse;
}
}
