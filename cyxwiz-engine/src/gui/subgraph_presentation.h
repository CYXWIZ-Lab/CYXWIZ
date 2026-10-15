#pragma once
#include "node_editor.h"

#include <map>

namespace gui::detail {
bool IsExpandedSubgraph(int node_id, const std::vector<SubgraphData>& subgraphs);
// Presentation only: canonical links retain their wrapper endpoints for persistence.
std::optional<NodeLink> DisplaySubgraphLink(const NodeLink& link, const std::vector<MLNode>& nodes,
                             const std::vector<SubgraphData>& subgraphs);
bool CrossesExpandedSubgraphBoundary(int from_node, int to_node,
                                    const std::vector<SubgraphData>& subgraphs);
// What a click on an expanded subgraph's frame asked for (subgraph ids, -1 for none).
struct SubgraphFrameActions {
    int collapse = -1;  // the Collapse button
    int select = -1;    // the header: show the subgraph in Properties
    bool header_hovered = false;
};
// Call after visible nodes are submitted, within the ImNodes editor scope.
// `selected` is the subgraph shown in Properties (its header is highlighted).
// The caller must mutate after EndNodeEditor.
SubgraphFrameActions DrawExpandedSubgraphFrames(const std::vector<MLNode>& nodes,
    const std::vector<SubgraphData>& subgraphs, float zoom, const std::string& busy_reason, int selected);

// Room for a subgraph that expands (TOFIX148): its frame, in grid space, is
// estimated from its steps' positions; the outside nodes it would cover move
// right of it, and so do the outside nodes right of them in the frame's band,
// keeping their order and spacing. Returns the new positions by node id.
struct PlacedNode {
    int id = -1;
    ImVec2 pos;   // grid space, top-left
    ImVec2 size;  // grid space (an estimate before the node is drawn)
};
ImVec4 EstimateExpandedFrame(const std::vector<ImVec2>& step_positions);  // x0, y0, x1, y1
std::map<int, ImVec2> MakeRoomForFrame(const ImVec4& frame, const std::vector<PlacedNode>& outside, float gap = 60.0f);
}
