#pragma once
#include "node_editor.h"

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
}
