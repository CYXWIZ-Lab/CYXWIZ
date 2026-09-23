#pragma once
#include "node_editor.h"

namespace gui::detail {
bool IsExpandedSubgraph(int node_id, const std::vector<SubgraphData>& subgraphs);
// Presentation only: canonical links retain their wrapper endpoints for persistence.
std::optional<NodeLink> DisplaySubgraphLink(const NodeLink& link, const std::vector<MLNode>& nodes,
                             const std::vector<SubgraphData>& subgraphs);
bool CrossesExpandedSubgraphBoundary(int from_node, int to_node,
                                    const std::vector<SubgraphData>& subgraphs);
// Call after visible nodes are submitted, within the ImNodes editor scope.
// Returns a requested collapse ID; the caller must mutate after EndNodeEditor.
int DrawExpandedSubgraphFrames(const std::vector<MLNode>& nodes,
    const std::vector<SubgraphData>& subgraphs, float zoom, const std::string& busy_reason);
}
