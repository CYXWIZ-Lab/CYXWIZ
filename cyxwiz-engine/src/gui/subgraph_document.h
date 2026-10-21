#pragma once

#include "node_editor.h"
#include <nlohmann/json.hpp>

namespace gui::detail {
// Visual composition persistence only; this does not lower subgraphs for execution.
void WriteEditorGraphContent(nlohmann::json& document,
    const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links,
    const std::vector<SubgraphData>& subgraphs, const std::map<int, ImVec2>& positions);
nlohmann::json FlattenSubgraphDocument(const nlohmann::json& document);
void RestoreSubgraphPins(const nlohmann::json& document,
    std::vector<MLNode>& nodes, int& next_pin_id);
std::vector<SubgraphData> RestoreSubgraphContents(const nlohmann::json& document,
    std::vector<MLNode>& nodes, std::vector<NodeLink>& links);
// Expanded nodes are the editable version; refresh the stored copy before collapse.
void CaptureExpandedSubgraph(SubgraphData& data,
    const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);
}
