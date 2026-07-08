#pragma once

#include "node_editor.h"

namespace cyxwiz {
struct NodeMetadata;
}

namespace gui::properties_contract {

enum class PanelContractPath {
    DialogOnly,
    CustomSequenceEditor,
    MetadataRenderer,
    CustomFallbackEditor
};

bool IsDialogOnlyPropertiesNode(NodeType type);
bool IsCustomSequencePropertiesNode(NodeType type);
PanelContractPath ClassifyPanelContractPath(
    NodeType type,
    const cyxwiz::NodeMetadata* metadata);
const char* PanelContractPathName(PanelContractPath path);

} // namespace gui::properties_contract
