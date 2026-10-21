#pragma once

#include "node_editor.h"

namespace cyxwiz {
struct NodeMetadata;
}

namespace gui::properties_contract {

enum class PanelContractPath {
    DialogOnly,
    CustomEditor,
    MetadataRenderer,
    CustomFallbackEditor
};

bool IsDialogOnlyPropertiesNode(const cyxwiz::NodeMetadata* metadata);
bool IsCustomPropertiesNode(const cyxwiz::NodeMetadata* metadata);
PanelContractPath ClassifyPanelContractPath(
    NodeType type,
    const cyxwiz::NodeMetadata* metadata);
const char* PanelContractPathName(PanelContractPath path);

} // namespace gui::properties_contract
