#include "properties_contract.h"

#include "../core/node_metadata.h"

namespace gui::properties_contract {

bool IsDialogOnlyPropertiesNode(const cyxwiz::NodeMetadata* metadata) {
    return metadata &&
           metadata->properties_editor ==
               cyxwiz::NodePropertiesEditor::Dialog;
}

bool IsCustomPropertiesNode(const cyxwiz::NodeMetadata* metadata) {
    return metadata &&
           metadata->properties_editor ==
               cyxwiz::NodePropertiesEditor::Custom;
}

PanelContractPath ClassifyPanelContractPath(
    NodeType type,
    const cyxwiz::NodeMetadata* metadata) {
    (void)type;
    if (IsDialogOnlyPropertiesNode(metadata)) {
        return PanelContractPath::DialogOnly;
    }
    if (IsCustomPropertiesNode(metadata)) {
        return PanelContractPath::CustomEditor;
    }
    if (metadata && !metadata->parameters.empty()) {
        return PanelContractPath::MetadataRenderer;
    }
    return PanelContractPath::CustomFallbackEditor;
}

const char* PanelContractPathName(PanelContractPath path) {
    switch (path) {
        case PanelContractPath::DialogOnly:
            return "dialog_only";
        case PanelContractPath::CustomEditor:
            return "custom_editor";
        case PanelContractPath::MetadataRenderer:
            return "metadata_renderer";
        case PanelContractPath::CustomFallbackEditor:
            return "custom_fallback_editor";
    }
    return "unknown";
}

} // namespace gui::properties_contract
