#include "properties_contract.h"

#include "../core/node_metadata.h"

namespace gui::properties_contract {

bool IsDialogOnlyPropertiesNode(NodeType type) {
    switch (type) {
        case NodeType::DataInput:
        case NodeType::DataOutput:
        case NodeType::DataConvert:
        case NodeType::TextTokenizer:
        case NodeType::TextVocabulary:
        case NodeType::TextPadding:
        case NodeType::Embedding:
            return true;
        default:
            return false;
    }
}

bool IsCustomSequencePropertiesNode(NodeType type) {
    switch (type) {
        case NodeType::NERSequenceBuilder:
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return true;
        default:
            return false;
    }
}

PanelContractPath ClassifyPanelContractPath(
    NodeType type,
    const cyxwiz::NodeMetadata* metadata) {
    if (IsDialogOnlyPropertiesNode(type)) {
        return PanelContractPath::DialogOnly;
    }
    if (IsCustomSequencePropertiesNode(type)) {
        return PanelContractPath::CustomSequenceEditor;
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
        case PanelContractPath::CustomSequenceEditor:
            return "custom_sequence_editor";
        case PanelContractPath::MetadataRenderer:
            return "metadata_renderer";
        case PanelContractPath::CustomFallbackEditor:
            return "custom_fallback_editor";
    }
    return "unknown";
}

} // namespace gui::properties_contract
