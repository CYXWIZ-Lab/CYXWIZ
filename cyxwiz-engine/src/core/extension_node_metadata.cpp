#include "extension_node_metadata.h"

#include "extension_node_registry.h"
#include "node_metadata_registry.h"

namespace cyxwiz {

namespace {

NodeMetadata MissingExtensionMetadata(const gui::MLNode& node) {
    NodeMetadata metadata;
    metadata.type = gui::NodeType::PluginCustom;
    metadata.category = gui::NodeCategory::Plugin;
    metadata.name = node.name;
    metadata.brief_description =
        "Extension node " + node.extension_type_id + " is not installed.";
    metadata.help_text =
        "The graph keeps this node's pins, links and parameters. Install and "
        "approve the extension that provides " + node.extension_type_id +
        " to use it.";
    metadata.status = NodeImplementationStatus::Template;
    metadata.badge = "Not installed";
    for (const auto& pin : node.inputs) {
        metadata.inputs.emplace_back(pin.name, pin.type, pin.is_required, pin.description);
    }
    for (const auto& pin : node.outputs) {
        metadata.outputs.emplace_back(pin.name, pin.type, pin.is_required, pin.description);
    }
    return metadata;
}

}  // namespace

std::optional<NodeMetadata> ResolveNodeMetadata(const gui::MLNode& node) {
    if (node.type != gui::NodeType::PluginCustom) {
        auto& catalog = NodeMetadataRegistry::Instance();
        if (!catalog.IsInitialized()) catalog.Initialize();
        const NodeMetadata* metadata = catalog.GetMetadata(node.type);
        if (metadata == nullptr) return std::nullopt;
        return *metadata;
    }

    if (const auto descriptor = ExtensionNodeRegistry::Instance().Find(node.extension_type_id)) {
        return descriptor->metadata;
    }
    return MissingExtensionMetadata(node);
}

}  // namespace cyxwiz
