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

std::optional<std::string> ExtensionCodeExportRefusal(const gui::MLNode& node) {
    if (node.type != gui::NodeType::PluginCustom) return std::nullopt;
    const std::string name = node.name.empty() ? node.extension_type_id : node.name;
    const auto descriptor = node.extension_missing ? std::nullopt
                                                   : ExtensionNodeRegistry::Instance().Find(node.extension_type_id);
    if (!descriptor) {
        return name + ": the extension that provides it (" + node.extension_type_id +
               ") is not installed, so code export cannot include it";
    }
    if (descriptor->kind == ExtensionNodeKind::TensorLayer) {
        return name + ": code export cannot include an extension node's own code yet (" + node.extension_type_id +
               "); train it in Studio, or export a CyxWiz package";
    }
    return std::nullopt;
}

std::string SimulationNodeCodeComment(const gui::MLNode& node, const char* framework, const std::string& indent) {
    std::string out = indent + "# " + node.name + " (" + node.extension_type_id +
                      ") is a simulation node, not part of this model";
    const std::string text = node.extension_type_id.empty()
        ? std::string()
        : ExtensionNodeRegistry::Instance().GenerateCode(node.extension_type_id, node.parameters, framework);
    if (text.empty()) return out + "\n";
    out += "; its plugin's code:\n";
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        const std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!(end == std::string::npos && line.empty())) out += indent + "#   " + line + "\n";
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

}  // namespace cyxwiz
