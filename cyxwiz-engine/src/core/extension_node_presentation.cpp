#include "extension_node_presentation.h"

#include "extension_node_document.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

namespace cyxwiz {

namespace {

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string JoinNames(const std::vector<PortDefinition>& ports) {
    std::string text;
    for (const auto& port : ports) {
        if (!text.empty()) text += ", ";
        text += port.name;
    }
    return text;
}

std::string PinsSummary(const ExtensionNodeDescriptor& descriptor) {
    if (descriptor.supports_dynamic_pins) {
        return "Pins follow the " + (descriptor.dynamic_pin_trigger.empty()
                                         ? std::string("settings")
                                         : descriptor.dynamic_pin_trigger) +
               " setting";
    }
    const auto& inputs = descriptor.metadata.inputs;
    const auto& outputs = descriptor.metadata.outputs;
    std::string text;
    if (!inputs.empty()) text += "In: " + JoinNames(inputs) + ".";
    if (!outputs.empty()) {
        if (!text.empty()) text += " ";
        text += "Out: " + JoinNames(outputs);
    }
    return text.empty() ? "No pins" : text;
}

const char* KindChip(ExtensionNodeKind kind) {
    return kind == ExtensionNodeKind::Signal ? "Simulation node" : "Trainable node";
}

std::string ProviderName(const std::string& provider_id,
                         const std::vector<std::pair<std::string, std::string>>& names) {
    for (const auto& [id, name] : names) {
        if (id == provider_id && !name.empty()) return name;
    }
    return provider_id;
}

}  // namespace

std::vector<ExtensionPaletteEntry> BuildExtensionPalette(
    const std::vector<ExtensionNodeDescriptor>& descriptors) {
    std::map<std::string, size_t> group_order;
    std::vector<ExtensionPaletteEntry> entries;
    entries.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        ExtensionPaletteEntry entry;
        entry.type_id = descriptor.type_id;
        entry.name = descriptor.metadata.name.empty() ? descriptor.type_name : descriptor.metadata.name;
        entry.group = descriptor.menu_category.empty() ? "Other" : descriptor.menu_category;
        entry.category_label = "Plugin/" + entry.group;
        entry.description = descriptor.metadata.brief_description;
        entry.keywords = Lower(descriptor.type_name + " " + entry.name + " " + entry.group + " " +
                               entry.description + " plugin " + descriptor.provider_id);
        entry.pins_summary = PinsSummary(descriptor);
        entry.hint = "Double-click or drag to add";
        entry.color = descriptor.color;
        group_order.emplace(entry.group, group_order.size());
        entries.push_back(std::move(entry));
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [&](const ExtensionPaletteEntry& a, const ExtensionPaletteEntry& b) {
                         const size_t ga = group_order[a.group];
                         const size_t gb = group_order[b.group];
                         if (ga != gb) return ga < gb;
                         return Lower(a.name) < Lower(b.name);
                     });
    return entries;
}

bool ExtensionPaletteEntryMatches(const ExtensionPaletteEntry& entry, const std::string& query) {
    const std::string needle = Lower(query);
    if (needle.empty()) return true;
    return entry.keywords.find(needle) != std::string::npos;
}

std::string ExtensionPaletteSummary(
    const std::vector<ExtensionPaletteEntry>& entries,
    const std::vector<std::pair<std::string, std::string>>& provider_names) {
    if (entries.empty()) return "No nodes";
    std::set<std::string> providers;
    for (const auto& entry : entries) {
        std::string provider_id;
        std::string type_name;
        if (SplitExtensionTypeId(entry.type_id, provider_id, type_name)) providers.insert(provider_id);
    }
    const std::string count =
        std::to_string(entries.size()) + (entries.size() == 1 ? " node" : " nodes");
    if (providers.size() == 1) {
        return count + " from " + ProviderName(*providers.begin(), provider_names);
    }
    return count + " from " + std::to_string(providers.size()) + " plugins";
}

ExtensionCanvasStyle BuildExtensionCanvasStyle(const gui::MLNode& node,
                                               const ExtensionNodeDescriptor* descriptor,
                                               uint32_t fallback_color) {
    ExtensionCanvasStyle style;
    if (descriptor == nullptr || node.extension_missing) {
        style.missing = true;
        style.box_color = kExtensionMissingBoxColor;
        style.outline_color = kExtensionMissingOutlineColor;
        style.status_line = kExtensionNotInstalledLabel;
        return style;
    }
    // A declared colour with no alpha means "not declared".
    style.box_color = (descriptor->color >> 24) != 0 ? descriptor->color : fallback_color;
    return style;
}

ExtensionInfoFacts BuildExtensionInfoFacts(const ExtensionNodeDescriptor& descriptor,
                                           const ExtensionProviderInfo& provider) {
    ExtensionInfoFacts facts;
    facts.category_line = descriptor.menu_category.empty()
        ? std::string("Plugins")
        : "Plugins / " + descriptor.menu_category;
    facts.chips.push_back(kExtensionPluginBadge);
    facts.chips.push_back(KindChip(descriptor.kind));
    if (descriptor.kind == ExtensionNodeKind::Signal) facts.chips.push_back("Cannot train");
    if (descriptor.supports_dynamic_pins) facts.chips.push_back("Pins follow its settings");

    const auto add = [&](const char* key, const std::string& value) {
        if (!value.empty()) facts.provided_by.emplace_back(key, value);
    };
    add("Plugin", provider.name.empty() ? descriptor.provider_id : provider.name);
    add("Version", provider.version);
    add("Author", provider.author);
    add("Node version", descriptor.version);
    add("Type id", descriptor.type_id);
    add("Source", descriptor.source_path);
    return facts;
}

ExtensionMissingCard BuildExtensionMissingCard(const gui::MLNode& node) {
    ExtensionMissingCard card;
    card.title = kExtensionNotInstalledLabel;
    card.lines.push_back(
        "This graph uses a node from an extension that is not loaded. The node keeps "
        "its pins, links and settings.");
    std::string provider_id;
    std::string type_name;
    const bool split = SplitExtensionTypeId(node.extension_type_id, provider_id, type_name);
    card.lines.push_back(split
        ? "Load the plugin " + provider_id + " in the Plugin Manager, or delete the node."
        : std::string("Load the plugin that provides it in the Plugin Manager, or delete the node."));

    card.details.emplace_back("Type id", node.extension_type_id.empty() ? "(none)" : node.extension_type_id);
    card.details.emplace_back("Saved version", node.extension_version.empty() ? "(none)" : node.extension_version);
    card.details.emplace_back(
        "Saved pins",
        std::to_string(node.inputs.size()) + (node.inputs.size() == 1 ? " input, " : " inputs, ") +
            std::to_string(node.outputs.size()) + (node.outputs.size() == 1 ? " output" : " outputs"));
    card.details.emplace_back("Code", "CW-X-0601");

    for (const auto& [key, value] : node.parameters) {
        if (key.rfind("_meta_", 0) == 0) continue;
        card.settings.emplace_back(key, value);
    }
    for (const auto& pin : node.inputs) {
        card.pins.emplace_back(pin.name, std::string("(") + PinTypeToText(pin.type) + ", in)");
    }
    for (const auto& pin : node.outputs) {
        card.pins.emplace_back(pin.name, std::string("(") + PinTypeToText(pin.type) + ", out)");
    }
    return card;
}

bool IsExtensionTypeName(const gui::MLNode& node, const std::string& type_name) {
    if (node.type != gui::NodeType::PluginCustom) return false;
    std::string provider_id;
    std::string name;
    return SplitExtensionTypeId(node.extension_type_id, provider_id, name) && name == type_name;
}

}  // namespace cyxwiz
