#include "extension_node_presentation.h"

#include "extension_node_document.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
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
        entry.python_bundle = descriptor.source == ExtensionSource::PythonBundle;
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
    std::set<std::string> plugins;
    std::set<std::string> bundles;
    for (const auto& entry : entries) {
        std::string provider_id;
        std::string type_name;
        if (SplitExtensionTypeId(entry.type_id, provider_id, type_name)) {
            (entry.python_bundle ? bundles : plugins).insert(provider_id);
        }
    }
    const std::string count =
        std::to_string(entries.size()) + (entries.size() == 1 ? " node" : " nodes");
    if (plugins.size() + bundles.size() == 1) {
        return count + " from " + ProviderName(plugins.empty() ? *bundles.begin() : *plugins.begin(), provider_names);
    }
    std::string from;
    if (!plugins.empty()) from = std::to_string(plugins.size()) + (plugins.size() == 1 ? " plugin" : " plugins");
    if (!bundles.empty()) {
        from += (from.empty() ? "" : " and ") + std::to_string(bundles.size()) +
                (bundles.size() == 1 ? " Python bundle" : " Python bundles");
    }
    return count + " from " + from;
}

ExtensionCanvasStyle BuildExtensionCanvasStyle(const gui::MLNode& node,
                                               const ExtensionNodeDescriptor* descriptor,
                                               uint32_t fallback_color) {
    ExtensionCanvasStyle style;
    if (descriptor == nullptr || node.extension_missing) {
        style.missing = true;
        style.status_line = kExtensionNotInstalledLabel;
        return style;
    }
    // A declared colour with no alpha means "not declared".
    style.box_color = (descriptor->color >> 24) != 0 ? descriptor->color : fallback_color;
    return style;
}

ExtensionInfoFacts BuildExtensionInfoFacts(const ExtensionNodeDescriptor& descriptor,
                                           const ExtensionProviderInfo& provider,
                                           const ExtensionPythonState* python) {
    ExtensionInfoFacts facts;
    if (descriptor.source == ExtensionSource::PythonBundle) {
        const bool approved = python && python->approved;
        facts.category_line = descriptor.menu_category.empty()
            ? std::string("Python nodes")
            : "Python nodes / " + descriptor.menu_category;
        facts.chips.push_back(kExtensionPythonBadge);
        facts.chips.push_back(KindChip(descriptor.kind));
        if (!approved) facts.chips.push_back("Code not approved");
        const auto add = [&](const char* key, const std::string& value) {
            if (!value.empty()) facts.provided_by.emplace_back(key, value);
        };
        std::string folder_name = std::filesystem::path(descriptor.source_path).filename().string();
        if (folder_name.size() > 8 && folder_name.compare(folder_name.size() - 8, 8, ".cyxnode") == 0) {
            folder_name.resize(folder_name.size() - 8);
        }
        add("Python bundle", folder_name.empty() ? descriptor.provider_id
                                                 : folder_name + " (" + descriptor.provider_id + ")");
        add("Node version", descriptor.version);
        add("Approval", approved ? (python->approved_on.empty() ? std::string("Approved")
                                                                : "Approved " + python->approved_on)
                                 : std::string("Not approved yet"));
        const auto colon = descriptor.content_hash.find(':');
        const std::string digits =
            colon == std::string::npos ? descriptor.content_hash : descriptor.content_hash.substr(colon + 1);
        if (!digits.empty()) add("Code sha256", digits.substr(0, 12) + "\xE2\x80\xA6");
        add("Type id", descriptor.type_id);
        add("Folder", descriptor.source_path);
        return facts;
    }
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
    if (!node.extension_content_hash.empty()) {
        // Saved with a code hash: a Python bundle node (plugins save none).
        card.lines.push_back("Put the Python bundle " + (split ? provider_id : std::string("that provides it")) +
                             " in this project's nodes folder or in your user nodes folder, or delete the node.");
    } else {
        card.lines.push_back(split
            ? "Load the plugin " + provider_id + " in the Plugin Manager, or delete the node."
            : std::string("Load the plugin that provides it in the Plugin Manager, or delete the node."));
    }

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
