#include "plugin_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace cyxwiz {

namespace {

constexpr int kStartupListVersion = 1;

std::string Count(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

PluginStartupEntry* Find(PluginStartupList& list, const std::string& id) {
    for (auto& entry : list.entries) {
        if (entry.id == id) return &entry;
    }
    return nullptr;
}

}  // namespace

PluginStartupList ParsePluginStartupList(const std::string& json_text, std::string* error) {
    PluginStartupList list;
    const auto fail = [&](const std::string& reason) {
        if (error) *error = reason;
        return PluginStartupList{};
    };
    const auto json = nlohmann::json::parse(json_text, nullptr, false);
    if (json.is_discarded() || !json.is_object()) return fail("not a JSON object");
    if (json.value("version", 0) != kStartupListVersion) return fail("unknown version");
    const auto plugins = json.find("plugins");
    if (plugins == json.end() || !plugins->is_array()) return fail("no plugin list");
    for (const auto& item : *plugins) {
        if (!item.is_object()) continue;
        PluginStartupEntry entry;
        entry.id = item.value("id", std::string());
        entry.folder = item.value("folder", std::string());
        entry.load_at_startup = item.value("load_at_startup", true);
        if (entry.id.empty() || Find(list, entry.id)) continue;
        list.entries.push_back(std::move(entry));
    }
    if (error) error->clear();
    return list;
}

std::string SerializePluginStartupList(const PluginStartupList& list) {
    nlohmann::json plugins = nlohmann::json::array();
    for (const auto& entry : list.entries) {
        plugins.push_back({{"id", entry.id}, {"folder", entry.folder}, {"load_at_startup", entry.load_at_startup}});
    }
    return nlohmann::json{{"version", kStartupListVersion}, {"plugins", plugins}}.dump(2);
}

void RecordPluginLoaded(PluginStartupList& list, const std::string& id, const std::string& folder) {
    if (id.empty()) return;
    if (auto* entry = Find(list, id)) {
        entry->folder = folder;
        return;
    }
    list.entries.push_back({id, folder, true});
}

void RecordPluginRemoved(PluginStartupList& list, const std::string& id) {
    list.entries.erase(std::remove_if(list.entries.begin(), list.entries.end(),
                                      [&](const PluginStartupEntry& e) { return e.id == id; }),
                       list.entries.end());
}

void SetPluginLoadAtStartup(PluginStartupList& list, const std::string& id, bool on) {
    if (auto* entry = Find(list, id)) entry->load_at_startup = on;
}

bool PluginLoadsAtStartup(const PluginStartupList& list, const std::string& id) {
    for (const auto& entry : list.entries) {
        if (entry.id == id) return entry.load_at_startup;
    }
    return false;
}

std::vector<PluginStartupEntry> PluginsToLoadAtStartup(const PluginStartupList& list) {
    std::vector<PluginStartupEntry> result;
    for (const auto& entry : list.entries) {
        if (entry.load_at_startup && !entry.folder.empty()) result.push_back(entry);
    }
    return result;
}

std::string PluginStartupNote(bool on) {
    return on ? "Graphs that use its nodes open ready after a restart."
              : "After a restart its nodes show as not installed until you load it here.";
}

std::string PluginAddsLine(const PluginContribution& c) {
    std::vector<std::string> parts;
    if (c.windows > 0) parts.push_back(Count(c.windows, "window", "windows"));
    if (c.trainable_nodes > 0) parts.push_back(Count(c.trainable_nodes, "trainable node", "trainable nodes"));
    if (c.simulation_nodes > 0) parts.push_back(Count(c.simulation_nodes, "simulation node", "simulation nodes"));
    if (parts.empty()) return {};
    std::string text = "Adds: ";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) text += ", ";
        text += parts[i];
    }
    if (!c.node_categories.empty()) {
        text += " (";
        for (size_t i = 0; i < c.node_categories.size(); ++i) {
            if (i > 0) text += ", ";
            text += c.node_categories[i];
        }
        text += ")";
    }
    return text;
}

std::string PluginMenuDetail(const PluginContribution& c) {
    const size_t nodes = c.trainable_nodes + c.simulation_nodes;
    std::string text;
    if (c.windows > 0) text = Count(c.windows, "window", "windows");
    if (nodes > 0) text += (text.empty() ? "" : ", ") + Count(nodes, "node", "nodes");
    return text;
}

}  // namespace cyxwiz
