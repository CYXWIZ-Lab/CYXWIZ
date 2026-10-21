#include "plugin_session.h"

#include "plugin_manager.h"
#include "registries/plugin_panel_registry.h"
#include "../core/extension_node_registry.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cyxwiz::plugin {

std::filesystem::path PluginStartupListPath() {
#ifdef _WIN32
    if (auto* appdata = std::getenv("APPDATA"))
        return std::filesystem::path(appdata) / "cyxwiz" / "plugin_startup.json";
#else
    if (auto* home = std::getenv("HOME"))
        return std::filesystem::path(home) / ".cyxwiz" / "plugin_startup.json";
#endif
    return "plugin_startup.json";
}

PluginStartupList ReadPluginStartupList() {
    const auto path = PluginStartupListPath();
    std::ifstream file(path);
    if (!file) return {};
    std::stringstream text;
    text << file.rdbuf();
    std::string error;
    auto list = ParsePluginStartupList(text.str(), &error);
    if (!error.empty()) spdlog::warn("Plugin startup list {} was not read: {}", path.string(), error);
    return list;
}

void WritePluginStartupList(const PluginStartupList& list) {
    const auto path = PluginStartupListPath();
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        spdlog::warn("Plugin startup list {} could not be saved", path.string());
        return;
    }
    file << SerializePluginStartupList(list);
}

void RecordLoadedPlugins(const PluginManager& manager) {
    auto list = ReadPluginStartupList();
    for (const auto* p : manager.GetAllPlugins()) {
        if (p) RecordPluginLoaded(list, p->manifest.id, p->plugin_dir.generic_string());
    }
    WritePluginStartupList(list);
}

void RecordPluginUnloaded(const std::string& plugin_id) {
    auto list = ReadPluginStartupList();
    RecordPluginRemoved(list, plugin_id);
    WritePluginStartupList(list);
}

void SetLoadAtStartup(const std::string& plugin_id, const std::string& folder, bool on) {
    auto list = ReadPluginStartupList();
    RecordPluginLoaded(list, plugin_id, folder);
    SetPluginLoadAtStartup(list, plugin_id, on);
    WritePluginStartupList(list);
}

bool LoadsAtStartup(const std::string& plugin_id) {
    return PluginLoadsAtStartup(ReadPluginStartupList(), plugin_id);
}

size_t LoadStartupPlugins(PluginManager& manager) {
    size_t loaded = 0;
    const auto entries = PluginsToLoadAtStartup(ReadPluginStartupList());
    for (const auto& entry : entries) {
        if (manager.GetLoadedPlugin(entry.id)) continue;
        if (!std::filesystem::exists(std::filesystem::path(entry.folder) / "plugin.json")) {
            spdlog::warn("Plugin {} was not loaded at startup: {} has no plugin.json", entry.id, entry.folder);
            continue;
        }
        if (!manager.LoadPlugin(entry.folder)) {
            spdlog::debug("Plugin {} could not be loaded at startup from {} (the loader said why)", entry.id, entry.folder);
            continue;
        }
        // A plugin whose OnLoad fails (or crashes) is not counted as loaded.
        if (!manager.InitializePlugin(entry.id)) {
            spdlog::debug("Plugin {} did not start: its OnLoad failed", entry.id);
            continue;
        }
        ++loaded;
    }
    if (!entries.empty())
        spdlog::info("Plugins: {} of {} loaded at startup", loaded, entries.size());
    return loaded;
}

PluginContribution DescribeContribution(const std::string& plugin_id) {
    PluginContribution c;
    c.windows = PluginPanelRegistry::Instance().CountForPlugin(plugin_id);
    for (const auto& d : ExtensionNodeRegistry::Instance().All()) {
        if (d.provider_id != plugin_id) continue;
        if (d.kind == ExtensionNodeKind::TensorLayer) ++c.trainable_nodes; else ++c.simulation_nodes;
        if (!d.menu_category.empty() &&
            std::find(c.node_categories.begin(), c.node_categories.end(), d.menu_category) == c.node_categories.end()) {
            c.node_categories.push_back(d.menu_category);
        }
    }
    return c;
}

}  // namespace cyxwiz::plugin
