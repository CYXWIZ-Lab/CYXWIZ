#pragma once

// Plugins across sessions and what each one adds (TOFIX125 P2 steps 2.9 and
// 2.10, approved mockup row 5, "Plugin Manager card" and "Apps menu").
//
// The startup list remembers the plugins loaded in the Engine and whether
// each loads again when the Engine starts. It is stored as JSON next to the
// plugin permissions (plugin_startup_store.h). Pure data in, data out.

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct PluginStartupEntry {
    std::string id;
    std::string folder;  // the folder holding plugin.json
    bool load_at_startup = true;
};

struct PluginStartupList {
    std::vector<PluginStartupEntry> entries;  // in the order they were first loaded
};

// Unknown or damaged text gives an empty list and the reason in `error`.
PluginStartupList ParsePluginStartupList(const std::string& json_text, std::string* error = nullptr);
std::string SerializePluginStartupList(const PluginStartupList& list);

// A plugin was loaded: listed with startup on; a listed one keeps its choice
// and takes the new folder.
void RecordPluginLoaded(PluginStartupList& list, const std::string& id, const std::string& folder);
// The user unloaded it: it no longer loads at startup.
void RecordPluginRemoved(PluginStartupList& list, const std::string& id);
void SetPluginLoadAtStartup(PluginStartupList& list, const std::string& id, bool on);
bool PluginLoadsAtStartup(const PluginStartupList& list, const std::string& id);
std::vector<PluginStartupEntry> PluginsToLoadAtStartup(const PluginStartupList& list);

// What a loaded plugin adds to the Engine.
struct PluginContribution {
    size_t windows = 0;
    size_t trainable_nodes = 0;
    size_t simulation_nodes = 0;
    std::vector<std::string> node_categories;  // the plugin's own grouping, in order
    size_t dataset_formats = 0;                // Data Input formats (IDataProvider)
    size_t analytics = 0;                      // Explore > Plugins analytics (IAnalyticsProvider)
};

// The card's note under "Load when the Engine starts", naming what the
// plugin adds (nodes, dataset formats, analytics, windows).
std::string PluginStartupNote(bool on, const PluginContribution& contribution);

// "Adds: 2 trainable nodes (Activations)", "Adds: 1 dataset format, 1
// analytic"; empty when it adds nothing.
std::string PluginAddsLine(const PluginContribution& contribution);
// The Apps menu entry's note: "2 nodes", "2 windows, 5 nodes", "1 format, 1 analytic".
std::string PluginMenuDetail(const PluginContribution& contribution);

}  // namespace cyxwiz
