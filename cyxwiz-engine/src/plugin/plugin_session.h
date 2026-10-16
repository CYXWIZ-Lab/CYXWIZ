#pragma once

// Plugins across Engine sessions (TOFIX125 P2 step 2.9) and what each loaded
// plugin adds (step 2.10). The list lives in plugin_startup.json beside
// plugin_permissions.json (%APPDATA%\cyxwiz or ~/.cyxwiz); the rules are in
// core/plugin_presentation.h.

#include "../core/plugin_presentation.h"

#include <filesystem>
#include <string>

namespace cyxwiz::plugin {

class PluginManager;

std::filesystem::path PluginStartupListPath();
PluginStartupList ReadPluginStartupList();
void WritePluginStartupList(const PluginStartupList& list);

// Records every loaded plugin (after Refresh or Install) and saves.
void RecordLoadedPlugins(const PluginManager& manager);
// The user unloaded the plugin: it no longer loads at startup.
void RecordPluginUnloaded(const std::string& plugin_id);
// Lists the plugin first when it is not listed yet.
void SetLoadAtStartup(const std::string& plugin_id, const std::string& folder, bool on);
bool LoadsAtStartup(const std::string& plugin_id);

// Loads and initialises the listed plugins that load at startup. A plugin
// needing a permission decision asks through the usual dialog. Returns the
// number loaded.
size_t LoadStartupPlugins(PluginManager& manager);

// Windows and extension nodes the plugin registered.
PluginContribution DescribeContribution(const std::string& plugin_id);

}  // namespace cyxwiz::plugin
