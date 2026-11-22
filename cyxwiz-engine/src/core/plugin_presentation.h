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
// The plugins to load when the Engine starts. `pack_plugins` are the plugin
// packs active in the installed runtime (id and their current folder): a listed
// one loads from that folder, whatever folder was recorded (a pack update moves
// it); an installed pack not listed yet loads too, since installing it was the
// user's choice. A plugin the user set not to load stays off.
std::vector<PluginStartupEntry> PluginsToLoadAtStartup(
    const PluginStartupList& list, const std::vector<PluginStartupEntry>& pack_plugins = {});

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

// ---- The Plugin Manager card (TOFIX129 C6, approved mockup "Plugin Manager")

enum class PluginRunState { Loaded, Running, Failed, Disabled, Unloaded };

struct PluginCardFacts {
    std::string id;
    std::string name;
    PluginRunState state = PluginRunState::Loaded;
    std::string error;          // the manager's message: why it failed, or "waits for X, which ..."
    bool awaiting_answer = false;  // a permission request for it is open or queued
};

enum class PluginTone { Good, Waiting, Problem, Off };
enum class PluginAction { None, Disable, Start, TryAgain, Enable };

struct PluginCardView {
    std::string state_word;  // "Active", "Not started", "Waiting", "Failed", "Disabled", "Unloaded"
    PluginTone tone = PluginTone::Off;
    std::string reason;      // the sentence under the description; empty when Active
    PluginAction action = PluginAction::None;
    std::string action_label;
};

PluginCardView DescribePluginCard(const PluginCardFacts& facts);

// The list in name order (case-insensitive, then id), so a card keeps its
// place when another plugin loads or unloads.
std::vector<size_t> PluginCardOrder(const std::vector<PluginCardFacts>& cards);

// "1 plugin: 1 active", "3 plugins: 1 active, 1 waiting, 1 failed"; every
// state that has a plugin, in a fixed order; "No plugins" for none.
std::string PluginFooter(const std::vector<PluginCardFacts>& cards);

// What a plugin asked for and what the user answered.
enum class PermissionAnswer { NotAsked, Allowed, Denied };

struct PluginPermissionFacts {
    std::string name;          // "Training Hooks"
    std::string description;   // "Hook into the training pipeline"
    bool needs_answer = false; // a dangerous permission: the user decides
    PermissionAnswer answer = PermissionAnswer::NotAsked;
};

struct PluginPermissionRow {
    std::string name;
    std::string description;
    std::string answer;  // "Allowed", "Denied", "Not asked yet", "Allowed (low risk, not asked)"
    bool can_change = false;
};

std::vector<PluginPermissionRow> PluginPermissionRows(const std::vector<PluginPermissionFacts>& permissions);
// The card's link: "Permissions (2 allowed)", "Permissions (1 allowed, 1 denied)"; empty when it asks for none.
std::string PluginPermissionsLink(const std::vector<PluginPermissionFacts>& permissions);

// The Unload confirmation: what closes and what stops working.
std::string PluginUnloadQuestion(const std::string& name, const PluginContribution& contribution);

}  // namespace cyxwiz
