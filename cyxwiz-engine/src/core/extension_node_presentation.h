#pragma once

// What the editor shows for extension nodes (TOFIX125 P1 step 1.5): the
// palette entries (search popup, right-click menu, Node Browser), the canvas
// look, the Info panel facts and the Properties card for a node whose
// extension is not installed. Pure data in and out: no ImGui, no registry
// singletons. Approved mockup: https://claude.ai/artifact/EtpzDhx9brMw43wA5Dt9Xy

#include "extension_node_contract.h"
#include "graph_model.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Words shared by every screen.
inline constexpr const char* kExtensionNotInstalledLabel = "Not installed";
inline constexpr const char* kExtensionPluginBadge = "Plugin";

// One addable extension node in a palette.
struct ExtensionPaletteEntry {
    std::string type_id;
    std::string name;
    std::string group;           // the provider's own category ("RL / Simulation")
    std::string category_label;  // "Plugin/" + group, as the search popup groups it
    std::string keywords;        // lower-case search text
    std::string description;
    std::string pins_summary;    // "In: qpos, qvel. Out: reward"
    std::string hint;            // "Double-click or drag to add"
    uint32_t color = 0;          // IM_COL32 layout (ABGR), as plugins declare it
};

// Entries ordered by group, then name ignoring case. Groups keep the order of
// their first appearance in the provider's list.
std::vector<ExtensionPaletteEntry> BuildExtensionPalette(
    const std::vector<ExtensionNodeDescriptor>& descriptors);

// Case-insensitive match on name, group, keywords and description.
bool ExtensionPaletteEntryMatches(const ExtensionPaletteEntry& entry, const std::string& query);

// "5 nodes from MuJoCo Simulation", "3 nodes from 2 plugins", "No nodes".
// provider_names: display name per provider id (the id when unknown).
std::string ExtensionPaletteSummary(
    const std::vector<ExtensionPaletteEntry>& entries,
    const std::vector<std::pair<std::string, std::string>>& provider_names);

// How the canvas draws an extension node.
struct ExtensionCanvasStyle {
    uint32_t box_color = 0;      // IM_COL32 layout
    bool missing = false;
    uint32_t outline_color = 0;  // 0: the normal outline
    std::string status_line;     // under the box; empty when installed
};

inline constexpr uint32_t kExtensionMissingBoxColor = 0xFF483D3AU;    // IM_COL32(58, 61, 72)
inline constexpr uint32_t kExtensionMissingOutlineColor = 0xFF4AA1F5U;  // IM_COL32(245, 161, 74)

// descriptor: the registered descriptor, or nullptr when not registered.
// fallback_color: the colour the canvas used before (steel blue).
ExtensionCanvasStyle BuildExtensionCanvasStyle(const gui::MLNode& node,
                                               const ExtensionNodeDescriptor* descriptor,
                                               uint32_t fallback_color);

// Who provides a node, as the Plugin Manager knows it.
struct ExtensionProviderInfo {
    std::string name;
    std::string version;
    std::string author;
};

// The Info panel's extra facts for an extension node.
struct ExtensionInfoFacts {
    std::string category_line;                 // "Plugins / Simulation / Control"
    std::vector<std::string> chips;            // "Plugin", "Simulation node", ...
    std::vector<std::pair<std::string, std::string>> provided_by;  // key/value rows
};

ExtensionInfoFacts BuildExtensionInfoFacts(const ExtensionNodeDescriptor& descriptor,
                                           const ExtensionProviderInfo& provider);

// The Properties card for a node whose extension is not installed.
struct ExtensionMissingCard {
    std::string title;                         // "Not installed"
    std::vector<std::string> lines;
    std::vector<std::pair<std::string, std::string>> details;   // under Details
    std::vector<std::pair<std::string, std::string>> settings;  // saved parameters
    std::vector<std::pair<std::string, std::string>> pins;      // name, "(Tensor, in)"
};

ExtensionMissingCard BuildExtensionMissingCard(const gui::MLNode& node);

// True when the node is an extension node whose type name (the part after
// "provider:") equals type_name exactly.
bool IsExtensionTypeName(const gui::MLNode& node, const std::string& type_name);

}  // namespace cyxwiz
