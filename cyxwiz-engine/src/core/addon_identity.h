#pragma once

// The add-ons this release knows (plugin packs: optional features such as
// Simulation (MuJoCo)): their names and words, one table for the add-on
// cards, the Plugin Manager and the Engine's log. Header-only, so the plugin
// host and the installer can name an add-on without the card model.

#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

struct AddOnIdentity {
    std::string_view plugin_id;
    std::string_view title;
    std::string_view short_name;
    std::string_view description;
};

inline constexpr AddOnIdentity kKnownAddOns[] = {
    {"com.cyxwiz.simulation.mujoco", "Simulation (MuJoCo)", "MuJoCo",
     "Physics simulation for reinforcement learning: MuJoCo environments, "
     "the 3D viewport and the Environment Library."},
};

inline const AddOnIdentity* FindAddOnIdentity(const std::string& plugin_id) {
    for (const auto& identity : kKnownAddOns) {
        if (identity.plugin_id == plugin_id) return &identity;
    }
    return nullptr;
}

// The add-ons this release knows by plugin id, in card order. A known
// add-on with no catalog pack for this platform still gets a Not published
// card, as the compute backends do.
inline const std::vector<std::string>& KnownAddOnPluginIds() {
    static const std::vector<std::string> ids = [] {
        std::vector<std::string> result;
        for (const auto& identity : kKnownAddOns) result.emplace_back(identity.plugin_id);
        return result;
    }();
    return ids;
}

// Display name of an add-on: its own wording for a known plugin id, else the
// plugin id itself.
inline std::string AddOnTitle(const std::string& plugin_id) {
    const auto* identity = FindAddOnIdentity(plugin_id);
    return identity ? std::string(identity->title) : plugin_id;
}

}  // namespace cyxwiz
