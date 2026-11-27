#pragma once

// What an add-on (a plugin pack: an optional feature such as Simulation
// (MuJoCo)) is called. Its name and description come from the plugin itself:
// packaging copies them from plugin.json into the signed pack manifest, and
// the pack record carries them. The table below is only for an add-on this
// release expects but has no pack for (its Not published card), and for a
// pack made before manifests carried the words. Header-only, so the plugin
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

// Display name of an add-on: the name its signed pack carries (published_name,
// empty when there is no pack or an older one), else this release's wording
// for an expected plugin id, else the plugin id itself. Screens that have the
// pack records do not pick a pack themselves: AddOnWordsFor
// (backend_pack_manager_model.h) chooses the add-on's words once.
inline std::string AddOnTitle(const std::string& plugin_id,
                              const std::string& published_name = {}) {
    if (!published_name.empty()) return published_name;
    const auto* identity = FindAddOnIdentity(plugin_id);
    return identity ? std::string(identity->title) : plugin_id;
}

// What an add-on does, in the same order: its signed pack's description, else
// this release's wording, else a plain line.
inline std::string AddOnDescription(const std::string& plugin_id,
                                    const std::string& published_description = {}) {
    if (!published_description.empty()) return published_description;
    const auto* identity = FindAddOnIdentity(plugin_id);
    return identity ? std::string(identity->description)
                    : std::string("An optional add-on from the signed catalog.");
}

}  // namespace cyxwiz
