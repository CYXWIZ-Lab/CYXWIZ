// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Settings a plugin adds to the Engine's Preferences dialog (plugin API 1.4).
// The Engine finds them through IPlugin::QueryInterface("IPreferencesProvider")
// when the plugin starts (UIModify permission) and draws each section at the
// end of the Preferences tab it names, every frame that tab is open, on the
// UI thread. Keep what the user sets with PluginContext::SetSetting, and read
// it back with GetSetting where your panels start.

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct PluginPreferencesSection {
    std::string section_id;  // unique within the plugin ("viewport")
    // The Preferences tab, by its title without the icon: "General",
    // "Editor", "Files", "Python" or "Simulation". Any other title puts the
    // section in General.
    std::string tab;
    std::string title;  // the section's heading ("MuJoCo Viewport")
    std::string hint;   // one muted sentence under the heading; may be empty
};

class IPreferencesProvider {
public:
    virtual ~IPreferencesProvider() = default;

    virtual std::vector<PluginPreferencesSection> GetPreferenceSections() = 0;
    // Draw the section's settings below its heading with ImGui and
    // PluginContext::GetUi (typed number fields, segmented controls,
    // checkboxes). Called every frame the tab is open: keep it cheap.
    virtual void RenderPreferences(const std::string& section_id) = 0;
};

}  // namespace cyxwiz::plugin
