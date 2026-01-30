#pragma once

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct PluginPanelInfo {
    std::string panel_id;               // Unique ID (e.g. "com.vendor.mypanel")
    std::string title;                  // Window title
    std::string category;               // Menu path (e.g. "Tools")
    std::string icon;                   // FontAwesome icon codepoint
    bool show_by_default = false;
};

class IPanelProvider {
public:
    virtual ~IPanelProvider() = default;

    virtual std::vector<PluginPanelInfo> GetPanels() = 0;

    // Render a panel. Called every frame while visible.
    // visible: pointer to visibility flag (set to false to close)
    virtual void RenderPanel(const std::string& panel_id, bool* visible) = 0;
};

} // namespace cyxwiz::plugin
