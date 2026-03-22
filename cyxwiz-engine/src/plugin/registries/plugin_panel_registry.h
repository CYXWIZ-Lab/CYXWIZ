#pragma once

#include "../interfaces/i_panel_provider.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace cyxwiz::plugin {

class PluginPanelRegistry {
public:
    static PluginPanelRegistry& Instance();

    PluginPanelRegistry(const PluginPanelRegistry&) = delete;
    PluginPanelRegistry& operator=(const PluginPanelRegistry&) = delete;

    void Register(const std::string& plugin_id, IPanelProvider* provider);
    // Direct registration — avoids calling GetPanels() across DLL boundary
    void RegisterDirect(const std::string& plugin_id, const std::string& panel_id,
                        const std::string& title, const std::string& category,
                        bool show_by_default, IPanelProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    // Query
    std::vector<PluginPanelInfo> GetAllPanels() const;
    bool HasPanel(const std::string& panel_id) const;

    // Rendering (called by MainWindow in Phase 3)
    void RenderAllVisible();

    // Visibility
    bool IsPanelVisible(const std::string& panel_id) const;
    void SetPanelVisible(const std::string& panel_id, bool visible);
    void TogglePanelVisible(const std::string& panel_id);
    bool* GetVisiblePtr(const std::string& panel_id);

    size_t GetCount() const;

private:
    PluginPanelRegistry() = default;

    struct PanelEntry {
        std::string plugin_id;
        PluginPanelInfo info;
        IPanelProvider* provider = nullptr;  // non-owning
        bool visible = false;
    };

    mutable std::mutex mutex_;
    std::map<std::string, PanelEntry> panels_;  // panel_id -> entry
};

} // namespace cyxwiz::plugin
