#include "plugin_panel_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

PluginPanelRegistry& PluginPanelRegistry::Instance() {
    static PluginPanelRegistry instance;
    return instance;
}

void PluginPanelRegistry::Register(const std::string& plugin_id, IPanelProvider* provider) {
    if (!provider) return;

    auto panels = provider->GetPanels();
    std::lock_guard lock(mutex_);

    for (auto& info : panels) {
        if (panels_.count(info.panel_id)) {
            spdlog::warn("PluginPanelRegistry: Duplicate panel {}, skipping", info.panel_id);
            continue;
        }
        bool vis = info.show_by_default;
        spdlog::info("PluginPanelRegistry: Registered panel '{}' ({})", info.title, info.panel_id);
        panels_[info.panel_id] = PanelEntry{plugin_id, std::move(info), provider, vis};
    }
}

void PluginPanelRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = panels_.begin(); it != panels_.end();) {
        if (it->second.plugin_id == plugin_id)
            it = panels_.erase(it);
        else
            ++it;
    }
}

std::vector<PluginPanelInfo> PluginPanelRegistry::GetAllPanels() const {
    std::lock_guard lock(mutex_);
    std::vector<PluginPanelInfo> result;
    result.reserve(panels_.size());
    for (const auto& [_, entry] : panels_)
        result.push_back(entry.info);
    return result;
}

bool PluginPanelRegistry::HasPanel(const std::string& panel_id) const {
    std::lock_guard lock(mutex_);
    return panels_.count(panel_id) > 0;
}

void PluginPanelRegistry::RenderAllVisible() {
    // Snapshot visible panels to avoid holding lock during render
    struct RenderItem {
        std::string id;
        IPanelProvider* provider;
        bool visible;
    };
    std::vector<RenderItem> to_render;
    {
        std::lock_guard lock(mutex_);
        for (auto& [id, entry] : panels_) {
            if (entry.visible && entry.provider)
                to_render.push_back({id, entry.provider, entry.visible});
        }
    }

    for (auto& item : to_render) {
        bool vis = item.visible;
        try {
            item.provider->RenderPanel(item.id, &vis);
        } catch (const std::exception& e) {
            spdlog::error("PluginPanelRegistry: RenderPanel({}) threw: {}", item.id, e.what());
            vis = false;
        }
        // Write back visibility change
        if (vis != item.visible) {
            std::lock_guard lock(mutex_);
            auto it = panels_.find(item.id);
            if (it != panels_.end())
                it->second.visible = vis;
        }
    }
}

bool PluginPanelRegistry::IsPanelVisible(const std::string& panel_id) const {
    std::lock_guard lock(mutex_);
    auto it = panels_.find(panel_id);
    return it != panels_.end() && it->second.visible;
}

void PluginPanelRegistry::SetPanelVisible(const std::string& panel_id, bool visible) {
    std::lock_guard lock(mutex_);
    auto it = panels_.find(panel_id);
    if (it != panels_.end())
        it->second.visible = visible;
}

void PluginPanelRegistry::TogglePanelVisible(const std::string& panel_id) {
    std::lock_guard lock(mutex_);
    auto it = panels_.find(panel_id);
    if (it != panels_.end())
        it->second.visible = !it->second.visible;
}

size_t PluginPanelRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return panels_.size();
}

} // namespace cyxwiz::plugin
