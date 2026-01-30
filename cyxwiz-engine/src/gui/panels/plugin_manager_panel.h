#pragma once

#include "../panel.h"
#include <string>

namespace cyxwiz::plugin {
struct LoadedPlugin;
enum class PluginPermission : uint32_t;
} // namespace cyxwiz::plugin

namespace cyxwiz {

class PluginManagerPanel : public Panel {
public:
    PluginManagerPanel();
    ~PluginManagerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override;

private:
    void RenderToolbar();
    void RenderPluginList();
    void RenderPluginCard(const plugin::LoadedPlugin* plugin);
    void RenderPermissionBadge(plugin::PluginPermission perm, bool is_dangerous);
    void RenderStatusBar();
    void RenderInstallPopup();

    char search_buf_[256] = {};
    char install_path_buf_[512] = {};
    bool show_install_popup_ = false;
};

} // namespace cyxwiz
