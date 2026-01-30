#pragma once

#include "../panel.h"
#include <string>
#include <vector>

namespace cyxwiz::plugin {
struct LoadedPlugin;
enum class PluginPermission : uint32_t;
enum class PluginState;
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
    void RenderPluginList(const std::vector<const plugin::LoadedPlugin*>& plugins);
    void RenderPluginCard(const plugin::LoadedPlugin* plugin);
    void RenderPermissionBadge(plugin::PluginPermission perm, bool is_dangerous);
    void RenderStatusBar(const std::vector<const plugin::LoadedPlugin*>& plugins);
    void RenderInstallPopup();

    char search_buf_[256] = {};
    char install_path_buf_[512] = {};
    bool show_install_popup_ = false;

    // Deferred actions (applied after iteration to avoid use-after-free)
    std::vector<std::string> deferred_unloads_;

    // Install error feedback
    std::string install_error_;
    bool show_install_error_ = false;
};

} // namespace cyxwiz
