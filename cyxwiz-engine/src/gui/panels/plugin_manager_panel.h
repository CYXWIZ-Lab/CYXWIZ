#pragma once

#include "../panel.h"
#include "../../core/plugin_presentation.h"
#include "../../plugin/plugin_pack.h"
#include <functional>
#include <set>
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

    // Opens Preferences > Devices, where Installed packs lists the add-ons.
    void SetOpenInstalledPacksCallback(std::function<void()> callback) {
        open_installed_packs_ = std::move(callback);
    }

private:
    void RenderToolbar();
    void RenderPluginList(const std::vector<const plugin::LoadedPlugin*>& plugins,
                          const std::vector<plugin::PackRefusal>& refusals);
    void RenderPluginCard(const plugin::LoadedPlugin* plugin);
    // An installed add-on the Engine did not load: why, and Repair.
    void RenderRefusedCard(const plugin::PackRefusal& refusal);
    void RenderTrustDetails(const PluginTrustView& view);
    void RenderPermissionBadge(plugin::PluginPermission perm, bool is_dangerous);
    void RenderStatusBar(const std::vector<const plugin::LoadedPlugin*>& plugins,
                         const std::vector<plugin::PackRefusal>& refusals);
    void RenderInstallPopup();

    std::function<void()> open_installed_packs_;
    // Cards whose Details are open, by plugin id.
    std::set<std::string> details_open_;

    char search_buf_[256] = {};
    char install_path_buf_[512] = {};
    bool show_install_popup_ = false;
    bool close_install_popup_ = false;
    bool is_installing_plugin_ = false;
    uint64_t install_task_id_ = 0;
    std::string install_status_;

    // Deferred actions (applied after iteration to avoid use-after-free)
    std::vector<std::string> deferred_unloads_;

    // Which plugins load when the Engine starts (TOFIX125 step 2.9), read
    // once and again after each change.
    cyxwiz::PluginStartupList startup_list_;
    bool startup_list_read_ = false;
    void ReloadStartupList();

    // Install error feedback
    std::string install_error_;
    bool show_install_error_ = false;
};

} // namespace cyxwiz
