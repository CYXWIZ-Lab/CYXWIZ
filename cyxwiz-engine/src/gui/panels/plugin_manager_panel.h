#pragma once

#include "../panel.h"
#include "../../core/plugin_presentation.h"
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz::plugin {
struct LoadedPlugin;
} // namespace cyxwiz::plugin

namespace cyxwiz {

// Plugin Manager: one card per plugin and per Python node bundle (the same
// look), each with its state in words, one main action and an Actions menu,
// its permissions and what it adds. Worded by core/plugin_presentation.
class PluginManagerPanel : public Panel {
public:
    PluginManagerPanel();
    ~PluginManagerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override;

    // Apps > a plugin: shows the window scrolled to that plugin's card.
    void ShowPlugin(const std::string& plugin_id);

private:
    void RenderToolbar();
    void RenderPluginCard(const plugin::LoadedPlugin& plugin, const PluginCardFacts& facts);
    void RenderPermissions(const plugin::LoadedPlugin& plugin);
    void RenderInstallDialog();
    void StartInstall();
    const PluginContribution& ContributionOf(const std::string& plugin_id);

    char search_buf_[256] = {};

    // Install dialog: its message stays inside it, under the path
    char install_path_buf_[512] = {};
    bool open_install_ = false;
    bool is_installing_plugin_ = false;
    uint64_t install_task_id_ = 0;
    enum class InstallTone { None, Problem, Partial };
    std::string install_message_;
    InstallTone install_tone_ = InstallTone::None;
    bool close_install_ = false;

    // Deferred actions (applied after iteration to avoid use-after-free)
    std::vector<std::string> deferred_unloads_;
    std::string unload_asked_;            // the card asking "Unload ...?"
    std::set<std::string> open_permissions_;  // cards whose Permissions part is open
    std::string scroll_to_;               // ShowPlugin's card, scrolled to once

    // Which plugins load when the Engine starts, read on open and after each change.
    cyxwiz::PluginStartupList startup_list_;
    bool startup_list_read_ = false;
    void ReloadStartupList();

    // What each plugin adds walks four registries: read at most once a second
    std::map<std::string, PluginContribution> contributions_;
    std::chrono::steady_clock::time_point contributions_read_at_{};
};

} // namespace cyxwiz
