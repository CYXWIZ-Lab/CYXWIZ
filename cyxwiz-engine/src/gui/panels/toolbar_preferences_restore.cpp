// Preferences: "Restore this page's defaults" (TOFIX129 B9-12). Each page
// with settings can go back to the shipped values, after a confirmation;
// what is the user's own data (the Python interpreter, Favourites and
// Recent nodes, plugin sections) is kept and the confirmation says so.

#include "toolbar.h"

#include "../../core/data_budget.h"
#include "../../core/engine_config.h"
#include "../../core/project_manager.h"
#include "../appearance_settings.h"
#include "../separate_windows.h"
#include "../ui_buttons.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {
// The pages that have settings of their own, and what a restore keeps
struct RestorablePage {
    const char* page;
    const char* keeps;  // what is kept or what the defaults are; nullptr: nothing to add
};
constexpr RestorablePage kRestorable[] = {
    {"General", nullptr},
    {"Appearance", "CyxWiz Dark, 15 px interface text, 16 px code text, sidebar on the right, Font Awesome node "
                   "icons."},
    {"Studio", nullptr},
    {"Script Editor", nullptr},
    {"Python", "The interpreter is kept; Packages to install is emptied."},
    {"Nodes panel", "Favourites and Recent are kept."},
    {"Training", nullptr},
    {"Search", nullptr},
    {"Studio Debugger", nullptr},
    {"Simulation", "Plugin sections keep their own settings."},
    {"Tools", nullptr},
};

const RestorablePage* FindRestorable(const std::string& page) {
    for (const auto& r : kRestorable)
        if (page == r.page) return &r;
    return nullptr;
}
}  // namespace

void ToolbarPanel::RenderRestoreDefaults(const std::string& page) {
    const RestorablePage* restorable = FindRestorable(page);
    if (!restorable) return;  // nothing on this page has a shipped value
    ImGui::Spacing();
    ImGui::Spacing();
    if (restore_confirm_page_ != page) {
        if (ui::LinkButton("Restore this page's defaults")) restore_confirm_page_ = page;
        return;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Restore the shipped values on this page?");
    if (restorable->keeps) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", restorable->keeps);
    }
    if (ui::DangerButton("Restore defaults")) {
        RestorePageDefaults(page);
        restore_confirm_page_.clear();
    }
    ImGui::SameLine();
    if (ui::LinkButton("Keep my settings")) restore_confirm_page_.clear();
}

void ToolbarPanel::RestorePageDefaults(const std::string& page) {
    auto& config = core::EngineConfig::Instance();
    if (page == "General") {
        config.SetStartWithLastProject(false);
        config.SetRecentProjectsKept(core::EngineConfig::kDefaultRecentProjects);
        ProjectManager::Instance().SetRecentProjectsLimit(config.RecentProjectsKept());
        config.SetAskBeforeClosingAlways(false);
        config.SetRecordDebugLogs(true);
        spdlog::set_level(spdlog::level::debug);
        config.SetEveryWindowMayLeave(false);
        ::gui::SetEveryWindowMayLeave(false);
        config.SetDataBudgetBytes(0);
        SetDataBudgetBytes(config.DataBudgetBytes());
    } else if (page == "Appearance") {
        ::gui::ResetAppearance();  // saves the appearance itself
    } else if (page == "Studio") {
        config.SetStudio({});
        config.SetRequireDebugBeforeTrain(false);
    } else if (page == "Script Editor") {
        config.SetEditor({});
        if (editor_settings_callback_) editor_settings_callback_(config.GetEditor());
    } else if (page == "Python") {
        config.SetAutoCreateVenv(true);
        config.SetDefaultVenvPackages({});
        config.SetPythonStart({});
        python_settings_panel_.Reload();
    } else if (page == "Nodes panel") {
        auto nodes = config.GetNodesPanel();
        const core::NodesPanelConfig shipped;
        nodes.show_recent = shipped.show_recent;
        nodes.recent_count = shipped.recent_count;
        nodes.list_view = shipped.list_view;
        config.SetNodesPanel(nodes);
    } else if (page == "Training") {
        config.SetTrainingDefaults({});
        config.SetTrainingDashboard({});
    } else if (page == "Search") {
        config.SetSearchDefaults({});
        config.SetArchitectureSearchDefaults({});
    } else if (page == "Studio Debugger") {
        config.SetStudioDebugger({});
    } else if (page == "Simulation") {
        config.SetSimulationRateHz(core::EngineConfig::kDefaultSimulationRateHz);
        config.SetOpenSimulationMonitorOnRun(true);
        config.SetPolicyPlaybackBestActions(true);
        config.SetPolicyPlaybackSpeed(0);
        config.SetPlayPolicyAfterTraining(false);
        config.SetPolicyStressTestDefaults({});
        config.SetWatchTraining(0, 20);
    } else if (page == "Tools") {
        config.SetUtilities({});
        config.SetMonitor({});
        if (idle_log_ptr_) *idle_log_ptr_ = false;
        if (verbose_python_log_ptr_) verbose_python_log_ptr_->store(false);
        if (frame_overlay_ptr_) *frame_overlay_ptr_ = false;
        SaveDiagnostics();
    } else {
        return;
    }
    config.Save();
    spdlog::info("Preferences > {}: the shipped defaults are back", page);
}

}  // namespace cyxwiz
