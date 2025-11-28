#pragma once

#include "../panel.h"
#include "plot_window.h"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Top Toolbar Panel
 * Renders main menu bar with File, Edit, View, Nodes, Train, Dataset, Script, Deploy, Plots, Help
 */
class ToolbarPanel : public Panel {
public:
    ToolbarPanel();
    ~ToolbarPanel() override = default;

    void Render() override;

    // Callbacks
    void SetResetLayoutCallback(std::function<void()> callback) { reset_layout_callback_ = callback; }
    void SetTogglePlotTestControlCallback(std::function<void()> callback) { toggle_plot_test_control_callback_ = callback; }
    void SetConnectToServerCallback(std::function<void()> callback) { connect_to_server_callback_ = callback; }
    void SetNewScriptCallback(std::function<void()> callback) { new_script_callback_ = callback; }
    void SetOpenScriptCallback(std::function<void()> callback) { open_script_callback_ = callback; }
    void SetOpenScriptInEditorCallback(std::function<void(const std::string&)> callback) { open_script_in_editor_callback_ = callback; }
    void SetSaveAllCallback(std::function<void()> callback) { save_all_callback_ = callback; }
    void SetAccountSettingsCallback(std::function<void()> callback) { account_settings_callback_ = callback; }
    void SetExitCallback(std::function<void()> callback) { exit_callback_ = callback; }
    void SetHasUnsavedChangesCallback(std::function<bool()> callback) { has_unsaved_changes_callback_ = callback; }

    // Auto-save state
    bool IsAutoSaveEnabled() const { return auto_save_enabled_; }
    void SetAutoSaveEnabled(bool enabled) { auto_save_enabled_ = enabled; }

    // Access to created plot windows
    const std::vector<std::shared_ptr<PlotWindow>>& GetPlotWindows() const { return plot_windows_; }

private:
    void RenderFileMenu();
    void RenderEditMenu();
    void RenderViewMenu();
    void RenderNodesMenu();
    void RenderTrainMenu();
    void RenderDatasetMenu();
    void RenderScriptMenu();
    void RenderPlotsMenu();
    void RenderDeployMenu();
    void RenderHelpMenu();

    // Helper functions
    std::string OpenFolderDialog();
    std::string OpenFileDialog(const char* filter, const char* title);
    void CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type);

    bool show_new_project_dialog_;
    bool show_about_dialog_;
    bool show_account_settings_dialog_ = false;
    bool show_exit_confirmation_dialog_ = false;
    bool auto_save_enabled_ = false;
    float auto_save_interval_ = 60.0f;  // Auto save every 60 seconds
    float auto_save_timer_ = 0.0f;      // Current timer countdown

    // Account/Auth state
    bool is_logged_in_ = false;
    char login_identifier_[256] = "";  // Email or phone (auto-detected)
    char login_password_[256] = "";
    std::string logged_in_user_;
    std::string login_error_message_;

    std::function<void()> reset_layout_callback_;
    std::function<void()> toggle_plot_test_control_callback_;
    std::function<void()> connect_to_server_callback_;
    std::function<void()> new_script_callback_;
    std::function<void()> open_script_callback_;
    std::function<void(const std::string&)> open_script_in_editor_callback_;
    std::function<void()> save_all_callback_;
    std::function<void()> account_settings_callback_;
    std::function<void()> exit_callback_;
    std::function<bool()> has_unsaved_changes_callback_;

    // Project creation state
    char project_name_buffer_[256];
    char project_path_buffer_[512];

    // Save As dialog state
    bool show_save_as_dialog_ = false;
    char save_as_name_buffer_[256] = "";
    char save_as_path_buffer_[512] = "";

    // New script dialog state
    bool show_new_script_dialog_ = false;
    char new_script_name_[256] = "";
    int new_script_type_ = 0;  // 0 = .cyx, 1 = .py

    // Plot windows management
    std::vector<std::shared_ptr<PlotWindow>> plot_windows_;
};

} // namespace cyxwiz
