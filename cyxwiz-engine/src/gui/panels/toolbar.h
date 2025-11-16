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
    bool CreateProjectOnDisk(const std::string& project_name, const std::string& project_path);
    void CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type);

    bool show_new_project_dialog_;
    bool show_about_dialog_;
    std::function<void()> reset_layout_callback_;
    std::function<void()> toggle_plot_test_control_callback_;

    // Project creation state
    char project_name_buffer_[256];
    char project_path_buffer_[512];

    // Plot windows management
    std::vector<std::shared_ptr<PlotWindow>> plot_windows_;
};

} // namespace cyxwiz
