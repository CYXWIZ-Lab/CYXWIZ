#pragma once

#include "../panel.h"
#include <functional>
#include <string>

namespace cyxwiz {

/**
 * Top Toolbar Panel
 * Renders main menu bar with File, Edit, View, Nodes, Train, Dataset, Script, Deploy, Help
 */
class ToolbarPanel : public Panel {
public:
    ToolbarPanel();
    ~ToolbarPanel() override = default;

    void Render() override;

    // Callback for resetting layout
    void SetResetLayoutCallback(std::function<void()> callback) { reset_layout_callback_ = callback; }

private:
    void RenderFileMenu();
    void RenderEditMenu();
    void RenderViewMenu();
    void RenderNodesMenu();
    void RenderTrainMenu();
    void RenderDatasetMenu();
    void RenderScriptMenu();
    void RenderDeployMenu();
    void RenderHelpMenu();

    // Helper functions
    std::string OpenFolderDialog();
    bool CreateProjectOnDisk(const std::string& project_name, const std::string& project_path);

    bool show_new_project_dialog_;
    bool show_about_dialog_;
    std::function<void()> reset_layout_callback_;

    // Project creation state
    char project_name_buffer_[256];
    char project_path_buffer_[512];
};

} // namespace cyxwiz
