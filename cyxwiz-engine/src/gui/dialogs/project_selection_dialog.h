#pragma once

#include <string>
#include <vector>
#include <optional>

namespace cyxwiz {

/**
 * @brief Mandatory project selection dialog shown on startup.
 *
 * This dialog is shown if no project is specified on the command line.
 * The user MUST select a project to continue or exit the application.
 * There is no "Skip" option - project is mandatory.
 *
 * Flow:
 * 1. Show list of recent projects
 * 2. Options: Create New, Open Existing, Select Recent, Exit
 * 3. Returns selected project path or empty (user wants to exit)
 */
class ProjectSelectionDialog {
public:
    enum class Result {
        InProgress,  // Dialog still showing
        ProjectSelected,  // User selected/created a project
        Exit,        // User wants to exit application
    };

    ProjectSelectionDialog();
    ~ProjectSelectionDialog() = default;

    // Render the dialog (call every frame)
    // Returns true if the dialog is still active
    bool Render();

    // Get the result of the dialog
    Result GetResult() const { return result_; }

    // Get the selected project path (valid only if result == ProjectSelected)
    std::string GetSelectedProjectPath() const { return selected_project_path_; }

private:
    void RenderRecentProjects();
    void RenderButtons();
    void ShowCreateProjectDialog();
    void ShowOpenProjectDialog();

    Result result_ = Result::InProgress;
    std::string selected_project_path_;

    // Recent projects list (loaded from config)
    std::vector<std::string> recent_projects_;

    // UI state
    int selected_recent_index_ = -1;
    bool show_create_dialog_ = false;
    bool show_open_dialog_ = false;

    // Create project dialog state
    char project_name_buf_[256] = {};
    char project_location_buf_[512] = {};
    bool create_venv_ = true;

    // Helper methods
    void LoadRecentProjects();
    bool CreateNewProject(const std::string& name, const std::string& location);
    void SelectProject(const std::string& path);
};

} // namespace cyxwiz
