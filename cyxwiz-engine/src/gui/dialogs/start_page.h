#pragma once

#include <string>
#include <vector>
#include <ctime>

namespace cyxwiz {

/**
 * @brief Professional start page inspired by Visual Studio 2026
 *
 * Full-screen start experience with:
 * - Search bar for filtering recent projects
 * - Chronological grouping (This week, This month, Older)
 * - Action cards (Create, Open, Clone, Continue)
 * - Professional styling
 */
class StartPage {
public:
    enum class Result {
        InProgress,        // Still showing the page
        ProjectSelected,   // User selected a project
        ContinueWithout,   // Continue without project
        Exit               // User closed the window
    };

    struct RecentProject {
        std::string name;
        std::string path;
        std::time_t last_opened;
    };

    StartPage();
    ~StartPage() = default;

    /**
     * @brief Render the start page
     * @return false when the page should close
     */
    bool Render();

    /**
     * @brief Get the result of the start page interaction
     */
    Result GetResult() const { return result_; }

    /**
     * @brief Get the selected project path (valid when result is ProjectSelected)
     */
    std::string GetSelectedProjectPath() const { return selected_project_path_; }

private:
    // Rendering sections
    void RenderSearchBar();
    void RenderRecentProjects();
    void RenderActionCards();
    void RenderBottomBar();
    void RenderCreateProjectDialog();

    // Project grouping by time
    void LoadRecentProjects();
    void GroupProjectsByTime();
    bool IsThisWeek(std::time_t time) const;
    bool IsThisMonth(std::time_t time) const;

    // Search filtering
    void FilterProjects();

    // Actions
    void OpenProject(const std::string& path);
    void CreateNewProject();
    void OpenExistingProject();
    void ContinueWithoutProject();

    // State
    Result result_ = Result::InProgress;
    std::string selected_project_path_;

    // Recent projects
    std::vector<RecentProject> all_projects_;
    std::vector<RecentProject> this_week_;
    std::vector<RecentProject> this_month_;
    std::vector<RecentProject> older_;

    // Search
    char search_buffer_[256] = {0};
    bool search_active_ = false;

    // UI state
    bool show_this_week_ = true;
    bool show_this_month_ = true;
    bool show_older_ = true;

    // Dialogs
    bool show_create_dialog_ = false;
    bool show_open_dialog_ = false;

    // Create project dialog state
    char project_name_buf_[256] = {0};
    char project_location_buf_[512] = {0};
    bool create_venv_ = true;
};

} // namespace cyxwiz
