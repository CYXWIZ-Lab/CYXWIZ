#pragma once

// Start page (TOFIX129 piece A2): recent projects, starter graphs, and the
// ways in (create, open, open folder, continue without a project). Grouping,
// search and the Create checks come from core/start_page_presentation; the
// look comes from the shared tokens and widgets, so the page follows the
// active theme.

#include "create_project_dialog.h"

#include <functional>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

class StartPage {
public:
    enum class Result {
        InProgress,            // Still showing the page
        ProjectSelected,       // A project was opened or created
        ExampleGraphSelected,  // A starter graph was chosen
        ContinueWithout        // Continue without a project
    };

    struct StarterGraph {
        std::string title;
        std::string description;
        std::string domain;
        std::string icon;
        std::string path;
    };

    // Python status shown as a chip at the top right; clicking it calls
    // `on_click` (the application opens the Python dialog).
    struct PythonStatus {
        std::string text;   // "Python 3.12.8 ready", "Checking Python...", "Python not found"
        int level = 0;      // 0 = ready, 1 = checking, 2 = needs attention
        std::function<void()> on_click;
    };

    StartPage();

    // Returns false when the page is done.
    bool Render();

    Result GetResult() const { return result_; }
    std::string GetSelectedProjectPath() const { return selected_project_path_; }
    std::string GetSelectedGraphPath() const { return selected_graph_path_; }

    void SetPythonStatus(PythonStatus status) { python_ = std::move(status); }

private:
    void LoadRecentProjects();
    void LoadStarterGraphs();

    void RenderHeader();
    void RenderRecentProjects(float height);
    void RenderStartActions();
    void RenderStarterGraphs(float height);
    void RenderFooter();
    void HandleKeys();

    void OpenProject(const std::string& path);
    void OpenExistingProject();
    void OpenProjectFolder();
    void SetStatus(std::string text, bool problem = false);

    Result result_ = Result::InProgress;
    std::string selected_project_path_;
    std::string selected_graph_path_;

    struct Recent {
        std::string name;
        std::string path;
        long long last_opened = 0;
    };
    std::vector<Recent> recent_;
    std::vector<StarterGraph> starter_graphs_;

    char search_[256] = {};
    std::string selected_path_;          // selected recent project (file path)
    std::string menu_for_path_;          // recent project whose Actions menu is open
    std::set<std::string> collapsed_groups_;

    CreateProjectDialog create_dialog_;
    PythonStatus python_;
    std::string status_ = "Ready";
    bool status_problem_ = false;
};

}  // namespace cyxwiz
