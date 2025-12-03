#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <ctime>
#include <nlohmann/json.hpp>

namespace cyxwiz {

/**
 * Editor settings stored in project file
 */
struct EditorSettings {
    // Script Editor settings
    int theme = 3;           // 0=Dark, 1=Light, 2=RetroBlu, 3=Monokai, 4=Dracula, 5=OneDark, 6=GitHub
    float font_scale = 1.6f; // 1.0=Small, 1.3=Medium, 1.6=Large, 2.0=Extra Large
    int tab_size = 4;        // 2, 4, or 8
    bool show_whitespace = true;
    bool syntax_highlighting = true;
    bool word_wrap = false;
    bool show_line_numbers = true;
    bool auto_indent = true;

    // Application-wide settings
    int app_theme = 0;       // 0=CyxWizDark, 1=CyxWizLight, 2=VSCodeDark, 3=UnrealEngine, 4=ModernDark, 5=HighContrast
    float ui_scale = 1.0f;   // Global UI scale (0.8 to 2.0)

    // JSON serialization
    static EditorSettings FromJson(const nlohmann::json& j);
    nlohmann::json ToJson() const;
};

/**
 * Project configuration stored in .cyxwiz files
 */
struct ProjectConfig {
    std::string name;
    std::string version;
    std::time_t created;
    std::string description;
    std::vector<std::string> recent_files;
    // Filter definitions (custom filters can be added)
    std::map<std::string, std::vector<std::string>> filters; // filter_name -> extensions

    // Editor/UI settings
    EditorSettings editor_settings;

    // Open files in Script Editor (to restore on project load)
    std::vector<std::string> open_scripts;
    int active_script_index = 0;

    // JSON serialization
    static ProjectConfig FromJson(const nlohmann::json& j);
    nlohmann::json ToJson() const;
};

/**
 * ProjectManager - Singleton for global project state management
 *
 * Manages the active project, working directory, and provides path utilities.
 * All components should use this to resolve asset paths.
 */
class ProjectManager {
public:
    // Singleton access
    static ProjectManager& Instance();

    // Delete copy/move constructors
    ProjectManager(const ProjectManager&) = delete;
    ProjectManager& operator=(const ProjectManager&) = delete;
    ProjectManager(ProjectManager&&) = delete;
    ProjectManager& operator=(ProjectManager&&) = delete;

    // Project lifecycle
    bool CreateProject(const std::string& name, const std::string& location);
    bool OpenProject(const std::string& cyxwiz_file_path);
    void CloseProject();
    bool SaveProject();
    bool SaveProjectAs(const std::string& new_name, const std::string& new_location);

    // Accessors
    bool HasActiveProject() const { return !project_root_.empty(); }
    const std::string& GetProjectRoot() const { return project_root_; }
    const std::string& GetProjectName() const { return project_name_; }
    const std::string& GetProjectFilePath() const { return project_file_path_; }
    const ProjectConfig& GetConfig() const { return config_; }
    ProjectConfig& GetConfig() { return config_; }

    // Path utilities (all return absolute paths)
    std::string GetScriptsPath() const;
    std::string GetModelsPath() const;
    std::string GetDatasetsPath() const;
    std::string GetCheckpointsPath() const;
    std::string GetExportsPath() const;
    std::string GetPluginsPath() const;
    std::string GetLayoutFilePath() const;  // Path to project-specific imgui.ini

    // Resolve relative path to absolute
    std::string ResolveAssetPath(const std::string& relative_path) const;

    // Convert absolute path to relative (for display)
    std::string MakeRelativePath(const std::string& absolute_path) const;

    // Callbacks for state changes
    using ProjectCallback = std::function<void(const std::string& project_root)>;
    void SetOnProjectOpened(ProjectCallback callback) { on_opened_ = std::move(callback); }
    void SetOnProjectClosed(ProjectCallback callback) { on_closed_ = std::move(callback); }

    // Default filter extensions
    static const std::map<std::string, std::vector<std::string>>& GetDefaultFilters();

    // Recent projects management
    struct RecentProject {
        std::string name;
        std::string path;  // Full path to .cyxwiz file
        std::time_t last_opened;
    };
    const std::vector<RecentProject>& GetRecentProjects() const { return recent_projects_; }
    void ClearRecentProjects();
    static constexpr size_t MAX_RECENT_PROJECTS = 10;

private:
    ProjectManager();
    ~ProjectManager() = default;

    // Internal helpers
    bool CreateDirectoryStructure(const std::string& project_dir);
    bool WriteProjectFile(const std::string& file_path);
    bool ReadProjectFile(const std::string& file_path);
    void InitializeDefaultFilters();

    // Recent projects helpers
    void AddToRecentProjects(const std::string& name, const std::string& path);
    void LoadRecentProjects();
    void SaveRecentProjects();
    std::string GetRecentProjectsFilePath() const;

    // State
    std::string project_root_;      // Absolute path to project directory
    std::string project_name_;      // Project name
    std::string project_file_path_; // Full path to .cyxwiz file
    ProjectConfig config_;

    // Recent projects
    std::vector<RecentProject> recent_projects_;

    // Callbacks
    ProjectCallback on_opened_;
    ProjectCallback on_closed_;
};

} // namespace cyxwiz
