#include "project_manager.h"
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

#ifdef _WIN32
#include <shlobj.h>
#endif

namespace cyxwiz {

// EditorSettings JSON serialization
EditorSettings EditorSettings::FromJson(const nlohmann::json& j) {
    EditorSettings settings;
    // Script Editor settings
    settings.theme = j.value("theme", 3);
    settings.font_scale = j.value("font_scale", 1.6f);
    settings.tab_size = j.value("tab_size", 4);
    settings.show_whitespace = j.value("show_whitespace", true);
    settings.syntax_highlighting = j.value("syntax_highlighting", true);
    settings.word_wrap = j.value("word_wrap", false);
    settings.show_line_numbers = j.value("show_line_numbers", true);
    settings.auto_indent = j.value("auto_indent", true);
    // Application-wide settings
    settings.app_theme = j.value("app_theme", 0);
    settings.ui_scale = j.value("ui_scale", 1.0f);
    return settings;
}

nlohmann::json EditorSettings::ToJson() const {
    nlohmann::json j;
    // Script Editor settings
    j["theme"] = theme;
    j["font_scale"] = font_scale;
    j["tab_size"] = tab_size;
    j["show_whitespace"] = show_whitespace;
    j["syntax_highlighting"] = syntax_highlighting;
    j["word_wrap"] = word_wrap;
    j["show_line_numbers"] = show_line_numbers;
    j["auto_indent"] = auto_indent;
    // Application-wide settings
    j["app_theme"] = app_theme;
    j["ui_scale"] = ui_scale;
    return j;
}

// Static default filters
static const std::map<std::string, std::vector<std::string>> s_default_filters = {
    {"Scripts", {".py", ".cyx"}},
    {"Models", {".h5", ".onnx", ".pt", ".safetensors", ".bin"}},
    {"Datasets", {".csv", ".json", ".parquet", ".h5", ".arrow", ".txt"}},
    {"Checkpoints", {".ckpt", ".pt", ".checkpoint"}},
    {"Exports", {".onnx", ".gguf", ".lora", ".safetensors"}},
    {"Plugins", {".dll", ".so", ".dylib"}}
};

ProjectConfig ProjectConfig::FromJson(const nlohmann::json& j) {
    ProjectConfig config;
    config.name = j.value("name", "");
    config.version = j.value("version", "0.1.0");
    config.created = j.value("created", std::time(nullptr));
    config.description = j.value("description", "");

    if (j.contains("recent_files") && j["recent_files"].is_array()) {
        for (const auto& file : j["recent_files"]) {
            config.recent_files.push_back(file.get<std::string>());
        }
    }

    if (j.contains("filters") && j["filters"].is_object()) {
        for (auto& [key, value] : j["filters"].items()) {
            std::vector<std::string> extensions;
            for (const auto& ext : value) {
                extensions.push_back(ext.get<std::string>());
            }
            config.filters[key] = extensions;
        }
    }

    // Load editor settings
    if (j.contains("editor_settings") && j["editor_settings"].is_object()) {
        config.editor_settings = EditorSettings::FromJson(j["editor_settings"]);
    }

    // Load open scripts
    if (j.contains("open_scripts") && j["open_scripts"].is_array()) {
        for (const auto& script : j["open_scripts"]) {
            config.open_scripts.push_back(script.get<std::string>());
        }
    }
    config.active_script_index = j.value("active_script_index", 0);

    return config;
}

nlohmann::json ProjectConfig::ToJson() const {
    nlohmann::json j;
    j["name"] = name;
    j["version"] = version;
    j["created"] = created;
    j["description"] = description;
    j["recent_files"] = recent_files;
    j["filters"] = filters;
    j["editor_settings"] = editor_settings.ToJson();
    j["open_scripts"] = open_scripts;
    j["active_script_index"] = active_script_index;
    return j;
}

ProjectManager::ProjectManager() {
    LoadRecentProjects();
}

ProjectManager& ProjectManager::Instance() {
    static ProjectManager instance;
    return instance;
}

const std::map<std::string, std::vector<std::string>>& ProjectManager::GetDefaultFilters() {
    return s_default_filters;
}

bool ProjectManager::CreateProject(const std::string& name, const std::string& location) {
    try {
        // Close any existing project
        if (HasActiveProject()) {
            CloseProject();
        }

        // Create project directory
        fs::path project_dir = fs::path(location) / name;
        if (fs::exists(project_dir)) {
            spdlog::error("Project directory already exists: {}", project_dir.string());
            return false;
        }

        // Create directory structure
        if (!CreateDirectoryStructure(project_dir.string())) {
            return false;
        }

        // Set up project state
        project_root_ = project_dir.string();
        project_name_ = name;
        project_file_path_ = (project_dir / (name + ".cyxwiz")).string();

        // Initialize config
        config_.name = name;
        config_.version = "0.1.0";
        config_.created = std::time(nullptr);
        config_.description = "CyxWiz Machine Learning Project";
        InitializeDefaultFilters();

        // Write project file
        if (!WriteProjectFile(project_file_path_)) {
            project_root_.clear();
            project_name_.clear();
            project_file_path_.clear();
            return false;
        }

        spdlog::info("Project created: {} at {}", name, project_root_);

        // Add to recent projects
        AddToRecentProjects(name, project_file_path_);

        // Fire callback
        if (on_opened_) {
            on_opened_(project_root_);
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to create project: {}", e.what());
        return false;
    }
}

bool ProjectManager::OpenProject(const std::string& cyxwiz_file_path) {
    try {
        // Close any existing project
        if (HasActiveProject()) {
            CloseProject();
        }

        // Verify file exists
        fs::path file_path(cyxwiz_file_path);
        if (!fs::exists(file_path)) {
            spdlog::error("Project file not found: {}", cyxwiz_file_path);
            return false;
        }

        // Read project file
        if (!ReadProjectFile(cyxwiz_file_path)) {
            return false;
        }

        // Set up project state
        project_file_path_ = fs::absolute(file_path).string();
        project_root_ = file_path.parent_path().string();
        project_name_ = config_.name;

        // Ensure default filters if not present
        if (config_.filters.empty()) {
            InitializeDefaultFilters();
        }

        spdlog::info("Project opened: {} from {}", project_name_, project_root_);

        // Add to recent projects
        AddToRecentProjects(project_name_, project_file_path_);

        // Fire callback
        if (on_opened_) {
            on_opened_(project_root_);
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to open project: {}", e.what());
        return false;
    }
}

void ProjectManager::CloseProject() {
    if (!HasActiveProject()) {
        return;
    }

    std::string old_root = project_root_;

    // Clear state
    project_root_.clear();
    project_name_.clear();
    project_file_path_.clear();
    config_ = ProjectConfig();

    spdlog::info("Project closed");

    // Fire callback
    if (on_closed_) {
        on_closed_(old_root);
    }
}

bool ProjectManager::SaveProject() {
    if (!HasActiveProject()) {
        spdlog::warn("No active project to save");
        return false;
    }

    return WriteProjectFile(project_file_path_);
}

bool ProjectManager::SaveProjectAs(const std::string& new_name, const std::string& new_location) {
    if (!HasActiveProject()) {
        spdlog::warn("No active project to save");
        return false;
    }

    try {
        // Create new project directory
        fs::path new_project_dir = fs::path(new_location) / new_name;
        if (fs::exists(new_project_dir)) {
            spdlog::error("Project directory already exists: {}", new_project_dir.string());
            return false;
        }

        // Copy the entire project directory to the new location
        fs::copy(project_root_, new_project_dir, fs::copy_options::recursive);

        // Delete the old .cyxwiz file in the new location (it has the old name)
        fs::path old_cyxwiz_in_new = new_project_dir / (project_name_ + ".cyxwiz");
        if (fs::exists(old_cyxwiz_in_new)) {
            fs::remove(old_cyxwiz_in_new);
        }

        // Update project state to point to new location
        project_root_ = new_project_dir.string();
        project_name_ = new_name;
        project_file_path_ = (new_project_dir / (new_name + ".cyxwiz")).string();

        // Update config with new name
        config_.name = new_name;

        // Write new project file
        if (!WriteProjectFile(project_file_path_)) {
            spdlog::error("Failed to write project file at new location");
            return false;
        }

        spdlog::info("Project saved as: {} at {}", new_name, project_root_);

        // Add to recent projects
        AddToRecentProjects(new_name, project_file_path_);

        // Fire callback
        if (on_opened_) {
            on_opened_(project_root_);
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to save project as: {}", e.what());
        return false;
    }
}

std::string ProjectManager::GetScriptsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "scripts").string();
}

std::string ProjectManager::GetModelsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "models").string();
}

std::string ProjectManager::GetDatasetsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "datasets").string();
}

std::string ProjectManager::GetCheckpointsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "checkpoints").string();
}

std::string ProjectManager::GetExportsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "exports").string();
}

std::string ProjectManager::GetPluginsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "plugins").string();
}

std::string ProjectManager::GetLayoutFilePath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "layout.ini").string();
}

std::string ProjectManager::ResolveAssetPath(const std::string& relative_path) const {
    if (!HasActiveProject()) return relative_path;
    return (fs::path(project_root_) / relative_path).string();
}

std::string ProjectManager::MakeRelativePath(const std::string& absolute_path) const {
    if (!HasActiveProject()) return absolute_path;

    try {
        fs::path abs_path(absolute_path);
        fs::path root_path(project_root_);
        return fs::relative(abs_path, root_path).string();
    } catch (...) {
        return absolute_path;
    }
}

bool ProjectManager::CreateDirectoryStructure(const std::string& project_dir) {
    try {
        fs::create_directories(project_dir);
        fs::create_directories(fs::path(project_dir) / "scripts");
        fs::create_directories(fs::path(project_dir) / "models");
        fs::create_directories(fs::path(project_dir) / "datasets");
        fs::create_directories(fs::path(project_dir) / "checkpoints");
        fs::create_directories(fs::path(project_dir) / "exports");
        fs::create_directories(fs::path(project_dir) / "plugins");

        spdlog::info("Created project directory structure at: {}", project_dir);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to create directory structure: {}", e.what());
        return false;
    }
}

bool ProjectManager::WriteProjectFile(const std::string& file_path) {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open project file for writing: {}", file_path);
            return false;
        }

        nlohmann::json j = config_.ToJson();
        file << j.dump(2);
        file.close();

        spdlog::info("Saved project file: {}", file_path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to write project file: {}", e.what());
        return false;
    }
}

bool ProjectManager::ReadProjectFile(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open project file: {}", file_path);
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        config_ = ProjectConfig::FromJson(j);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to read project file: {}", e.what());
        return false;
    }
}

void ProjectManager::InitializeDefaultFilters() {
    config_.filters = s_default_filters;
}

// Recent projects implementation
void ProjectManager::AddToRecentProjects(const std::string& name, const std::string& path) {
    // Remove if already exists
    recent_projects_.erase(
        std::remove_if(recent_projects_.begin(), recent_projects_.end(),
            [&path](const RecentProject& p) { return p.path == path; }),
        recent_projects_.end()
    );

    // Add at the front
    RecentProject rp;
    rp.name = name;
    rp.path = path;
    rp.last_opened = std::time(nullptr);
    recent_projects_.insert(recent_projects_.begin(), rp);

    // Trim to max size
    if (recent_projects_.size() > MAX_RECENT_PROJECTS) {
        recent_projects_.resize(MAX_RECENT_PROJECTS);
    }

    // Save to disk
    SaveRecentProjects();
}

void ProjectManager::ClearRecentProjects() {
    recent_projects_.clear();
    SaveRecentProjects();
}

std::string ProjectManager::GetRecentProjectsFilePath() const {
    // Store in user's app data directory
#ifdef _WIN32
    char appdata[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, appdata))) {
        fs::path config_dir = fs::path(appdata) / "CyxWiz";
        fs::create_directories(config_dir);
        return (config_dir / "recent_projects.json").string();
    }
    return "recent_projects.json";
#else
    // Linux/Mac - use ~/.config/cyxwiz/
    const char* home = std::getenv("HOME");
    if (home) {
        fs::path config_dir = fs::path(home) / ".config" / "cyxwiz";
        fs::create_directories(config_dir);
        return (config_dir / "recent_projects.json").string();
    }
    return "recent_projects.json";
#endif
}

void ProjectManager::LoadRecentProjects() {
    try {
        std::string file_path = GetRecentProjectsFilePath();
        if (!fs::exists(file_path)) {
            return;
        }

        std::ifstream file(file_path);
        if (!file.is_open()) {
            return;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        recent_projects_.clear();
        if (j.contains("recent_projects") && j["recent_projects"].is_array()) {
            for (const auto& item : j["recent_projects"]) {
                RecentProject rp;
                rp.name = item.value("name", "");
                rp.path = item.value("path", "");
                rp.last_opened = item.value("last_opened", std::time_t(0));

                // Only add if the file still exists
                if (!rp.path.empty() && fs::exists(rp.path)) {
                    recent_projects_.push_back(rp);
                }
            }
        }

        spdlog::info("Loaded {} recent projects", recent_projects_.size());

    } catch (const std::exception& e) {
        spdlog::warn("Failed to load recent projects: {}", e.what());
    }
}

void ProjectManager::SaveRecentProjects() {
    try {
        std::string file_path = GetRecentProjectsFilePath();

        nlohmann::json j;
        j["recent_projects"] = nlohmann::json::array();

        for (const auto& rp : recent_projects_) {
            nlohmann::json item;
            item["name"] = rp.name;
            item["path"] = rp.path;
            item["last_opened"] = rp.last_opened;
            j["recent_projects"].push_back(item);
        }

        std::ofstream file(file_path);
        if (file.is_open()) {
            file << j.dump(2);
            file.close();
        }

    } catch (const std::exception& e) {
        spdlog::warn("Failed to save recent projects: {}", e.what());
    }
}

} // namespace cyxwiz
