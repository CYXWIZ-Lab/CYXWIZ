#include "project_manager.h"
#include "async_task_manager.h"
#include "core/engine_config.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <algorithm>
#include <cctype>
#include <system_error>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

#ifdef _WIN32
#include <shlobj.h>
#endif

namespace {

std::string GetVenvInterpreterPath(const fs::path& venv_dir) {
#ifdef _WIN32
    return (venv_dir / "Scripts" / "python.exe").string();
#else
    return (venv_dir / "bin" / "python").string();
#endif
}

std::string BuildVenvCommand(const std::string& python_exe, const fs::path& venv_dir, bool with_pip) {
    // Don't use -I flag - it causes venv to miss standard library modules
    std::string cmd = "\"" + python_exe + "\" -m venv \"" + venv_dir.string() + "\"";
    if (!with_pip) {
        cmd += " --without-pip";
    }
    return cmd;
}

int RunVenvCommand(const std::string& command) {
    if (command.empty()) {
        return -1;
    }
#ifdef _WIN32
    // On Windows, wrap the entire command in quotes for cmd.exe
    std::string wrapped_cmd = "cmd.exe /c \"" + command + "\"";
    return std::system(wrapped_cmd.c_str());
#else
    return std::system(command.c_str());
#endif
}

bool IsVenvBinDir(const fs::path& path) {
    auto name = path.filename().string();
#ifdef _WIN32
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name == "scripts";
#else
    return name == "bin";
#endif
}

fs::path ResolveVenvRoot(const fs::path& interpreter_path) {
    auto parent = interpreter_path.parent_path();
    if (!IsVenvBinDir(parent)) {
        return {};
    }
    auto root = parent.parent_path();
    if (root.empty()) {
        return {};
    }
    if (fs::exists(root / "pyvenv.cfg")) {
        return root;
    }
    return {};
}

fs::path ResolvePythonHomeFromInterpreter(const fs::path& interpreter_path) {
    auto venv_root = ResolveVenvRoot(interpreter_path);
    if (!venv_root.empty()) {
        return venv_root;
    }
    return interpreter_path.parent_path();
}

bool HasVenvModule(const fs::path& python_home) {
#ifdef _WIN32
    fs::path venv_dir = python_home / "Lib" / "venv";
    return fs::exists(venv_dir);
#else
    fs::path lib_dir = python_home / "lib";
    if (!fs::exists(lib_dir)) {
        return false;
    }
    for (const auto& entry : fs::directory_iterator(lib_dir)) {
        if (!entry.is_directory()) {
            continue;
        }
        auto name = entry.path().filename().string();
        if (name.rfind("python", 0) != 0) {
            continue;
        }
        if (fs::exists(entry.path() / "venv")) {
            return true;
        }
    }
    return false;
#endif
}

fs::path NormalizePath(const fs::path& path) {
    try {
        return fs::absolute(path).lexically_normal();
    } catch (...) {
        return path.lexically_normal();
    }
}

bool IsPathUnderRoot(const fs::path& path, const fs::path& root) {
    if (root.empty()) {
        return false;
    }
    auto norm_path = NormalizePath(path);
    auto norm_root = NormalizePath(root);

#ifdef _WIN32
    std::string path_str = norm_path.string();
    std::string root_str = norm_root.string();
    std::transform(path_str.begin(), path_str.end(), path_str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(root_str.begin(), root_str.end(), root_str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!root_str.empty() && root_str.back() != '\\' && root_str.back() != '/') {
        root_str += "\\";
    }
    return path_str.rfind(root_str, 0) == 0;
#else
    auto pit = norm_path.begin();
    auto rit = norm_root.begin();
    for (; rit != norm_root.end(); ++rit, ++pit) {
        if (pit == norm_path.end() || *pit != *rit) {
            return false;
        }
    }
    return true;
#endif
}

std::optional<fs::path> ReadPythonEnvInterpreterPath(const fs::path& project_dir) {
    fs::path env_file = project_dir / "python_env.json";
    if (!fs::exists(env_file)) {
        return std::nullopt;
    }

    try {
        std::ifstream file(env_file.string());
        if (!file.is_open()) {
            return std::nullopt;
        }
        nlohmann::json j;
        file >> j;

        if (!j.contains("python") || !j["python"].is_object()) {
            return std::nullopt;
        }
        const auto& python = j["python"];
        if (!python.contains("interpreter_path") || !python["interpreter_path"].is_string()) {
            return std::nullopt;
        }
        std::string interp = python["interpreter_path"].get<std::string>();
        if (interp.empty()) {
            return std::nullopt;
        }
        fs::path interp_path(interp);
        if (interp_path.is_relative()) {
            interp_path = project_dir / interp_path;
        }
        return fs::absolute(interp_path);
    } catch (...) {
        return std::nullopt;
    }
}

bool WritePythonEnvFile(const fs::path& project_dir, const std::string& interpreter_path) {
    if (interpreter_path.empty()) {
        return false;
    }

    std::string stored_path = interpreter_path;
    try {
        fs::path interp_path(interpreter_path);
        fs::path abs_interp = fs::absolute(interp_path);
        if (IsPathUnderRoot(abs_interp, project_dir)) {
            std::error_code ec;
            fs::path rel = fs::relative(abs_interp, project_dir, ec);
            if (!ec && !rel.empty()) {
                stored_path = rel.string();
            }
        }
    } catch (...) {
        // Keep the original path if normalization fails
    }

    nlohmann::json j;
    j["python"] = {
        {"interpreter_path", stored_path}
    };

    fs::path env_file = project_dir / "python_env.json";
    std::ofstream file(env_file.string());
    if (!file.is_open()) {
        spdlog::error("Failed to write python_env.json: {}", env_file.string());
        return false;
    }
    file << j.dump(2);
    return true;
}

void MaybeUpdateProjectPythonEnv(const fs::path& project_dir) {
    fs::path venv_dir = project_dir / "python";
    fs::path venv_interpreter = GetVenvInterpreterPath(venv_dir);
    if (!fs::exists(venv_interpreter)) {
        return;
    }

    auto existing = ReadPythonEnvInterpreterPath(project_dir);
    if (existing.has_value() && !IsPathUnderRoot(*existing, project_dir)) {
        // Preserve custom interpreter outside the project root
        return;
    }

    WritePythonEnvFile(project_dir, venv_interpreter.string());
}

void UpdatePythonEnvAfterSaveAs(const fs::path& old_root, const fs::path& new_root) {
    fs::path venv_dir = new_root / "python";
    fs::path venv_interpreter = GetVenvInterpreterPath(venv_dir);
    if (!fs::exists(venv_interpreter)) {
        return;
    }

    auto existing = ReadPythonEnvInterpreterPath(new_root);
    if (!existing.has_value()) {
        WritePythonEnvFile(new_root, venv_interpreter.string());
        return;
    }

    if (IsPathUnderRoot(*existing, old_root) || IsPathUnderRoot(*existing, new_root)) {
        WritePythonEnvFile(new_root, venv_interpreter.string());
    }
}

bool CreateProjectVenv(const fs::path& project_dir, std::string* error_out) {
    fs::path venv_dir = project_dir / "python";
    auto& config = cyxwiz::core::EngineConfig::Instance();

    // Use system Python from EngineConfig
    std::string system_python = config.GetSystemPythonPath();
    if (system_python.empty()) {
        if (error_out) {
            *error_out = "No system Python configured. Please configure Python in settings.";
        }
        return false;
    }

    // Verify system Python exists
    if (!fs::exists(system_python)) {
        if (error_out) {
            *error_out = "System Python not found: " + system_python;
        }
        return false;
    }

    // Verify system Python has venv module
    fs::path python_home = ResolvePythonHomeFromInterpreter(fs::path(system_python));
    if (!python_home.empty() && !HasVenvModule(python_home)) {
        if (error_out) {
            *error_out = "System Python is missing the venv module at " + python_home.string();
        }
        return false;
    }

    // Create venv using system Python
    spdlog::info("Creating project venv using system Python: {}", system_python);
    std::string command_with_pip = BuildVenvCommand(system_python, venv_dir, true);
    std::string command_without_pip = BuildVenvCommand(system_python, venv_dir, false);

    spdlog::info("Venv command: {}", command_with_pip);
    int result = RunVenvCommand(command_with_pip);
    if (result != 0) {
        spdlog::warn("Venv creation command failed, exit_code={}", result);
        spdlog::info("Retrying venv creation without pip");
        spdlog::info("Venv command (no pip): {}", command_without_pip);
        result = RunVenvCommand(command_without_pip);
    }

    if (result == 0) {
        std::string venv_interpreter = GetVenvInterpreterPath(venv_dir);
        if (!fs::exists(venv_interpreter)) {
            spdlog::warn("Venv created but interpreter not found: {}", venv_interpreter);
            std::error_code ec;
            fs::remove_all(venv_dir, ec);
            if (error_out) {
                *error_out = "Venv created but interpreter not found";
            }
            return false;
        }
        return WritePythonEnvFile(project_dir, venv_interpreter);
    }

    // Venv creation failed
    spdlog::error("Venv creation failed, exit_code={}", result);
    if (fs::exists(venv_dir)) {
        std::error_code ec;
        fs::remove_all(venv_dir, ec);
    }
    if (error_out) {
        *error_out = "Failed to create venv using system Python: " + system_python +
            ". Ensure Python has venv module installed.";
    }
    return false;
}

void QueueProjectVenvCreation(const fs::path& project_dir, const std::string& project_name) {
    std::string project_root = project_dir.string();
    std::string task_name = "Create Python venv";
    if (!project_name.empty()) {
        task_name += " (" + project_name + ")";
    }

    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        task_name,
        [project_root](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.05f, "Preparing virtual environment...");
            std::string error;
            if (!CreateProjectVenv(fs::path(project_root), &error)) {
                throw std::runtime_error(error.empty() ? "Failed to create project venv" : error);
            }
            task.ReportProgress(1.0f, "Virtual environment ready");
        },
        nullptr,
        [project_root](bool success, const std::string& error) {
            if (success) {
                spdlog::info("Project venv created: {}", project_root);
                cyxwiz::ProjectManager::Instance().NotifyProjectVenvReady(project_root);
            } else {
                spdlog::error("Project venv creation failed for {}: {}", project_root, error);
            }
        });
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool HasProjectExtension(const fs::path& path) {
    return ToLower(path.extension().string()) == ".cyxwiz";
}

} // namespace

namespace cyxwiz {

// EditorSettings JSON serialization
EditorSettings EditorSettings::FromJson(const nlohmann::json& j) {
    EditorSettings settings;
    // Script Editor settings
    settings.theme = j.value("theme", 3);
    settings.font_scale = j.value("font_scale", 1.3f);
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
    {"Graphs", {".cyxgraph"}},
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

void ProjectManager::NotifyProjectVenvReady(const std::string& project_root) {
    if (on_venv_ready_) {
        on_venv_ready_(project_root);
    }
}

const std::map<std::string, std::vector<std::string>>& ProjectManager::GetDefaultFilters() {
    return s_default_filters;
}

std::optional<std::string> ProjectManager::ResolveProjectFilePath(const std::string& path) {
    if (path.empty()) {
        return std::nullopt;
    }

    std::error_code ec;
    fs::path selected(path);
    if (!fs::exists(selected, ec) || ec) {
        return std::nullopt;
    }

    if (fs::is_regular_file(selected, ec) && HasProjectExtension(selected)) {
        return NormalizePath(fs::absolute(selected, ec)).string();
    }

    if (!fs::is_directory(selected, ec) || ec) {
        return std::nullopt;
    }

    fs::path named = selected / (selected.filename().string() + ".cyxwiz");
    if (fs::is_regular_file(named, ec) && !ec) {
        return NormalizePath(fs::absolute(named, ec)).string();
    }

    fs::path first_match;
    for (const auto& entry : fs::directory_iterator(selected, ec)) {
        if (ec) {
            spdlog::warn("Could not scan project directory {}: {}", selected.string(), ec.message());
            return std::nullopt;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!HasProjectExtension(entry.path())) {
            continue;
        }
        if (!first_match.empty()) {
            spdlog::warn("Multiple .cyxwiz files found in {}; using {}", selected.string(), first_match.string());
            return NormalizePath(fs::absolute(first_match, ec)).string();
        }
        first_match = entry.path();
    }

    if (!first_match.empty()) {
        return NormalizePath(fs::absolute(first_match, ec)).string();
    }

    return std::nullopt;
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

        QueueProjectVenvCreation(project_dir, name);

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
        // Verify file exists
        fs::path file_path(cyxwiz_file_path);
        if (!fs::exists(file_path)) {
            spdlog::error("Project file not found: {}", cyxwiz_file_path);
            return false;
        }

        // The Start Page may select/open a project before the main window is
        // constructed. Application startup then asks to open that same path
        // after the project callbacks are installed. Reopening used to close
        // the active project and restart its asynchronous asset scan, leaving
        // the first scan racing the close/reopen cycle. Make same-project open
        // idempotent and preserve the already initialized project state.
        if (HasActiveProject()) {
            std::error_code equivalent_error;
            const bool already_active = fs::equivalent(
                fs::path(project_file_path_), file_path, equivalent_error);
            if (!equivalent_error && already_active) {
                spdlog::info(
                    "Project already active; skipping duplicate open: {}",
                    project_file_path_);
                return true;
            }
            CloseProject();
        }

        // Read project file
        if (!ReadProjectFile(cyxwiz_file_path)) {
            return false;
        }

        // Set up project state
        project_file_path_ = fs::absolute(file_path).string();
        project_root_ = file_path.parent_path().string();
        project_name_ = config_.name;

        // Ensure the canonical graph directory exists for legacy projects.
        // This is additive: older user-created folders are never moved or
        // deleted during project open.
        std::error_code graph_dir_error;
        fs::create_directories(
            fs::path(project_root_) / "cyxgraph", graph_dir_error);
        if (graph_dir_error) {
            spdlog::warn(
                "Could not create canonical cyxgraph directory in '{}': {}",
                project_root_, graph_dir_error.message());
        }

        // Ensure default filters if not present. Merge newly introduced
        // canonical filters into older projects without discarding custom
        // filters stored in the project file.
        if (config_.filters.empty()) {
            InitializeDefaultFilters();
        } else {
            for (const auto& [name, extensions] : s_default_filters) {
                config_.filters.try_emplace(name, extensions);
            }
        }

        spdlog::info("Project opened: {} from {}", project_name_, project_root_);

        // Add to recent projects
        AddToRecentProjects(project_name_, project_file_path_);

        // Fire callback
        if (on_opened_) {
            on_opened_(project_root_);
        }

        // Check if project has python_env.json (legacy project check)
        fs::path python_env_file = fs::path(project_root_) / "python_env.json";
        fs::path python_dir = fs::path(project_root_) / "python";

        // Only create venv if both python_env.json AND python folder are missing
        // (if python folder exists, venv creation is likely already in progress)
        if (!fs::exists(python_env_file) && !fs::exists(python_dir)) {
            spdlog::warn("Legacy project detected (no python_env.json) - creating virtual environment");

            // Create venv for legacy project
            auto& config = cyxwiz::core::EngineConfig::Instance();
            if (config.HasSystemPython()) {
                // Use the async task manager to create venv in the background
                spdlog::info("Creating venv for legacy project: {}", project_name_);
                auto& task_mgr = cyxwiz::AsyncTaskManager::Instance();
                task_mgr.RunAsync(
                    "Create venv for legacy project",
                    [project_root = project_root_](cyxwiz::LambdaTask& task) {
                        task.ReportProgress(0.05f, "Creating virtual environment for legacy project...");
                        std::string error;
                        if (!CreateProjectVenv(fs::path(project_root), &error)) {
                            spdlog::error("Failed to create venv for legacy project: {}", error);
                            task.MarkFailed(error);
                        } else {
                            spdlog::info("Virtual environment created for legacy project");
                            task.ReportProgress(1.0f, "Virtual environment ready");
                        }
                    }
                );
            } else {
                spdlog::error("Cannot create venv for legacy project: no system Python configured");
            }
        } else if (!fs::exists(python_env_file) && fs::exists(python_dir)) {
            spdlog::info("Virtual environment creation already in progress for project: {}", project_name_);
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

    MaybeUpdateProjectPythonEnv(fs::path(project_root_));

    return WriteProjectFile(project_file_path_);
}

bool ProjectManager::SaveProjectAs(const std::string& new_name, const std::string& new_location) {
    if (!HasActiveProject()) {
        spdlog::warn("No active project to save");
        return false;
    }

    try {
        std::string old_root = project_root_;
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

        UpdatePythonEnvAfterSaveAs(fs::path(old_root), fs::path(project_root_));

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

std::string ProjectManager::GetCyxGraphsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "cyxgraph").string();
}

std::string ProjectManager::GetModelsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "models").string();
}

std::string ProjectManager::GetDatasetsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "datasets").string();
}

std::string ProjectManager::GetIngestionCachePath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "cache" / "ingestion").string();
}

std::string ProjectManager::GetArtifactsPath() const {
    if (!HasActiveProject()) return "";
    return (fs::path(project_root_) / "artifacts").string();
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
        fs::create_directories(fs::path(project_dir) / "cyxgraph");
        fs::create_directories(fs::path(project_dir) / "scripts");
        fs::create_directories(fs::path(project_dir) / "models");
        fs::create_directories(fs::path(project_dir) / "datasets");
        fs::create_directories(fs::path(project_dir) / "cache" / "ingestion");
        fs::create_directories(fs::path(project_dir) / "artifacts");
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
