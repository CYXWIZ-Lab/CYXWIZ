// engine_config.cpp - Centralized configuration management implementation
#include "core/engine_config.h"

#include <fstream>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <ShlObj.h>
#include <windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#include <limits.h>  // PATH_MAX
#endif

namespace cyxwiz::core {

using json = nlohmann::json;

namespace {

bool IsVenvBinDir(const std::filesystem::path& path) {
    auto name = path.filename().string();
#ifdef _WIN32
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name == "scripts";
#else
    return name == "bin";
#endif
}

std::filesystem::path ResolveVenvRoot(const std::filesystem::path& interpreter_path) {
    auto parent = interpreter_path.parent_path();
    if (!IsVenvBinDir(parent)) {
        return {};
    }
    auto root = parent.parent_path();
    if (root.empty()) {
        return {};
    }
    if (std::filesystem::exists(root / "pyvenv.cfg")) {
        return root;
    }
    return {};
}

std::filesystem::path ResolvePythonHomeFromInterpreter(const std::filesystem::path& interpreter_path) {
    auto venv_root = ResolveVenvRoot(interpreter_path);
    if (!venv_root.empty()) {
        return venv_root;
    }
    return interpreter_path.parent_path();
}

std::filesystem::path FindPosixSitePackages(const std::filesystem::path& base_root) {
#ifndef _WIN32
    std::filesystem::path lib_dir = base_root / "lib";
    if (!std::filesystem::exists(lib_dir)) {
        return {};
    }

    for (const auto& entry : std::filesystem::directory_iterator(lib_dir)) {
        if (!entry.is_directory()) {
            continue;
        }
        auto name = entry.path().filename().string();
        if (name.rfind("python", 0) != 0) {
            continue;
        }
        std::filesystem::path candidate = entry.path() / "site-packages";
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
#endif
    return {};
}

std::filesystem::path ResolveSitePackagesFromInterpreter(const std::filesystem::path& interpreter_path) {
#ifdef _WIN32
    auto venv_root = ResolveVenvRoot(interpreter_path);
    if (!venv_root.empty()) {
        return venv_root / "Lib" / "site-packages";
    }
    return interpreter_path.parent_path() / "Lib" / "site-packages";
#else
    auto venv_root = ResolveVenvRoot(interpreter_path);
    std::filesystem::path base_root = venv_root.empty() ? interpreter_path.parent_path().parent_path() : venv_root;
    auto site_packages = FindPosixSitePackages(base_root);
    if (site_packages.empty()) {
        site_packages = base_root / "lib" / "python3" / "site-packages";
    }
    return site_packages;
#endif
}

std::filesystem::path GetExecutableDir() {
#ifdef _WIN32
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::filesystem::path exe_path(path);
    return exe_path.parent_path();
#else
    char path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    std::filesystem::path exe_path(count > 0 ? std::string(path, count) : ".");
    return exe_path.parent_path();
#endif
}

}  // namespace

EngineConfig& EngineConfig::Instance() {
    static EngineConfig instance;
    return instance;
}

EngineConfig::EngineConfig() {
    SetDefaults();
    Load();
}

void EngineConfig::SetDefaults() {
    central_server_address_ = "localhost:50051";
    auth_api_url_ = "http://127.0.0.1:3002/api";
    default_deployment_address_ = "localhost:50056";
    default_p2p_port_ = 50052;
    use_secure_connection_ = false;
    auto_connect_on_startup_ = false;
    connection_timeout_ = 10;
    request_timeout_ = 30;
    system_python_path_ = "";  // Will be auto-detected on first launch
}

std::filesystem::path EngineConfig::GetUserConfigDir() const {
#ifdef _WIN32
    wchar_t* appdata = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &appdata))) {
        std::filesystem::path path(appdata);
        CoTaskMemFree(appdata);
        return path / "CyxWiz";
    }
    // Fallback
    const char* appdata_env = std::getenv("APPDATA");
    if (appdata_env) {
        return std::filesystem::path(appdata_env) / "CyxWiz";
    }
    return std::filesystem::path(".");
#else
    // Linux/macOS
    const char* home = std::getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        home = pw ? pw->pw_dir : ".";
    }
    return std::filesystem::path(home) / ".cyxwiz";
#endif
}

std::filesystem::path EngineConfig::FindConfigFile() const {
    const std::string config_name = "engine_config.json";

    // Prefer the executable directory over the current working directory.
    std::filesystem::path exe_dir = GetExecutableDir();
    std::filesystem::path base_dir = exe_dir;
    if (base_dir.empty()) {
        base_dir = std::filesystem::current_path();
    }

    // Search paths in order of priority
    std::vector<std::filesystem::path> search_paths;
    if (!base_dir.empty()) {
        search_paths.push_back(base_dir / config_name);
        search_paths.push_back(base_dir / "config" / config_name);
    }
    search_paths.push_back(GetUserConfigDir() / config_name);

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            spdlog::info("Found config file: {}", path.string());
            return path;
        }
    }

    // Default location for creation
    return GetUserConfigDir() / config_name;
}

bool EngineConfig::Load() {
    return Load(FindConfigFile());
}

bool EngineConfig::Load(const std::filesystem::path& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!std::filesystem::exists(config_path)) {
        spdlog::info("Config file not found at {}, creating default config", config_path.string());
        config_path_ = config_path;

        try {
            // Ensure parent directory exists
            auto parent = config_path.parent_path();
            if (!parent.empty() && !std::filesystem::exists(parent)) {
                std::filesystem::create_directories(parent);
            }

            // Build default config using current values
            json defaults;
            defaults["servers"] = {
                {"central", central_server_address_},
                {"deployment", default_deployment_address_},
                {"default_p2p_port", default_p2p_port_}
            };
            defaults["auth"] = {
                {"api_url", auth_api_url_}
            };
            defaults["connection"] = {
                {"use_tls", use_secure_connection_},
                {"auto_connect", auto_connect_on_startup_},
                {"timeout", connection_timeout_},
                {"request_timeout", request_timeout_}
            };
            defaults["python"] = {
                {"system_python_path", system_python_path_},
                {"auto_create_venv", auto_create_venv_},
                {"default_venv_packages", default_venv_packages_}
            };
            defaults["recent_projects"] = recent_projects_;

            json config = defaults;

            // If a packaged template exists, merge it over defaults
            try {
                auto exe_dir = GetExecutableDir();
                if (!exe_dir.empty()) {
                    std::filesystem::path template_path = exe_dir / "resources" / "engine_config.json";
                    if (std::filesystem::exists(template_path)) {
                        std::ifstream tmpl_file(template_path);
                        if (tmpl_file.is_open()) {
                            json tmpl = json::parse(tmpl_file, nullptr, false);
                            if (tmpl.is_object()) {
                                config.merge_patch(tmpl);
                                spdlog::info("Seeding default config from template: {}", template_path.string());
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::debug("Failed to load config template: {}", e.what());
            }

            std::ofstream file(config_path);
            if (file.is_open()) {
                file << config.dump(4);
                spdlog::info("Wrote default config to: {}", config_path.string());
            } else {
                spdlog::warn("Failed to write default config to: {}", config_path.string());
            }
        } catch (const std::exception& e) {
            spdlog::warn("Failed to create default config: {}", e.what());
        }

        // Write a template next to the executable for portable installs
        try {
            std::filesystem::path exe_dir = GetExecutableDir();
            std::filesystem::path template_path = exe_dir / "engine_config.template.json";
            if (!exe_dir.empty() && !std::filesystem::exists(template_path)) {
                json template_config;
                template_config["servers"] = {
                    {"central", central_server_address_},
                    {"deployment", default_deployment_address_},
                    {"default_p2p_port", default_p2p_port_}
                };
                template_config["auth"] = {
                    {"api_url", auth_api_url_}
                };
                template_config["connection"] = {
                    {"use_tls", use_secure_connection_},
                    {"auto_connect", auto_connect_on_startup_},
                    {"timeout", connection_timeout_},
                    {"request_timeout", request_timeout_}
                };
                template_config["python"] = {
                    {"system_python_path", system_python_path_},
                    {"auto_create_venv", auto_create_venv_},
                    {"default_venv_packages", default_venv_packages_}
                };
                template_config["recent_projects"] = recent_projects_;

                std::ofstream tmpl(template_path);
                if (tmpl.is_open()) {
                    tmpl << template_config.dump(4);
                    spdlog::info("Wrote config template to: {}", template_path.string());
                }
            }
        } catch (const std::exception& e) {
            spdlog::debug("Failed to write config template: {}", e.what());
        }

        return false;
    }

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open config file: {}", config_path.string());
            return false;
        }

        json config = json::parse(file);

        // Server addresses
        if (config.contains("servers")) {
            const auto& servers = config["servers"];
            if (servers.contains("central")) {
                central_server_address_ = servers["central"].get<std::string>();
            }
            if (servers.contains("deployment")) {
                default_deployment_address_ = servers["deployment"].get<std::string>();
            }
            if (servers.contains("default_p2p_port")) {
                default_p2p_port_ = servers["default_p2p_port"].get<int>();
            }
        }

        // Auth settings
        if (config.contains("auth")) {
            const auto& auth = config["auth"];
            if (auth.contains("api_url")) {
                auth_api_url_ = auth["api_url"].get<std::string>();
            }
        }

        // Connection settings
        if (config.contains("connection")) {
            const auto& conn = config["connection"];
            if (conn.contains("use_tls")) {
                use_secure_connection_ = conn["use_tls"].get<bool>();
            }
            if (conn.contains("auto_connect")) {
                auto_connect_on_startup_ = conn["auto_connect"].get<bool>();
            }
            if (conn.contains("timeout")) {
                connection_timeout_ = conn["timeout"].get<int>();
            }
            if (conn.contains("request_timeout")) {
                request_timeout_ = conn["request_timeout"].get<int>();
            }
        }

        // Python settings
        if (config.contains("python")) {
            const auto& python = config["python"];
            if (python.contains("system_python_path")) {
                system_python_path_ = python["system_python_path"].get<std::string>();
            }
            // Legacy support: migrate old config format
            else if (python.contains("interpreter_path")) {
                system_python_path_ = python["interpreter_path"].get<std::string>();
                spdlog::info("Migrated legacy Python config to system_python_path");
            }
            if (python.contains("auto_create_venv")) {
                auto_create_venv_ = python["auto_create_venv"].get<bool>();
            }
            if (python.contains("default_venv_packages")) {
                default_venv_packages_ = python["default_venv_packages"].get<std::vector<std::string>>();
            }
        }

        // Recent projects
        if (config.contains("recent_projects")) {
            recent_projects_ = config["recent_projects"].get<std::vector<std::string>>();
        }

        config_path_ = config_path;
        modified_ = false;

        spdlog::info("Loaded config from: {}", config_path.string());
        spdlog::debug("  Central Server: {}", central_server_address_);
        spdlog::debug("  Auth API: {}", auth_api_url_);
        spdlog::debug("  Default Deployment: {}", default_deployment_address_);
        spdlog::debug("  Default P2P Port: {}", default_p2p_port_);
        spdlog::debug("  System Python: {}", system_python_path_.empty() ? "not configured" : system_python_path_);

        return true;

    } catch (const json::exception& e) {
        spdlog::error("Failed to parse config file: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Error loading config: {}", e.what());
        return false;
    }
}

bool EngineConfig::Save() {
    if (config_path_.empty()) {
        config_path_ = FindConfigFile();
    }
    return Save(config_path_);
}

bool EngineConfig::Save(const std::filesystem::path& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        // Create parent directory if it doesn't exist
        auto parent = config_path.parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent)) {
            std::filesystem::create_directories(parent);
        }

        json config;

        // Server addresses
        config["servers"] = {
            {"central", central_server_address_},
            {"deployment", default_deployment_address_},
            {"default_p2p_port", default_p2p_port_}
        };

        // Auth settings
        config["auth"] = {
            {"api_url", auth_api_url_}
        };

        // Connection settings
        config["connection"] = {
            {"use_tls", use_secure_connection_},
            {"auto_connect", auto_connect_on_startup_},
            {"timeout", connection_timeout_},
            {"request_timeout", request_timeout_}
        };

        // Python settings
        config["python"] = {
            {"system_python_path", system_python_path_},
            {"auto_create_venv", auto_create_venv_},
            {"default_venv_packages", default_venv_packages_}
        };

        // Recent projects
        config["recent_projects"] = recent_projects_;

        std::ofstream file(config_path);
        if (!file.is_open()) {
            spdlog::error("Failed to create config file: {}", config_path.string());
            return false;
        }

        file << config.dump(4);  // Pretty print with 4-space indent

        config_path_ = config_path;
        modified_ = false;

        spdlog::info("Saved config to: {}", config_path.string());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving config: {}", e.what());
        return false;
    }
}

bool EngineConfig::Reload() {
    return Load(config_path_);
}

std::filesystem::path EngineConfig::GetConfigPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_path_;
}

// ===== Getters and Setters =====

std::string EngineConfig::GetCentralServerAddress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return central_server_address_;
}

void EngineConfig::SetCentralServerAddress(const std::string& address) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (central_server_address_ != address) {
        central_server_address_ = address;
        modified_ = true;
    }
}

std::string EngineConfig::GetAuthApiUrl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return auth_api_url_;
}

void EngineConfig::SetAuthApiUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auth_api_url_ != url) {
        auth_api_url_ = url;
        modified_ = true;
    }
}

std::string EngineConfig::GetDefaultDeploymentAddress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return default_deployment_address_;
}

void EngineConfig::SetDefaultDeploymentAddress(const std::string& address) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (default_deployment_address_ != address) {
        default_deployment_address_ = address;
        modified_ = true;
    }
}

bool EngineConfig::UseSecureConnection() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return use_secure_connection_;
}

void EngineConfig::SetUseSecureConnection(bool secure) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (use_secure_connection_ != secure) {
        use_secure_connection_ = secure;
        modified_ = true;
    }
}

bool EngineConfig::AutoConnectOnStartup() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return auto_connect_on_startup_;
}

void EngineConfig::SetAutoConnectOnStartup(bool auto_connect) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto_connect_on_startup_ != auto_connect) {
        auto_connect_on_startup_ = auto_connect;
        modified_ = true;
    }
}

int EngineConfig::GetConnectionTimeout() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return connection_timeout_;
}

void EngineConfig::SetConnectionTimeout(int seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (connection_timeout_ != seconds) {
        connection_timeout_ = seconds;
        modified_ = true;
    }
}

int EngineConfig::GetRequestTimeout() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_timeout_;
}

void EngineConfig::SetRequestTimeout(int seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_timeout_ != seconds) {
        request_timeout_ = seconds;
        modified_ = true;
    }
}

int EngineConfig::GetDefaultP2PPort() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return default_p2p_port_;
}

void EngineConfig::SetDefaultP2PPort(int port) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (default_p2p_port_ != port) {
        default_p2p_port_ = port;
        modified_ = true;
    }
}


// ===== Python Settings =====

std::string EngineConfig::GetPythonPackagesDir() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (system_python_path_.empty()) {
        return "";  // No system Python configured
    }

    std::filesystem::path interp(system_python_path_);
    auto site_packages = ResolveSitePackagesFromInterpreter(interp);

    if (std::filesystem::exists(site_packages)) {
        return site_packages.string();
    }

    return "";  // Fall back to system default
}

std::string EngineConfig::GetSystemPythonPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return system_python_path_;
}

void EngineConfig::SetSystemPythonPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (system_python_path_ != path) {
        system_python_path_ = path;
        modified_ = true;
    }
}

bool EngineConfig::HasSystemPython() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !system_python_path_.empty() && std::filesystem::exists(system_python_path_);
}

bool EngineConfig::GetAutoCreateVenv() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return auto_create_venv_;
}

void EngineConfig::SetAutoCreateVenv(bool auto_create) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto_create_venv_ != auto_create) {
        auto_create_venv_ = auto_create;
        modified_ = true;
    }
}

std::vector<std::string> EngineConfig::GetDefaultVenvPackages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return default_venv_packages_;
}

void EngineConfig::SetDefaultVenvPackages(const std::vector<std::string>& packages) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (default_venv_packages_ != packages) {
        default_venv_packages_ = packages;
        modified_ = true;
    }
}

std::vector<std::string> EngineConfig::GetRecentProjects() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recent_projects_;
}

void EngineConfig::AddRecentProject(const std::string& project_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Remove if already exists
    auto it = std::find(recent_projects_.begin(), recent_projects_.end(), project_path);
    if (it != recent_projects_.end()) {
        recent_projects_.erase(it);
    }

    // Add to front
    recent_projects_.insert(recent_projects_.begin(), project_path);

    // Limit to 10 recent projects
    if (recent_projects_.size() > 10) {
        recent_projects_.resize(10);
    }

    modified_ = true;
}

} // namespace cyxwiz::core
