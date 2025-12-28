// engine_config.cpp - Centralized configuration management implementation
#include "core/engine_config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <ShlObj.h>
#include <windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace cyxwiz::core {

using json = nlohmann::json;

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

    // Search paths in order of priority
    std::vector<std::filesystem::path> search_paths = {
        std::filesystem::current_path() / config_name,
        std::filesystem::current_path() / "config" / config_name,
        GetUserConfigDir() / config_name
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            spdlog::info("Found config file: {}", path.string());
            return path;
        }
    }

    // Return default location for creation
    return std::filesystem::current_path() / config_name;
}

bool EngineConfig::Load() {
    return Load(FindConfigFile());
}

bool EngineConfig::Load(const std::filesystem::path& config_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!std::filesystem::exists(config_path)) {
        spdlog::info("Config file not found at {}, using defaults", config_path.string());
        config_path_ = config_path;
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

        config_path_ = config_path;
        modified_ = false;

        spdlog::info("Loaded config from: {}", config_path.string());
        spdlog::debug("  Central Server: {}", central_server_address_);
        spdlog::debug("  Auth API: {}", auth_api_url_);
        spdlog::debug("  Default Deployment: {}", default_deployment_address_);
        spdlog::debug("  Default P2P Port: {}", default_p2p_port_);

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

} // namespace cyxwiz::core
