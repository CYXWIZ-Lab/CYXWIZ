// engine_config.h - Centralized configuration management for CyxWiz Engine
#pragma once

#include <string>
#include <mutex>
#include <filesystem>

namespace cyxwiz::core {

/**
 * @brief Centralized configuration management for the CyxWiz Engine.
 *
 * Manages server addresses and other configuration settings.
 * Loads from engine_config.json and supports runtime modifications.
 *
 * Config file search order:
 * 1. ./engine_config.json (next to executable)
 * 2. ./config/engine_config.json
 * 3. User config directory (~/.cyxwiz/engine_config.json on Linux/macOS,
 *    %APPDATA%/CyxWiz/engine_config.json on Windows)
 *
 * Usage:
 *   auto& config = EngineConfig::Instance();
 *   std::string central = config.GetCentralServerAddress();
 */
class EngineConfig {
public:
    static EngineConfig& Instance();

    // Prevent copying
    EngineConfig(const EngineConfig&) = delete;
    EngineConfig& operator=(const EngineConfig&) = delete;

    // Load configuration from file
    bool Load();
    bool Load(const std::filesystem::path& config_path);

    // Save current configuration to file
    bool Save();
    bool Save(const std::filesystem::path& config_path);

    // Reload configuration from disk
    bool Reload();

    // Get config file path
    std::filesystem::path GetConfigPath() const;

    // ===== Server Addresses =====

    // Central Server (gRPC orchestrator)
    std::string GetCentralServerAddress() const;
    void SetCentralServerAddress(const std::string& address);

    // Auth API Server (JWT login)
    std::string GetAuthApiUrl() const;
    void SetAuthApiUrl(const std::string& url);

    // Default deployment server node
    std::string GetDefaultDeploymentAddress() const;
    void SetDefaultDeploymentAddress(const std::string& address);

    // Default P2P port (used when server node doesn't specify port)
    int GetDefaultP2PPort() const;
    void SetDefaultP2PPort(int port);

    // ===== Feature Flags =====

    // Whether to use secure (TLS) connections
    bool UseSecureConnection() const;
    void SetUseSecureConnection(bool secure);

    // Auto-connect to central server on startup
    bool AutoConnectOnStartup() const;
    void SetAutoConnectOnStartup(bool auto_connect);

    // ===== Timeouts (in seconds) =====

    int GetConnectionTimeout() const;
    void SetConnectionTimeout(int seconds);

    int GetRequestTimeout() const;
    void SetRequestTimeout(int seconds);

    // Check if configuration has been modified
    bool IsModified() const { return modified_; }

private:
    EngineConfig();
    ~EngineConfig() = default;

    // Find config file in search paths
    std::filesystem::path FindConfigFile() const;

    // Get user config directory
    std::filesystem::path GetUserConfigDir() const;

    // Set default values
    void SetDefaults();

    mutable std::mutex mutex_;
    std::filesystem::path config_path_;
    bool modified_ = false;

    // Server addresses
    std::string central_server_address_;
    std::string auth_api_url_;
    std::string default_deployment_address_;
    int default_p2p_port_ = 50052;

    // Feature flags
    bool use_secure_connection_ = false;
    bool auto_connect_on_startup_ = false;

    // Timeouts
    int connection_timeout_ = 10;  // seconds
    int request_timeout_ = 30;     // seconds
};

} // namespace cyxwiz::core
