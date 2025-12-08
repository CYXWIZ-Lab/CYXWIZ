// config_manager.h - YAML configuration loading/saving
#pragma once

#include "core/backend_manager.h"
#include <string>

namespace cyxwiz::servernode::core {

class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager() = default;

    // Load config from YAML file into internal cache
    bool Load(const std::string& path);

    // Load config from YAML file
    bool LoadConfig(const std::string& path, NodeConfig& config);

    // Save cached config to file
    bool Save();

    // Save config to YAML file
    bool SaveConfig(const NodeConfig& config, const std::string& path);

    // Get cached config
    const NodeConfig& GetConfig() const { return cached_config_; }

    // Set cached config
    void SetConfig(const NodeConfig& config) { cached_config_ = config; }

    // Get default config
    static NodeConfig GetDefaultConfig();

    // Get config file path (checks multiple locations)
    static std::string FindConfigFile();

private:
    std::string last_loaded_path_;
    NodeConfig cached_config_;
};

} // namespace cyxwiz::servernode::core
