// config_manager.cpp - YAML configuration implementation
#include "core/config_manager.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::core {

ConfigManager::ConfigManager() {
    cached_config_ = GetDefaultConfig();
    spdlog::debug("ConfigManager created");
}

bool ConfigManager::Load(const std::string& path) {
    return LoadConfig(path, cached_config_);
}

bool ConfigManager::Save() {
    if (last_loaded_path_.empty()) {
        last_loaded_path_ = FindConfigFile();
    }
    return SaveConfig(cached_config_, last_loaded_path_);
}

bool ConfigManager::LoadConfig(const std::string& path, NodeConfig& config) {
    try {
        if (!std::filesystem::exists(path)) {
            spdlog::warn("Config file not found: {}, using defaults", path);
            config = GetDefaultConfig();
            return true;
        }

        YAML::Node yaml = YAML::LoadFile(path);
        last_loaded_path_ = path;

        // Node settings
        if (yaml["node"]) {
            auto node = yaml["node"];
            if (node["id"]) config.node_id = node["id"].as<std::string>();
            if (node["name"]) config.node_name = node["name"].as<std::string>();
            if (node["region"]) config.region = node["region"].as<std::string>();
        }

        // Network settings
        if (yaml["network"]) {
            auto network = yaml["network"];
            if (network["central_server"]) config.central_server = network["central_server"].as<std::string>();
            if (network["p2p_port"]) config.p2p_port = network["p2p_port"].as<int>();
            if (network["terminal_port"]) config.terminal_port = network["terminal_port"].as<int>();
            if (network["node_service_port"]) config.node_service_port = network["node_service_port"].as<int>();
            if (network["deployment_port"]) config.deployment_port = network["deployment_port"].as<int>();
            if (network["http_api_port"]) config.http_api_port = network["http_api_port"].as<int>();
            if (network["enable_tls"]) config.enable_tls = network["enable_tls"].as<bool>();
            if (network["cert_path"]) config.cert_path = network["cert_path"].as<std::string>();
            if (network["key_path"]) config.key_path = network["key_path"].as<std::string>();
            if (network["p2p_secret"]) config.p2p_secret = network["p2p_secret"].as<std::string>();
        }

        // Training settings
        if (yaml["training"]) {
            auto training = yaml["training"];
            if (training["enabled"]) config.training_enabled = training["enabled"].as<bool>();
            if (training["max_concurrent_jobs"]) config.max_concurrent_jobs = training["max_concurrent_jobs"].as<int>();
            if (training["gpu_allocation"]) config.gpu_allocation = training["gpu_allocation"].as<float>();
        }

        // Deployment settings
        if (yaml["deployment"]) {
            auto deployment = yaml["deployment"];
            if (deployment["enabled"]) config.deployment_enabled = deployment["enabled"].as<bool>();
            if (deployment["models_directory"]) config.models_directory = deployment["models_directory"].as<std::string>();
            if (deployment["max_loaded_models"]) config.max_loaded_models = deployment["max_loaded_models"].as<int>();
        }

        // API settings
        if (yaml["api"]) {
            auto api = yaml["api"];
            if (api["require_authentication"]) config.api_require_auth = api["require_authentication"].as<bool>();
            if (api["default_rate_limit"]) config.api_default_rate_limit = api["default_rate_limit"].as<int>();
        }

        // Pool mining settings
        if (yaml["pool_mining"]) {
            auto pool = yaml["pool_mining"];
            if (pool["enabled"]) config.pool_mining_enabled = pool["enabled"].as<bool>();
            if (pool["pool_address"]) config.pool_address = pool["pool_address"].as<std::string>();
            if (pool["intensity"]) config.mining_intensity = pool["intensity"].as<float>();
            if (pool["mine_when_idle_only"]) config.mine_when_idle_only = pool["mine_when_idle_only"].as<bool>();
        }

        // Wallet settings
        if (yaml["wallet"]) {
            auto wallet = yaml["wallet"];
            if (wallet["address"]) config.wallet_address = wallet["address"].as<std::string>();
            if (wallet["auto_withdraw_threshold"]) config.auto_withdraw_threshold = wallet["auto_withdraw_threshold"].as<double>();
        }

        // Logging settings
        if (yaml["logging"]) {
            auto logging = yaml["logging"];
            if (logging["level"]) config.log_level = logging["level"].as<std::string>();
            if (logging["file"]) config.log_file = logging["file"].as<std::string>();
        }

        spdlog::info("Config loaded from: {}", path);
        return true;

    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to parse config file: {}", e.what());
        config = GetDefaultConfig();
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        config = GetDefaultConfig();
        return false;
    }
}

bool ConfigManager::SaveConfig(const NodeConfig& config, const std::string& path) {
    try {
        // Create parent directories if needed
        std::filesystem::path file_path(path);
        if (file_path.has_parent_path()) {
            std::filesystem::create_directories(file_path.parent_path());
        }

        YAML::Emitter out;
        out << YAML::BeginMap;

        // Node settings
        out << YAML::Key << "node" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "id" << YAML::Value << config.node_id;
        out << YAML::Key << "name" << YAML::Value << config.node_name;
        out << YAML::Key << "region" << YAML::Value << config.region;
        out << YAML::EndMap;

        // Network settings
        out << YAML::Key << "network" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "central_server" << YAML::Value << config.central_server;
        out << YAML::Key << "p2p_port" << YAML::Value << config.p2p_port;
        out << YAML::Key << "terminal_port" << YAML::Value << config.terminal_port;
        out << YAML::Key << "node_service_port" << YAML::Value << config.node_service_port;
        out << YAML::Key << "deployment_port" << YAML::Value << config.deployment_port;
        out << YAML::Key << "http_api_port" << YAML::Value << config.http_api_port;
        out << YAML::Key << "enable_tls" << YAML::Value << config.enable_tls;
        out << YAML::Key << "cert_path" << YAML::Value << config.cert_path;
        out << YAML::Key << "key_path" << YAML::Value << config.key_path;
        out << YAML::Key << "p2p_secret" << YAML::Value << config.p2p_secret;
        out << YAML::EndMap;

        // Training settings
        out << YAML::Key << "training" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "enabled" << YAML::Value << config.training_enabled;
        out << YAML::Key << "max_concurrent_jobs" << YAML::Value << config.max_concurrent_jobs;
        out << YAML::Key << "gpu_allocation" << YAML::Value << config.gpu_allocation;
        out << YAML::EndMap;

        // Deployment settings
        out << YAML::Key << "deployment" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "enabled" << YAML::Value << config.deployment_enabled;
        out << YAML::Key << "models_directory" << YAML::Value << config.models_directory;
        out << YAML::Key << "max_loaded_models" << YAML::Value << config.max_loaded_models;
        out << YAML::EndMap;

        // API settings
        out << YAML::Key << "api" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "require_authentication" << YAML::Value << config.api_require_auth;
        out << YAML::Key << "default_rate_limit" << YAML::Value << config.api_default_rate_limit;
        out << YAML::EndMap;

        // Pool mining settings
        out << YAML::Key << "pool_mining" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "enabled" << YAML::Value << config.pool_mining_enabled;
        out << YAML::Key << "pool_address" << YAML::Value << config.pool_address;
        out << YAML::Key << "intensity" << YAML::Value << config.mining_intensity;
        out << YAML::Key << "mine_when_idle_only" << YAML::Value << config.mine_when_idle_only;
        out << YAML::EndMap;

        // Wallet settings
        out << YAML::Key << "wallet" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "address" << YAML::Value << config.wallet_address;
        out << YAML::Key << "auto_withdraw_threshold" << YAML::Value << config.auto_withdraw_threshold;
        out << YAML::EndMap;

        // Logging settings
        out << YAML::Key << "logging" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "level" << YAML::Value << config.log_level;
        out << YAML::Key << "file" << YAML::Value << config.log_file;
        out << YAML::EndMap;

        out << YAML::EndMap;

        std::ofstream file(path);
        file << out.c_str();
        file.close();

        spdlog::info("Config saved to: {}", path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to save config: {}", e.what());
        return false;
    }
}

NodeConfig ConfigManager::GetDefaultConfig() {
    NodeConfig config;
    config.node_id = "auto";
    return config;
}

std::string ConfigManager::FindConfigFile() {
    // Check in order of priority
    std::vector<std::string> paths = {
        "./config/server_config.yaml",
        "./server_config.yaml",
        "../config/server_config.yaml",
#ifdef _WIN32
        std::string(getenv("APPDATA") ? getenv("APPDATA") : "") + "/CyxWiz/server_config.yaml",
#else
        std::string(getenv("HOME") ? getenv("HOME") : "") + "/.config/cyxwiz/server_config.yaml",
#endif
    };

    for (const auto& path : paths) {
        if (!path.empty() && std::filesystem::exists(path)) {
            return path;
        }
    }

    return "./config/server_config.yaml";  // Default location
}

} // namespace cyxwiz::servernode::core
