// backend_manager.cpp - Singleton service container implementation
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "core/config_manager.h"
#include "core/metrics_collector.h"
#include "api/api_key_manager.h"

#ifdef CYXWIZ_IS_DAEMON
#include "http/openai_api_server.h"
#include "job_executor.h"
#include "deployment_manager.h"
#include "node_client.h"
#endif

#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::core {

BackendManager& BackendManager::Instance() {
    static BackendManager instance;
    return instance;
}

BackendManager::~BackendManager() {
    if (initialized_) {
        Shutdown();
    }
}

bool BackendManager::Initialize(const NodeConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        spdlog::warn("BackendManager already initialized");
        return true;
    }

    config_ = config;
    spdlog::info("Initializing BackendManager with node_id: {}", config_.node_id);

    try {
        // Initialize state manager first (other services depend on it)
        state_manager_ = std::make_shared<StateManager>();

        // Initialize config manager
        config_manager_ = std::make_shared<ConfigManager>();

        // Initialize metrics collector
        metrics_collector_ = std::make_shared<MetricsCollector>();
        metrics_collector_->StartCollection(1000);  // 1 second interval

        // Initialize API key manager
        api_key_manager_ = std::make_shared<api::APIKeyManager>("./config/api_keys.json");
        api_key_manager_->Load();

#ifdef CYXWIZ_IS_DAEMON
        // Note: JobExecutor, DeploymentManager, and NodeClient are created in main.cpp
        // and will be set via setter methods or accessed via existing instances

        // Initialize HTTP API server
        if (config_.deployment_enabled) {
            api_server_ = std::make_unique<OpenAIAPIServer>(config_.http_api_port);
        }
#endif

        initialized_ = true;
        spdlog::info("BackendManager initialized successfully");
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize BackendManager: {}", e.what());
        Shutdown();
        return false;
    }
}

void BackendManager::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    spdlog::info("Shutting down BackendManager");

#ifdef CYXWIZ_IS_DAEMON
    // Stop HTTP API server
    if (api_server_) {
        api_server_->Stop();
        api_server_.reset();
    }
#endif

    // Save API keys
    if (api_key_manager_) {
        api_key_manager_->Save();
    }

    // Stop metrics collection
    if (metrics_collector_) {
        metrics_collector_->StopCollection();
    }

    // Clear all service pointers
    api_key_manager_.reset();
    metrics_collector_.reset();
    config_manager_.reset();
    state_manager_.reset();

#ifdef CYXWIZ_IS_DAEMON
    // Note: job_executor_, deployment_manager_, node_client_ are managed by main.cpp
#endif

    initialized_ = false;
    spdlog::info("BackendManager shutdown complete");
}

#ifdef CYXWIZ_IS_DAEMON
JobExecutor* BackendManager::GetJobExecutor() {
    return job_executor_.get();
}

DeploymentManager* BackendManager::GetDeploymentManager() {
    return deployment_manager_.get();
}

NodeClient* BackendManager::GetNodeClient() {
    return node_client_.get();
}
#endif

MetricsCollector* BackendManager::GetMetricsCollector() {
    return metrics_collector_.get();
}

StateManager* BackendManager::GetStateManager() {
    return state_manager_.get();
}

ConfigManager* BackendManager::GetConfigManager() {
    return config_manager_.get();
}

api::APIKeyManager* BackendManager::GetAPIKeyManager() {
    return api_key_manager_.get();
}

#ifdef CYXWIZ_IS_DAEMON
OpenAIAPIServer* BackendManager::GetAPIServer() {
    return api_server_.get();
}
#endif

void BackendManager::UpdateConfig(const NodeConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;

    // Save updated config
    if (config_manager_) {
        config_manager_->SaveConfig(config, "./config/server_config.yaml");
    }
}

} // namespace cyxwiz::servernode::core
