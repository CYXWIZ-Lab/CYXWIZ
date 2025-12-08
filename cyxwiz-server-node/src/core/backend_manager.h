// backend_manager.h - Singleton service container for server node
#pragma once

#include <memory>
#include <string>
#include <mutex>

// Forward declarations
namespace cyxwiz::servernode::api {
    class APIKeyManager;
}
#ifdef CYXWIZ_IS_DAEMON
namespace cyxwiz::servernode {
    class JobExecutor;
    class DeploymentManager;
    class NodeClient;
    class OpenAIAPIServer;
}
#endif

namespace cyxwiz::servernode::core {

class MetricsCollector;
class StateManager;
class ConfigManager;

struct NodeConfig {
    // Node identity
    std::string node_id;
    std::string node_name = "CyxWiz Server Node";
    std::string region = "us-west-2";

    // Network
    std::string central_server = "localhost:50051";
    int p2p_port = 50052;
    int terminal_port = 50053;
    int node_service_port = 50054;
    int deployment_port = 50055;
    int http_api_port = 8080;
    bool enable_tls = false;
    std::string cert_path;
    std::string key_path;

    // Training
    bool training_enabled = true;
    int max_concurrent_jobs = 2;
    float gpu_allocation = 0.8f;

    // Deployment
    bool deployment_enabled = true;
    std::string models_directory = "./models";
    int max_loaded_models = 3;

    // API
    bool api_require_auth = true;
    int api_default_rate_limit = 100;

    // Pool mining
    bool pool_mining_enabled = false;
    std::string pool_address = "pool.cyxwiz.io:3333";
    float mining_intensity = 0.5f;
    bool mine_when_idle_only = true;

    // Wallet
    std::string wallet_address;
    double auto_withdraw_threshold = 100.0;

    // Logging
    std::string log_level = "info";
    std::string log_file = "./logs/server-node.log";
};

class BackendManager {
public:
    static BackendManager& Instance();

    // Delete copy/move
    BackendManager(const BackendManager&) = delete;
    BackendManager& operator=(const BackendManager&) = delete;
    BackendManager(BackendManager&&) = delete;
    BackendManager& operator=(BackendManager&&) = delete;

    // Lifecycle
    bool Initialize(const NodeConfig& config);
    void Shutdown();
    bool IsInitialized() const { return initialized_; }

    // Service access
    MetricsCollector* GetMetricsCollector();
    StateManager* GetStateManager();
    ConfigManager* GetConfigManager();
    api::APIKeyManager* GetAPIKeyManager();
#ifdef CYXWIZ_IS_DAEMON
    JobExecutor* GetJobExecutor();
    DeploymentManager* GetDeploymentManager();
    NodeClient* GetNodeClient();
    OpenAIAPIServer* GetAPIServer();
#endif

    // Configuration
    const NodeConfig& GetConfig() const { return config_; }
    void UpdateConfig(const NodeConfig& config);

private:
    BackendManager() = default;
    ~BackendManager();

    bool initialized_ = false;
    NodeConfig config_;

    // Services
    std::shared_ptr<MetricsCollector> metrics_collector_;
    std::shared_ptr<StateManager> state_manager_;
    std::shared_ptr<ConfigManager> config_manager_;
    std::shared_ptr<api::APIKeyManager> api_key_manager_;
#ifdef CYXWIZ_IS_DAEMON
    std::shared_ptr<JobExecutor> job_executor_;
    std::shared_ptr<DeploymentManager> deployment_manager_;
    std::shared_ptr<NodeClient> node_client_;
    std::unique_ptr<OpenAIAPIServer> api_server_;
#endif

    mutable std::mutex mutex_;
};

} // namespace cyxwiz::servernode::core
