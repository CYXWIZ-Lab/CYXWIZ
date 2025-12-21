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
    int http_api_port = 8082;
    bool enable_tls = false;
    std::string cert_path;
    std::string key_path;

    // P2P Authentication
    // Secret for validating P2P JWT tokens from Central Server
    // Must match Central Server's jwt.secret or jwt.p2p_secret
    std::string p2p_secret;

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

    // Pricing - what this node charges for compute
    enum class BillingModel {
        Hourly = 0,        // Charged per hour of compute time
        PerEpoch = 1,      // Charged per training epoch
        PerJob = 2,        // Fixed price per job
        PerInference = 3   // Charged per inference request
    };
    BillingModel billing_model = BillingModel::Hourly;

    // Base prices (in CYXWIZ tokens)
    double price_per_hour = 0.10;           // Default: 0.10 CYXWIZ/hour
    double price_per_epoch = 0.01;          // Default: 0.01 CYXWIZ/epoch
    double price_per_job_base = 1.0;        // Default: 1.0 CYXWIZ flat rate
    double price_per_inference = 0.0001;    // Default: 0.0001 CYXWIZ/inference

    // Minimum charges
    double minimum_charge = 0.01;           // Minimum 0.01 CYXWIZ per job
    int minimum_duration_minutes = 1;       // Minimum 1 minute billing

    // Volume discounts (as percentages, e.g., 0.05 = 5% off)
    double discount_1h_plus = 0.0;          // Discount for jobs > 1 hour
    double discount_24h_plus = 0.10;        // 10% off for jobs > 24 hours
    double discount_bulk = 0.15;            // 15% off for repeat customers

    // Accepted payment methods
    bool accepts_cyxwiz_token = true;
    bool accepts_sol = false;
    bool accepts_usdc = false;

    // Free tier
    bool free_tier_enabled = false;
    int free_tier_minutes = 5;              // 5 minutes free for new users

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
