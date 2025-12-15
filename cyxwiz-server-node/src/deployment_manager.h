#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include "deployment.pb.h"
#include "model_loader.h"
#include <cyxwiz/tensor.h>

namespace cyxwiz {
namespace servernode {

// Inference request structure
struct InferenceRequest {
    std::string request_id;
    std::unordered_map<std::string, cyxwiz::Tensor> inputs;
    std::promise<std::pair<bool, std::unordered_map<std::string, cyxwiz::Tensor>>> promise;
};

// Represents a single deployment instance
struct DeploymentInstance {
    std::string id;
    std::string model_id;
    protocol::DeploymentType type;
    protocol::DeploymentStatus status;
    protocol::DeploymentConfig config;
    std::unique_ptr<ModelLoader> model_loader;
    std::thread worker_thread;
    std::atomic<bool> should_stop{false};

    // Request queue for inference
    std::queue<std::shared_ptr<InferenceRequest>> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Metrics
    std::atomic<uint64_t> request_count{0};
    std::atomic<double> total_latency_ms{0.0};
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point started_at;
};

// Metrics snapshot for a deployment
struct DeploymentMetrics {
    int64_t timestamp_ms;        // Unix timestamp in milliseconds
    double cpu_usage;
    double gpu_usage;
    double memory_usage;
    uint64_t request_count;
    double avg_latency_ms;
};

// Manages all deployments on this Server Node
class DeploymentManager {
public:
    explicit DeploymentManager(const std::string& node_id);
    ~DeploymentManager();

    // Accept a new deployment from Central Server
    std::string AcceptDeployment(
        const std::string& model_id,
        protocol::DeploymentType type,
        const protocol::DeploymentConfig& config
    );

    // Stop a running deployment
    void StopDeployment(const std::string& deployment_id);

    // Remove a deployment completely (must be stopped first)
    bool RemoveDeployment(const std::string& deployment_id);

    // Get deployment status
    protocol::DeploymentStatus GetDeploymentStatus(const std::string& deployment_id) const;

    // Get deployment metrics
    std::vector<DeploymentMetrics> GetDeploymentMetrics(const std::string& deployment_id) const;

    // Get node ID
    const std::string& GetNodeId() const { return node_id_; }

    // Get number of active deployments
    size_t GetActiveDeploymentCount() const;

    // Check if deployment exists
    bool HasDeployment(const std::string& deployment_id) const;

    // Run inference on a deployment (thread-safe, queues request)
    // Returns true if inference succeeded, outputs are populated
    bool RunInference(
        const std::string& deployment_id,
        const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
        std::unordered_map<std::string, cyxwiz::Tensor>& outputs
    );

    // Get input/output specifications for a deployment
    std::vector<servernode::TensorSpec> GetInputSpecs(const std::string& deployment_id) const;
    std::vector<servernode::TensorSpec> GetOutputSpecs(const std::string& deployment_id) const;

    // Get list of all deployments (for ListDeployments RPC)
    struct DeploymentInfo {
        std::string id;
        std::string model_id;
        protocol::DeploymentType type;
        protocol::DeploymentStatus status;
    };
    std::vector<DeploymentInfo> GetAllDeployments() const;

private:
    // Execute a deployment (runs in worker thread)
    void ExecuteDeployment(DeploymentInstance* instance);

    // Load model for deployment
    bool LoadModel(DeploymentInstance* instance);

    // Run inference loop for deployment
    void RunInferenceLoop(DeploymentInstance* instance);

    // Cleanup deployment resources
    void CleanupDeployment(DeploymentInstance* instance);

    // Generate unique deployment ID
    std::string GenerateDeploymentId() const;

private:
    std::string node_id_;
    std::unordered_map<std::string, std::unique_ptr<DeploymentInstance>> deployments_;
    mutable std::mutex deployments_mutex_;
    std::atomic<bool> shutdown_{false};
};

} // namespace servernode
} // namespace cyxwiz
