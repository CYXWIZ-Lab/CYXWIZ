#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include "deployment.pb.h"
#include "model_loader.h"

namespace cyxwiz {
namespace servernode {

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

    // Metrics
    std::atomic<uint64_t> request_count{0};
    std::atomic<double> total_latency_ms{0.0};
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point started_at;
};

// Metrics snapshot for a deployment
struct DeploymentMetrics {
    double cpu_usage;
    double gpu_usage;
    uint64_t memory_usage;
    uint64_t request_count;
    double avg_latency_ms;
    std::string timestamp;
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
