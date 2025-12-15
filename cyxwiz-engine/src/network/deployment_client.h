// deployment_client.h - gRPC client for deploying models to Server Node
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>

// Fix Windows macro conflict with protobuf DeviceCapabilities message
#ifdef _WIN32
#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif
#endif

#include <grpcpp/grpcpp.h>
#include "deployment.grpc.pb.h"

namespace network {

/**
 * Configuration for model deployment
 */
struct DeploymentConfig {
    std::string model_path;         // Path to .cyxmodel file
    std::string model_name;         // Display name (optional, derived from path if empty)
    int port = 8080;                // Inference server port
    int gpu_layers = 0;             // GPU layers (0 = CPU only)
    int context_size = 2048;        // Context window size
    bool enable_terminal = false;   // Enable terminal access
};

/**
 * Result of a deployment operation
 */
struct DeploymentResult {
    bool success = false;
    std::string deployment_id;
    std::string endpoint_url;
    std::string error_message;
};

/**
 * Summary of an active deployment
 */
struct DeploymentSummary {
    std::string id;
    std::string model_name;
    std::string model_path;
    int status;  // DeploymentStatus enum
    int port;
    int gpu_layers;
    uint64_t request_count;
    double avg_latency_ms;
};

/**
 * DeploymentClient - Connect to Server Node and manage deployments
 *
 * Uses gRPC to communicate with the Server Node's DeploymentService
 * on port 50055 (default).
 */
class DeploymentClient {
public:
    DeploymentClient();
    ~DeploymentClient();

    // Non-copyable
    DeploymentClient(const DeploymentClient&) = delete;
    DeploymentClient& operator=(const DeploymentClient&) = delete;

    /**
     * Connect to Server Node
     * @param server_address Address in format "host:port" (e.g., "localhost:50055")
     * @return true if connected successfully
     */
    bool Connect(const std::string& server_address);

    /**
     * Disconnect from Server Node
     */
    void Disconnect();

    /**
     * Check if connected
     */
    bool IsConnected() const { return connected_; }

    /**
     * Get server address
     */
    const std::string& GetServerAddress() const { return server_address_; }

    /**
     * Deploy a model (blocking)
     * @param config Deployment configuration
     * @return Result with deployment_id and endpoint_url on success
     */
    DeploymentResult Deploy(const DeploymentConfig& config);

    /**
     * Deploy a model asynchronously
     * @param config Deployment configuration
     * @param callback Called when deployment completes
     */
    void DeployAsync(const DeploymentConfig& config,
                     std::function<void(const DeploymentResult&)> callback);

    /**
     * List active deployments
     * @param deployments Output vector of deployment summaries
     * @return true if successful
     */
    bool ListDeployments(std::vector<DeploymentSummary>& deployments);

    /**
     * Get deployment status
     * @param deployment_id Deployment ID
     * @param summary Output deployment summary
     * @return true if found
     */
    bool GetDeployment(const std::string& deployment_id, DeploymentSummary& summary);

    /**
     * Stop a deployment
     * @param deployment_id Deployment ID to stop
     * @return true if stopped successfully
     */
    bool StopDeployment(const std::string& deployment_id);

    /**
     * Delete a deployment (removes from server)
     * @param deployment_id Deployment ID to delete
     * @return true if deleted successfully
     */
    bool DeleteDeployment(const std::string& deployment_id);

    /**
     * Get last error message
     */
    const std::string& GetLastError() const { return last_error_; }

private:
    bool connected_ = false;
    std::string server_address_;
    std::string last_error_;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::DeploymentService::Stub> stub_;
};

} // namespace network
