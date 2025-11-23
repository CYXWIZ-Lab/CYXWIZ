#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "execution.grpc.pb.h"
#include "job_executor.h"

namespace cyxwiz {
namespace server_node {

/**
 * JobExecutionService - P2P service for direct Engine→Node communication
 *
 * This service runs on port 50052 and handles:
 * - Direct connections from Engines with auth tokens
 * - Job acceptance and execution
 * - Real-time streaming of training metrics
 * - Model weights transfer
 */
class JobExecutionServiceImpl final : public cyxwiz::protocol::JobExecutionService::Service {
public:
    JobExecutionServiceImpl();
    ~JobExecutionServiceImpl();

    // Initialize the service with dependencies
    void Initialize(std::shared_ptr<JobExecutor> executor,
                   const std::string& central_server_address);

    // Start the P2P server on the specified port
    bool StartServer(const std::string& listen_address = "0.0.0.0:50052");

    // Stop the P2P server
    void StopServer();

    // ========== gRPC Service Methods ==========

    /**
     * ConnectToNode - Engine establishes connection with auth token
     */
    grpc::Status ConnectToNode(
        grpc::ServerContext* context,
        const cyxwiz::protocol::ConnectRequest* request,
        cyxwiz::protocol::ConnectResponse* response) override;

    /**
     * SendJob - Engine sends job details and dataset
     */
    grpc::Status SendJob(
        grpc::ServerContext* context,
        const cyxwiz::protocol::SendJobRequest* request,
        cyxwiz::protocol::SendJobResponse* response) override;

    /**
     * StreamTrainingMetrics - Bidirectional streaming for real-time updates
     */
    grpc::Status StreamTrainingMetrics(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<cyxwiz::protocol::TrainingUpdate,
                                cyxwiz::protocol::TrainingCommand>* stream) override;

    /**
     * DownloadWeights - Engine downloads final model weights
     */
    grpc::Status DownloadWeights(
        grpc::ServerContext* context,
        const cyxwiz::protocol::DownloadRequest* request,
        grpc::ServerWriter<cyxwiz::protocol::WeightsChunk>* writer) override;

private:
    // Connection tracking
    struct ConnectionInfo {
        std::string job_id;
        std::string engine_address;
        std::string auth_token;
        int64_t connected_at;
        bool is_authenticated;
    };

    // Job tracking
    struct JobSession {
        std::string job_id;
        std::string engine_address;
        cyxwiz::protocol::JobConfig job_config;
        std::atomic<bool> is_running;
        std::atomic<bool> is_paused;
        std::atomic<bool> should_stop;
        std::string final_weights_path;
        std::mutex metrics_mutex;
        cyxwiz::protocol::TrainingProgress latest_progress;
    };

    // Helper methods
    bool VerifyAuthToken(const std::string& token, const std::string& job_id);
    bool NotifyCentralServer(const std::string& job_id, const std::string& node_id);
    void UpdateJobMetrics(const std::string& job_id,
                         const cyxwiz::protocol::TrainingProgress& progress);
    std::string SaveDatasetToFile(const std::string& job_id,
                                  const std::string& dataset_data);

    // Server management
    std::unique_ptr<grpc::Server> server_;
    std::atomic<bool> is_running_{false};

    // Dependencies
    std::shared_ptr<JobExecutor> job_executor_;
    std::string central_server_address_;

    // Connection and job tracking
    std::mutex connections_mutex_;
    std::unordered_map<std::string, ConnectionInfo> connections_;  // context_id -> info

    std::mutex jobs_mutex_;
    std::unordered_map<std::string, std::unique_ptr<JobSession>> active_jobs_;  // job_id -> session

    // Node information
    std::string node_id_;
    cyxwiz::protocol::NodeCapabilities capabilities_;
};

} // namespace server_node
} // namespace cyxwiz