#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "execution.grpc.pb.h"
#include "auth/p2p_jwt_validator.h"

// Forward declarations - JobExecutor is in servernode namespace
namespace cyxwiz {
namespace servernode {
class JobExecutor;
class NodeClient;  // For communicating with Central Server
}
namespace server_node {
class RemoteDataLoader;
}
}

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
    void Initialize(std::shared_ptr<cyxwiz::servernode::JobExecutor> executor,
                   const std::string& central_server_address,
                   const std::string& node_id,
                   const std::string& p2p_secret);

    // Set the NodeClient for Central Server communication
    void SetNodeClient(std::shared_ptr<cyxwiz::servernode::NodeClient> client) {
        node_client_ = client;
    }

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

    // ========== P2P Training Control RPCs ==========

    /**
     * PauseTraining - Pause training and save checkpoint
     */
    grpc::Status PauseTraining(
        grpc::ServerContext* context,
        const cyxwiz::protocol::PauseTrainingRequest* request,
        cyxwiz::protocol::PauseTrainingResponse* response) override;

    /**
     * ResumeTraining - Resume training from checkpoint
     */
    grpc::Status ResumeTraining(
        grpc::ServerContext* context,
        const cyxwiz::protocol::ResumeTrainingRequest* request,
        cyxwiz::protocol::ResumeTrainingResponse* response) override;

    /**
     * CancelTraining - Cancel training and optionally save partial model
     */
    grpc::Status CancelTraining(
        grpc::ServerContext* context,
        const cyxwiz::protocol::CancelTrainingRequest* request,
        cyxwiz::protocol::CancelTrainingResponse* response) override;

    /**
     * StartNewJob - Start a new training job within the same reservation
     */
    grpc::Status StartNewJob(
        grpc::ServerContext* context,
        const cyxwiz::protocol::StartNewJobRequest* request,
        cyxwiz::protocol::StartNewJobResponse* response) override;

private:
    // Connection tracking
    struct ConnectionInfo {
        std::string job_id;
        std::string engine_address;
        std::string auth_token;
        int64_t connected_at;
        bool is_authenticated;
    };

    // Reservation limits (minimal - user paid for time, can use it freely)
    static constexpr int MIN_EPOCHS_PER_JOB = 1;             // Minimum epochs per job
    // NO max jobs limit - user paid for time, can submit unlimited jobs

    // Job tracking
    struct JobSession {
        std::string job_id;
        std::string reservation_id;
        std::string engine_address;
        cyxwiz::protocol::JobConfig job_config;
        std::atomic<bool> is_running{false};
        std::atomic<bool> is_paused{false};
        std::atomic<bool> should_stop{false};
        std::string final_weights_path;
        std::mutex metrics_mutex;
        cyxwiz::protocol::TrainingProgress latest_progress;

        // Remote data loaders for lazy-loading datasets
        std::shared_ptr<RemoteDataLoader> train_loader;
        std::shared_ptr<RemoteDataLoader> val_loader;

        // Checkpoint state for pause/resume
        std::string checkpoint_path;
        std::atomic<int> paused_at_epoch{0};
        std::atomic<int> paused_at_batch{0};
        std::atomic<int> completed_epochs{0};

        // Condition variable for pause/resume synchronization
        std::condition_variable pause_cv;
        std::mutex pause_mutex;

        // Reservation-based tracking (multiple jobs per reservation)
        std::atomic<int> jobs_completed_in_reservation{0};
        std::atomic<bool> waiting_for_new_job{false};
        std::chrono::steady_clock::time_point reservation_start;
        std::chrono::seconds reservation_duration{0};
    };

    // Helper methods
    bool VerifyAuthToken(const std::string& token, const std::string& job_id);
    bool NotifyCentralServer(const std::string& job_id, const std::string& node_id);
    void NotifyJobEnded(const std::string& job_id, bool success, const std::string& reason);
    void CleanupJob(const std::string& job_id);

    // Reservation-based job management
    void ResetJobSession(JobSession* session);
    void ReportJobComplete(const std::string& job_id, bool success);
    void ReportReservationEnd(const std::string& reservation_id, int jobs_completed);
    bool RunTrainingLoop(JobSession* session,
                        grpc::ServerReaderWriter<cyxwiz::protocol::TrainingUpdate,
                                                  cyxwiz::protocol::TrainingCommand>* stream,
                        std::shared_ptr<std::mutex> stream_mutex);
    void UpdateJobMetrics(const std::string& job_id,
                         const cyxwiz::protocol::TrainingProgress& progress);
    std::string SaveDatasetToFile(const std::string& job_id,
                                  const std::string& dataset_data);

    // Checkpoint helpers
    std::string SaveCheckpoint(const std::string& job_id, int epoch, int batch);
    bool LoadCheckpoint(const std::string& job_id, const std::string& checkpoint_path);
    std::string SavePartialModel(const std::string& job_id);

    // Server management
    std::unique_ptr<grpc::Server> server_;
    std::atomic<bool> is_running_{false};

    // Dependencies
    std::shared_ptr<cyxwiz::servernode::JobExecutor> job_executor_;
    std::string central_server_address_;

    // Connection and job tracking
    std::mutex connections_mutex_;
    std::unordered_map<std::string, ConnectionInfo> connections_;  // context_id -> info

    std::mutex jobs_mutex_;
    std::unordered_map<std::string, std::unique_ptr<JobSession>> active_jobs_;  // job_id -> session

    // Node information
    std::string node_id_;
    cyxwiz::protocol::NodeCapabilities capabilities_;

    // P2P JWT validator
    std::unique_ptr<P2PJwtValidator> jwt_validator_;

    // Central Server communication client
    std::shared_ptr<cyxwiz::servernode::NodeClient> node_client_;
};

} // namespace server_node
} // namespace cyxwiz