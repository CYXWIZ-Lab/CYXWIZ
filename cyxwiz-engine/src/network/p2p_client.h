#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <grpcpp/grpcpp.h>
#include "execution.grpc.pb.h"
#include "job.pb.h"
#include "dataset_provider.h"

namespace network {

// Forward declarations
struct NodeCapabilities {
    uint64_t max_memory;
    uint32_t max_batch_size;
    std::vector<cyxwiz::protocol::DeviceType> supported_devices;
    bool supports_checkpointing;
    bool supports_distributed;
};

struct TrainingProgress {
    uint32_t current_epoch;
    uint32_t total_epochs;
    uint32_t current_batch;
    uint32_t total_batches;
    float progress_percentage;
    std::map<std::string, float> metrics;  // loss, accuracy, etc.
    float gpu_usage;
    float memory_usage;
};

struct CheckpointInfo {
    uint32_t epoch;
    std::string checkpoint_hash;
    std::string storage_uri;
    uint64_t size_bytes;
};

struct TrainingComplete {
    bool success;
    std::map<std::string, float> final_metrics;
    uint64_t total_training_time;
    std::string result_hash;
    std::string model_uri;
};

// Callback types for P2P events
using ProgressCallback = std::function<void(const TrainingProgress&)>;
using CheckpointCallback = std::function<void(const CheckpointInfo&)>;
using CompletionCallback = std::function<void(const TrainingComplete&)>;
using ErrorCallback = std::function<void(const std::string& error_message, bool is_fatal)>;
using LogCallback = std::function<void(const std::string& source, const std::string& message)>;

/**
 * P2PClient - Direct communication with Server Node for job execution
 *
 * This client handles the P2P workflow:
 * 1. Connect to assigned node (with JWT token from Central Server)
 * 2. Send job configuration and dataset
 * 3. Stream real-time training metrics (bidirectional)
 * 4. Download trained model weights
 *
 * Usage:
 *   P2PClient client;
 *   client.SetProgressCallback([](const TrainingProgress& prog) { ... });
 *   client.ConnectToNode("localhost:50052", "job_123", "jwt_token");
 *   client.SendJob(job_config, dataset);
 *   client.StartTrainingStream("job_123");
 *   // ... wait for completion
 *   client.DownloadWeights("job_123", "./model.pt");
 */
class P2PClient {
public:
    P2PClient();
    ~P2PClient();

    // Connection management
    bool ConnectToNode(const std::string& node_address,
                      const std::string& job_id,
                      const std::string& auth_token,
                      const std::string& engine_version = "CyxWiz-Engine/1.0.0");

    void Disconnect();
    bool NotifyDisconnect(const std::string& reason = "");  // Notify server before disconnect
    bool IsConnected() const { return connected_; }

    // Get node information
    const std::string& GetNodeId() const { return node_id_; }
    const std::string& GetNodeAddress() const { return node_address_; }
    const NodeCapabilities& GetNodeCapabilities() const { return capabilities_; }

    // Job submission
    bool SendJob(const cyxwiz::protocol::JobConfig& config,
                const std::string& initial_dataset = "");

    bool SendJobWithDatasetURI(const cyxwiz::protocol::JobConfig& config,
                              const std::string& dataset_uri);

    // Training control and monitoring
    bool StartTrainingStream(const std::string& job_id);
    void StopTrainingStream();
    bool IsStreaming() const { return streaming_; }

    // Interactive training control (sent via bidirectional stream)
    bool PauseTraining();
    bool ResumeTraining();
    bool StopTraining();
    bool RequestCheckpoint();

    // Reservation-based job management
    bool SendNewJobConfig(const cyxwiz::protocol::JobConfig& config);
    bool SendReservationEnd();
    bool IsWaitingForNewJob() const { return waiting_for_new_job_; }
    void SetWaitingForNewJob(bool waiting) { waiting_for_new_job_ = waiting; }

    // Weights download
    bool DownloadWeights(const std::string& job_id,
                        const std::string& output_path,
                        size_t chunk_size = 1024 * 1024);  // Default 1MB chunks

    bool DownloadWeightsWithOffset(const std::string& job_id,
                                   const std::string& output_path,
                                   size_t offset,
                                   size_t chunk_size = 1024 * 1024);

    // Event callbacks
    void SetProgressCallback(ProgressCallback callback) { progress_callback_ = callback; }
    void SetCheckpointCallback(CheckpointCallback callback) { checkpoint_callback_ = callback; }
    void SetCompletionCallback(CompletionCallback callback) { completion_callback_ = callback; }
    void SetErrorCallback(ErrorCallback callback) { error_callback_ = callback; }
    void SetLogCallback(LogCallback callback) { log_callback_ = callback; }

    // Error handling
    std::string GetLastError() const { return last_error_; }

    // Authentication
    void SetAuthToken(const std::string& token) { auth_token_ = token; }
    void ClearAuthToken() { auth_token_.clear(); }
    bool HasAuthToken() const { return !auth_token_.empty(); }

    // Dataset provider for lazy-loading streaming
    DatasetProvider& GetDatasetProvider() { return dataset_provider_; }
    const DatasetProvider& GetDatasetProvider() const { return dataset_provider_; }

    // Register a dataset for streaming to Server Node
    void RegisterDatasetForJob(const std::string& job_id, cyxwiz::DatasetHandle dataset);
    void UnregisterDatasetForJob(const std::string& job_id);

private:
    // Add authorization header to gRPC context
    void AddAuthMetadata(grpc::ClientContext& context);
    // Internal streaming thread function
    void StreamingThreadFunc(const std::string& job_id);

    // Send control command during streaming
    bool SendTrainingCommand(const cyxwiz::protocol::TrainingCommand& command);

    // Handle dataset requests from Server Node
    void HandleDatasetRequest(const cyxwiz::protocol::TrainingUpdate& update);

    // Connection state
    bool connected_;
    std::atomic<bool> streaming_;
    std::atomic<bool> waiting_for_new_job_{false};  // Job complete, waiting for new config
    std::string node_address_;
    std::string node_id_;
    std::string current_job_id_;
    NodeCapabilities capabilities_;
    std::string last_error_;
    std::string auth_token_;  // JWT token for authentication

    // gRPC communication
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::JobExecutionService::Stub> stub_;

    // Bidirectional streaming
    std::unique_ptr<grpc::ClientContext> stream_context_;
    std::shared_ptr<grpc::ClientReaderWriter<cyxwiz::protocol::TrainingCommand,
                                             cyxwiz::protocol::TrainingUpdate>> stream_;
    std::thread streaming_thread_;
    std::atomic<bool> streaming_thread_done_{true};  // Set when thread exits
    mutable std::mutex stream_mutex_;  // Protects stream_ access for thread safety

    // Event callbacks
    ProgressCallback progress_callback_;
    CheckpointCallback checkpoint_callback_;
    CompletionCallback completion_callback_;
    ErrorCallback error_callback_;
    LogCallback log_callback_;

    // Dataset provider for lazy-loading streaming
    DatasetProvider dataset_provider_;
};

} // namespace network
