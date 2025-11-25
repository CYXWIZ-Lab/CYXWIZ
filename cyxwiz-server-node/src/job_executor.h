#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

// Protocol includes
#include "job.pb.h"

// Backend includes
#include <cyxwiz/cyxwiz.h>

namespace cyxwiz {
namespace servernode {

// Forward declarations
class NodeClient;

/**
 * Training metrics for progress reporting
 */
struct TrainingMetrics {
    int current_epoch = 0;
    int total_epochs = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    double learning_rate = 0.0;
    int64_t samples_processed = 0;
    int64_t time_elapsed_ms = 0;
    std::unordered_map<std::string, double> custom_metrics;
};

/**
 * Callback types for progress reporting
 */
using ProgressCallback = std::function<void(const std::string& job_id, double progress, const TrainingMetrics& metrics)>;
using CompletionCallback = std::function<void(const std::string& job_id, bool success, const std::string& error_msg)>;

/**
 * Job Executor - Executes ML training jobs using cyxwiz-backend
 *
 * Features:
 * - Asynchronous job execution in background threads
 * - Dataset loading from local files or URIs
 * - Progress reporting with metrics
 * - Job cancellation support
 * - Result saving
 */
class JobExecutor {
public:
    explicit JobExecutor(const std::string& node_id, cyxwiz::Device* device = nullptr);
    ~JobExecutor();

    // Execute a job asynchronously
    bool ExecuteJobAsync(const protocol::JobConfig& job_config);

    // Cancel a running job
    bool CancelJob(const std::string& job_id);

    // Check if a job is running
    bool IsJobRunning(const std::string& job_id) const;

    // Get active job count
    size_t GetActiveJobCount() const;

    // Get list of active job IDs
    std::vector<std::string> GetActiveJobIds() const;

    // Set progress callback
    void SetProgressCallback(ProgressCallback callback);

    // Set completion callback
    void SetCompletionCallback(CompletionCallback callback);

    // Set node client for progress reporting
    void SetNodeClient(NodeClient* client);

private:
    // Job execution state
    struct JobState {
        std::thread worker_thread;
        std::atomic<bool> should_cancel{false};
        std::atomic<bool> is_running{false};
        protocol::JobConfig config;
        TrainingMetrics current_metrics;
        std::chrono::steady_clock::time_point start_time;
    };

    // Execute job in worker thread (synchronous)
    void ExecuteJob(const std::string& job_id);

    // Dataset loading
    bool LoadDataset(const std::string& dataset_uri,
                    std::vector<cyxwiz::Tensor>& train_data,
                    std::vector<cyxwiz::Tensor>& train_labels);

    // Model building from definition
    std::unique_ptr<cyxwiz::Model> BuildModel(const std::string& model_definition);

    // Training loop
    bool RunTraining(const std::string& job_id, JobState* state);

    // Result saving
    bool SaveResults(const std::string& job_id, cyxwiz::Model* model, const TrainingMetrics& final_metrics);

    // Progress reporting
    void ReportProgress(const std::string& job_id, JobState* state);

    // Helper: Parse hyperparameters
    std::unordered_map<std::string, double> ParseHyperparameters(
        const google::protobuf::Map<std::string, std::string>& hyper_params);

    // Helper: Create optimizer from hyperparameters
    std::unique_ptr<cyxwiz::Optimizer> CreateOptimizer(
        const std::unordered_map<std::string, double>& hyperparameters);

    // Dataset loading helpers
    bool LoadMockDataset(std::vector<cyxwiz::Tensor>& train_data,
                        std::vector<cyxwiz::Tensor>& train_labels);

    bool LoadMNISTDataset(const std::string& path,
                         std::vector<cyxwiz::Tensor>& train_data,
                         std::vector<cyxwiz::Tensor>& train_labels);

    bool LoadCIFAR10Dataset(const std::string& path,
                           std::vector<cyxwiz::Tensor>& train_data,
                           std::vector<cyxwiz::Tensor>& train_labels);

    bool LoadCSVDataset(const std::string& path,
                       std::vector<cyxwiz::Tensor>& train_data,
                       std::vector<cyxwiz::Tensor>& train_labels);

    // Member variables
    std::string node_id_;
    cyxwiz::Device* device_;

    // Job management
    std::unordered_map<std::string, std::unique_ptr<JobState>> active_jobs_;
    mutable std::mutex jobs_mutex_;

    // Callbacks
    ProgressCallback progress_callback_;
    CompletionCallback completion_callback_;
    std::mutex callback_mutex_;

    // Node client for reporting
    NodeClient* node_client_ = nullptr;

    // Progress reporting interval (milliseconds)
    int progress_interval_ms_ = 1000;
};

} // namespace servernode
} // namespace cyxwiz
