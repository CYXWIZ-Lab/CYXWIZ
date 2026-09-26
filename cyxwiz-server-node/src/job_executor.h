#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <optional>
#include <queue>

// Protocol includes
#include "job.pb.h"

// Backend includes
#include <cyxwiz/cyxwiz.h>

// Core includes
#include "core/device_pool.h"
#include "core/graph_training_job_timing.h"
#include "core/training_failure.h"

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

    // Pause a running job
    bool PauseJob(const std::string& job_id);

    // Resume a paused job
    bool ResumeJob(const std::string& job_id);

    // Stop a running job (same as cancel, for P2P API consistency)
    bool StopJob(const std::string& job_id);

    // Get current model weights (for checkpointing)
    std::vector<char> GetCurrentWeights(const std::string& job_id);

    // Load weights into a job's model
    bool LoadWeights(const std::string& job_id, const std::vector<char>& weights);

    // Check if a job is running
    bool IsJobRunning(const std::string& job_id) const;

    // Check if a job is paused
    bool IsJobPaused(const std::string& job_id) const;

    // Get active job count
    size_t GetActiveJobCount() const;

    // Get list of active job IDs
    std::vector<std::string> GetActiveJobIds() const;

    // Set progress callback
    void SetProgressCallback(ProgressCallback callback);

    // Set completion callback
    void SetCompletionCallback(CompletionCallback callback);

    // The job's prepare/train split once training ended; readable from the
    // completion callback (TOFIX118 P3).
    std::optional<cyxwiz::GraphTrainingJobTiming> GetJobRunTiming(const std::string& job_id);
    // Why the job's training failed (None when it did not); readable from the
    // completion callback (TOFIX118 P4a).
    cyxwiz::TrainingFailureKind GetJobFailure(const std::string& job_id);

    // Set node client for progress reporting
    void SetNodeClient(NodeClient* client);


private:
    // Job execution state
    struct JobState {
        std::thread worker_thread;
        std::atomic<bool> should_cancel{false};
        std::atomic<bool> is_running{false};
        std::atomic<bool> is_paused{false};  // Pause flag for P2P control
        protocol::JobConfig config;
        TrainingMetrics current_metrics;
        std::chrono::steady_clock::time_point start_time;
        int assigned_device_id = -1;  // Device from pool

        // Model storage for weights extraction
        std::unique_ptr<cyxwiz::SequentialModel> model;
        std::mutex model_mutex;
        // The shared runner's time split and failure category (guarded by
        // model_mutex).
        std::optional<cyxwiz::GraphTrainingJobTiming> run_timing;
        cyxwiz::TrainingFailureKind failure = cyxwiz::TrainingFailureKind::None;

        // Pause/resume synchronization
        std::condition_variable pause_cv;
        std::mutex pause_mutex;
    };

    // Execute job in worker thread (synchronous)
    void ExecuteJob(const std::string& job_id);


    // Training loop
    bool RunTraining(const std::string& job_id, JobState* state);

    // Result saving
    bool SaveResults(const std::string& job_id, cyxwiz::Model* model, const TrainingMetrics& final_metrics);

    // Progress reporting
    void ReportProgress(const std::string& job_id, JobState* state);

    // Process pending jobs when a device becomes available
    void ProcessPendingJobs();

    // Member variables
    std::string node_id_;
    cyxwiz::Device* device_;  // Legacy single device (for backward compatibility)

    // Device pool for multi-GPU management
    std::unique_ptr<core::DevicePool> device_pool_;
    bool use_device_pool_ = false;

    // Job management
    std::unordered_map<std::string, std::unique_ptr<JobState>> active_jobs_;
    mutable std::mutex jobs_mutex_;

    // Pending jobs queue (when all devices are busy)
    std::queue<protocol::JobConfig> pending_jobs_;
    mutable std::mutex pending_mutex_;

    // Callbacks
    ProgressCallback progress_callback_;
    CompletionCallback completion_callback_;
    std::mutex callback_mutex_;

    // Node client for reporting
    NodeClient* node_client_ = nullptr;

    // Progress reporting interval (milliseconds)
    int progress_interval_ms_ = 1000;

public:
    // Device pool access
    core::DevicePool* GetDevicePool() { return device_pool_.get(); }
    const core::DevicePool* GetDevicePool() const { return device_pool_.get(); }
    bool IsUsingDevicePool() const { return use_device_pool_; }

    // Get pending job count
    size_t GetPendingJobCount() const {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        return pending_jobs_.size();
    }
};

} // namespace servernode
} // namespace cyxwiz
