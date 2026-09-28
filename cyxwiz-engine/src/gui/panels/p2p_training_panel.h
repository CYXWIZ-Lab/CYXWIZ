#pragma once

#include "../../core/remote_job_presentation.h"
#include <functional>
#include <set>
#include "../panel.h"
#include "../../network/p2p_client.h"
#include "../../plotting/plot_manager.h"
#include <imgui.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <deque>
#include <thread>
#include <atomic>

namespace cyxwiz {

/**
 * P2PTrainingPanel - Real-time P2P training progress monitoring and control
 *
 * Features:
 * - Live training metrics visualization (loss, accuracy, GPU usage)
 * - Interactive training control (pause, resume, stop, checkpoint)
 * - Connection status display
 * - Progress bar and ETA
 * - Training logs display
 * - Checkpoint history
 */
class P2PTrainingPanel : public Panel {
public:
    P2PTrainingPanel();
    ~P2PTrainingPanel() override;

    void Render() override;

    // P2P Client management
    void SetP2PClient(std::shared_ptr<network::P2PClient> client);
    std::shared_ptr<network::P2PClient> GetP2PClient() const { return p2p_client_; }

    // Job control
    void StartMonitoring(const std::string& job_id, const std::string& node_address);
    void StopMonitoring();
    void CancelDownloadAndWait();  // Cancel any pending download and wait for thread
    bool IsMonitoring() const { return is_monitoring_; }
    bool IsDownloading() const { return downloading_model_; }

    // Training config (stored for .cyxmodel export)
    void SetTrainingConfig(const std::string& graph_json, int epochs, int batch_size, float learning_rate = 0.001f);
    const std::string& GetGraphJson() const { return graph_json_; }

    // Get current training state
    const std::string& GetJobId() const { return job_id_; }
    const std::string& GetNodeAddress() const { return node_address_; }
    float GetProgress() const { return current_progress_; }
    bool IsTrainingComplete() const { return training_complete_; }
    bool HasError() const { return has_error_; }

    // Clear data
    void Clear();

    // Public methods to receive progress updates from external sources
    void OnProgressUpdate(const network::TrainingProgress& progress) { OnProgress(progress); }
    void OnTrainingComplete(const network::TrainingComplete& complete) { OnComplete(complete); }
    void OnErrorUpdate(const std::string& error_message, bool is_fatal) { OnError(error_message, is_fatal); }

    // Recovery actions (TOFIX118 P4 GUI), wired by the Server Connection
    // window: resume sends this job again (the node continues from its
    // checkpoint); start over sends it as a new job.
    void SetRecoveryActions(std::function<void(const std::string& job_id)> resume,
                            std::function<void()> start_over) {
        resume_action_ = std::move(resume);
        start_over_action_ = std::move(start_over);
    }

private:
    // Rendering sub-components
    void RenderConnectionStatus();
    // Failure category and Resume from checkpoint cards
    // (p2p_training_panel_recovery.cpp, TOFIX118 P4 GUI).
    void RenderRecoveryCards();
    void OnFailure(const network::TrainingFailureReport& report);
    void RenderTrainingControls();
    void RenderProgressBar();
    void RenderMetricsPlots();
    void RenderLogs();
    void RenderCheckpoints();
    void RenderStopConfirmPopup();
    void RenderModelExport();

    // P2P callbacks (called from P2P client thread)
    void OnProgress(const network::TrainingProgress& progress);
    void OnCheckpoint(const network::CheckpointInfo& checkpoint);
    void OnComplete(const network::TrainingComplete& complete);
    void OnError(const std::string& error_message, bool is_fatal);
    void OnLog(const std::string& source, const std::string& message);

    // Helper methods
    void UpdateETA();
    void AddLogEntry(const std::string& level, const std::string& message);
    std::string FormatDuration(uint64_t seconds);

    // P2P client
    std::shared_ptr<network::P2PClient> p2p_client_;

    // Job state
    std::string job_id_;
    std::string node_address_;
    bool is_monitoring_;
    bool training_complete_;
    bool has_error_;
    std::string error_message_;

    // Training progress
    struct MetricHistory {
        std::vector<float> epochs;
        std::vector<float> values;
        size_t max_points = 500;

        void AddPoint(float epoch, float value) {
            epochs.push_back(epoch);
            values.push_back(value);
            if (epochs.size() > max_points) {
                epochs.erase(epochs.begin());
                values.erase(values.begin());
            }
        }

        void Clear() {
            epochs.clear();
            values.clear();
        }
    };

    network::TrainingProgress latest_progress_;
    float current_progress_;
    uint32_t current_epoch_;
    uint32_t total_epochs_;
    uint32_t current_batch_;
    uint32_t total_batches_;

    // Metrics history for plotting
    MetricHistory loss_history_;
    MetricHistory accuracy_history_;
    MetricHistory gpu_usage_history_;
    MetricHistory memory_usage_history_;

    // ETA calculation
    std::chrono::steady_clock::time_point training_start_time_;
    uint64_t estimated_time_remaining_;  // seconds

    // Checkpoints
    struct CheckpointEntry {
        uint32_t epoch;
        std::string checkpoint_hash;
        std::string storage_uri;
        uint64_t size_bytes;
        std::chrono::system_clock::time_point timestamp;
        int64_t next_batch = -1;       // resume point inside the epoch; -1 = epoch end
        uint32_t total_batches = 0;    // of that epoch, from the latest progress
        double loss = -1.0;            // loss when it arrived
    };
    std::vector<CheckpointEntry> checkpoint_history_;
    std::string checkpoints_job_id_;   // checkpoints stay listed when this job is sent again

    // The node's failure category for the current job (TOFIX118 P4a).
    bool has_failure_ = false;
    cyxwiz::RemoteFailureCard failure_card_;
    bool stopped_by_user_ = false;
    std::set<size_t> open_checkpoint_rows_;
    std::function<void(const std::string&)> resume_action_;
    std::function<void()> start_over_action_;
    static constexpr size_t kMaxCheckpointHistory = 100;

    // Logs
    struct LogEntry {
        std::string timestamp;
        std::string level;  // INFO, WARN, ERROR
        std::string source;
        std::string message;
    };
    std::deque<LogEntry> log_entries_;
    size_t max_log_entries_ = 200;

    // UI State
    bool show_logs_ = true;
    bool show_checkpoints_ = false;
    bool auto_scroll_logs_ = true;
    int selected_metric_plot_ = 0;  // 0=Loss, 1=Accuracy, 2=GPU, 3=Memory
    bool show_stop_confirm_popup_ = false;

    // Thread safety
    mutable std::mutex data_mutex_;

    // Control button state
    bool pause_button_enabled_ = false;
    bool resume_button_enabled_ = false;
    bool stop_button_enabled_ = true;
    bool checkpoint_button_enabled_ = true;

    // Model download state
    std::string model_weights_location_;
    bool model_available_ = false;
    bool downloading_model_ = false;
    std::atomic<bool> cancel_download_{false};  // Signal download thread to stop
    std::thread download_thread_;               // Track download thread for cleanup
    float download_progress_ = 0.0f;
    std::string download_error_;
    char download_path_[512] = "";
    int export_format_ = 0;  // 0 = raw weights, 1 = .cyxmodel

    // Node capabilities
    network::NodeCapabilities node_capabilities_;

    // Training configuration (for .cyxmodel export)
    std::string graph_json_;
    int training_epochs_ = 0;
    int training_batch_size_ = 32;
    float training_learning_rate_ = 0.001f;

    // Helper for model download/export
    void DownloadModel(const std::string& output_path);
    void ExportAsCyxModel(const std::string& weights_path, const std::string& output_path);
};

} // namespace cyxwiz
