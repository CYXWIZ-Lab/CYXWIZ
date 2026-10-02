#pragma once

#include "../panel.h"
#include "../../core/training_run_comparison_record.h"
#include "../../core/training_progress_estimate.h"
#include <chrono>
#include <functional>
#include "../../plotting/plot_manager.h"
#include <imgui.h>
#include <vector>
#include <string>
#include <mutex>
#include <utility>

namespace cyxwiz {

struct TrainingMetrics;

struct TrainingStatusSnapshot {
    bool has_data = false;
    bool is_training = false;
    bool is_preparing = false;
    bool preparation_failed = false;
    std::string status_message;
    std::string terminal_status;
    std::string terminal_reason;
    int current_epoch = 0;
    int last_executed_epoch = 0;
    int total_epochs = 0;
    int current_batch = 0;
    int total_batches = 0;
    double train_loss = -1.0;
    double val_loss = -1.0;
    double train_accuracy = -1.0;
    double val_accuracy = -1.0;
    float preparation_progress = 0.0f;
    float samples_per_second = 0.0f;
    float total_training_time = 0.0f;
    int checkpoint_epoch = 0;
    int checkpoint_step = 0;
    std::string checkpoint_used;
    std::string active_model_provenance;
    size_t metric_points = 0;
    std::string materialization_status;
    std::string materialization_message;
    std::vector<std::pair<std::string, double>> latest_custom_metrics;
};

/**
 * TrainingPlotPanel - Real-time visualization of training metrics
 * Displays loss, accuracy, and other metrics as training progresses
 */
class TrainingPlotPanel : public Panel {
public:
    TrainingPlotPanel();
    ~TrainingPlotPanel() override;

    void Render() override;

    // Training data updates (thread-safe).
    // epoch is a double so callers can pass fractional epochs from per-batch
    // callbacks (epoch - 1 + batch/total_batches), making the loss curve draw
    // smoothly during a long epoch instead of waiting for epoch boundaries.
    // Integer values still work — callers passing int are implicitly promoted.
    void AddLossPoint(double epoch, double train_loss, double val_loss = -1.0);
    void AddAccuracyPoint(double epoch, double train_acc, double val_acc = -1.0);
    void AddCustomMetric(const std::string& metric_name, int epoch, double value);

    // Control
    void Clear();
    void ResetPlots();
    void SetMaxPoints(size_t max_points);

    // Export
    void ExportToCSV(const std::string& filepath);
    void ExportPlotImage(const std::string& filepath);
    void ExportRunComparisonCSV(const std::string& filepath);

    // Configuration
    void ShowLossPlot(bool show) { show_loss_plot_ = show; }
    void ShowAccuracyPlot(bool show) { show_accuracy_plot_ = show; }
    void ShowCustomMetrics(bool show) { show_custom_metrics_ = show; }
    void SetAutoScale(bool auto_scale) { auto_scale_ = auto_scale; }

    // Training state updates (thread-safe) - call from TrainingManager
    void SetTrainingState(bool is_training, int current_epoch, int total_epochs,
                          float epoch_time_seconds, float samples_per_second);
    void SetPreparationState(bool is_preparing,
                             const std::string& status_message = "",
                             float progress = 0.0f);
    void SetPreparationFailed(const std::string& error_message);
    void RecordMaterializationProgress(const std::string& stage,
                                       const std::string& message,
                                       float progress,
                                       uint64_t estimated_memory_bytes = 0,
                                       uint64_t processed_items = 0,
                                       uint64_t total_items = 0,
                                       int node_id = -1,
                                       const std::string& node_name = "",
                                       const std::string& memory_risk_level = "",
                                       const std::string& status = "running",
                                       uint64_t available_memory_bytes = 0,
                                       uint64_t safe_memory_budget_bytes = 0,
                                       bool process_memory_detected = false,
                                       uint64_t process_resident_memory_bytes = 0,
                                       uint64_t process_private_memory_bytes = 0,
                                       uint64_t process_resident_growth_bytes = 0,
                                       const std::string& process_private_memory_name = "",
                                       const std::string& process_memory_source = "",
                                       const std::string& cache_key = "",
                                       const std::string& cache_artifact_path = "",
                                       const std::string& cache_manifest_path = "",
                                       int64_t cache_row_count = 0,
                                       int64_t cache_column_count = 0);
    void SetMaterializationComplete(const std::string& output_dataset,
                                    int operators_applied,
                                    const std::string& status = "completed");
    // Prepared-data cache on disk (Data preparation card).
    void SetMaterializationCacheInfo(const std::string& cache_directory,
                                     int entries,
                                     uint64_t total_bytes,
                                     uint64_t size_limit_bytes,
                                     int pruned_entries = 0,
                                     uint64_t pruned_bytes = 0,
                                     const std::string& rebuild_reason = "");
    // Something the user must know, e.g. preprocessing nodes not applied.
    void SetMaterializationNotice(const std::string& notice);
    void SetMaterializationClearResult(int removed_entries,
                                       uint64_t freed_bytes,
                                       const std::string& error);
    // Actions: "rebuild", "cancel_rebuild", "clear", "refresh".
    void SetMaterializationActionCallback(
        std::function<void(const std::string&)> callback) {
        materialization_action_callback_ = std::move(callback);
    }
    void SetTrainingComplete(float total_time_seconds,
                             const std::string& terminal_status = "completed",
                             const std::string& terminal_reason = "",
                             const std::string& checkpoint_used = "",
                             bool has_validation_metrics = false,
                             float checkpoint_val_loss = 0.0f,
                             float checkpoint_val_accuracy = 0.0f,
                             int checkpoint_epoch = 0);
    void SetTrainingComplete(float total_time_seconds,
                             const TrainingMetrics& metrics);
    void SetActiveCheckpointLoaded(const std::string& checkpoint_path,
                                   int checkpoint_epoch,
                                   float validation_loss,
                                   float validation_accuracy,
                                   bool has_validation_metrics);

    // Per-batch progress updates (thread-safe). Called from TrainingManager's
    // batch_cb so the dashboard shows live activity inside an epoch instead of
    // freezing at "Epoch 0/N" for several minutes until the first epoch finishes.
    // Also updates current_epoch_ so the epoch counter advances as soon as the
    // first batch of that epoch runs (not at epoch end).
    void SetBatchProgress(int current_epoch, int current_batch, int total_batches,
                          float running_loss);
    void SetMetricReportingCadence(int batch_interval);
    // Samples per batch and batches per optimizer update, so the batch counter
    // can name its unit ("8 samples each") instead of reading as samples.
    void SetBatchComposition(int samples_per_batch, int batches_per_update);
    void AddRunComparisonRecord(const TrainingRunComparisonRecord& record);
    void ClearRunComparisonRecords();

    // Getters for live metrics (thread-safe)
    bool HasData() const;
    bool IsTraining() const;
    int GetCurrentEpoch() const;
    double GetCurrentTrainLoss() const;
    double GetCurrentValLoss() const;
    double GetCurrentTrainAccuracy() const;
    double GetCurrentValAccuracy() const;
    size_t GetDataPointCount() const;
    TrainingStatusSnapshot GetStatusSnapshot() const;

private:
    struct MetricSeries {
        std::vector<double> epochs;
        std::vector<double> values;
        std::string name;
        ImVec4 color;
    };

    struct ValueRange {
        double min = 0.0;
        double max = 0.0;
        bool has_values = false;
    };

    struct MaterializationProgress {
        // When the step started and was last updated (step durations).
        std::chrono::steady_clock::time_point started_at{};
        std::chrono::steady_clock::time_point updated_at{};
        std::string stage;
        std::string message;
        std::string status = "running";
        std::string node_name;
        int node_id = -1;
        float progress = 0.0f;
        uint64_t estimated_memory_bytes = 0;
        uint64_t available_memory_bytes = 0;
        uint64_t safe_memory_budget_bytes = 0;
        std::string memory_risk_level;
        bool process_memory_detected = false;
        uint64_t process_resident_memory_bytes = 0;
        uint64_t process_private_memory_bytes = 0;
        uint64_t process_resident_growth_bytes = 0;
        std::string process_private_memory_name;
        std::string process_memory_source;
        uint64_t processed_items = 0;
        uint64_t total_items = 0;
        std::string cache_key;
        std::string cache_artifact_path;
        std::string cache_manifest_path;
        int64_t cache_row_count = 0;
        int64_t cache_column_count = 0;
    };

    // Plot IDs
    std::string loss_plot_id_;
    std::string accuracy_plot_id_;
    std::string custom_plot_id_;

    // Data storage
    MetricSeries train_loss_;
    MetricSeries val_loss_;
    MetricSeries train_accuracy_;
    MetricSeries val_accuracy_;
    std::vector<MetricSeries> custom_metrics_;
    std::vector<TrainingRunComparisonRecord> run_comparison_records_;
    std::vector<MaterializationProgress> materialization_events_;
    std::string materialization_output_dataset_;
    std::string materialization_status_;
    std::string materialization_cache_key_;
    std::string materialization_cache_artifact_path_;
    std::string materialization_cache_manifest_path_;
    int64_t materialization_cache_row_count_ = 0;
    int64_t materialization_cache_column_count_ = 0;
    int materialization_operators_applied_ = 0;
    std::string materialization_notice_;
    std::string materialization_rebuild_reason_;
    std::string materialization_cache_directory_;
    int materialization_cache_entries_ = -1;  // -1: not measured yet
    uint64_t materialization_cache_bytes_ = 0;
    uint64_t materialization_cache_limit_bytes_ = 0;
    int materialization_pruned_entries_ = 0;
    uint64_t materialization_pruned_bytes_ = 0;
    std::string materialization_clear_message_;
    bool materialization_rebuild_pending_ = false;
    bool materialization_details_open_ = false;
    bool materialization_cache_refresh_requested_ = false;
    bool materialization_clear_confirm_ = false;

public:
    // Tools > Monitoring > Clear Cache: shows the panel and asks the same
    // question as the dashboard card (TOFIX129 G3).
    void RequestClearPreparedDataCache() {
        SetVisible(true);
        materialization_clear_confirm_ = true;
    }

private:
    std::function<void(const std::string&)> materialization_action_callback_;

    // UI state
    bool show_loss_plot_ = true;
    // Charts opened in their own dockable windows.
    bool loss_window_open_ = false;
    bool accuracy_window_open_ = false;
    bool custom_window_open_ = false;
    bool show_accuracy_plot_ = true;
    bool show_custom_metrics_ = false;
    bool log_loss_scale_ = false;
    bool auto_scale_ = true;
    bool follow_current_epoch_ = false;
    bool show_smoothed_curves_ = false;
    int smoothing_window_ = 5;
    int visible_epoch_window_ = 30;
    size_t max_points_ = 100000;

    // Training state
    bool is_training_ = false;
    bool is_preparing_ = false;
    bool preparation_failed_ = false;
    std::string preparation_status_message_;
    std::string preparation_error_message_;
    float preparation_progress_ = 0.0f;
    int current_epoch_ = 0;
    int last_executed_epoch_ = 0;
    int total_epochs_ = 0;
    int current_batch_ = 0;
    int total_batches_ = 0;
    float current_batch_loss_ = 0.0f;
    int metric_reporting_interval_ = 10;
    int samples_per_batch_ = 0;
    int batches_per_update_ = 1;
    float last_epoch_time_ = 0.0f;
    float avg_epoch_time_ = 0.0f;
    float samples_per_second_ = 0.0f;
    // Remaining-time estimate from batch progress across all epochs (rate over a
    // recent window, plus measured epoch-boundary overhead).
    TrainingEtaEstimator eta_estimator_;
    std::chrono::steady_clock::time_point eta_clock_start_ = std::chrono::steady_clock::now();
    float total_training_time_ = 0.0f;
    std::string terminal_status_;
    std::string terminal_reason_;
    std::string checkpoint_used_;
    bool has_checkpoint_validation_metrics_ = false;
    float checkpoint_val_loss_ = 0.0f;
    float checkpoint_val_accuracy_ = 0.0f;
    int checkpoint_epoch_ = 0;
    int checkpoint_step_ = 0;
    std::string active_model_provenance_;
    bool active_checkpoint_loaded_ = false;
    std::vector<float> epoch_times_;  // For averaging
    bool last_render_visible_ = false;
    mutable size_t sampled_read_events_ = 0;

    // Thread safety
    mutable std::mutex data_mutex_;

    // Drawn copies of the series (TOFIX134 P0 item 8): reduced to a few
    // thousand points with their smoothed curves, rebuilt only when the data
    // or the smoothing changes. Drawing 100k points and smoothing them every
    // frame held data_mutex_ (and so the training thread) for the frame.
    struct DrawnSeries {
        MetricSeries line;
        std::vector<double> smooth_x;
        std::vector<double> smooth_y;
    };
    struct DrawnCache {
        uint64_t version = ~0ull;
        int smoothing = 0;
        bool smoothed = false;
        DrawnSeries train_loss, val_loss, train_accuracy, val_accuracy;
        std::vector<MetricSeries> custom;
    };
    uint64_t data_version_ = 0;
    DrawnCache drawn_;
    // Call with data_mutex_ held.
    const DrawnCache& Drawn();

    // Helper methods
    void RenderTrainingStatus();
    void RenderLossPlot(float plot_height);
    void RenderAccuracyPlot(float plot_height);
    void RenderCustomMetricsPlot(float plot_height);
    void RenderKpiCards();
    void DrawLossPlot(const ImVec2& size, bool fit);
    void DrawAccuracyPlot(const ImVec2& size, bool fit);
    void DrawCustomMetricsPlot(const ImVec2& size, bool fit);
    void RenderChartWindows();
    void RenderEmptyState();
    void RenderControls();
    void RenderCurveSummary();
    void RenderSequenceMetricsSummary();
    void RenderActiveTaskSummary();
    void RenderMaterializationSummary();
    void RenderTrainingWarningSummary();
    void RenderRunComparisonTable();
    void RenderStatistics();

    // Internal helpers
    // Render() already owns data_mutex_; these helpers must only be called
    // while that lock is held. Public entry points acquire the lock first.
    void ClearLocked();
    void ExportToCSVLocked(const std::string& filepath) const;
    std::pair<double, double> CalculateEpochWindow(const MetricSeries& series) const;
    ValueRange CalculateVisibleRange(const MetricSeries& primary,
                                     const MetricSeries& secondary,
                                     double min_epoch,
                                     double max_epoch) const;
    std::vector<double> CalculateMovingAverage(
        const std::vector<double>& values,
        int window) const;
    void TrimDataIfNeeded(MetricSeries& series);
    void RecordPanelEvent(const std::string& action,
                          const std::string& detail = "") const;
    double CalculateMean(const std::vector<double>& values, size_t last_n = 10) const;
    double CalculateMin(const std::vector<double>& values) const;
    double CalculateMax(const std::vector<double>& values) const;
};

// Global accessor functions for Python integration
void set_training_plot_panel(cyxwiz::TrainingPlotPanel* panel);
cyxwiz::TrainingPlotPanel* get_training_plot_panel();

} // namespace cyxwiz
