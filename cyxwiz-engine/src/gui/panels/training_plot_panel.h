#pragma once

#include "../panel.h"
#include "../../core/training_run_comparison_record.h"
#include "../../plotting/plot_manager.h"
#include <imgui.h>
#include <vector>
#include <string>
#include <mutex>

namespace cyxwiz {

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
                                       const std::string& memory_risk_level = "");
    void SetMaterializationComplete(const std::string& output_dataset,
                                    int operators_applied,
                                    const std::string& status = "completed");
    void SetTrainingComplete(float total_time_seconds,
                             const std::string& terminal_status = "completed",
                             const std::string& terminal_reason = "",
                             const std::string& checkpoint_used = "",
                             bool has_validation_metrics = false,
                             float checkpoint_val_loss = 0.0f,
                             float checkpoint_val_accuracy = 0.0f,
                             int checkpoint_epoch = 0);

    // Per-batch progress updates (thread-safe). Called from TrainingManager's
    // batch_cb so the dashboard shows live activity inside an epoch instead of
    // freezing at "Epoch 0/N" for several minutes until the first epoch finishes.
    // Also updates current_epoch_ so the epoch counter advances as soon as the
    // first batch of that epoch runs (not at epoch end).
    void SetBatchProgress(int current_epoch, int current_batch, int total_batches,
                          float running_loss);
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
        std::string stage;
        std::string message;
        std::string node_name;
        int node_id = -1;
        float progress = 0.0f;
        uint64_t estimated_memory_bytes = 0;
        std::string memory_risk_level;
        uint64_t processed_items = 0;
        uint64_t total_items = 0;
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
    int materialization_operators_applied_ = 0;

    // UI state
    bool show_loss_plot_ = true;
    bool show_accuracy_plot_ = true;
    bool show_custom_metrics_ = false;
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
    int total_epochs_ = 0;
    int current_batch_ = 0;
    int total_batches_ = 0;
    float current_batch_loss_ = 0.0f;
    float last_epoch_time_ = 0.0f;
    float avg_epoch_time_ = 0.0f;
    float samples_per_second_ = 0.0f;
    float total_training_time_ = 0.0f;
    std::string terminal_status_;
    std::string terminal_reason_;
    std::string checkpoint_used_;
    bool has_checkpoint_validation_metrics_ = false;
    float checkpoint_val_loss_ = 0.0f;
    float checkpoint_val_accuracy_ = 0.0f;
    int checkpoint_epoch_ = 0;
    std::vector<float> epoch_times_;  // For averaging
    bool last_render_visible_ = false;
    mutable size_t sampled_read_events_ = 0;

    // Thread safety
    mutable std::mutex data_mutex_;

    // Helper methods
    void RenderTrainingStatus();
    void RenderLossPlot();
    void RenderAccuracyPlot();
    void RenderCustomMetricsPlot();
    void RenderControls();
    void RenderCurveSummary();
    void RenderSequenceMetricsSummary();
    void RenderActiveTaskSummary();
    void RenderMaterializationSummary();
    void RenderTrainingWarningSummary();
    void RenderRunComparisonTable();
    void RenderStatistics();

    // Internal helpers
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
