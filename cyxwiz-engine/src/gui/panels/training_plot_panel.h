#pragma once

#include "../panel.h"
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

    // Training data updates (thread-safe)
    void AddLossPoint(int epoch, double train_loss, double val_loss = -1.0);
    void AddAccuracyPoint(int epoch, double train_acc, double val_acc = -1.0);
    void AddCustomMetric(const std::string& metric_name, int epoch, double value);

    // Control
    void Clear();
    void ResetPlots();
    void SetMaxPoints(size_t max_points);

    // Export
    void ExportToCSV(const std::string& filepath);
    void ExportPlotImage(const std::string& filepath);

    // Configuration
    void ShowLossPlot(bool show) { show_loss_plot_ = show; }
    void ShowAccuracyPlot(bool show) { show_accuracy_plot_ = show; }
    void SetAutoScale(bool auto_scale) { auto_scale_ = auto_scale; }

    // Training state updates (thread-safe) - call from TrainingManager
    void SetTrainingState(bool is_training, int current_epoch, int total_epochs,
                          float epoch_time_seconds, float samples_per_second);
    void SetTrainingComplete(float total_time_seconds);

    // Getters for live metrics (thread-safe)
    bool HasData() const { return !train_loss_.values.empty(); }
    bool IsTraining() const { return is_training_; }
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

    // UI state
    bool show_loss_plot_ = true;
    bool show_accuracy_plot_ = true;
    bool show_custom_metrics_ = false;
    bool auto_scale_ = true;
    size_t max_points_ = 1000;

    // Training state
    bool is_training_ = false;
    int current_epoch_ = 0;
    int total_epochs_ = 0;
    float last_epoch_time_ = 0.0f;
    float avg_epoch_time_ = 0.0f;
    float samples_per_second_ = 0.0f;
    float total_training_time_ = 0.0f;
    std::vector<float> epoch_times_;  // For averaging

    // Thread safety
    mutable std::mutex data_mutex_;

    // Helper methods
    void RenderTrainingStatus();
    void RenderLossPlot();
    void RenderAccuracyPlot();
    void RenderCustomMetricsPlot();
    void RenderControls();
    void RenderStatistics();

    // Internal helpers
    void TrimDataIfNeeded(MetricSeries& series);
    double CalculateMean(const std::vector<double>& values, size_t last_n = 10);
    double CalculateMin(const std::vector<double>& values);
    double CalculateMax(const std::vector<double>& values);
};

// Global accessor functions for Python integration
void set_training_plot_panel(cyxwiz::TrainingPlotPanel* panel);
cyxwiz::TrainingPlotPanel* get_training_plot_panel();

} // namespace cyxwiz
