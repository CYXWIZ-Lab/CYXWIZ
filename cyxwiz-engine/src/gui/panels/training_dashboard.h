#pragma once

#include "../panel.h"
#include <vector>
#include <string>

namespace cyxwiz {

/**
 * Training Dashboard Panel
 * Displays real-time training metrics with plots and statistics
 */
class TrainingDashboardPanel : public Panel {
public:
    TrainingDashboardPanel();
    ~TrainingDashboardPanel() override = default;

    void Render() override;

    // Update methods (called by training system)
    void UpdateLoss(float loss);
    void UpdateAccuracy(float accuracy);
    void UpdateThroughput(float samples_per_sec);
    void UpdateLearningRate(float lr);

    // Training state
    void SetTrainingState(bool is_training);
    void ResetMetrics();

private:
    void RenderMetricsOverview();
    void RenderLossChart();
    void RenderAccuracyChart();
    void RenderThroughputChart();
    void RenderHyperparameters();
    void RenderTrainingControls();

    // Training state
    bool is_training_;
    float current_epoch_;
    float total_epochs_;
    float progress_;

    // Metrics history (circular buffers)
    static constexpr int MAX_HISTORY = 1000;
    std::vector<float> loss_history_;
    std::vector<float> accuracy_history_;
    std::vector<float> throughput_history_;

    // Current values
    float current_loss_;
    float current_accuracy_;
    float current_throughput_;
    float current_lr_;

    // Statistics
    float min_loss_;
    float max_loss_;
    float avg_loss_;
    float best_accuracy_;

    // UI state
    bool show_loss_chart_;
    bool show_accuracy_chart_;
    bool show_throughput_chart_;
    int chart_history_length_;
};

} // namespace cyxwiz
