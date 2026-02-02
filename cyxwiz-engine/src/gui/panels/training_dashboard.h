#pragma once

#include "../panel.h"
#include "../../plotting/plot_manager.h"
#include <imgui.h>
#include <vector>
#include <string>
#include <map>
#include <mutex>

namespace cyxwiz {

class TrainingDashboardPanel : public Panel {
public:
    TrainingDashboardPanel();
    ~TrainingDashboardPanel() override = default;

    void Render() override;

    void UpdateLoss(float loss);
    void UpdateAccuracy(float accuracy);
    void UpdateThroughput(float samples_per_sec);
    void UpdateLearningRate(float lr);

    void RegisterCustomPlot(const std::string& name, const std::string& display_name, ImVec4 color);
    void UpdateCustomMetric(const std::string& name, float value);

    void SetTrainingState(bool is_training);
    void SetRLTrainingState(bool is_rl_training);
    void ResetMetrics();
    void ResetRLMetrics();

private:
    void RenderMetricsOverview();
    void RenderLossChart();
    void RenderAccuracyChart();
    void RenderThroughputChart();
    void RenderHyperparameters();
    void RenderTrainingControls();
    void RenderCustomPlot(const std::string& name);
    void RenderRLMetricsTab();
    void RenderPolicyDiagnosticsTab();

    void InitializePlots();
    void InitializeRLPlots();

    bool is_training_;
    bool is_rl_training_ = false;
    float current_epoch_;
    float total_epochs_;
    float progress_;

    std::string loss_plot_id_;
    std::string accuracy_plot_id_;
    std::string throughput_plot_id_;

    static constexpr int MAX_HISTORY = 1000;
    std::vector<float> loss_history_;
    std::vector<float> accuracy_history_;
    std::vector<float> throughput_history_;

    float current_loss_;
    float current_accuracy_;
    float current_throughput_;
    float current_lr_;

    float min_loss_;
    float max_loss_;
    float avg_loss_;
    float best_accuracy_;

    bool show_loss_chart_;
    bool show_accuracy_chart_;
    bool show_throughput_chart_;
    int chart_history_length_;

    struct CustomMetric {
        std::string display_name;
        ImVec4 color;
        std::string plot_id;
        std::vector<float> history;
        float current_value = 0.0f;
        int update_count = 0;
    };
    std::map<std::string, CustomMetric> custom_metrics_;
    mutable std::mutex custom_metrics_mutex_;
    bool rl_plots_initialized_ = false;
};

} // namespace cyxwiz
