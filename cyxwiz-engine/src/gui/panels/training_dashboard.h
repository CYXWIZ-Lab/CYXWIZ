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

    void RegisterCustomPlot(const std::string& name, const std::string& display_name, ImVec4 color);
    void UpdateCustomMetric(const std::string& name, float value);

    void SetTrainingState(bool is_training);
    void SetRLTrainingState(bool is_rl_training);
    void ResetRLMetrics();

private:
    void RenderMetricsOverview();
    void RenderTrainingControls();
    void RenderCustomPlot(const std::string& name);
    void RenderRLMetricsTab();
    void RenderPolicyDiagnosticsTab();

    void InitializeRLPlots();

    bool is_training_;
    bool is_rl_training_ = false;

    static constexpr int MAX_HISTORY = 1000;

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
