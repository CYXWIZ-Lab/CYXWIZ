#pragma once

// RL Training Dashboard (approved board 3, TOFIX134 P1 step 1.7): episode
// and policy metrics a run reports (pycyxwiz.rl_update_metric), drawn with
// the shared PlotView in the theme colours. UI thread only, except
// SetRLTrainingState (a run's completion callback may call it from the
// script worker).

#include "../panel.h"
#include "../plot/plot_view.h"

#include <imgui.h>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class TrainingDashboardPanel : public Panel {
public:
    TrainingDashboardPanel();
    ~TrainingDashboardPanel() override = default;

    void Render() override;

    void RegisterCustomPlot(const std::string& name, const std::string& display_name);
    void UpdateCustomMetric(const std::string& name, float value);

    void SetTrainingState(bool is_training);
    void SetRLTrainingState(bool is_rl_training);
    void ResetRLMetrics();

private:
    struct Metric {
        std::string display_name;
        std::vector<double> history;  // one value per report
        bool dirty = true;            // the plot needs new data
        std::unique_ptr<plot::PlotView> view;
    };

    void RenderKpis();
    void RenderMetric(const std::string& name, float height);

    // Long runs keep every value; the plot draws a reduced copy.
    static constexpr size_t kMaxHistory = 200000;

    std::atomic<bool> training_{false};
    std::map<std::string, Metric> metrics_;
    std::vector<std::string> order_;  // registration order (series colours)
};

}  // namespace cyxwiz
