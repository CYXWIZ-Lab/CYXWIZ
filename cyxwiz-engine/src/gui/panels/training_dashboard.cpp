#include "training_dashboard.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

namespace {

void ClearRealtimeDataset(const std::string& plot_id) {
    auto& plot_mgr = plotting::PlotManager::GetInstance();
    if (plot_mgr.GetDataset(plot_id, "realtime") != nullptr) {
        plot_mgr.RemoveDataset(plot_id, "realtime");
    }
}

} // namespace

TrainingDashboardPanel::TrainingDashboardPanel()
    : Panel("RL Training Dashboard", true)
    , is_training_(false)
{
}

void TrainingDashboardPanel::InitializeRLPlots() {
    if (rl_plots_initialized_) return;

    RegisterCustomPlot("episode_reward", "Episode Reward", ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
    RegisterCustomPlot("episode_length", "Episode Length", ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
    RegisterCustomPlot("policy_loss", "Policy Loss", ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
    RegisterCustomPlot("value_loss", "Value Loss", ImVec4(0.2f, 0.2f, 0.8f, 1.0f));
    RegisterCustomPlot("explained_variance", "Explained Variance", ImVec4(0.4f, 0.4f, 1.0f, 1.0f));

    rl_plots_initialized_ = true;
    spdlog::info("Training Dashboard RL plots initialized");
}

void TrainingDashboardPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    InitializeRLPlots();

    RenderMetricsOverview();
    ImGui::Separator();

    RenderTrainingControls();
    ImGui::Separator();

    // This panel is reserved for RL metrics. Supervised training telemetry is
    // owned by TrainingPlotPanel, which is wired to graph/Python training.
    if (ImGui::BeginTabBar("MetricsTabs")) {
        if (ImGui::BeginTabItem("Episodes")) {
            RenderRLMetricsTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Policy Diagnostics")) {
            RenderPolicyDiagnosticsTab();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void TrainingDashboardPanel::RenderMetricsOverview() {
    ImGui::Text("RL Training Status: %s", is_training_ ? "RUNNING" : "STOPPED");
    ImGui::Spacing();

    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    ImGui::Columns(3, "rl_metrics_overview", false);

    auto it = custom_metrics_.find("episode_reward");
    ImGui::Text("Episode Reward");
    ImGui::Text("%.2f", it != custom_metrics_.end() ? it->second.current_value : 0.0f);
    ImGui::NextColumn();

    it = custom_metrics_.find("episode_length");
    ImGui::Text("Episode Length");
    ImGui::Text("%.0f", it != custom_metrics_.end() ? it->second.current_value : 0.0f);
    ImGui::NextColumn();

    it = custom_metrics_.find("policy_loss");
    ImGui::Text("Policy Loss");
    ImGui::Text("%.6f", it != custom_metrics_.end() ? it->second.current_value : 0.0f);
    ImGui::NextColumn();

    ImGui::Columns(1);
}

void TrainingDashboardPanel::RenderTrainingControls() {
    ImGui::Text("Controls:");
    ImGui::SameLine();

    ImGui::TextDisabled("%s", is_training_ ? "Active" : "Idle");
    ImGui::SameLine();

    if (ImGui::Button("Reset Metrics")) {
        ResetRLMetrics();
    }
}

void TrainingDashboardPanel::RenderCustomPlot(const std::string& name) {
    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    auto it = custom_metrics_.find(name);
    if (it == custom_metrics_.end()) {
        ImGui::TextDisabled("Metric '%s' not registered", name.c_str());
        return;
    }

    auto& metric = it->second;

    ImGui::Text("%s", metric.display_name.c_str());

    if (metric.history.empty()) {
        ImGui::TextDisabled("No data yet");
        return;
    }

    // Stats
    float min_val = *std::min_element(metric.history.begin(), metric.history.end());
    float max_val = *std::max_element(metric.history.begin(), metric.history.end());
    float avg_val = std::accumulate(metric.history.begin(), metric.history.end(), 0.0f) / metric.history.size();

    ImGui::Text("Current: %.4f  Min: %.4f  Max: %.4f  Avg: %.4f",
                metric.current_value, min_val, max_val, avg_val);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.RenderImPlot(metric.plot_id);

    ImGui::Separator();
}

void TrainingDashboardPanel::RenderRLMetricsTab() {
    if (!rl_plots_initialized_) {
        InitializeRLPlots();
    }

    if (!is_rl_training_) {
        ImGui::TextDisabled("No RL training active. Start RL training from CyxWiz Studio.");
        ImGui::Spacing();
    }

    RenderCustomPlot("episode_reward");
    RenderCustomPlot("episode_length");
}

void TrainingDashboardPanel::RenderPolicyDiagnosticsTab() {
    if (!rl_plots_initialized_) {
        InitializeRLPlots();
    }

    if (!is_rl_training_) {
        ImGui::TextDisabled("No RL training active.");
        ImGui::Spacing();
    }

    RenderCustomPlot("policy_loss");
    RenderCustomPlot("value_loss");
    RenderCustomPlot("explained_variance");
}

void TrainingDashboardPanel::RegisterCustomPlot(const std::string& name,
                                                 const std::string& display_name,
                                                 ImVec4 color) {
    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    if (custom_metrics_.count(name)) return;

    auto& plot_mgr = plotting::PlotManager::GetInstance();

    plotting::PlotManager::PlotConfig config;
    config.title = display_name;
    config.x_label = "Step";
    config.y_label = display_name;
    config.type = plotting::PlotManager::PlotType::Line;
    config.backend = plotting::PlotManager::BackendType::ImPlot;
    config.auto_fit = true;
    config.show_legend = true;
    config.show_grid = true;
    config.width = 600;
    config.height = 180;

    std::string plot_id = plot_mgr.CreatePlot(config);

    CustomMetric metric;
    metric.display_name = display_name;
    metric.color = color;
    metric.plot_id = plot_id;
    metric.history.reserve(MAX_HISTORY);

    custom_metrics_[name] = std::move(metric);

    spdlog::debug("Registered custom plot: {} ({})", name, display_name);
}

void TrainingDashboardPanel::UpdateCustomMetric(const std::string& name, float value) {
    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    auto it = custom_metrics_.find(name);
    if (it == custom_metrics_.end()) return;

    auto& metric = it->second;
    metric.current_value = value;
    metric.update_count++;
    metric.history.push_back(value);

    if (metric.history.size() > MAX_HISTORY) {
        metric.history.erase(metric.history.begin());
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.UpdateRealtimePlot(metric.plot_id,
                                static_cast<double>(metric.update_count),
                                static_cast<double>(value),
                                name);
}

void TrainingDashboardPanel::SetTrainingState(bool is_training) {
    is_training_ = is_training;
}

void TrainingDashboardPanel::SetRLTrainingState(bool is_rl_training) {
    is_rl_training_ = is_rl_training;
    if (is_rl_training) {
        is_training_ = true;
        if (!rl_plots_initialized_) {
            InitializeRLPlots();
        }
    }
}

void TrainingDashboardPanel::ResetRLMetrics() {
    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    for (auto& [name, metric] : custom_metrics_) {
        metric.history.clear();
        metric.current_value = 0.0f;
        metric.update_count = 0;
        ClearRealtimeDataset(metric.plot_id);
    }

    is_rl_training_ = false;
    is_training_ = false;
}

} // namespace cyxwiz
