#include "training_dashboard.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz {

TrainingDashboardPanel::TrainingDashboardPanel()
    : Panel("Training Dashboard", true)
    , is_training_(false)
    , current_epoch_(0.0f)
    , total_epochs_(100.0f)
    , progress_(0.0f)
    , current_loss_(0.0f)
    , current_accuracy_(0.0f)
    , current_throughput_(0.0f)
    , current_lr_(0.001f)
    , min_loss_(0.0f)
    , max_loss_(0.0f)
    , avg_loss_(0.0f)
    , best_accuracy_(0.0f)
    , show_loss_chart_(true)
    , show_accuracy_chart_(true)
    , show_throughput_chart_(true)
    , chart_history_length_(100)
{
    loss_history_.reserve(MAX_HISTORY);
    accuracy_history_.reserve(MAX_HISTORY);
    throughput_history_.reserve(MAX_HISTORY);

    // Initialize plots using PlotManager
    InitializePlots();

    // Add some sample data for visualization
    for (int i = 0; i < 50; i++) {
        float t = i / 50.0f;
        float loss = 2.0f * std::exp(-t * 2.0f) + 0.1f;
        float accuracy = 1.0f - std::exp(-t * 3.0f);
        float throughput = 1000.0f + 200.0f * std::sin(t * 6.28f);

        loss_history_.push_back(loss);
        accuracy_history_.push_back(accuracy);
        throughput_history_.push_back(throughput);

        // Update plots
        auto& plot_mgr = plotting::PlotManager::GetInstance();
        plot_mgr.UpdateRealtimePlot(loss_plot_id_, static_cast<double>(i), static_cast<double>(loss), "loss");
        plot_mgr.UpdateRealtimePlot(accuracy_plot_id_, static_cast<double>(i), static_cast<double>(accuracy), "accuracy");
        plot_mgr.UpdateRealtimePlot(throughput_plot_id_, static_cast<double>(i), static_cast<double>(throughput), "throughput");
    }
}

void TrainingDashboardPanel::InitializePlots() {
    auto& plot_mgr = plotting::PlotManager::GetInstance();

    // Create loss plot
    plotting::PlotManager::PlotConfig loss_config;
    loss_config.title = "Training Loss";
    loss_config.x_label = "Epoch";
    loss_config.y_label = "Loss";
    loss_config.type = plotting::PlotManager::PlotType::Line;
    loss_config.backend = plotting::PlotManager::BackendType::ImPlot;
    loss_config.auto_fit = true;
    loss_config.show_legend = true;
    loss_config.show_grid = true;
    loss_config.width = 600;
    loss_config.height = 200;
    loss_plot_id_ = plot_mgr.CreatePlot(loss_config);

    // Create accuracy plot
    plotting::PlotManager::PlotConfig acc_config;
    acc_config.title = "Training Accuracy";
    acc_config.x_label = "Epoch";
    acc_config.y_label = "Accuracy";
    acc_config.type = plotting::PlotManager::PlotType::Line;
    acc_config.backend = plotting::PlotManager::BackendType::ImPlot;
    acc_config.auto_fit = true;
    acc_config.show_legend = true;
    acc_config.show_grid = true;
    acc_config.width = 600;
    acc_config.height = 200;
    accuracy_plot_id_ = plot_mgr.CreatePlot(acc_config);

    // Create throughput plot
    plotting::PlotManager::PlotConfig throughput_config;
    throughput_config.title = "Training Throughput";
    throughput_config.x_label = "Epoch";
    throughput_config.y_label = "Samples/sec";
    throughput_config.type = plotting::PlotManager::PlotType::Line;
    throughput_config.backend = plotting::PlotManager::BackendType::ImPlot;
    throughput_config.auto_fit = true;
    throughput_config.show_legend = true;
    throughput_config.show_grid = true;
    throughput_config.width = 600;
    throughput_config.height = 200;
    throughput_plot_id_ = plot_mgr.CreatePlot(throughput_config);

    spdlog::info("Training Dashboard plots initialized");
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

    RenderMetricsOverview();
    ImGui::Separator();

    RenderTrainingControls();
    ImGui::Separator();

    // Tabbed layout for different training modes
    if (ImGui::BeginTabBar("MetricsTabs")) {
        if (ImGui::BeginTabItem("Supervised")) {
            // Chart selection
            ImGui::Checkbox("Loss", &show_loss_chart_);
            ImGui::SameLine();
            ImGui::Checkbox("Accuracy", &show_accuracy_chart_);
            ImGui::SameLine();
            ImGui::Checkbox("Throughput", &show_throughput_chart_);
            ImGui::SameLine();
            ImGui::SliderInt("History", &chart_history_length_, 10, 500);

            ImGui::Separator();

            if (show_loss_chart_) RenderLossChart();
            if (show_accuracy_chart_) RenderAccuracyChart();
            if (show_throughput_chart_) RenderThroughputChart();

            RenderHyperparameters();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Reinforcement Learning")) {
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
    if (is_rl_training_) {
        ImGui::Text("RL Training Status: %s", is_training_ ? "RUNNING" : "STOPPED");
    } else {
        ImGui::Text("Training Status: %s", is_training_ ? "RUNNING" : "STOPPED");
    }
    ImGui::SameLine(250);
    ImGui::Text("Epoch: %.0f / %.0f", current_epoch_, total_epochs_);

    // Progress bar
    ImGui::ProgressBar(progress_, ImVec2(-1.0f, 0.0f));

    // Metrics in columns
    if (is_rl_training_) {
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
    } else {
        ImGui::Columns(4, "metrics", false);

        ImGui::Text("Loss");
        ImGui::Text("%.6f", current_loss_);
        ImGui::NextColumn();

        ImGui::Text("Accuracy");
        ImGui::Text("%.2f%%", current_accuracy_ * 100.0f);
        ImGui::NextColumn();

        ImGui::Text("Throughput");
        ImGui::Text("%.0f samples/s", current_throughput_);
        ImGui::NextColumn();

        ImGui::Text("Learning Rate");
        ImGui::Text("%.6f", current_lr_);
        ImGui::NextColumn();

        ImGui::Columns(1);
    }
}

void TrainingDashboardPanel::RenderLossChart() {
    if (loss_history_.empty()) return;

    ImGui::Text("Loss Over Time");

    if (!loss_history_.empty()) {
        min_loss_ = *std::min_element(loss_history_.begin(), loss_history_.end());
        max_loss_ = *std::max_element(loss_history_.begin(), loss_history_.end());
        avg_loss_ = std::accumulate(loss_history_.begin(), loss_history_.end(), 0.0f) / loss_history_.size();
    }

    ImGui::Text("Min: %.6f  Max: %.6f  Avg: %.6f", min_loss_, max_loss_, avg_loss_);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.RenderImPlot(loss_plot_id_);

    ImGui::Separator();
}

void TrainingDashboardPanel::RenderAccuracyChart() {
    if (accuracy_history_.empty()) return;

    ImGui::Text("Accuracy Over Time");

    if (!accuracy_history_.empty()) {
        best_accuracy_ = *std::max_element(accuracy_history_.begin(), accuracy_history_.end());
    }

    ImGui::Text("Current: %.2f%%  Best: %.2f%%", current_accuracy_ * 100.0f, best_accuracy_ * 100.0f);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.RenderImPlot(accuracy_plot_id_);

    ImGui::Separator();
}

void TrainingDashboardPanel::RenderThroughputChart() {
    if (throughput_history_.empty()) return;

    ImGui::Text("Training Throughput");

    float avg_throughput = 0.0f;
    if (!throughput_history_.empty()) {
        avg_throughput = std::accumulate(throughput_history_.begin(), throughput_history_.end(), 0.0f) / throughput_history_.size();
    }

    ImGui::Text("Current: %.0f samples/s  Average: %.0f samples/s", current_throughput_, avg_throughput);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.RenderImPlot(throughput_plot_id_);

    ImGui::Separator();
}

void TrainingDashboardPanel::RenderHyperparameters() {
    if (ImGui::CollapsingHeader("Hyperparameters")) {
        ImGui::Columns(2, "hyperparams", false);

        ImGui::Text("Batch Size");
        ImGui::NextColumn();
        ImGui::Text("32");
        ImGui::NextColumn();

        ImGui::Text("Optimizer");
        ImGui::NextColumn();
        ImGui::Text("Adam");
        ImGui::NextColumn();

        ImGui::Text("Learning Rate");
        ImGui::NextColumn();
        ImGui::Text("%.6f", current_lr_);
        ImGui::NextColumn();

        ImGui::Text("Weight Decay");
        ImGui::NextColumn();
        ImGui::Text("0.0001");
        ImGui::NextColumn();

        ImGui::Text("Momentum");
        ImGui::NextColumn();
        ImGui::Text("0.9");
        ImGui::NextColumn();

        ImGui::Columns(1);
    }
}

void TrainingDashboardPanel::RenderTrainingControls() {
    ImGui::Text("Controls:");
    ImGui::SameLine();

    if (is_training_) {
        if (ImGui::Button("Pause")) {
            is_training_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            is_training_ = false;
            ResetMetrics();
        }
    } else {
        if (ImGui::Button("Start")) {
            is_training_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Resume")) {
            is_training_ = true;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        if (is_rl_training_) {
            ResetRLMetrics();
        } else {
            ResetMetrics();
        }
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

void TrainingDashboardPanel::UpdateLoss(float loss) {
    current_loss_ = loss;
    loss_history_.push_back(loss);

    if (loss_history_.size() > MAX_HISTORY) {
        loss_history_.erase(loss_history_.begin());
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.UpdateRealtimePlot(loss_plot_id_, static_cast<double>(current_epoch_),
                               static_cast<double>(loss), "loss");
}

void TrainingDashboardPanel::UpdateAccuracy(float accuracy) {
    current_accuracy_ = accuracy;
    accuracy_history_.push_back(accuracy);

    if (accuracy_history_.size() > MAX_HISTORY) {
        accuracy_history_.erase(accuracy_history_.begin());
    }

    if (accuracy > best_accuracy_) {
        best_accuracy_ = accuracy;
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.UpdateRealtimePlot(accuracy_plot_id_, static_cast<double>(current_epoch_),
                               static_cast<double>(accuracy), "accuracy");
}

void TrainingDashboardPanel::UpdateThroughput(float samples_per_sec) {
    current_throughput_ = samples_per_sec;
    throughput_history_.push_back(samples_per_sec);

    if (throughput_history_.size() > MAX_HISTORY) {
        throughput_history_.erase(throughput_history_.begin());
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plot_mgr.UpdateRealtimePlot(throughput_plot_id_, static_cast<double>(current_epoch_),
                               static_cast<double>(samples_per_sec), "throughput");
}

void TrainingDashboardPanel::UpdateLearningRate(float lr) {
    current_lr_ = lr;
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

void TrainingDashboardPanel::ResetMetrics() {
    loss_history_.clear();
    accuracy_history_.clear();
    throughput_history_.clear();

    current_loss_ = 0.0f;
    current_accuracy_ = 0.0f;
    current_throughput_ = 0.0f;
    current_epoch_ = 0.0f;
    progress_ = 0.0f;

    min_loss_ = 0.0f;
    max_loss_ = 0.0f;
    avg_loss_ = 0.0f;
    best_accuracy_ = 0.0f;
}

void TrainingDashboardPanel::ResetRLMetrics() {
    std::lock_guard<std::mutex> lock(custom_metrics_mutex_);

    for (auto& [name, metric] : custom_metrics_) {
        metric.history.clear();
        metric.current_value = 0.0f;
        metric.update_count = 0;
    }

    is_rl_training_ = false;
    is_training_ = false;
}

} // namespace cyxwiz
