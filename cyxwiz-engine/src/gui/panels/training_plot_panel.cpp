#include "training_plot_panel.h"
#ifndef CYXWIZ_PLOTTING_MODULE
#include "../../core/crash_run_recorder.h"
#include "../../core/training_manager.h"
#endif
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {

bool IsSequenceMetricName(const std::string& name) {
    return name == "Train Token Accuracy" ||
           name == "Val Token Accuracy" ||
           name == "Train Entity F1" ||
           name == "Val Entity F1";
}

} // namespace

TrainingPlotPanel::TrainingPlotPanel()
    : Panel("Training Dashboard") {

    // Initialize metric series
    train_loss_.name = "Training Loss";
    train_loss_.color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red

    val_loss_.name = "Validation Loss";
    val_loss_.color = ImVec4(0.3f, 0.5f, 1.0f, 1.0f);  // Blue

    train_accuracy_.name = "Training Accuracy";
    train_accuracy_.color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);  // Green

    val_accuracy_.name = "Validation Accuracy";
    val_accuracy_.color = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);  // Yellow

    visible_ = true;
    RecordPanelEvent("TrainingPlotPanel.Created");
}

TrainingPlotPanel::~TrainingPlotPanel() {
    RecordPanelEvent("TrainingPlotPanel.Destroyed");
}

void TrainingPlotPanel::Render() {
    if (!visible_) {
        if (last_render_visible_) {
            RecordPanelEvent("TrainingPlotPanel.Hidden");
            last_render_visible_ = false;
        }
        return;
    }
    if (!last_render_visible_) {
        RecordPanelEvent("TrainingPlotPanel.Visible");
        last_render_visible_ = true;
    }

    // Larger default size for better visibility
    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin(name_.c_str(), &visible_)) {
        ImGui::End();
        return;
    }

    // Lock data for reading
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Render controls at the top
    RenderControls();

    ImGui::Separator();

    // Render training status (always visible)
    RenderTrainingStatus();

    ImGui::Separator();

    // Check if we have any training data
    bool has_data = !train_loss_.values.empty() || !train_accuracy_.values.empty() || !custom_metrics_.empty();

    if (has_data) {
        // Render plots
        if (show_loss_plot_ && !train_loss_.values.empty()) {
            RenderLossPlot();
        }

        if (show_accuracy_plot_ && !train_accuracy_.values.empty()) {
            RenderAccuracyPlot();
        }

        if (show_custom_metrics_ && !custom_metrics_.empty()) {
            RenderCustomMetricsPlot();
        }

        RenderCurveSummary();
        RenderSequenceMetricsSummary();

        // Render statistics
        if (!train_loss_.values.empty()) {
            RenderStatistics();
        }
    } else {
        // Show placeholder when no data
        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));

        float window_width = ImGui::GetContentRegionAvail().x;
        const char* msg1 = "No training data yet";
        const char* msg2 = "Run a training script to see real-time metrics";
        const char* msg3 = "Try: scripts/train_xor_simple.py";

        float text_width1 = ImGui::CalcTextSize(msg1).x;
        float text_width2 = ImGui::CalcTextSize(msg2).x;
        float text_width3 = ImGui::CalcTextSize(msg3).x;

        ImGui::SetCursorPosX((window_width - text_width1) * 0.5f);
        ImGui::Text("%s", msg1);

        ImGui::Spacing();

        ImGui::SetCursorPosX((window_width - text_width2) * 0.5f);
        ImGui::Text("%s", msg2);

        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::SetCursorPosX((window_width - text_width3) * 0.5f);
        ImGui::Text("%s", msg3);
        ImGui::PopStyleColor();

        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::Spacing();

        // Show example plot area
        ImGui::BeginChild("PlaceholderPlot", ImVec2(0, 300), true);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        ImGui::SetCursorPos(ImVec2(ImGui::GetContentRegionAvail().x * 0.5f - 100, ImGui::GetContentRegionAvail().y * 0.5f - 10));
        ImGui::Text("Loss/Accuracy plots will appear here");
        ImGui::PopStyleColor();
        ImGui::EndChild();
    }

    ImGui::End();
}

void TrainingPlotPanel::AddLossPoint(double epoch, double train_loss, double val_loss) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    RecordPanelEvent("TrainingPlotPanel.WriteLoss",
                     "epoch=" + std::to_string(epoch) +
                     " train_loss=" + std::to_string(train_loss));

    train_loss_.epochs.push_back(epoch);
    train_loss_.values.push_back(train_loss);
    TrimDataIfNeeded(train_loss_);

    if (val_loss >= 0.0) {
        val_loss_.epochs.push_back(epoch);
        val_loss_.values.push_back(val_loss);
        TrimDataIfNeeded(val_loss_);
    }
}

void TrainingPlotPanel::AddAccuracyPoint(double epoch, double train_acc, double val_acc) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    RecordPanelEvent("TrainingPlotPanel.WriteAccuracy",
                     "epoch=" + std::to_string(epoch) +
                     " train_acc=" + std::to_string(train_acc));

    train_accuracy_.epochs.push_back(epoch);
    train_accuracy_.values.push_back(train_acc);
    TrimDataIfNeeded(train_accuracy_);

    if (val_acc >= 0.0) {
        val_accuracy_.epochs.push_back(epoch);
        val_accuracy_.values.push_back(val_acc);
        TrimDataIfNeeded(val_accuracy_);
    }
}

void TrainingPlotPanel::AddCustomMetric(const std::string& metric_name, int epoch, double value) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Find or create metric series
    auto it = std::find_if(custom_metrics_.begin(), custom_metrics_.end(),
        [&metric_name](const MetricSeries& series) {
            return series.name == metric_name;
        });

    if (it == custom_metrics_.end()) {
        // Create new metric series
        MetricSeries new_series;
        new_series.name = metric_name;
        // Generate a unique color based on index
        float hue = (custom_metrics_.size() * 0.618034f);  // Golden ratio
        hue = hue - std::floor(hue);  // Wrap to [0, 1]
        ImGui::ColorConvertHSVtoRGB(hue, 0.7f, 1.0f,
                                    new_series.color.x,
                                    new_series.color.y,
                                    new_series.color.z);
        new_series.color.w = 1.0f;
        custom_metrics_.push_back(new_series);
        it = custom_metrics_.end() - 1;
    }

    it->epochs.push_back(epoch);
    it->values.push_back(value);
    TrimDataIfNeeded(*it);
}

void TrainingPlotPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    train_loss_.epochs.clear();
    train_loss_.values.clear();
    val_loss_.epochs.clear();
    val_loss_.values.clear();
    train_accuracy_.epochs.clear();
    train_accuracy_.values.clear();
    val_accuracy_.epochs.clear();
    val_accuracy_.values.clear();
    custom_metrics_.clear();

    // Reset training state
    is_training_ = false;
    current_epoch_ = 0;
    total_epochs_ = 0;
    last_epoch_time_ = 0.0f;
    avg_epoch_time_ = 0.0f;
    samples_per_second_ = 0.0f;
    total_training_time_ = 0.0f;
    epoch_times_.clear();
}

void TrainingPlotPanel::SetTrainingState(bool is_training, int current_epoch, int total_epochs,
                                          float epoch_time_seconds, float samples_per_second) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    is_training_ = is_training;
    current_epoch_ = current_epoch;
    total_epochs_ = total_epochs;
    last_epoch_time_ = epoch_time_seconds;
    samples_per_second_ = samples_per_second;

    if (epoch_time_seconds > 0) {
        epoch_times_.push_back(epoch_time_seconds);
        // Calculate moving average of epoch times
        float sum = 0.0f;
        for (float t : epoch_times_) sum += t;
        avg_epoch_time_ = sum / epoch_times_.size();
    }
}

void TrainingPlotPanel::SetTrainingComplete(float total_time_seconds) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    is_training_ = false;
    total_training_time_ = total_time_seconds;
}

void TrainingPlotPanel::SetBatchProgress(int current_epoch, int current_batch,
                                          int total_batches, float running_loss) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    RecordPanelEvent("TrainingPlotPanel.WriteBatchProgress",
                     "epoch=" + std::to_string(current_epoch) +
                     " batch=" + std::to_string(current_batch) +
                     "/" + std::to_string(total_batches));
    // Advance epoch counter as soon as the first batch of that epoch fires,
    // so the UI doesn't show "Epoch 0/N" while batch N of epoch 1 is running.
    // Don't regress the counter (epoch_callback may have already set it higher).
    if (current_epoch > current_epoch_) {
        current_epoch_ = current_epoch;
    }
    current_batch_ = current_batch;
    total_batches_ = total_batches;
    current_batch_loss_ = running_loss;
}

void TrainingPlotPanel::ResetPlots() {
    Clear();
}

void TrainingPlotPanel::SetMaxPoints(size_t max_points) {
    max_points_ = max_points;
}

void TrainingPlotPanel::ExportToCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        return;
    }

    // Write header
    file << "Epoch,TrainLoss,ValLoss,TrainAccuracy,ValAccuracy";
    for (const auto& metric : custom_metrics_) {
        file << "," << metric.name;
    }
    file << "\n";

    // Find max number of rows
    size_t max_rows = std::max({train_loss_.epochs.size(),
                                 val_loss_.epochs.size(),
                                 train_accuracy_.epochs.size(),
                                 val_accuracy_.epochs.size()});

    // Write data
    for (size_t i = 0; i < max_rows; ++i) {
        file << (i < train_loss_.epochs.size() ? train_loss_.epochs[i] : -1) << ",";
        file << (i < train_loss_.values.size() ? train_loss_.values[i] : 0.0) << ",";
        file << (i < val_loss_.values.size() ? val_loss_.values[i] : 0.0) << ",";
        file << (i < train_accuracy_.values.size() ? train_accuracy_.values[i] : 0.0) << ",";
        file << (i < val_accuracy_.values.size() ? val_accuracy_.values[i] : 0.0);

        for (const auto& metric : custom_metrics_) {
            file << "," << (i < metric.values.size() ? metric.values[i] : 0.0);
        }
        file << "\n";
    }

    file.close();
}

void TrainingPlotPanel::ExportPlotImage(const std::string& /*filepath*/) {
    // TODO: Implement screenshot/export functionality
    // This would require rendering to a framebuffer and saving as image
}

void TrainingPlotPanel::RenderLossPlot() {
    // Calculate plot height based on available space (at least 300px, or half available minus some padding)
    float available_height = ImGui::GetContentRegionAvail().y;
    float plot_height = std::max(300.0f, (available_height - 100.0f) / 2.0f);

    if (ImPlot::BeginPlot("Loss", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

        if (auto_scale_ && !train_loss_.epochs.empty()) {
            const auto [min_epoch, max_epoch] = CalculateEpochWindow(train_loss_);
            ImPlot::SetupAxisLimits(ImAxis_X1, min_epoch, max_epoch, ImGuiCond_Always);

            ValueRange range = CalculateVisibleRange(train_loss_, val_loss_, min_epoch, max_epoch);
            double padding = (range.max - range.min) * 0.1;
            if (padding < 0.01) padding = 0.1;
            ImPlot::SetupAxisLimits(ImAxis_Y1,
                std::max(0.0, range.min - padding),
                range.max + padding,
                ImGuiCond_Always);
        }

        // Plot training loss
        if (!train_loss_.values.empty()) {
            ImPlot::SetNextLineStyle(train_loss_.color, 2.0f);
            ImPlot::PlotLine(train_loss_.name.c_str(),
                           train_loss_.epochs.data(),
                           train_loss_.values.data(),
                           static_cast<int>(train_loss_.values.size()));
        }

        // Plot validation loss
        if (!val_loss_.values.empty()) {
            ImPlot::SetNextLineStyle(val_loss_.color, 2.0f);
            ImPlot::PlotLine(val_loss_.name.c_str(),
                           val_loss_.epochs.data(),
                           val_loss_.values.data(),
                           static_cast<int>(val_loss_.values.size()));
        }

        ImPlot::EndPlot();
    }
}

void TrainingPlotPanel::RenderAccuracyPlot() {
    // Calculate plot height based on available space (at least 300px, or remaining available space)
    float available_height = ImGui::GetContentRegionAvail().y;
    float plot_height = std::max(300.0f, available_height - 80.0f);

    if (ImPlot::BeginPlot("Accuracy", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("Epoch", "Accuracy (%)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

        if (auto_scale_ && !train_accuracy_.epochs.empty()) {
            const auto [min_epoch, max_epoch] = CalculateEpochWindow(train_accuracy_);
            ImPlot::SetupAxisLimits(ImAxis_X1, min_epoch, max_epoch, ImGuiCond_Always);

            ValueRange range = CalculateVisibleRange(train_accuracy_, val_accuracy_, min_epoch, max_epoch);
            double padding = (range.max - range.min) * 0.1;
            if (padding < 1.0) padding = 5.0;
            ImPlot::SetupAxisLimits(ImAxis_Y1,
                std::max(0.0, range.min - padding),
                std::min(100.0, range.max + padding),
                ImGuiCond_Always);
        }

        // Plot training accuracy
        if (!train_accuracy_.values.empty()) {
            ImPlot::SetNextLineStyle(train_accuracy_.color, 2.0f);
            ImPlot::PlotLine(train_accuracy_.name.c_str(),
                           train_accuracy_.epochs.data(),
                           train_accuracy_.values.data(),
                           static_cast<int>(train_accuracy_.values.size()));
        }

        // Plot validation accuracy
        if (!val_accuracy_.values.empty()) {
            ImPlot::SetNextLineStyle(val_accuracy_.color, 2.0f);
            ImPlot::PlotLine(val_accuracy_.name.c_str(),
                           val_accuracy_.epochs.data(),
                           val_accuracy_.values.data(),
                           static_cast<int>(val_accuracy_.values.size()));
        }

        ImPlot::EndPlot();
    }
}

void TrainingPlotPanel::RenderCustomMetricsPlot() {
    bool has_sequence_metrics = false;
    bool has_non_sequence_metrics = false;
    for (const auto& metric : custom_metrics_) {
        if (metric.values.empty()) {
            continue;
        }
        if (IsSequenceMetricName(metric.name)) {
            has_sequence_metrics = true;
        } else {
            has_non_sequence_metrics = true;
        }
    }

    const char* plot_title =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Sequence Metrics"
            : "Custom Metrics";
    const char* y_label =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Score (%)"
            : "Value";

    if (ImPlot::BeginPlot(plot_title, ImVec2(-1, 250))) {
        // Enable zoom and pan on both axes
        ImPlot::SetupAxes("Epoch", y_label, ImPlotAxisFlags_None, ImPlotAxisFlags_None);

        for (const auto& metric : custom_metrics_) {
            if (!metric.values.empty()) {
                ImPlot::SetNextLineStyle(metric.color, 2.0f);
                ImPlot::PlotLine(metric.name.c_str(),
                               metric.epochs.data(),
                               metric.values.data(),
                               static_cast<int>(metric.values.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void TrainingPlotPanel::RenderControls() {
#ifndef CYXWIZ_PLOTTING_MODULE
    auto& tm = TrainingManager::Instance();
    const bool training_active = tm.IsTrainingActive();
    const bool training_paused = tm.IsPaused();
#endif

    if (ImGui::Button("Clear All")) {
        Clear();
    }
    ImGui::SameLine();

    if (ImGui::Button("Export CSV")) {
        ExportToCSV("training_metrics.csv");
    }
    ImGui::SameLine();

    ImGui::Checkbox("Auto Scale", &auto_scale_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("When enabled, axes adapt to the live training data.\nDisable to manually zoom/pan.");
    }
    ImGui::SameLine();
    ImGui::Checkbox("Follow Current", &follow_current_epoch_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Keep the epoch axis scrolled to the latest batch/epoch.");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderInt("Epoch Window", &visible_epoch_window_, 3, 50);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Number of epochs visible while following live training.");
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show Loss", &show_loss_plot_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Accuracy", &show_accuracy_plot_);

#ifndef CYXWIZ_PLOTTING_MODULE
    ImGui::SameLine();
    ImGui::TextUnformatted("Training");
    ImGui::SameLine();

    if (training_active) {
        if (training_paused) {
            if (ImGui::Button("Continue")) {
                tm.ResumeTraining();
            }
        } else {
            if (ImGui::Button("Pause")) {
                tm.PauseTraining();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Early Stop")) {
            tm.StopTraining();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stops training at the next batch boundary and keeps the current model state.");
        }
    } else {
        ImGui::TextDisabled("Idle");
    }
#endif

    if (!custom_metrics_.empty()) {
        bool has_sequence_metrics = false;
        bool has_non_sequence_metrics = false;
        for (const auto& metric : custom_metrics_) {
            if (metric.values.empty()) {
                continue;
            }
            if (IsSequenceMetricName(metric.name)) {
                has_sequence_metrics = true;
            } else {
                has_non_sequence_metrics = true;
            }
        }

        ImGui::SameLine();
        const char* label = has_sequence_metrics && !has_non_sequence_metrics
            ? "Show Sequence"
            : "Show Custom";
        ImGui::Checkbox(label, &show_custom_metrics_);
    }

    // Show zoom/pan help
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Plot Controls:");
        ImGui::BulletText("Scroll wheel: Zoom both axes");
        ImGui::BulletText("Scroll on axis: Zoom that axis only");
        ImGui::BulletText("Right-drag: Pan view");
        ImGui::BulletText("Double-click: Reset zoom");
        ImGui::BulletText("Disable Auto Scale or Follow Current for manual control");
        ImGui::EndTooltip();
    }
}

void TrainingPlotPanel::RenderCurveSummary() {
    ImGui::Separator();
    ImGui::Text("Curve Summary");

    auto slope_last = [](const MetricSeries& series, size_t window) -> double {
        const size_t count = std::min(series.epochs.size(), series.values.size());
        if (count < 2) return 0.0;

        const size_t start = count > window ? count - window : 0;
        const double dx = series.epochs[count - 1] - series.epochs[start];
        if (dx <= 0.0) return 0.0;
        return (series.values[count - 1] - series.values[start]) / dx;
    };

    auto volatility_last = [](const MetricSeries& series, size_t window) -> double {
        const size_t count = std::min(series.epochs.size(), series.values.size());
        if (count < 3) return 0.0;

        const size_t start = count > window ? count - window : 1;
        double total_delta = 0.0;
        size_t deltas = 0;
        for (size_t i = start; i < count; ++i) {
            total_delta += std::abs(series.values[i] - series.values[i - 1]);
            ++deltas;
        }
        return deltas > 0 ? total_delta / static_cast<double>(deltas) : 0.0;
    };

    auto best_index = [](const MetricSeries& series, bool lower_is_better) -> int {
        const size_t count = std::min(series.epochs.size(), series.values.size());
        if (count == 0) return -1;

        size_t best = 0;
        for (size_t i = 1; i < count; ++i) {
            if ((lower_is_better && series.values[i] < series.values[best]) ||
                (!lower_is_better && series.values[i] > series.values[best])) {
                best = i;
            }
        }
        return static_cast<int>(best);
    };

    auto closest_value = [](const MetricSeries& series, double epoch) -> double {
        const size_t count = std::min(series.epochs.size(), series.values.size());
        if (count == 0) return 0.0;

        size_t best = 0;
        double best_distance = std::abs(series.epochs[0] - epoch);
        for (size_t i = 1; i < count; ++i) {
            const double distance = std::abs(series.epochs[i] - epoch);
            if (distance < best_distance) {
                best = i;
                best_distance = distance;
            }
        }
        return series.values[best];
    };

    auto trend_label = [](double slope, bool lower_is_better) -> const char* {
        const double threshold = lower_is_better ? 0.002 : 0.05;
        if (std::abs(slope) < threshold) return "flat";
        const bool improving = lower_is_better ? slope < 0.0 : slope > 0.0;
        return improving ? "improving" : "worsening";
    };

    const double train_loss_slope = slope_last(train_loss_, 120);
    const double val_loss_slope = slope_last(val_loss_, 5);
    const double train_acc_slope = slope_last(train_accuracy_, 120);
    const double val_acc_slope = slope_last(val_accuracy_, 5);
    const double val_loss_volatility = volatility_last(val_loss_, 5);

    const int best_val_loss_idx = best_index(val_loss_, true);
    const int best_val_acc_idx = best_index(val_accuracy_, false);

    double rough_epoch = -1.0;
    if (best_val_loss_idx >= 0) {
        int rising_streak = 0;
        for (size_t i = static_cast<size_t>(best_val_loss_idx) + 1;
             i < val_loss_.values.size();
             ++i) {
            if (val_loss_.values[i] > val_loss_.values[i - 1]) {
                ++rising_streak;
                if (rising_streak >= 2) {
                    rough_epoch = val_loss_.epochs[i - 1];
                    break;
                }
            } else {
                rising_streak = 0;
            }
        }
    }

    const bool has_validation = !val_loss_.values.empty() || !val_accuracy_.values.empty();
    const bool val_loss_worsening = val_loss_slope > 0.0;
    const bool val_acc_worsening = val_acc_slope < 0.0;
    const bool gap_large = !val_loss_.values.empty() && !train_loss_.values.empty() &&
        (val_loss_.values.back() - closest_value(train_loss_, val_loss_.epochs.back())) > 0.25;

    const char* recommendation = "continue";
    ImVec4 recommendation_color = ImVec4(0.45f, 0.85f, 0.55f, 1.0f);
    if (has_validation && (gap_large || val_loss_worsening || val_acc_worsening || rough_epoch >= 0.0)) {
        recommendation = "inspect validation";
        recommendation_color = ImVec4(1.0f, 0.75f, 0.25f, 1.0f);
    }
    if (has_validation && rough_epoch >= 0.0 && val_loss_volatility > 0.02) {
        recommendation = "consider early stop";
        recommendation_color = ImVec4(1.0f, 0.55f, 0.35f, 1.0f);
    }

    ImGui::Columns(2, "curve_summary", false);

    ImGui::Text("Train Curve:");
    ImGui::NextColumn();
    ImGui::Text("loss %s, accuracy %s",
        trend_label(train_loss_slope, true),
        trend_label(train_acc_slope, false));
    ImGui::NextColumn();

    ImGui::Text("Validation Curve:");
    ImGui::NextColumn();
    if (!val_loss_.values.empty() || !val_accuracy_.values.empty()) {
        ImGui::Text("loss %s, accuracy %s",
            trend_label(val_loss_slope, true),
            trend_label(val_acc_slope, false));
    } else {
        ImGui::TextDisabled("waiting for validation points");
    }
    ImGui::NextColumn();

    ImGui::Text("Best Validation:");
    ImGui::NextColumn();
    if (best_val_loss_idx >= 0) {
        ImGui::Text("loss %.4f at epoch %.2f",
            val_loss_.values[best_val_loss_idx],
            val_loss_.epochs[best_val_loss_idx]);
        if (best_val_acc_idx >= 0) {
            ImGui::SameLine();
            ImGui::Text("| acc %.2f%% at epoch %.2f",
                val_accuracy_.values[best_val_acc_idx],
                val_accuracy_.epochs[best_val_acc_idx]);
        }
    } else {
        ImGui::TextDisabled("no validation data yet");
    }
    ImGui::NextColumn();

    ImGui::Text("Rough Point:");
    ImGui::NextColumn();
    if (rough_epoch >= 0.0) {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f),
            "validation loss starts rising around epoch %.2f", rough_epoch);
    } else if (val_loss_.values.size() >= 3) {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f),
            "no sustained validation rise detected");
    } else {
        ImGui::TextDisabled("need more validation points");
    }
    ImGui::NextColumn();

    ImGui::Text("Generalization Gap:");
    ImGui::NextColumn();
    if (!val_loss_.values.empty() && !train_loss_.values.empty()) {
        const double epoch = val_loss_.epochs.back();
        const double train_near_val = closest_value(train_loss_, epoch);
        const double gap = val_loss_.values.back() - train_near_val;
        ImGui::Text("val_loss - train_loss = %.4f", gap);
        ImGui::SameLine();
        if (gap > 0.25) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f), "possible overfit");
        } else {
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f), "controlled");
        }
    } else {
        ImGui::TextDisabled("waiting for train/validation loss");
    }
    ImGui::NextColumn();

    ImGui::Text("Validation Roughness:");
    ImGui::NextColumn();
    if (val_loss_.values.size() >= 3) {
        ImGui::Text("recent avg delta %.4f", val_loss_volatility);
    } else {
        ImGui::TextDisabled("need more validation points");
    }
    ImGui::NextColumn();

    ImGui::Text("Suggested Action:");
    ImGui::NextColumn();
    ImGui::TextColored(recommendation_color, "%s", recommendation);

    ImGui::Columns(1);
}

void TrainingPlotPanel::RenderSequenceMetricsSummary() {
    auto find_latest = [this](const char* metric_name, double& value, bool& found) {
        for (const auto& metric : custom_metrics_) {
            if (metric.name != metric_name || metric.values.empty()) {
                continue;
            }
            value = metric.values.back();
            found = true;
            return;
        }
    };

    double train_token_accuracy = 0.0;
    double val_token_accuracy = 0.0;
    double train_entity_f1 = 0.0;
    double val_entity_f1 = 0.0;
    bool has_train_token_accuracy = false;
    bool has_val_token_accuracy = false;
    bool has_train_entity_f1 = false;
    bool has_val_entity_f1 = false;

    find_latest("Train Token Accuracy", train_token_accuracy, has_train_token_accuracy);
    find_latest("Val Token Accuracy", val_token_accuracy, has_val_token_accuracy);
    find_latest("Train Entity F1", train_entity_f1, has_train_entity_f1);
    find_latest("Val Entity F1", val_entity_f1, has_val_entity_f1);

    if (!has_train_token_accuracy && !has_val_token_accuracy &&
        !has_train_entity_f1 && !has_val_entity_f1) {
        return;
    }

    ImGui::Separator();
    ImGui::Text("Sequence Metrics");
    ImGui::Columns(2, "sequence_metrics", false);

    ImGui::Text("Token Accuracy");
    ImGui::NextColumn();
    if (has_train_token_accuracy || has_val_token_accuracy) {
        if (has_train_token_accuracy) {
            ImGui::Text("train %.2f%%", train_token_accuracy);
            ImGui::SameLine();
        }
        if (has_val_token_accuracy) {
            ImGui::Text("val %.2f%%", val_token_accuracy);
        }
    } else {
        ImGui::TextDisabled("no data");
    }
    ImGui::NextColumn();

    ImGui::Text("Entity F1");
    ImGui::NextColumn();
    if (has_train_entity_f1 || has_val_entity_f1) {
        if (has_train_entity_f1) {
            ImGui::Text("train %.2f%%", train_entity_f1);
            ImGui::SameLine();
        }
        if (has_val_entity_f1) {
            ImGui::Text("val %.2f%%", val_entity_f1);
        }
    } else {
        ImGui::TextDisabled("no data");
    }
    ImGui::NextColumn();

    ImGui::Columns(1);
}

void TrainingPlotPanel::RenderTrainingStatus() {
    // Training status header with colored indicator
    ImGui::BeginChild("TrainingStatus", ImVec2(0, 100), true);

    // Status indicator
    if (is_training_) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
        ImGui::Text("TRAINING");
        ImGui::PopStyleColor();
    } else if (total_training_time_ > 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::Text("COMPLETED");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        ImGui::Text("IDLE");
        ImGui::PopStyleColor();
    }

    ImGui::SameLine(120);

    // Progress info
    if (total_epochs_ > 0) {
        float progress = static_cast<float>(current_epoch_) / total_epochs_;
        ImGui::Text("Epoch %d / %d", current_epoch_, total_epochs_);
        ImGui::SameLine(280);
        ImGui::ProgressBar(progress, ImVec2(200, 0));
    }

    // Batch-level progress within the current epoch (live feedback during training)
    if (is_training_ && total_batches_ > 0) {
        float batch_progress = static_cast<float>(current_batch_) /
                                std::max(1, total_batches_);
        ImGui::Text("Batch %d / %d", current_batch_, total_batches_);
        ImGui::SameLine(280);
        ImGui::ProgressBar(batch_progress, ImVec2(200, 0));
        ImGui::SameLine();
        ImGui::Text("running loss: %.4f", current_batch_loss_);
    }

    // Second row: timing info
    ImGui::Spacing();

    if (is_training_ && avg_epoch_time_ > 0) {
        int remaining_epochs = total_epochs_ - current_epoch_;
        float eta_seconds = remaining_epochs * avg_epoch_time_;

        // Format ETA nicely
        int eta_hours = static_cast<int>(eta_seconds / 3600);
        int eta_mins = static_cast<int>((eta_seconds - eta_hours * 3600) / 60);
        int eta_secs = static_cast<int>(eta_seconds) % 60;

        ImGui::Text("Last Epoch: %.1fs", last_epoch_time_);
        ImGui::SameLine(150);
        ImGui::Text("Avg: %.1fs/epoch", avg_epoch_time_);
        ImGui::SameLine(300);

        if (eta_hours > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                "ETA: %dh %dm %ds", eta_hours, eta_mins, eta_secs);
        } else if (eta_mins > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                "ETA: %dm %ds", eta_mins, eta_secs);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                "ETA: %ds", eta_secs);
        }

        ImGui::SameLine(480);
        if (samples_per_second_ > 0) {
            ImGui::Text("%.0f samples/sec", samples_per_second_);
        }
    } else if (total_training_time_ > 0) {
        // Training completed
        int total_hours = static_cast<int>(total_training_time_ / 3600);
        int total_mins = static_cast<int>((total_training_time_ - total_hours * 3600) / 60);
        int total_secs = static_cast<int>(total_training_time_) % 60;

        if (total_hours > 0) {
            ImGui::Text("Total Time: %dh %dm %ds", total_hours, total_mins, total_secs);
        } else if (total_mins > 0) {
            ImGui::Text("Total Time: %dm %ds", total_mins, total_secs);
        } else {
            ImGui::Text("Total Time: %ds", total_secs);
        }
    }

    // Third row: current metrics
    if (!train_loss_.values.empty() || !train_accuracy_.values.empty()) {
        ImGui::Spacing();

        if (!train_loss_.values.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                "Loss: %.6f", train_loss_.values.back());
        }

        if (!train_accuracy_.values.empty()) {
            ImGui::SameLine(180);
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                "Accuracy: %.2f%%", train_accuracy_.values.back());
        }

        if (!val_loss_.values.empty()) {
            ImGui::SameLine(360);
            ImGui::TextColored(ImVec4(0.3f, 0.5f, 1.0f, 1.0f),
                    "Val Loss: %.6f", val_loss_.values.back());
        }

        if (!val_accuracy_.values.empty()) {
            ImGui::SameLine(540);
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                    "Val Acc: %.2f%%", val_accuracy_.values.back());
        }

        if (!val_loss_.values.empty() || !val_accuracy_.values.empty()) {
            ImGui::Spacing();
            ImGui::Text("Validation Signal:");
            ImGui::SameLine(150);
            if (!val_loss_.values.empty()) {
                const double recent_val_loss = val_loss_.values.back();
                const double recent_train_loss = train_loss_.values.empty() ? recent_val_loss : train_loss_.values.back();
                const double gap = recent_val_loss - recent_train_loss;
                if (gap > 0.25) {
                    ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
                        "val_loss is above train_loss by %.4f", gap);
                } else {
                    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f),
                        "validation gap is controlled");
                }
            } else {
                ImGui::TextDisabled("validation metrics not available yet");
            }
        }
    }

    ImGui::EndChild();
}

void TrainingPlotPanel::RenderStatistics() {
    ImGui::Separator();
    ImGui::Text("Statistics (last 10 epochs):");

    ImGui::Columns(2, "stats", false);

    if (!train_loss_.values.empty()) {
        double mean_loss = CalculateMean(train_loss_.values);
        double min_loss = CalculateMin(train_loss_.values);
        double max_loss = CalculateMax(train_loss_.values);

        ImGui::Text("Train Loss:");
        ImGui::NextColumn();
        ImGui::Text("Mean: %.6f | Min: %.6f | Max: %.6f", mean_loss, min_loss, max_loss);
        ImGui::NextColumn();
    }

    if (!val_loss_.values.empty()) {
        double mean_loss = CalculateMean(val_loss_.values);
        double min_loss = CalculateMin(val_loss_.values);
        double max_loss = CalculateMax(val_loss_.values);

        ImGui::Text("Val Loss:");
        ImGui::NextColumn();
        ImGui::Text("Mean: %.6f | Min: %.6f | Max: %.6f", mean_loss, min_loss, max_loss);
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
}

void TrainingPlotPanel::TrimDataIfNeeded(MetricSeries& series) {
    if (series.epochs.size() > max_points_) {
        size_t to_remove = series.epochs.size() - max_points_;
        RecordPanelEvent("TrainingPlotPanel.TrimData",
                         series.name + " remove=" + std::to_string(to_remove));
        series.epochs.erase(series.epochs.begin(), series.epochs.begin() + to_remove);
        series.values.erase(series.values.begin(), series.values.begin() + to_remove);
    }
}

void TrainingPlotPanel::RecordPanelEvent(const std::string& action,
                                         const std::string& detail) const {
#ifndef CYXWIZ_PLOTTING_MODULE
    CrashRunRecorder::Instance().MarkPanelEvent(action, detail);
#else
    (void)action;
    (void)detail;
#endif
}

std::pair<double, double> TrainingPlotPanel::CalculateEpochWindow(const MetricSeries& series) const {
    const double latest_epoch = series.epochs.empty() ? 0.0 : series.epochs.back();
    const double planned_epochs = total_epochs_ > 0 ? static_cast<double>(total_epochs_) : latest_epoch;
    const double window = static_cast<double>(std::max(3, visible_epoch_window_));

    if (!follow_current_epoch_) {
        const double max_epoch = std::max({1.0, planned_epochs, latest_epoch});
        return {0.0, max_epoch + 0.25};
    }

    if (latest_epoch <= window) {
        const double max_epoch = std::max(window, std::min(planned_epochs, window));
        return {0.0, max_epoch + 0.25};
    }

    const double min_epoch = std::max(0.0, latest_epoch - window);
    const double max_epoch = std::max(latest_epoch + 0.25, min_epoch + window);
    return {min_epoch, max_epoch};
}

TrainingPlotPanel::ValueRange TrainingPlotPanel::CalculateVisibleRange(
    const MetricSeries& primary,
    const MetricSeries& secondary,
    double min_epoch,
    double max_epoch) const
{
    ValueRange range;
    range.min = std::numeric_limits<double>::max();
    range.max = std::numeric_limits<double>::lowest();

    auto add_series = [&](const MetricSeries& series) {
        const size_t count = std::min(series.epochs.size(), series.values.size());
        for (size_t i = 0; i < count; ++i) {
            const double epoch = series.epochs[i];
            if (epoch < min_epoch || epoch > max_epoch) {
                continue;
            }
            range.min = std::min(range.min, series.values[i]);
            range.max = std::max(range.max, series.values[i]);
            range.has_values = true;
        }
    };

    add_series(primary);
    add_series(secondary);

    if (!range.has_values) {
        range.min = primary.values.empty() ? 0.0 : CalculateMin(primary.values);
        range.max = primary.values.empty() ? 1.0 : CalculateMax(primary.values);
        range.has_values = true;
    }

    if (range.min == range.max) {
        range.min -= 0.5;
        range.max += 0.5;
    }

    return range;
}

bool TrainingPlotPanel::HasData() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ++sampled_read_events_;
    if (sampled_read_events_ == 1 || sampled_read_events_ % 60 == 0) {
        RecordPanelEvent("TrainingPlotPanel.ReadHasData",
                         "points=" + std::to_string(train_loss_.values.size()));
    }
    return !train_loss_.values.empty();
}

bool TrainingPlotPanel::IsTraining() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return is_training_;
}

double TrainingPlotPanel::CalculateMean(const std::vector<double>& values, size_t last_n) const {
    if (values.empty()) return 0.0;

    size_t start = values.size() > last_n ? values.size() - last_n : 0;
    double sum = std::accumulate(values.begin() + start, values.end(), 0.0);
    return sum / (values.size() - start);
}

double TrainingPlotPanel::CalculateMin(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

double TrainingPlotPanel::CalculateMax(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return *std::max_element(values.begin(), values.end());
}

int TrainingPlotPanel::GetCurrentEpoch() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (train_loss_.epochs.empty()) return 0;
    return static_cast<int>(train_loss_.epochs.back());
}

double TrainingPlotPanel::GetCurrentTrainLoss() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (train_loss_.values.empty()) return 0.0;
    return train_loss_.values.back();
}

double TrainingPlotPanel::GetCurrentValLoss() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (val_loss_.values.empty()) return -1.0;
    return val_loss_.values.back();
}

double TrainingPlotPanel::GetCurrentTrainAccuracy() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (train_accuracy_.values.empty()) return -1.0;
    return train_accuracy_.values.back();
}

double TrainingPlotPanel::GetCurrentValAccuracy() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (val_accuracy_.values.empty()) return -1.0;
    return val_accuracy_.values.back();
}

size_t TrainingPlotPanel::GetDataPointCount() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return train_loss_.values.size();
}

} // namespace cyxwiz
