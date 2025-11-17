#include "training_plot_panel.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <fstream>
#include <numeric>

namespace cyxwiz {

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
}

TrainingPlotPanel::~TrainingPlotPanel() {
    // Cleanup handled by base class
}

void TrainingPlotPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin(name_.c_str(), &visible_)) {
        ImGui::End();
        return;
    }

    // Lock data for reading
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Render controls at the top
    RenderControls();

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

void TrainingPlotPanel::AddLossPoint(int epoch, double train_loss, double val_loss) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    train_loss_.epochs.push_back(epoch);
    train_loss_.values.push_back(train_loss);
    TrimDataIfNeeded(train_loss_);

    if (val_loss >= 0.0) {
        val_loss_.epochs.push_back(epoch);
        val_loss_.values.push_back(val_loss);
        TrimDataIfNeeded(val_loss_);
    }
}

void TrainingPlotPanel::AddAccuracyPoint(int epoch, double train_acc, double val_acc) {
    std::lock_guard<std::mutex> lock(data_mutex_);

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

void TrainingPlotPanel::ExportPlotImage(const std::string& filepath) {
    // TODO: Implement screenshot/export functionality
    // This would require rendering to a framebuffer and saving as image
}

void TrainingPlotPanel::RenderLossPlot() {
    if (ImPlot::BeginPlot("Loss", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Epoch", "Loss");

        if (auto_scale_ && !train_loss_.epochs.empty()) {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0,
                std::max(1, static_cast<int>(train_loss_.epochs.back())),
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
    if (ImPlot::BeginPlot("Accuracy", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Epoch", "Accuracy (%)");

        if (auto_scale_ && !train_accuracy_.epochs.empty()) {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0,
                std::max(1, static_cast<int>(train_accuracy_.epochs.back())),
                ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImGuiCond_Always);
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
    if (ImPlot::BeginPlot("Custom Metrics", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Epoch", "Value");

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
    if (ImGui::Button("Clear All")) {
        Clear();
    }
    ImGui::SameLine();

    if (ImGui::Button("Export CSV")) {
        ExportToCSV("training_metrics.csv");
    }
    ImGui::SameLine();

    ImGui::Checkbox("Auto Scale", &auto_scale_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Loss", &show_loss_plot_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Accuracy", &show_accuracy_plot_);

    if (!custom_metrics_.empty()) {
        ImGui::SameLine();
        ImGui::Checkbox("Show Custom", &show_custom_metrics_);
    }
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
        series.epochs.erase(series.epochs.begin(), series.epochs.begin() + to_remove);
        series.values.erase(series.values.begin(), series.values.begin() + to_remove);
    }
}

double TrainingPlotPanel::CalculateMean(const std::vector<double>& values, size_t last_n) {
    if (values.empty()) return 0.0;

    size_t start = values.size() > last_n ? values.size() - last_n : 0;
    double sum = std::accumulate(values.begin() + start, values.end(), 0.0);
    return sum / (values.size() - start);
}

double TrainingPlotPanel::CalculateMin(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

double TrainingPlotPanel::CalculateMax(const std::vector<double>& values) {
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
