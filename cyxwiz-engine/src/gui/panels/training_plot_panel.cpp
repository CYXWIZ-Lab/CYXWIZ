#include "training_plot_panel.h"
#ifndef CYXWIZ_PLOTTING_MODULE
#include "../../core/async_task_manager.h"
#include "../../core/crash_run_recorder.h"
#include "../../core/training_manager.h"
#include "../../core/training_trace_collector.h"
#endif
#include "../../core/training_run_comparison.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cyxwiz {

namespace {

bool IsSequenceMetricName(const std::string& name) {
    return name == "Train Token Accuracy" ||
           name == "Val Token Accuracy" ||
           name == "Train Entity F1" ||
           name == "Val Entity F1";
}

bool IsRegressionMetricName(const std::string& name) {
    return name == "Train MAE" || name == "Val MAE" ||
           name == "Train RMSE" || name == "Val RMSE";
}

bool IsValidationMetricName(const std::string& name) {
    return name.rfind("Val ", 0) == 0 ||
           name.rfind("Validation ", 0) == 0;
}

constexpr float kTrainingPlotMinHeight = 390.0f;
constexpr float kTrainingPlotMaxHeight = 520.0f;

std::string FormatTraceBytes(uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    if (bytes >= 1024ull * 1024ull * 1024ull) {
        out.precision(2);
        out << static_cast<double>(bytes) /
                   static_cast<double>(1024ull * 1024ull * 1024ull)
            << " GB";
    } else if (bytes >= 1024ull * 1024ull) {
        out.precision(1);
        out << static_cast<double>(bytes) /
                   static_cast<double>(1024ull * 1024ull)
            << " MB";
    } else if (bytes >= 1024ull) {
        out.precision(1);
        out << static_cast<double>(bytes) / 1024.0 << " KB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string ParentDirectoryForPath(std::string path) {
    while (path.size() > 1 &&
           (path.back() == '\\' || path.back() == '/') &&
           !(path.size() == 3 && path[1] == ':')) {
        path.pop_back();
    }

    const size_t separator = path.find_last_of("\\/");
    if (separator == std::string::npos) {
        return ".";
    }
    if (separator == 0) {
        return path.substr(0, 1);
    }
    if (separator == 2 && path[1] == ':') {
        return path.substr(0, 3);
    }
    return path.substr(0, separator);
}

bool OpenDirectoryInFileBrowser(const std::string& directory) {
    if (directory.empty()) {
        return false;
    }

#ifdef _WIN32
    const HINSTANCE result = ShellExecuteA(
        nullptr, "open", directory.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(result) > 32;
#elif defined(__APPLE__)
    const pid_t pid = fork();
    if (pid == 0) {
        execlp("open", "open", directory.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
    const pid_t pid = fork();
    if (pid == 0) {
        execlp("xdg-open", "xdg-open", directory.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

const char* ClassifyTrainingWarning(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (lower.find("pin") != std::string::npos &&
        lower.find("memory") != std::string::npos) {
        return "Data transfer";
    }
    if (lower.find("fallback") != std::string::npos ||
        (lower.find("gpu") != std::string::npos &&
         lower.find("cpu") != std::string::npos)) {
        return "Device fallback";
    }
    if (lower.find("cuda") != std::string::npos ||
        lower.find("arrayfire") != std::string::npos ||
        lower.find("gpu") != std::string::npos) {
        return "GPU";
    }
    if (lower.find("memory") != std::string::npos ||
        lower.find("allocation") != std::string::npos ||
        lower.find("alloc") != std::string::npos) {
        return "Memory";
    }
    return "Warning";
}

#ifndef CYXWIZ_PLOTTING_MODULE
const TrainingTraceEvent* FindLatestPinMemoryTransferEvent(
    const TrainingTraceSummary& trace) {
    for (auto it = trace.recent_events.rbegin();
         it != trace.recent_events.rend();
         ++it) {
        if (it->pin_memory_requested || !it->transfer_mode.empty()) {
            return &(*it);
        }
    }
    return nullptr;
}

bool IsPinMemoryTransferWarning(const std::string& warning) {
    return warning.rfind("DataLoader.PinMemoryTransfer", 0) == 0;
}
#endif

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
    ImGui::SetNextWindowSize(ImVec2(1000, 800), ImGuiCond_FirstUseEver);

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
    RenderActiveTaskSummary();
    RenderMaterializationSummary();
    RenderTrainingWarningSummary();

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

    RenderRunComparisonTable();

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

void TrainingPlotPanel::AddRunComparisonRecord(
    const TrainingRunComparisonRecord& record) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    run_comparison_records_.push_back(record);
    run_comparison_records_ =
        SortTrainingRunComparisonsByBestMetric(run_comparison_records_);
}

void TrainingPlotPanel::ClearRunComparisonRecords() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    run_comparison_records_.clear();
}

void TrainingPlotPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ClearLocked();
}

void TrainingPlotPanel::ClearLocked() {
    train_loss_.epochs.clear();
    train_loss_.values.clear();
    val_loss_.epochs.clear();
    val_loss_.values.clear();
    train_accuracy_.epochs.clear();
    train_accuracy_.values.clear();
    val_accuracy_.epochs.clear();
    val_accuracy_.values.clear();
    custom_metrics_.clear();
    materialization_events_.clear();
    materialization_output_dataset_.clear();
    materialization_status_.clear();
    materialization_cache_key_.clear();
    materialization_cache_artifact_path_.clear();
    materialization_cache_manifest_path_.clear();
    materialization_cache_row_count_ = 0;
    materialization_cache_column_count_ = 0;
    materialization_operators_applied_ = 0;

    // Reset training state
    is_training_ = false;
    current_epoch_ = 0;
    total_epochs_ = 0;
    current_batch_ = 0;
    total_batches_ = 0;
    current_batch_loss_ = 0.0f;
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
    if (is_training) {
        is_preparing_ = false;
        preparation_failed_ = false;
        preparation_status_message_.clear();
        preparation_error_message_.clear();
        preparation_progress_ = 0.0f;
        terminal_status_.clear();
        terminal_reason_.clear();
        total_training_time_ = 0.0f;
    }
    current_epoch_ = current_epoch;
    if (total_epochs > 0) {
        total_epochs_ = total_epochs;
    }
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

void TrainingPlotPanel::SetTrainingComplete(float total_time_seconds,
                                            const std::string& terminal_status,
                                            const std::string& terminal_reason,
                                            const std::string& checkpoint_used,
                                            bool has_validation_metrics,
                                            float checkpoint_val_loss,
                                            float checkpoint_val_accuracy,
                                            int checkpoint_epoch) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    is_training_ = false;
    is_preparing_ = false;
    preparation_failed_ = false;
    preparation_status_message_.clear();
    preparation_error_message_.clear();
    preparation_progress_ = 0.0f;
    total_training_time_ = total_time_seconds;
    terminal_status_ = terminal_status;
    terminal_reason_ = terminal_reason;
    checkpoint_used_ = checkpoint_used;
    has_checkpoint_validation_metrics_ = has_validation_metrics;
    checkpoint_val_loss_ = checkpoint_val_loss;
    checkpoint_val_accuracy_ = checkpoint_val_accuracy;
    checkpoint_epoch_ = checkpoint_epoch;
    active_checkpoint_loaded_ = false;
}

void TrainingPlotPanel::SetActiveCheckpointLoaded(
    const std::string& checkpoint_path,
    int checkpoint_epoch,
    float validation_loss,
    float validation_accuracy,
    bool has_validation_metrics) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    is_training_ = false;
    is_preparing_ = false;
    preparation_failed_ = false;
    checkpoint_used_ = checkpoint_path;
    checkpoint_epoch_ = checkpoint_epoch;
    checkpoint_val_loss_ = validation_loss;
    checkpoint_val_accuracy_ = validation_accuracy;
    has_checkpoint_validation_metrics_ = has_validation_metrics;
    active_checkpoint_loaded_ = true;
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
    is_training_ = true;
    is_preparing_ = false;
    preparation_failed_ = false;
    preparation_status_message_.clear();
    preparation_error_message_.clear();
    preparation_progress_ = 0.0f;
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
    ExportToCSVLocked(filepath);
}

void TrainingPlotPanel::ExportToCSVLocked(
    const std::string& filepath) const {
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

void TrainingPlotPanel::ExportRunComparisonCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::string error;
    if (!WriteTrainingRunComparisonCsv(filepath, run_comparison_records_, &error)) {
        RecordPanelEvent("TrainingPlotPanel.ExportRunComparisonFailed", error);
    }
}

void TrainingPlotPanel::SetPreparationState(bool is_preparing,
                                            const std::string& status_message,
                                            float progress) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    is_preparing_ = is_preparing;
    preparation_status_message_ = status_message;
    preparation_progress_ = std::clamp(progress, 0.0f, 1.0f);
    if (is_preparing_) {
        preparation_failed_ = false;
        preparation_error_message_.clear();
        is_training_ = false;
        total_training_time_ = 0.0f;
        terminal_status_.clear();
        terminal_reason_.clear();
    }
}

void TrainingPlotPanel::SetPreparationFailed(
    const std::string& error_message) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (is_training_) {
        return;
    }

    is_preparing_ = false;
    is_training_ = false;
    preparation_failed_ = true;
    preparation_error_message_ = error_message;
    preparation_status_message_.clear();
    preparation_progress_ = 0.0f;
    total_training_time_ = 0.0f;
    terminal_status_ = "failed";
    terminal_reason_ = error_message;
}

void TrainingPlotPanel::RecordMaterializationProgress(
    const std::string& stage,
    const std::string& message,
    float progress,
    uint64_t estimated_memory_bytes,
    uint64_t processed_items,
    uint64_t total_items,
    int node_id,
    const std::string& node_name,
    const std::string& memory_risk_level,
    const std::string& status,
    const std::string& cache_key,
    const std::string& cache_artifact_path,
    const std::string& cache_manifest_path,
    int64_t cache_row_count,
    int64_t cache_column_count) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    MaterializationProgress event;
    event.stage = stage.empty() ? "Materializing" : stage;
    event.message = message.empty() ? event.stage : message;
    event.status = status.empty() ? "running" : status;
    event.node_name = node_name;
    event.node_id = node_id;
    event.progress = std::clamp(progress, 0.0f, 1.0f);
    event.estimated_memory_bytes = estimated_memory_bytes;
    event.memory_risk_level = memory_risk_level;
    event.processed_items = processed_items;
    event.total_items = total_items;
    event.cache_key = cache_key;
    event.cache_artifact_path = cache_artifact_path;
    event.cache_manifest_path = cache_manifest_path;
    event.cache_row_count = cache_row_count;
    event.cache_column_count = cache_column_count;

    if (!cache_key.empty()) {
        materialization_cache_key_ = cache_key;
    }
    if (!cache_artifact_path.empty()) {
        materialization_cache_artifact_path_ = cache_artifact_path;
    }
    if (!cache_manifest_path.empty()) {
        materialization_cache_manifest_path_ = cache_manifest_path;
    }
    if (cache_row_count > 0 || cache_column_count > 0) {
        materialization_cache_row_count_ = cache_row_count;
        materialization_cache_column_count_ = cache_column_count;
    }

    if (!materialization_events_.empty() &&
        materialization_events_.back().stage == event.stage) {
        materialization_events_.back() = std::move(event);
    } else {
        materialization_events_.push_back(std::move(event));
        if (materialization_events_.size() > 24) {
            materialization_events_.erase(materialization_events_.begin());
        }
    }
}

void TrainingPlotPanel::SetMaterializationComplete(
    const std::string& output_dataset,
    int operators_applied,
    const std::string& status) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    materialization_output_dataset_ = output_dataset;
    materialization_status_ = status.empty() ? "completed" : status;
    materialization_operators_applied_ = operators_applied;

    MaterializationProgress event;
    event.stage = "Complete";
    event.message = "Materialization completed";
    event.progress = 1.0f;
    materialization_events_.push_back(std::move(event));
    if (materialization_events_.size() > 24) {
        materialization_events_.erase(materialization_events_.begin());
    }
}

void TrainingPlotPanel::RenderLossPlot() {
    // Keep loss and accuracy visible together during active training. If both
    // plots are enabled, each plot shrinks instead of pushing the other below
    // the fold; the user needs both curves to judge overfitting.
    float available_height = ImGui::GetContentRegionAvail().y;
    const bool show_pair =
        show_accuracy_plot_ && !train_accuracy_.values.empty();
    float plot_height = show_pair
        ? std::clamp((available_height - 36.0f) * 0.50f,
                     kTrainingPlotMinHeight, kTrainingPlotMaxHeight)
        : std::max(kTrainingPlotMinHeight, available_height - 90.0f);

    if (ImPlot::BeginPlot("Loss", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        if (log_loss_scale_) {
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        }

        if (auto_scale_ && !train_loss_.epochs.empty()) {
            const auto [min_epoch, max_epoch] = CalculateEpochWindow(train_loss_);
            ImPlot::SetupAxisLimits(
                ImAxis_X1, min_epoch, max_epoch,
                follow_current_epoch_ ? ImGuiCond_Always : ImGuiCond_Once);

            ValueRange range = CalculateVisibleRange(train_loss_, val_loss_, min_epoch, max_epoch);
            if (log_loss_scale_) {
                double min_positive = std::numeric_limits<double>::max();
                const auto include_positive = [&](const MetricSeries& series) {
                    const size_t count =
                        std::min(series.epochs.size(), series.values.size());
                    for (size_t i = 0; i < count; ++i) {
                        if (series.epochs[i] < min_epoch ||
                            series.epochs[i] > max_epoch ||
                            series.values[i] <= 0.0) {
                            continue;
                        }
                        min_positive = std::min(min_positive, series.values[i]);
                    }
                };
                include_positive(train_loss_);
                include_positive(val_loss_);
                if (min_positive == std::numeric_limits<double>::max()) {
                    min_positive = 1.0e-6;
                }
                const double lower = std::max(1.0e-12, min_positive / 1.25);
                const double upper = std::max(lower * 10.0, range.max * 1.25);
                ImPlot::SetupAxisLimits(
                    ImAxis_Y1, lower, upper, ImGuiCond_Always);
            } else {
                double padding = (range.max - range.min) * 0.1;
                if (padding < 0.01) padding = 0.1;
                ImPlot::SetupAxisLimits(
                    ImAxis_Y1,
                    std::max(0.0, range.min - padding),
                    range.max + padding,
                    ImGuiCond_Always);
            }
        }

        // Plot training loss
        if (!train_loss_.values.empty()) {
            ImPlot::SetNextLineStyle(train_loss_.color, 2.0f);
            ImPlot::PlotLine(train_loss_.name.c_str(),
                           train_loss_.epochs.data(),
                           train_loss_.values.data(),
                           static_cast<int>(train_loss_.values.size()));
            if (show_smoothed_curves_ &&
                static_cast<int>(train_loss_.values.size()) >= smoothing_window_) {
                auto smoothed =
                    CalculateMovingAverage(train_loss_.values, smoothing_window_);
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.65f, 0.65f, 1.0f), 3.0f);
                ImPlot::PlotLine("Training Loss (smoothed)",
                                 train_loss_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        // Plot validation loss
        if (!val_loss_.values.empty()) {
            ImPlot::SetNextLineStyle(val_loss_.color, 2.0f);
            ImPlot::PlotLine(val_loss_.name.c_str(),
                           val_loss_.epochs.data(),
                           val_loss_.values.data(),
                           static_cast<int>(val_loss_.values.size()));
            ImPlot::SetNextMarkerStyle(
                ImPlotMarker_Circle, 5.0f, val_loss_.color,
                1.5f, val_loss_.color);
            ImPlot::PlotScatter("##Validation Loss Points",
                                val_loss_.epochs.data(),
                                val_loss_.values.data(),
                                static_cast<int>(val_loss_.values.size()));
            if (show_smoothed_curves_ &&
                static_cast<int>(val_loss_.values.size()) >= smoothing_window_) {
                auto smoothed =
                    CalculateMovingAverage(val_loss_.values, smoothing_window_);
                ImPlot::SetNextLineStyle(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), 3.0f);
                ImPlot::PlotLine("Validation Loss (smoothed)",
                                 val_loss_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void TrainingPlotPanel::RenderAccuracyPlot() {
    // Share the dashboard area with the loss plot instead of requiring a tall
    // window before accuracy becomes visible.
    float available_height = ImGui::GetContentRegionAvail().y;
    const bool show_pair =
        show_loss_plot_ && !train_loss_.values.empty();
    float plot_height = show_pair
        ? std::clamp(available_height - 40.0f,
                     kTrainingPlotMinHeight, kTrainingPlotMaxHeight)
        : std::max(kTrainingPlotMinHeight, available_height - 90.0f);

    if (ImPlot::BeginPlot("Accuracy", ImVec2(-1, plot_height))) {
        ImPlot::SetupAxes("Epoch", "Accuracy (%)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

        if (auto_scale_ && follow_current_epoch_ && !train_accuracy_.epochs.empty()) {
            const auto [min_epoch, max_epoch] = CalculateEpochWindow(train_accuracy_);
            ImPlot::SetupAxisLimits(ImAxis_X1, min_epoch, max_epoch, ImGuiCond_Always);

            ValueRange range = CalculateVisibleRange(train_accuracy_, val_accuracy_, min_epoch, max_epoch);
            double padding = (range.max - range.min) * 0.1;
            if (padding < 1.0) padding = 5.0;
            ImPlot::SetupAxisLimits(ImAxis_Y1,
                std::max(0.0, range.min - padding),
                std::min(100.0, range.max + padding),
                ImGuiCond_Always);
        } else if (auto_scale_ && !train_accuracy_.epochs.empty()) {
            const double max_epoch = std::max(1.0, train_accuracy_.epochs.back());
            ImPlot::SetupAxisLimits(
                ImAxis_X1, 0.0, max_epoch + 1.0, ImGuiCond_Once);
        }

        // Plot training accuracy
        if (!train_accuracy_.values.empty()) {
            ImPlot::SetNextLineStyle(train_accuracy_.color, 2.0f);
            ImPlot::PlotLine(train_accuracy_.name.c_str(),
                           train_accuracy_.epochs.data(),
                           train_accuracy_.values.data(),
                           static_cast<int>(train_accuracy_.values.size()));
            if (show_smoothed_curves_ &&
                static_cast<int>(train_accuracy_.values.size()) >= smoothing_window_) {
                auto smoothed = CalculateMovingAverage(
                    train_accuracy_.values, smoothing_window_);
                ImPlot::SetNextLineStyle(ImVec4(0.65f, 1.0f, 0.65f, 1.0f), 3.0f);
                ImPlot::PlotLine("Training Accuracy (smoothed)",
                                 train_accuracy_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        // Plot validation accuracy
        if (!val_accuracy_.values.empty()) {
            ImPlot::SetNextLineStyle(val_accuracy_.color, 2.0f);
            ImPlot::PlotLine(val_accuracy_.name.c_str(),
                           val_accuracy_.epochs.data(),
                           val_accuracy_.values.data(),
                           static_cast<int>(val_accuracy_.values.size()));
            if (show_smoothed_curves_ &&
                static_cast<int>(val_accuracy_.values.size()) >= smoothing_window_) {
                auto smoothed =
                    CalculateMovingAverage(val_accuracy_.values, smoothing_window_);
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.9f, 0.45f, 1.0f), 3.0f);
                ImPlot::PlotLine("Validation Accuracy (smoothed)",
                                 val_accuracy_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void TrainingPlotPanel::RenderCustomMetricsPlot() {
    bool has_sequence_metrics = false;
    bool has_regression_metrics = false;
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
        if (IsRegressionMetricName(metric.name)) {
            has_regression_metrics = true;
        }
    }

    const char* plot_title =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Sequence Metrics"
            : (has_regression_metrics ? "Regression Metrics" : "Custom Metrics");
    const char* y_label =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Score (%)"
            : (has_regression_metrics ? "Error" : "Value");

    if (ImPlot::BeginPlot(
            plot_title, ImVec2(-1, kTrainingPlotMinHeight))) {
        // Enable zoom and pan on both axes
        ImPlot::SetupAxes("Epoch", y_label, ImPlotAxisFlags_None, ImPlotAxisFlags_None);

        for (const auto& metric : custom_metrics_) {
            if (!metric.values.empty()) {
                ImPlot::SetNextLineStyle(metric.color, 2.0f);
                ImPlot::PlotLine(metric.name.c_str(),
                               metric.epochs.data(),
                               metric.values.data(),
                               static_cast<int>(metric.values.size()));
                if (IsValidationMetricName(metric.name)) {
                    ImPlot::SetNextMarkerStyle(
                        ImPlotMarker_Circle, 5.0f, metric.color,
                        1.5f, metric.color);
                    const std::string point_id =
                        "##" + metric.name + " Points";
                    ImPlot::PlotScatter(
                        point_id.c_str(), metric.epochs.data(),
                        metric.values.data(),
                        static_cast<int>(metric.values.size()));
                }
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
        ClearLocked();
    }
    ImGui::SameLine();

    if (ImGui::Button("Export CSV")) {
        ExportToCSVLocked("training_metrics.csv");
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
    ImGui::Checkbox("Log Loss", &log_loss_scale_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Use a logarithmic loss axis to compare large early losses with "
            "smaller later losses. Metric values are unchanged.");
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show Accuracy", &show_accuracy_plot_);
    ImGui::SameLine();
    ImGui::Checkbox("Smooth", &show_smoothed_curves_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Display-only moving average overlay. Raw curves are unchanged.");
    }
    if (show_smoothed_curves_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::SliderInt("Smooth Window", &smoothing_window_, 2, 50);
    }

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

void TrainingPlotPanel::RenderActiveTaskSummary() {
#ifndef CYXWIZ_PLOTTING_MODULE
    auto tasks = AsyncTaskManager::Instance().GetActiveTasks();
    if (tasks.empty()) {
        return;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Active engine tasks:");
    int rendered = 0;
    for (const auto& task : tasks) {
        if (rendered >= 3) {
            ImGui::TextDisabled("+ %d more task(s)",
                                static_cast<int>(tasks.size()) - rendered);
            break;
        }

        ImGui::BulletText("%s", task.name.c_str());
        ImGui::SameLine(220);
        ImGui::ProgressBar(task.progress, ImVec2(160, 0));
        ImGui::SameLine();
        if (!task.status_message.empty()) {
            ImGui::TextDisabled("%s", task.status_message.c_str());
        } else {
            ImGui::TextDisabled("running");
        }
        ++rendered;
    }
#endif
}

void TrainingPlotPanel::RenderMaterializationSummary() {
    if (materialization_events_.empty()) {
        return;
    }

    ImGui::Spacing();
    if (!ImGui::CollapsingHeader("Materialization",
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const auto& latest = materialization_events_.back();
    ImGui::TextColored(ImVec4(0.35f, 0.75f, 1.0f, 1.0f),
                       "Latest stage: %s", latest.stage.c_str());

    if (!latest.node_name.empty()) {
        ImGui::TextWrapped("Node: %s", latest.node_name.c_str());
    }

    if (!materialization_status_.empty()) {
        const ImVec4 color = materialization_status_ == "completed"
            ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
            : ImVec4(1.0f, 0.75f, 0.25f, 1.0f);
        ImGui::TextColored(color, "Status: %s", materialization_status_.c_str());
    }
    if (!materialization_output_dataset_.empty()) {
        ImGui::TextWrapped("Output dataset: %s",
                           materialization_output_dataset_.c_str());
    }
    if (materialization_operators_applied_ > 0) {
        ImGui::Text("Operators: %d applied",
                    materialization_operators_applied_);
    }
    if (!materialization_cache_key_.empty()) {
        const std::string short_key = materialization_cache_key_.substr(
            0, std::min<size_t>(12, materialization_cache_key_.size()));
        ImGui::TextWrapped("Cache key: %s", materialization_cache_key_.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("short %s", short_key.c_str());
    }
    if (!materialization_cache_artifact_path_.empty()) {
        ImGui::TextWrapped("Prepared dataset artifact: %s",
                           materialization_cache_artifact_path_.c_str());
        if (ImGui::SmallButton("Copy artifact path")) {
            ImGui::SetClipboardText(materialization_cache_artifact_path_.c_str());
        }
    }
    if (!materialization_cache_manifest_path_.empty()) {
        ImGui::TextWrapped("Cache manifest: %s",
                           materialization_cache_manifest_path_.c_str());
        if (ImGui::SmallButton("Copy manifest path")) {
            ImGui::SetClipboardText(materialization_cache_manifest_path_.c_str());
        }
    }
    const std::string cache_location_source =
        !materialization_cache_manifest_path_.empty()
            ? materialization_cache_manifest_path_
            : materialization_cache_artifact_path_;
    if (!cache_location_source.empty()) {
        const std::string cache_directory =
            ParentDirectoryForPath(cache_location_source);
        ImGui::SameLine();
        if (ImGui::SmallButton("Open cache location")) {
            if (OpenDirectoryInFileBrowser(cache_directory)) {
                RecordPanelEvent("TrainingPlotPanel.OpenCacheLocation",
                                 cache_directory);
            } else {
                RecordPanelEvent("TrainingPlotPanel.OpenCacheLocationFailed",
                                 cache_directory);
            }
        }
    }
    if (materialization_cache_row_count_ > 0 ||
        materialization_cache_column_count_ > 0) {
        ImGui::Text("Prepared dataset: %lld rows, %lld columns",
                    static_cast<long long>(materialization_cache_row_count_),
                    static_cast<long long>(materialization_cache_column_count_));
    }
    ImGui::TextWrapped("Message: %s", latest.message.c_str());
    ImGui::ProgressBar(latest.progress, ImVec2(-1.0f, 0.0f));

    if (latest.estimated_memory_bytes > 0 ||
        latest.total_items > 0 ||
        latest.processed_items > 0) {
        if (latest.estimated_memory_bytes > 0) {
            ImGui::Text("Estimated memory: %s",
                        FormatTraceBytes(latest.estimated_memory_bytes).c_str());
        }
        if (!latest.memory_risk_level.empty()) {
            ImGui::Text("Memory risk: %s", latest.memory_risk_level.c_str());
        }
        if (!latest.status.empty() && latest.status != "running") {
            ImGui::Text("Decision status: %s", latest.status.c_str());
        }
        if (latest.total_items > 0) {
            ImGui::Text("Work: %llu / %llu",
                        static_cast<unsigned long long>(latest.processed_items),
                        static_cast<unsigned long long>(latest.total_items));
        }
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Stages");
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
    int stage_index = 1;
    for (const auto& event : materialization_events_) {
        ImGui::PushID(stage_index);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.85f, 0.92f, 1.0f, 1.0f),
                           "%d. %s", stage_index, event.stage.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(%.0f%%)", event.progress * 100.0f);

        if (!event.node_name.empty()) {
            ImGui::TextWrapped("Node: %s", event.node_name.c_str());
        }
        if (event.estimated_memory_bytes > 0) {
            ImGui::Text("Estimated memory: %s",
                        FormatTraceBytes(event.estimated_memory_bytes).c_str());
        }
        if (!event.memory_risk_level.empty()) {
            ImGui::Text("Memory risk: %s", event.memory_risk_level.c_str());
        }
        if (!event.status.empty() && event.status != "running") {
            ImGui::Text("Decision status: %s", event.status.c_str());
        }
        if (!event.cache_key.empty()) {
            ImGui::TextWrapped("Cache key: %s", event.cache_key.c_str());
        }
        if (!event.cache_artifact_path.empty()) {
            ImGui::TextWrapped("Prepared dataset artifact: %s",
                               event.cache_artifact_path.c_str());
        }
        if (!event.cache_manifest_path.empty()) {
            ImGui::TextWrapped("Cache manifest: %s",
                               event.cache_manifest_path.c_str());
        }
        if (event.cache_row_count > 0 || event.cache_column_count > 0) {
            ImGui::Text("Prepared dataset: %lld rows, %lld columns",
                        static_cast<long long>(event.cache_row_count),
                        static_cast<long long>(event.cache_column_count));
        }
        if (event.total_items > 0 || event.processed_items > 0) {
            if (event.total_items > 0) {
                ImGui::Text("Work: %llu / %llu",
                            static_cast<unsigned long long>(event.processed_items),
                            static_cast<unsigned long long>(event.total_items));
            } else {
                ImGui::Text("Processed: %llu",
                            static_cast<unsigned long long>(event.processed_items));
            }
        }
        if (!event.message.empty()) {
            ImGui::TextWrapped("Message: %s", event.message.c_str());
        }
        ImGui::ProgressBar(event.progress, ImVec2(-1.0f, 0.0f));
        ImGui::Separator();
        ImGui::PopID();
        ++stage_index;
    }
    ImGui::PopTextWrapPos();
}

void TrainingPlotPanel::RenderTrainingWarningSummary() {
#ifndef CYXWIZ_PLOTTING_MODULE
    const auto trace = TrainingTraceCollector::Instance().Snapshot();
    if (!trace.available) {
        return;
    }

    const TrainingTraceEvent* transfer = FindLatestPinMemoryTransferEvent(trace);
    if (!transfer && trace.warnings.empty()) {
        return;
    }

    ImGui::Spacing();
    if (transfer) {
        const ImVec4 color = transfer->status == "warning"
            ? ImVec4(1.0f, 0.82f, 0.35f, 1.0f)
            : ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
        ImGui::TextColored(color, "Pin-memory transfer:");
        ImGui::SameLine(170);
        ImGui::TextWrapped(
            "mode=%s reason=%s backend=%s batch=%d",
            transfer->transfer_mode.empty()
                ? "unknown"
                : transfer->transfer_mode.c_str(),
            transfer->transfer_reason.empty()
                ? "unknown"
                : transfer->transfer_reason.c_str(),
            transfer->transfer_backend.empty()
                ? "unknown"
                : transfer->transfer_backend.c_str(),
            transfer->transfer_batch_size);
    }

    bool rendered_warning_header = false;
    if (!trace.warnings.empty()) {
        const int start =
            std::max(0, static_cast<int>(trace.warnings.size()) - 3);
        for (int i = start; i < static_cast<int>(trace.warnings.size()); ++i) {
            const auto& warning = trace.warnings[i];
            if (transfer && IsPinMemoryTransferWarning(warning)) {
                continue;
            }
            if (!rendered_warning_header) {
                ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.35f, 1.0f),
                                   "Training warnings:");
                rendered_warning_header = true;
            }
            ImGui::BulletText("%s", ClassifyTrainingWarning(warning));
            ImGui::SameLine(170);
            ImGui::TextWrapped("%s", warning.c_str());
        }
    }
#endif
}

void TrainingPlotPanel::RenderRunComparisonTable() {
    ImGui::Separator();
    if (!ImGui::CollapsingHeader("Run Comparison",
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (run_comparison_records_.empty()) {
        ImGui::TextDisabled(
            "No completed training runs recorded in this session.");
        ImGui::TextDisabled(
            "Completed graph training runs will appear here for comparison.");
        if (active_checkpoint_loaded_) {
            ImGui::TextDisabled(
                "The loaded checkpoint is active for testing, but loading is "
                "not a new training run and is not inserted here.");
        }
        return;
    }

    if (ImGui::Button("Export Run CSV")) {
        std::string error;
        if (!WriteTrainingRunComparisonCsv(
                "training_run_comparison.csv",
                run_comparison_records_,
                &error)) {
            RecordPanelEvent("TrainingPlotPanel.ExportRunComparisonFailed",
                             error);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Runs")) {
        run_comparison_records_.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled(
        "Sorted by test accuracy, then validation metrics, then elapsed time.");
    ImGui::TextDisabled(
        "Partition Match is relative to the top-ranked run; different manifests "
        "are not directly comparable.");
    if (active_checkpoint_loaded_) {
        ImGui::TextDisabled(
            "A checkpoint loaded for testing is shown under Active model state; "
            "it is not counted as a new run.");
    }

    if (ImGui::BeginTable(
            "TrainingRunComparisonTable",
            27,
            ImGuiTableFlags_Borders |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_ScrollX)) {
        ImGui::TableSetupColumn("Run");
        ImGui::TableSetupColumn("Status");
        ImGui::TableSetupColumn("Dataset");
        ImGui::TableSetupColumn("Domain");
        ImGui::TableSetupColumn("Seq");
        ImGui::TableSetupColumn("Split");
        ImGui::TableSetupColumn("Samples");
        ImGui::TableSetupColumn("Role Sources");
        ImGui::TableSetupColumn("Role Origins");
        ImGui::TableSetupColumn("Role Labels");
        ImGui::TableSetupColumn("Partition ID");
        ImGui::TableSetupColumn("Partition Match");
        ImGui::TableSetupColumn("Role Checks");
        ImGui::TableSetupColumn("Model");
        ImGui::TableSetupColumn("Architecture");
        ImGui::TableSetupColumn("Epochs");
        ImGui::TableSetupColumn("Batch");
        ImGui::TableSetupColumn("LR");
        ImGui::TableSetupColumn("Best Val Loss");
        ImGui::TableSetupColumn("Best Val Acc");
        ImGui::TableSetupColumn("Best Epoch");
        ImGui::TableSetupColumn("Test Loss");
        ImGui::TableSetupColumn("Test Acc");
        ImGui::TableSetupColumn("Elapsed");
        ImGui::TableSetupColumn("Best Ckpt");
        ImGui::TableSetupColumn("Patience");
        ImGui::TableSetupColumn("Checkpoint");
        ImGui::TableHeadersRow();

        const auto& partition_reference = run_comparison_records_.front();
        for (const auto& record : run_comparison_records_) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.run_id.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.run_status.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.dataset_name.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.preprocessing_domain.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.sequence_batch_enabled ? "yes" : "no");

            ImGui::TableNextColumn();
            ImGui::Text("%.0f/%.0f/%.0f%%",
                record.train_ratio * 100.0f,
                record.val_ratio * 100.0f,
                record.test_ratio * 100.0f);

            ImGui::TableNextColumn();
            ImGui::Text("%zu/%zu/%zu",
                record.train_sample_count,
                record.val_sample_count,
                record.test_sample_count);

            const std::string role_sources = record.train_source_name + " / " +
                record.dev_source_name + " / " + record.test_source_name;
            const std::string role_origins = record.train_origin + " / " +
                record.dev_origin + " / " + record.test_origin;
            const std::string role_labels = record.train_label_column + " / " +
                record.dev_label_column + " / " + record.test_label_column;
            const std::string partition_display =
                record.partition_manifest_fingerprint.empty()
                    ? std::string("-")
                    : record.partition_manifest_fingerprint.substr(0, 8);

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(role_sources.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(role_origins.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(role_labels.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(partition_display.c_str());
            if (!record.partition_manifest_fingerprint.empty() &&
                ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s",
                                  record.partition_manifest_fingerprint.c_str());
            }

            const auto partition_compatibility =
                CompareTrainingRunPartitions(partition_reference, record);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(
                TrainingRunPartitionCompatibilityLabel(partition_compatibility));
            if (partition_compatibility ==
                    TrainingRunPartitionCompatibility::DifferentManifest &&
                ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Not directly comparable: this run used a different "
                    "partition manifest than the top-ranked run.");
            }

            ImGui::TableNextColumn();
            const std::string role_checks =
                "Dev " + record.dev_schema_compatibility + "/" +
                record.dev_leakage_status + " | Test " +
                record.test_schema_compatibility + "/" +
                record.test_leakage_status;
            ImGui::TextUnformatted(role_checks.c_str());
            if (ImGui::IsItemHovered() &&
                (!record.dev_partition_status_reason.empty() ||
                 !record.test_partition_status_reason.empty())) {
                ImGui::SetTooltip(
                    "Dev: %s\nTest: %s",
                    record.dev_partition_status_reason.empty()
                        ? "no additional detail"
                        : record.dev_partition_status_reason.c_str(),
                    record.test_partition_status_reason.empty()
                        ? "no additional detail"
                        : record.test_partition_status_reason.c_str());
            }

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.model_family.c_str());

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(record.architecture_summary.c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%d", record.epochs);

            ImGui::TableNextColumn();
            ImGui::Text("%d", record.batch_size);

            ImGui::TableNextColumn();
            ImGui::Text("%.6f", record.learning_rate);

            ImGui::TableNextColumn();
            if (record.has_validation_metrics) {
                ImGui::Text("%.4f", record.best_val_loss);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            if (record.has_validation_metrics) {
                ImGui::Text("%.2f%%", record.best_val_accuracy * 100.0f);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            if (record.has_validation_metrics) {
                ImGui::Text("loss %d / acc %d",
                    record.best_val_loss_epoch,
                    record.best_val_accuracy_epoch);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            if (record.has_test_metrics) {
                ImGui::Text("%.4f", record.final_test_loss);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            if (record.has_test_metrics) {
                ImGui::Text("%.2f%%", record.final_test_accuracy * 100.0f);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            ImGui::Text("%.1fs", record.elapsed_seconds);

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(
                record.save_best_checkpoint ? "yes" : "no");

            ImGui::TableNextColumn();
            if (record.early_stopping_patience > 0) {
                ImGui::Text("%d", record.early_stopping_patience);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableNextColumn();
            if (!record.checkpoint_used.empty()) {
                ImGui::TextUnformatted(record.checkpoint_used.c_str());
            } else {
                ImGui::TextDisabled("-");
            }
        }

        ImGui::EndTable();
    }
}

void TrainingPlotPanel::RenderTrainingStatus() {
    // Training status header with colored indicator
    ImGui::BeginGroup();

    // Status indicator
    if (is_training_) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
        ImGui::Text("TRAINING");
        ImGui::PopStyleColor();
    } else if (preparation_failed_) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.35f, 0.30f, 1.0f));
        ImGui::Text("PREPARATION FAILED");
        ImGui::PopStyleColor();
    } else if (is_preparing_) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f, 0.75f, 1.0f, 1.0f));
        ImGui::Text("PREPARING");
        ImGui::PopStyleColor();
    } else if (total_training_time_ > 0) {
        const bool early_stopped = terminal_status_ == "early_stopped";
        const bool cancelled = terminal_status_ == "cancelled" ||
                               terminal_status_ == "stopped";
        const bool failed = terminal_status_ == "failed";
        if (early_stopped) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
            ImGui::Text("EARLY STOPPED");
        } else if (cancelled) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.55f, 0.35f, 1.0f));
            ImGui::Text("CANCELLED");
        } else if (failed) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::Text("FAILED");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("COMPLETED");
        }
        ImGui::PopStyleColor();
    } else if (active_checkpoint_loaded_) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.85f, 0.55f, 1.0f));
        ImGui::Text("MODEL LOADED");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        ImGui::Text("IDLE");
        ImGui::PopStyleColor();
    }

    ImGui::SameLine();

    // Progress info. Keep terminal-state explanations as explicit rows instead
    // of hiding them on one compressed line.
    if (preparation_failed_) {
        ImGui::TextWrapped("%s",
                           preparation_error_message_.empty()
                               ? "Training preparation failed."
                               : preparation_error_message_.c_str());
    } else if (is_preparing_) {
        ImGui::TextWrapped("%s",
                           preparation_status_message_.empty()
                               ? "Preparing training..."
                               : preparation_status_message_.c_str());
        ImGui::ProgressBar(preparation_progress_, ImVec2(-1.0f, 0.0f));
    } else if (total_epochs_ > 0) {
        const float progress =
            static_cast<float>(current_epoch_) / std::max(1, total_epochs_);
        if (!is_training_ && total_training_time_ > 0) {
            ImGui::Text("Final epoch state: %d / %d",
                        current_epoch_, total_epochs_);
        } else {
            const int display_epoch = std::max(1, current_epoch_);
            const int remaining_epochs =
                std::max(0, total_epochs_ - display_epoch);
            ImGui::Text("Epoch %d / %d (%d remaining)",
                        display_epoch, total_epochs_, remaining_epochs);
        }
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f));
    } else if (is_training_) {
        ImGui::Text("Epoch %d / ?", std::max(1, current_epoch_));
    }

    if (!is_training_ && total_training_time_ > 0) {
        const bool early_stopped = terminal_status_ == "early_stopped";
        const bool cancelled = terminal_status_ == "cancelled" ||
                               terminal_status_ == "stopped";
        const bool failed = terminal_status_ == "failed";

        ImGui::Spacing();
        if (early_stopped) {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f),
                               "Stop reason: early stopping triggered.");
        } else if (cancelled) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
                               "Stop reason: training was cancelled.");
        } else if (failed) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               "Stop reason: training failed.");
        } else {
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f),
                               "Stop reason: training reached its terminal epoch.");
        }

        if (!terminal_reason_.empty()) {
            ImGui::TextWrapped("%s", terminal_reason_.c_str());
        }
    }

    if (!is_training_ && !checkpoint_used_.empty()) {
        ImGui::Spacing();
        ImGui::SeparatorText("Active model state");
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.55f, 1.0f),
                           active_checkpoint_loaded_
                               ? "Checkpoint loaded for testing"
                               : "Best validation checkpoint restored");
        if (checkpoint_epoch_ > 0 && has_checkpoint_validation_metrics_) {
            ImGui::Text("Checkpoint epoch: %d", checkpoint_epoch_);
            ImGui::Text("Checkpoint validation: loss %.4f, accuracy %.2f%%",
                        checkpoint_val_loss_,
                        checkpoint_val_accuracy_ * 100.0f);
        } else if (checkpoint_epoch_ > 0) {
            ImGui::Text("Checkpoint epoch: %d", checkpoint_epoch_);
        }
        ImGui::TextWrapped("Path: %s", checkpoint_used_.c_str());
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

    ImGui::EndGroup();
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

std::vector<double> TrainingPlotPanel::CalculateMovingAverage(
    const std::vector<double>& values,
    int window) const {
    std::vector<double> smoothed;
    smoothed.reserve(values.size());
    const int safe_window = std::max(1, window);
    double running_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        running_sum += values[i];
        if (i >= static_cast<size_t>(safe_window)) {
            running_sum -= values[i - static_cast<size_t>(safe_window)];
        }
        const size_t denom = std::min(
            i + 1,
            static_cast<size_t>(safe_window));
        smoothed.push_back(running_sum / static_cast<double>(denom));
    }
    return smoothed;
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

TrainingStatusSnapshot TrainingPlotPanel::GetStatusSnapshot() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    TrainingStatusSnapshot snapshot;
    snapshot.has_data = !train_loss_.values.empty();
    snapshot.is_training = is_training_;
    snapshot.is_preparing = is_preparing_;
    snapshot.preparation_failed = preparation_failed_;
    snapshot.status_message = preparation_failed_
        ? preparation_error_message_
        : preparation_status_message_;
    snapshot.terminal_status = terminal_status_;
    snapshot.current_epoch = current_epoch_;
    snapshot.total_epochs = total_epochs_;
    snapshot.current_batch = current_batch_;
    snapshot.total_batches = total_batches_;
    snapshot.train_loss = train_loss_.values.empty()
        ? -1.0
        : train_loss_.values.back();
    snapshot.val_loss = val_loss_.values.empty()
        ? -1.0
        : val_loss_.values.back();
    snapshot.train_accuracy = train_accuracy_.values.empty()
        ? -1.0
        : train_accuracy_.values.back();
    snapshot.val_accuracy = val_accuracy_.values.empty()
        ? -1.0
        : val_accuracy_.values.back();
    snapshot.preparation_progress = preparation_progress_;
    snapshot.samples_per_second = samples_per_second_;
    snapshot.total_training_time = total_training_time_;
    snapshot.checkpoint_epoch = checkpoint_epoch_;
    snapshot.metric_points = train_loss_.values.size();
    snapshot.latest_custom_metrics.reserve(custom_metrics_.size());
    for (const auto& metric : custom_metrics_) {
        if (!metric.values.empty()) {
            snapshot.latest_custom_metrics.emplace_back(
                metric.name,
                metric.values.back());
        }
    }
    return snapshot;
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
