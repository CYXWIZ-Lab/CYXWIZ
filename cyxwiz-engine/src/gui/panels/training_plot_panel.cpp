#include "../ui_tokens.h"
#include "../../core/series_decimation.h"
#include "training_plot_panel.h"
#include "../missing_values_view.h"
#include "../../core/training_dashboard_presentation.h"
#include "../separate_windows.h"
// Test builds that draw the dashboard without the training system define
// CYXWIZ_DASHBOARD_WITHOUT_TRAINING (no TrainingManager, task manager or crash recorder).
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
#include "../../core/async_task_manager.h"
#include "../../core/crash_run_recorder.h"
#include "../../core/execution_policy.h"
#include "../../core/training_manager.h"
#include "../../core/training_trace_collector.h"
#endif
#include "../../core/training_run_comparison.h"
#include "../../core/route_qualification_snapshot.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
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
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
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

constexpr float kTrainingPlotMinHeight = 320.0f;
constexpr float kTrainingPlotMaxHeight = 480.0f;
// Loss and accuracy go side by side from this content width.
constexpr float kTrainingPlotSideBySideWidth = 860.0f;

// Title and y-axis label (when asked) of the custom metrics chart from the
// metric names.
template <typename Series>
void ClassifyCustomMetrics(const std::vector<Series>& metrics,
                           const char** plot_title,
                           const char** y_label) {
    bool has_sequence_metrics = false;
    bool has_regression_metrics = false;
    bool has_non_sequence_metrics = false;
    for (const auto& metric : metrics) {
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
    *plot_title =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Sequence Metrics"
            : (has_regression_metrics ? "Regression Metrics" : "Custom Metrics");
    if (!y_label) return;
    *y_label =
        has_sequence_metrics && !has_non_sequence_metrics
            ? "Score (%)"
            : (has_regression_metrics ? "Error" : "Value");
}

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

const char* MaterializationStatusDisplayName(const std::string& status) {
    if (status == "cache_hit") {
        return "Cache hit - preprocessing skipped";
    }
    if (status == "cache_saved") {
        return "Prepared and cached";
    }
    if (status == "cache_miss") {
        return "Cache miss - preprocessing rebuilt";
    }
    if (status == "cache_stale") {
        return "Cache stale - preprocessing rebuilt";
    }
    if (status == "cache_corrupt") {
        return "Cache invalid - preprocessing rebuilt";
    }
    if (status == "cache_save_failed") {
        return "Prepared; cache save failed";
    }
    if (status == "cache_unsupported") {
        return "Cache unavailable";
    }
    return status.empty() ? "Completed" : status.c_str();
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

// The header and Run details read the execution trace (training builds only).
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
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

const char* ResidencyVerdictDisplayName(const std::string& verdict) {
    if (verdict == "strict_arrayfire_declared_boundaries") {
        return "Strict ArrayFire residency";
    }
    if (verdict == "compatibility_completed") {
        return "Compatibility; completed";
    }
    if (verdict == "terminal_without_residency_pass") {
        return "Residency not proven";
    }
    if (verdict == "in_progress") {
        return "In progress";
    }
    return verdict.empty() || verdict == "unavailable"
        ? "Not available"
        : verdict.c_str();
}

std::string FormatRouteQualification(
    bool evidence_available,
    bool qualified,
    const std::string& matrix_id,
    const std::string& message) {
    std::string value;
    if (!evidence_available) {
        value = "No evidence";
    } else {
        value = qualified ? "Passed" : "Failed";
    }
    value += " | ";
    value += RouteQualificationEvidenceLabel(matrix_id);
    if (!message.empty()) {
        value += " | " + message;
    }
    return value;
}

// The latest execution trace, read at most every 500 ms: LatestTrace copies
// the whole trace (events, timings) and may read the saved one from disk; it
// ran every frame with data_mutex_ held (TOFIX134 P0 item 8). UI thread only.
const TrainingTraceSummary& LatestTraceCached() {
    static TrainingTraceSummary cached;
    static std::chrono::steady_clock::time_point read_at{};
    const auto now = std::chrono::steady_clock::now();
    if (now - read_at > std::chrono::milliseconds(500)) {
        cached = TrainingTraceCollector::LatestTrace();
        read_at = now;
    }
    return cached;
}

std::optional<training_dashboard::Route> TraceRoute(const TrainingTraceSummary& trace) {
    if (trace.effective_backend.empty()) return std::nullopt;
    training_dashboard::Route route;
    route.requested_backend = trace.requested_backend;
    route.effective_backend = trace.effective_backend;
    route.effective_device_id = trace.effective_device_id;
    route.device_name = trace.effective_device_name;
    route.policy = trace.execution_policy;
    route.fallback = trace.selection_fallback_applied;
    return route;
}

// Run details > Where it ran: Execution Truth's rows, the placement stages
// and the latest pin-memory transfer.
std::vector<training_dashboard::DetailRow> TraceRows(const TrainingTraceSummary& trace) {
    std::vector<training_dashboard::DetailRow> rows;
    if (!trace.available && trace.run_id.empty()) return rows;
    const auto add = [&rows](const char* key, const std::string& value) {
        rows.push_back({key, value.empty() ? std::string("Not recorded") : value});
    };
    const bool has_truth = !trace.requested_backend.empty() || !trace.effective_backend.empty() ||
                           !trace.placement_fingerprint.empty();
    if (has_truth || !trace.residency_verdict.empty()) {
        add("Residency", ResidencyVerdictDisplayName(trace.residency_verdict));
        if (!trace.residency_verdict.empty() && trace.residency_verdict != "unavailable")
            add("Verdict code", trace.residency_verdict);
        add("Requested backend", trace.requested_backend);
        add("Requested qualification",
            FormatRouteQualification(trace.requested_qualification_evidence_available, trace.requested_route_qualified,
                                     trace.requested_qualification_matrix_id, trace.requested_qualification_message));
        std::string effective = trace.effective_backend;
        if (!trace.effective_device_name.empty()) effective += " | " + trace.effective_device_name;
        if (!trace.effective_backend.empty()) effective += " | device " + std::to_string(trace.effective_device_id);
        add("Effective backend", effective);
        add("Effective qualification",
            FormatRouteQualification(trace.effective_qualification_evidence_available, trace.effective_route_qualified,
                                     trace.effective_qualification_matrix_id, trace.effective_qualification_message));
        std::string identity = trace.identity_confidence;
        if (!trace.physical_fingerprint.empty()) identity += (identity.empty() ? "" : " | ") + trace.physical_fingerprint;
        add("Physical identity", identity);
        add("Execution preflight", trace.execution_validated
                                       ? std::string("Validated")
                                       : (trace.preflight_stage.empty() ? std::string("Not recorded")
                                                                        : "Not validated | " + trace.preflight_stage));
        if (trace.selection_fallback_applied) add("Selection fallback", "Applied to ArrayFire CPU");
        add("Execution policy", RecordedExecutionPolicyDisplayName(trace.execution_policy));
        std::string placement = trace.placement_fingerprint;
        if (!placement.empty()) placement += " | " + std::to_string(trace.placement_entry_count) + " entries";
        add("Placement", placement);
        add("Host boundaries", std::to_string(trace.declared_output_boundary_count));
        add("Transfers", std::to_string(trace.transfer_event_count) + " events | " +
                             FormatTraceBytes(trace.transfer_known_bytes));
        add("Transfer reasons", trace.transfer_summary.empty() ? "None recorded" : trace.transfer_summary);
        add("Synchronizations", std::to_string(trace.synchronization_event_count) + " events | " +
                                    FormatTraceBytes(trace.synchronization_known_bytes));
        add("Sync reasons", trace.synchronization_summary.empty() ? "None recorded" : trace.synchronization_summary);
        if (!trace.placement_summary.empty()) add("Placement stages", trace.placement_summary);
    }
    if (const TrainingTraceEvent* transfer = FindLatestPinMemoryTransferEvent(trace)) {
        add("Pin-memory transfer", transfer->status == "warning" ? "with a warning" : "done");
        add("Transfer mode", transfer->transfer_mode);
        add("Transfer reason", transfer->transfer_reason);
        add("Transfer backend", transfer->transfer_backend);
        add("Transfer batch size", std::to_string(transfer->transfer_batch_size));
    }
    return rows;
}

// The trace's latest warnings (up to 3) as "Class: text", on the surface.
std::vector<std::string> TraceWarnings(const TrainingTraceSummary& trace) {
    std::vector<std::string> lines;
    if (!trace.available && trace.run_id.empty()) return lines;
    const bool has_transfer = FindLatestPinMemoryTransferEvent(trace) != nullptr;
    const int start = std::max(0, static_cast<int>(trace.warnings.size()) - 3);
    for (int i = start; i < static_cast<int>(trace.warnings.size()); ++i) {
        const std::string& warning = trace.warnings[static_cast<size_t>(i)];
        if (has_transfer && IsPinMemoryTransferWarning(warning)) continue;
        lines.push_back(std::string(ClassifyTrainingWarning(warning)) + ": " + warning);
    }
    return lines;
}
#endif

// One group of Run details: its title and key / value rows, the key column
// as wide as its longest key, the values wrapped.
void DetailGroup(const char* title, const std::vector<training_dashboard::DetailRow>& rows) {
    if (rows.empty()) return;
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    cyxwiz::ui::SettingsSection(title);
    float key_width = 0.0f;
    for (const auto& row : rows) key_width = std::max(key_width, ImGui::CalcTextSize(row.key.c_str()).x);
    key_width = std::min(key_width, ImGui::GetContentRegionAvail().x * 0.45f);
    if (!ImGui::BeginTable(title, 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) return;
    ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, key_width);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    for (const auto& row : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", row.key.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        cyxwiz::ui::TextWrapAnywhere(row.value, t.text);  // paths and cache keys break too
    }
    ImGui::EndTable();
}

// A line and its muted tail ("(188 rows the model never saw)"): together
// when they fit, the tail on the next line otherwise; both wrap.
void LineWithTail(const std::string& text, const ImVec4& colour, const std::string& tail, const ImVec4& tail_colour) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    if (!tail.empty()) {
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float end = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + spacing;
        if (end + ImGui::CalcTextSize(tail.c_str()).x <= ImGui::GetWindowContentRegionMax().x) ImGui::SameLine();
        ImGui::TextColored(tail_colour, "%s", tail.c_str());
    }
    ImGui::PopTextWrapPos();
}

// A chip that turns a choice on and off (the Charts row, Rebuild on next
// run): ui::ToggleChip in the accent text colour. True when clicked.
bool ChipToggle(const char* label, bool* on) {
    const bool clicked = ui::ToggleChip(label, label, "", *on, ui::ToU32(ui::CurrentTokens().accent_text));
    if (clicked) *on = !*on;
    return clicked;
}

float ChipToggleWidth(const char* label) { return ui::ToggleChipWidth(label, ""); }

// "Epoch window [ 10 ] - +": a label and a typed whole-number field with
// steps (B4-6: no sliders), its range on hover.
void EpochsField(const char* label, const char* id, int* value, int min, int max, const char* hint) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine();
    ui::NumberField(id, value, min, max, 1, 44.0f, hint);
}

} // namespace

namespace {
// The curves keep their roles in theme colours (series index): training
// loss coral, validation loss blue, training accuracy green, validation
// accuracy amber.
constexpr int kTrainLossColour = 4;
constexpr int kValLossColour = 1;
constexpr int kTrainAccuracyColour = 3;
constexpr int kValAccuracyColour = 2;

// The series dots (metric cards, stats table) use the same theme colours as
// the curves. Set every frame: the theme can change at any time.
template <typename Series>
void ApplySeriesColours(Series& train_loss, Series& val_loss, Series& train_accuracy,
                        Series& val_accuracy, std::vector<Series>& custom) {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    train_loss.color = cyxwiz::ui::SeriesColour(t, kTrainLossColour);
    val_loss.color = cyxwiz::ui::SeriesColour(t, kValLossColour);
    train_accuracy.color = cyxwiz::ui::SeriesColour(t, kTrainAccuracyColour);
    val_accuracy.color = cyxwiz::ui::SeriesColour(t, kValAccuracyColour);
    for (size_t i = 0; i < custom.size(); ++i) {
        custom[i].color = cyxwiz::ui::SeriesColour(t, i % 6);
    }
}
}  // namespace

TrainingPlotPanel::TrainingPlotPanel()
    : Panel("Training Dashboard") {

    // Initialize metric series
    // Colours come from the theme each frame (ApplySeriesColours).
    train_loss_.name = "Training Loss";
    val_loss_.name = "Validation Loss";
    train_accuracy_.name = "Training Accuracy";
    val_accuracy_.name = "Validation Accuracy";

    visible_ = true;
    RecordPanelEvent("TrainingPlotPanel.Created");
}

TrainingPlotPanel::~TrainingPlotPanel() {
    RecordPanelEvent("TrainingPlotPanel.Destroyed");
}

void TrainingPlotPanel::Render() {
    RenderWindow();
    RunPendingExport();
}

void TrainingPlotPanel::RenderWindow() {
    if (!visible_) {
        if (last_render_visible_) {
            RecordPanelEvent("TrainingPlotPanel.Hidden");
            last_render_visible_ = false;
        }
        std::lock_guard<std::mutex> lock(data_mutex_);
        ApplySeriesColours(train_loss_, val_loss_, train_accuracy_, val_accuracy_, custom_metrics_);
        RenderChartWindows();
        return;
    }
    if (!last_render_visible_) {
        RecordPanelEvent("TrainingPlotPanel.Visible");
        last_render_visible_ = true;
    }

    // Larger default size for better visibility
    ImGui::SetNextWindowSize(ImVec2(1000, 800), ImGuiCond_FirstUseEver);

    if (focus_requested_) {
        // A node dialog's "Show in Training Dashboard".
        ImGui::SetNextWindowFocus();
        focus_requested_ = false;
    }
    ::gui::NextWindowMayLeave(name_.c_str());
    const bool expanded = ImGui::Begin(name_.c_str(), &visible_);
    ::gui::TabMenu(name_.c_str(), &visible_);
    if (!expanded) {
        ImGui::End();
        std::lock_guard<std::mutex> lock(data_mutex_);
        ApplySeriesColours(train_loss_, val_loss_, train_accuracy_, val_accuracy_, custom_metrics_);
        RenderChartWindows();
        return;
    }

    // Lock data for reading
    std::lock_guard<std::mutex> lock(data_mutex_);
    ApplySeriesColours(train_loss_, val_loss_, train_accuracy_, val_accuracy_, custom_metrics_);

    // The Run switcher once a word-vector run exists; such a run has its own
    // view (training_word_vector_runs.cpp) and the model run's stays as it is.
    if (!word_vector_runs_.Runs().empty()) {
        RenderRunSwitcher();
        if (const WordVectorRun* run = word_vector_runs_.Find(shown_run_)) {
            RenderWordVectorRun(*run);
            ImGui::End();
            RenderChartWindows();
            return;
        }
    }

    // What the header, the cards and the insights read, once a frame
    // (data_mutex_ is held).
    facts_ = DashboardFacts();
    // Header: run status, progress and run actions (always visible).
    RenderTrainingStatus();

    // Generations (TOFIX159 A4): the run's samples, beside the metrics.
    if (!generation_folder_.empty()) {
        generation_view_.SetFolder(generation_folder_);
        generation_view_.Poll();
        if (ImGui::BeginTabBar("##dash_tabs")) {
            if (ImGui::BeginTabItem("Overview")) {
                show_generations_ = false;
                ImGui::EndTabItem();
            }
            const std::string label = "Generations (" + std::to_string(generation_view_.ReportCount()) + ")###generations";
            if (ImGui::BeginTabItem(label.c_str())) {
                show_generations_ = true;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        if (show_generations_) {
            generation_view_.Render();
            ImGui::End();
            RenderChartWindows();
            return;
        }
    }

    // Metric cards: latest values, change and timing.
    RenderKpiCards();
    if (health_notice_) {
        const HealthNotice notice = health_notice_();
        if (notice.show) {
            ui::NoticeLine(notice.text, ui::CurrentTokens().caution);
            if (open_health_ && ui::LinkButton("Open Model > Health")) open_health_();
        }
    }

    // Check if we have any training data
    bool has_data = !train_loss_.values.empty() || !train_accuracy_.values.empty() || !custom_metrics_.empty();

    if (has_data) {
        RenderControls();

        const bool loss_on = show_loss_plot_ && !train_loss_.values.empty();
        const bool accuracy_on = show_accuracy_plot_ && !train_accuracy_.values.empty();
        const float available_width = ImGui::GetContentRegionAvail().x;
        const float available_height = ImGui::GetContentRegionAvail().y;

        // Loss and accuracy side by side when there is room, so both curves
        // stay visible together; stacked (each shrinking) otherwise.
        if (loss_on && accuracy_on && available_width >= kTrainingPlotSideBySideWidth) {
            const float plot_height = std::clamp(
                available_height - 70.0f, kTrainingPlotMinHeight, kTrainingPlotMaxHeight);
            if (ImGui::BeginTable("##dash_charts", 2,
                                  ImGuiTableFlags_SizingStretchSame |
                                      ImGuiTableFlags_NoPadOuterX)) {
                ImGui::TableNextColumn();
                RenderLossPlot(plot_height);
                ImGui::TableNextColumn();
                RenderAccuracyPlot(plot_height);
                ImGui::EndTable();
            }
        } else {
            if (loss_on) {
                const float plot_height = accuracy_on
                    ? std::clamp((available_height - 36.0f) * 0.50f,
                                 kTrainingPlotMinHeight, kTrainingPlotMaxHeight)
                    : std::max(kTrainingPlotMinHeight, available_height - 90.0f);
                RenderLossPlot(plot_height);
            }
            if (accuracy_on) {
                const float plot_height = loss_on
                    ? std::clamp(ImGui::GetContentRegionAvail().y - 40.0f,
                                 kTrainingPlotMinHeight, kTrainingPlotMaxHeight)
                    : std::max(kTrainingPlotMinHeight, available_height - 90.0f);
                RenderAccuracyPlot(plot_height);
            }
        }

        if (show_custom_metrics_ && !custom_metrics_.empty()) {
            RenderCustomMetricsPlot(kTrainingPlotMinHeight);
        }

        // Insights next to statistics when wide, stacked otherwise.
        if (ImGui::GetContentRegionAvail().x >= kTrainingPlotSideBySideWidth &&
            ImGui::BeginTable("##dash_insights_row", 2,
                              ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_NoPadOuterX)) {
            ImGui::TableSetupColumn("insights", ImGuiTableColumnFlags_WidthStretch, 1.25f);
            ImGui::TableSetupColumn("stats", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableNextColumn();
            RenderCurveSummary();
            ImGui::TableNextColumn();
            if (!train_loss_.values.empty()) {
                RenderStatistics();
            }
            RenderSequenceMetricsSummary();
            ImGui::EndTable();
        } else {
            RenderCurveSummary();
            RenderSequenceMetricsSummary();
            if (!train_loss_.values.empty()) {
                RenderStatistics();
            }
        }
    }

    // Data preparation below the charts (its details are under Run details).
    RenderMaterializationSummary();

    if (!has_data) {
        RenderEmptyState();
    }

    RenderRunComparisonTable();

    ImGui::End();

    // Charts opened in their own windows.
    RenderChartWindows();
}

void TrainingPlotPanel::RenderEmptyState() {
    const ui::Tokens& c = ui::CurrentTokens();
    ui::BeginCard("##dash_empty");
    {
        const auto centered = [](const char* text, const ImVec4& color) {
            const float width = ImGui::GetContentRegionAvail().x;
            const float text_width = ImGui::CalcTextSize(text).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                 std::max(0.0f, (width - text_width) * 0.5f));
            ImGui::TextColored(color, "%s", text);
        };
        ImGui::Dummy(ImVec2(0.0f, 48.0f));
        {
            const char* icon = ICON_FA_CHART_LINE;
            ImFont* font = ImGui::GetFont();
            const float size = ImGui::GetFontSize() * 2.0f;
            const ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, icon);
            const float width = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                 std::max(0.0f, (width - extent.x) * 0.5f));
            ImGui::GetWindowDrawList()->AddText(font, size, ImGui::GetCursorScreenPos(),
                                                ImGui::GetColorU32(c.text_faint), icon);
            ImGui::Dummy(extent);
        }
        ImGui::Spacing();
        centered("No training data yet", c.text);
        // Only canvas training reports here; the script hint named a file
        // that does not exist and scripts cannot feed this panel.
        centered("Press Train on the canvas to see real-time metrics", c.text_dim);
        ImGui::Spacing();
        centered("Loss and accuracy charts will appear here", c.text_faint);
        ImGui::Dummy(ImVec2(0.0f, 48.0f));
    }
    ui::EndCard();
}

void TrainingPlotPanel::RenderKpiCards() {
    const ui::Tokens& c = ui::CurrentTokens();
    namespace td = training_dashboard;
    const std::vector<td::Card> cards = td::BuildCards(facts_);
    if (cards.empty()) return;

    const auto tone = [&](td::Tone t) {
        switch (t) {
        case td::Tone::Good: return c.success;
        case td::Tone::Bad: return c.caution;
        case td::Tone::Error: return c.error;
        case td::Tone::Plain: return c.text;
        case td::Tone::Faint: break;
        }
        return c.text_dim;
    };
    const ImVec4 dots[] = {train_loss_.color, val_loss_.color, train_accuracy_.color, val_accuracy_.color};

    // As many columns as cards of at least 150 px fit (two in the 340 px slot).
    const float min_card_width = 150.0f;
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(available / min_card_width), 1, static_cast<int>(cards.size()));
    ImGui::Spacing();
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("##dash_kpis", columns, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
        for (const auto& card : cards) {
            ImGui::TableNextColumn();
            const ImVec4 dot = card.series >= 0 && card.series < 4 ? dots[card.series] : c.info;
            std::string change = card.change;
            if (card.direction != 0) change = std::string(card.direction < 0 ? ICON_FA_CARET_DOWN : ICON_FA_CARET_UP) + " " + change;
            ui::KpiCard(card.id.c_str(), card.label.c_str(), dot, card.value.c_str(), card.tooltip.c_str(), change.c_str(),
                        tone(card.change_tone), card.line2.c_str(), tone(card.line2_tone),
                        card.line3.empty() ? nullptr : card.line3.c_str(), tone(card.line3_tone));
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

training_dashboard::RunFacts TrainingPlotPanel::DashboardFacts() const {
    namespace td = training_dashboard;
    const auto series = [](const MetricSeries& s) {
        td::Series out;
        out.x = s.epochs;
        out.y = s.values;
        return out;
    };
    const auto latest = [](const MetricSeries& s) -> std::optional<td::Point> {
        if (s.epochs.empty() || s.values.empty()) return std::nullopt;
        const size_t n = std::min(s.epochs.size(), s.values.size());
        return td::Point{s.epochs[n - 1], s.values[n - 1]};
    };
    td::RunFacts f;
    f.training = is_training_;
    // The batch curves can hold 100,000 points: only their last point is read.
    f.latest_loss_point = latest(train_loss_);
    f.latest_accuracy_point = latest(train_accuracy_);
    f.train_loss_epochs = series(train_loss_epochs_);
    f.train_accuracy_epochs = series(train_accuracy_epochs_);
    f.val_loss = series(val_loss_);
    f.val_accuracy = series(val_accuracy_);
    f.current_epoch = current_epoch_;
    f.total_epochs = total_epochs_;
    f.last_executed_epoch = last_executed_epoch_;
    f.current_batch = current_batch_;
    f.total_batches = total_batches_;
    f.remaining_seconds = eta_estimator_.RemainingSeconds();
    f.batches_per_second = eta_estimator_.BatchesPerSecond();
    f.samples_per_second = samples_per_second_;
    f.avg_epoch_seconds = avg_epoch_time_;
    if (eta_estimator_.HasMeasuredEpochOverhead()) f.epoch_end_seconds = eta_estimator_.MeanEpochOverheadSeconds();
    f.total_seconds = total_training_time_;
    return f;
}

void TrainingPlotPanel::AddLossPoint(double epoch, double train_loss, double val_loss) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    RecordPanelEvent("TrainingPlotPanel.WriteLoss",
                     "epoch=" + std::to_string(epoch) +
                     " train_loss=" + std::to_string(train_loss));

    ++data_version_;
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

    ++data_version_;
    train_accuracy_.epochs.push_back(epoch);
    train_accuracy_.values.push_back(train_acc);
    TrimDataIfNeeded(train_accuracy_);

    if (val_acc >= 0.0) {
        val_accuracy_.epochs.push_back(epoch);
        val_accuracy_.values.push_back(val_acc);
        TrimDataIfNeeded(val_accuracy_);
    }
}

namespace {
// A batch point already drawn inside this epoch (x in (epoch - 1, epoch]).
bool HasPointInEpoch(const std::vector<double>& xs, int epoch) {
    return std::any_of(xs.begin(), xs.end(), [epoch](double x) { return x > epoch - 1 && x <= epoch; });
}
}  // namespace

void TrainingPlotPanel::AddEpochLoss(int epoch, double train_loss, double val_loss) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ++data_version_;
    train_loss_epochs_.epochs.push_back(epoch);
    train_loss_epochs_.values.push_back(train_loss);
    if (!HasPointInEpoch(train_loss_.epochs, epoch)) {
        train_loss_.epochs.push_back(epoch);
        train_loss_.values.push_back(train_loss);
        TrimDataIfNeeded(train_loss_);
    }
    if (val_loss >= 0.0) {
        val_loss_.epochs.push_back(epoch);
        val_loss_.values.push_back(val_loss);
        TrimDataIfNeeded(val_loss_);
    }
}

void TrainingPlotPanel::AddEpochAccuracy(int epoch, double train_acc, double val_acc) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ++data_version_;
    train_accuracy_epochs_.epochs.push_back(epoch);
    train_accuracy_epochs_.values.push_back(train_acc);
    if (!HasPointInEpoch(train_accuracy_.epochs, epoch)) {
        train_accuracy_.epochs.push_back(epoch);
        train_accuracy_.values.push_back(train_acc);
        TrimDataIfNeeded(train_accuracy_);
    }
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
        // Its colour comes from the theme each frame (ApplySeriesColours).
        custom_metrics_.push_back(new_series);
        it = custom_metrics_.end() - 1;
    }

    ++data_version_;
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

void TrainingPlotPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ClearLocked();
}

void TrainingPlotPanel::ClearLocked() {
    ++data_version_;
    train_loss_.epochs.clear();
    train_loss_.values.clear();
    val_loss_.epochs.clear();
    val_loss_.values.clear();
    train_accuracy_.epochs.clear();
    train_accuracy_.values.clear();
    train_loss_epochs_ = {};
    train_accuracy_epochs_ = {};
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
    materialization_notice_.clear();
    materialization_rebuild_reason_.clear();
    materialization_pruned_entries_ = 0;
    materialization_pruned_bytes_ = 0;

    // Reset training state
    is_training_ = false;
    current_epoch_ = 0;
    last_executed_epoch_ = 0;
    total_epochs_ = 0;
    current_batch_ = 0;
    total_batches_ = 0;
    current_batch_loss_ = 0.0f;
    last_epoch_time_ = 0.0f;
    avg_epoch_time_ = 0.0f;
    samples_per_second_ = 0.0f;
    eta_estimator_.Reset();
    total_training_time_ = 0.0f;
    terminal_status_.clear();
    terminal_reason_.clear();
    ClearMissingValues();
    checkpoint_used_.clear();
    checkpoint_epoch_ = 0;
    checkpoint_step_ = 0;
    active_model_provenance_.clear();
    has_checkpoint_validation_metrics_ = false;
    checkpoint_val_loss_ = 0.0f;
    checkpoint_val_accuracy_ = 0.0f;
    active_checkpoint_loaded_ = false;
    run_dataset_.clear();
    run_train_rows_ = run_validation_rows_ = run_test_rows_ = 0;
    has_test_metrics_ = false;
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
        ClearMissingValues();
        last_executed_epoch_ = 0;
        checkpoint_used_.clear();
        checkpoint_epoch_ = 0;
        checkpoint_step_ = 0;
        active_model_provenance_.clear();
        has_checkpoint_validation_metrics_ = false;
        active_checkpoint_loaded_ = false;
        has_test_metrics_ = false;
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
    checkpoint_step_ = 0;
    last_executed_epoch_ = checkpoint_epoch;
    active_model_provenance_ = checkpoint_used.empty()
        ? "run_final_state"
        : "restored_best_checkpoint";
    active_checkpoint_loaded_ = false;
    has_test_metrics_ = false;
}

void TrainingPlotPanel::SetTrainingComplete(
    float total_time_seconds,
    const TrainingMetrics& metrics) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    is_training_ = false;
    is_preparing_ = false;
    preparation_failed_ = false;
    preparation_status_message_.clear();
    preparation_error_message_.clear();
    preparation_progress_ = 0.0f;
    total_training_time_ = total_time_seconds;
    terminal_status_ = metrics.terminal_status;
    terminal_reason_ = metrics.terminal_reason;
    ClearMissingValues();
    missing_columns_ = metrics.missing_columns;
    missing_values_total_rows_ = metrics.missing_values_total_rows;
    missing_values_supplied_role_ = metrics.missing_values_supplied_role;
    current_epoch_ = metrics.current_epoch;
    last_executed_epoch_ = metrics.last_executed_epoch;
    if (metrics.total_epochs > 0) {
        total_epochs_ = metrics.total_epochs;
    }
    checkpoint_used_ = metrics.checkpoint_used;
    checkpoint_epoch_ = metrics.restored_checkpoint_epoch;
    checkpoint_step_ = metrics.restored_checkpoint_step;
    active_model_provenance_ = metrics.active_model_provenance;
    has_checkpoint_validation_metrics_ = metrics.has_validation_metrics;
    checkpoint_val_loss_ = metrics.val_loss;
    checkpoint_val_accuracy_ = metrics.val_accuracy;
    active_checkpoint_loaded_ = false;
    if (metrics.train_sample_count > 0) {
        run_train_rows_ = metrics.train_sample_count;
        run_validation_rows_ = metrics.val_sample_count;
        run_test_rows_ = metrics.test_sample_count;
    }
    has_test_metrics_ = metrics.has_test_metrics;
    test_loss_ = metrics.test_loss;
    test_accuracy_ = metrics.test_accuracy;
    test_mae_ = metrics.test_mae;
}

void TrainingPlotPanel::SetRunData(const std::string& dataset, size_t train_rows, size_t validation_rows,
                                   size_t test_rows) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    run_dataset_ = dataset;
    run_train_rows_ = train_rows;
    run_validation_rows_ = validation_rows;
    run_test_rows_ = test_rows;
}

void TrainingPlotPanel::SetGenerationPreviewFolder(const std::filesystem::path& folder) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    generation_folder_ = folder;
    if (folder.empty()) show_generations_ = false;
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
    checkpoint_step_ = 0;
    has_test_metrics_ = false;  // the test was the run's model, not this one
    active_checkpoint_loaded_ = true;
    active_model_provenance_ = "loaded_checkpoint_for_testing";
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
    const double now_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - eta_clock_start_).count();
    eta_estimator_.Observe(now_seconds, current_epoch, current_batch, total_batches,
                           std::max(1, total_epochs_));
}

void TrainingPlotPanel::SetBatchComposition(int samples_per_batch, int batches_per_update) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    samples_per_batch_ = std::max(0, samples_per_batch);
    batches_per_update_ = std::max(1, batches_per_update);
}

void TrainingPlotPanel::SetMetricReportingCadence(int batch_interval) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    metric_reporting_interval_ = std::max(0, batch_interval);
}

std::string TrainingPlotPanel::MetricsCsvLocked() const {
    // The charts' points (training: the batch windows inside each epoch and
    // the epoch ends), then the whole-epoch training values the cards show.
    const auto column = [](const char* name, const MetricSeries& series) {
        training_dashboard::CsvColumn c;
        c.name = name;
        c.values.x = series.epochs;
        c.values.y = series.values;
        return c;
    };
    std::vector<training_dashboard::CsvColumn> columns = {
        column("TrainLoss", train_loss_),          column("ValLoss", val_loss_),
        column("TrainAccuracy", train_accuracy_), column("ValAccuracy", val_accuracy_),
        column("TrainLossEpoch", train_loss_epochs_), column("TrainAccuracyEpoch", train_accuracy_epochs_)};
    for (const auto& metric : custom_metrics_) columns.push_back(column(metric.name.c_str(), metric));
    return training_dashboard::MetricsCsv(columns);
}

void TrainingPlotPanel::RunPendingExport() {
    const PendingExport kind = pending_export_;  // UI thread only
    if (kind == PendingExport::None) return;
    pending_export_ = PendingExport::None;
    if (!save_path_callback_) return;
    const bool metrics = kind == PendingExport::Metrics;

    // Copy what to write, then ask with the lock released.
    std::string metrics_csv;
    std::vector<TrainingRunComparisonRecord> runs;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (metrics) {
            metrics_csv = MetricsCsvLocked();
        } else {
            runs = run_comparison_records_;
        }
    }
    const auto path = metrics ? save_path_callback_("Export metrics", "training_metrics.csv")
                              : save_path_callback_("Export runs", "training_run_comparison.csv");
    if (!path) return;  // cancelled

    std::string error;
    bool ok = false;
    if (metrics) {
        std::ofstream file(*path, std::ios::binary | std::ios::trunc);
        ok = file && (file << metrics_csv) && file.flush();
        if (!ok) error = "could not write " + *path;
    } else {
        ok = WriteTrainingRunComparisonCsv(*path, runs, &error);
    }
    const char* what = metrics ? "metrics" : "runs";
    std::lock_guard<std::mutex> lock(data_mutex_);
    export_note_at_ = pending_note_at_;
    export_failed_ = !ok;
    export_note_ = ok ? std::string("Saved the ") + what + " to " + *path
                      : std::string("Could not save the ") + what + ": " + error;
    if (!ok) {
        RecordPanelEvent(metrics ? "TrainingPlotPanel.ExportMetricsFailed" : "TrainingPlotPanel.ExportRunComparisonFailed",
                         error);
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
        ClearMissingValues();
        last_executed_epoch_ = 0;
        checkpoint_used_.clear();
        checkpoint_epoch_ = 0;
        checkpoint_step_ = 0;
        active_model_provenance_.clear();
        active_checkpoint_loaded_ = false;
        run_dataset_.clear();
        run_train_rows_ = run_validation_rows_ = run_test_rows_ = 0;
        has_test_metrics_ = false;
    } else {
        // Preparation over: a word-vector run it trained that never finished
        // was stopped with it.
        word_vector_runs_.EndAllLive(true, "");
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
    ClearMissingValues();
    word_vector_runs_.EndAllLive(false, error_message);
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
    uint64_t available_memory_bytes,
    uint64_t safe_memory_budget_bytes,
    bool process_memory_detected,
    uint64_t process_resident_memory_bytes,
    uint64_t process_private_memory_bytes,
    uint64_t process_resident_growth_bytes,
    const std::string& process_private_memory_name,
    const std::string& process_memory_source,
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
    event.available_memory_bytes = available_memory_bytes;
    event.safe_memory_budget_bytes = safe_memory_budget_bytes;
    event.memory_risk_level = memory_risk_level;
    event.process_memory_detected = process_memory_detected;
    event.process_resident_memory_bytes = process_resident_memory_bytes;
    event.process_private_memory_bytes = process_private_memory_bytes;
    event.process_resident_growth_bytes = process_resident_growth_bytes;
    event.process_private_memory_name = process_private_memory_name;
    event.process_memory_source = process_memory_source;
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

    const auto now = std::chrono::steady_clock::now();
    event.started_at = now;
    event.updated_at = now;
    if (!materialization_events_.empty() &&
        materialization_events_.back().stage == event.stage) {
        event.started_at = materialization_events_.back().started_at;
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
    // The run that just prepared its data used up a pending rebuild request.
    materialization_rebuild_pending_ = false;

    MaterializationProgress event;
    event.status = materialization_status_;
    if (materialization_status_ == "cache_hit") {
        event.stage = "Cache reused";
        event.message =
            "Preprocessing skipped: reused cached materialization.";
        if (!materialization_events_.empty() &&
            materialization_events_.back().status == "cache_hit" &&
            !materialization_events_.back().message.empty()) {
            event.message = materialization_events_.back().message;
        }
    } else if (materialization_status_ == "cache_saved") {
        event.stage = "Prepared and cached";
        event.message = "Preprocessing completed and saved to cache.";
    } else {
        event.stage = "Complete";
        event.message = "Materialization completed";
    }
    event.progress = 1.0f;
    const auto now = std::chrono::steady_clock::now();
    event.started_at = now;
    event.updated_at = now;
    if (!materialization_events_.empty() &&
        materialization_events_.back().stage == event.stage) {
        event.started_at = materialization_events_.back().started_at;
        materialization_events_.back() = std::move(event);
    } else {
        materialization_events_.push_back(std::move(event));
        if (materialization_events_.size() > 24) {
            materialization_events_.erase(materialization_events_.begin());
        }
    }
}

void TrainingPlotPanel::SetMaterializationCacheInfo(
    const std::string& cache_directory,
    int entries,
    uint64_t total_bytes,
    uint64_t size_limit_bytes,
    int pruned_entries,
    uint64_t pruned_bytes,
    const std::string& rebuild_reason) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    materialization_cache_directory_ = cache_directory;
    materialization_cache_entries_ = entries;
    materialization_cache_bytes_ = total_bytes;
    materialization_cache_limit_bytes_ = size_limit_bytes;
    materialization_pruned_entries_ = pruned_entries;
    materialization_pruned_bytes_ = pruned_bytes;
    if (!rebuild_reason.empty()) {
        materialization_rebuild_reason_ = rebuild_reason;
    }
}

void TrainingPlotPanel::SetMaterializationNotice(const std::string& notice) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    materialization_notice_ = notice;
}

void TrainingPlotPanel::SetMaterializationClearResult(
    int removed_entries,
    uint64_t freed_bytes,
    const std::string& error) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!error.empty()) {
        materialization_clear_message_ = error;
    } else {
        materialization_clear_message_ =
            "Removed " + std::to_string(removed_entries) + " prepared dataset" +
            (removed_entries == 1 ? "" : "s") + " (" +
            FormatTraceBytes(freed_bytes) + ").";
    }
}

void TrainingPlotPanel::RenderLossPlot(float plot_height) {
    ui::BeginCard("##dash_loss_card");
    {
        ui::CardTitle("Loss", log_loss_scale_ ? "training and validation, log axis" : "training and validation");
        // Open this chart in its own dockable window.
        ImGui::SameLine();
        const char* pop_label = ICON_FA_WINDOW_RESTORE "##pop_loss";
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x -
                                          ImGui::CalcTextSize(ICON_FA_WINDOW_RESTORE).x - 12.0f));
        if (ui::GhostButton(pop_label, true, nullptr, loss_window_open_)) {
            loss_window_open_ = !loss_window_open_;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(loss_window_open_ ? "Close the chart window"
                                     : "Open this chart in its own window (dock it or make it full size)");
        }
        DrawLossPlot(ImVec2(-1, plot_height), false);
    }
    ui::EndCard();
}

void TrainingPlotPanel::DrawLossPlot(const ImVec2& size, bool fit, bool in_window) {
    DrawChart(0, size, fit, in_window);
}


void TrainingPlotPanel::RenderAccuracyPlot(float plot_height) {
    ui::BeginCard("##dash_accuracy_card");
    {
        ui::CardTitle("Accuracy", "training and validation, percent");
        // Open this chart in its own dockable window.
        ImGui::SameLine();
        const char* pop_label = ICON_FA_WINDOW_RESTORE "##pop_accuracy";
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x -
                                          ImGui::CalcTextSize(ICON_FA_WINDOW_RESTORE).x - 12.0f));
        if (ui::GhostButton(pop_label, true, nullptr, accuracy_window_open_)) {
            accuracy_window_open_ = !accuracy_window_open_;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(accuracy_window_open_ ? "Close the chart window"
                                     : "Open this chart in its own window (dock it or make it full size)");
        }
        DrawAccuracyPlot(ImVec2(-1, plot_height), false);
    }
    ui::EndCard();
}

void TrainingPlotPanel::DrawAccuracyPlot(const ImVec2& size, bool fit, bool in_window) {
    DrawChart(1, size, fit, in_window);
}


void TrainingPlotPanel::RenderCustomMetricsPlot(float plot_height) {
    const char* plot_title = nullptr;
    ClassifyCustomMetrics(custom_metrics_, &plot_title, nullptr);
    ui::BeginCard("##dash_custom_card");
    {
        ui::CardTitle(plot_title, "per epoch");
        // Open this chart in its own dockable window.
        ImGui::SameLine();
        const char* pop_label = ICON_FA_WINDOW_RESTORE "##pop_custom";
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x -
                                          ImGui::CalcTextSize(ICON_FA_WINDOW_RESTORE).x - 12.0f));
        if (ui::GhostButton(pop_label, true, nullptr, custom_window_open_)) {
            custom_window_open_ = !custom_window_open_;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(custom_window_open_ ? "Close the chart window"
                                     : "Open this chart in its own window (dock it or make it full size)");
        }
        DrawCustomMetricsPlot(ImVec2(-1, plot_height), false);
    }
    ui::EndCard();
}

void TrainingPlotPanel::DrawCustomMetricsPlot(const ImVec2& size, bool fit, bool in_window) {
    DrawChart(2, size, fit, in_window);
}

plot::Series TrainingPlotPanel::ToPlotSeries(const MetricSeries& full, const MetricSeries& drawn,
                                             const std::vector<double>& smooth_x, const std::vector<double>& smooth_y,
                                             int colour, bool markers) const {
    plot::Series s;
    s.label = full.name;
    s.x = drawn.epochs;
    s.y = drawn.values;
    s.smooth_x = smooth_x;
    s.smooth_y = smooth_y;
    // Hover values and exports use every point of a reduced curve.
    if (drawn.epochs.size() < full.epochs.size()) {
        s.all_x = full.epochs;
        s.all_y = full.values;
    }
    s.x_sorted = true;  // epochs ascend
    s.colour = colour;
    s.markers = markers;
    return s;
}

plot::Prepared TrainingPlotPanel::ChartData(int chart) const {
    const DrawnCache& d = drawn_;
    plot::Prepared p;
    p.spec.kind = plot::Kind::Line;
    p.spec.x_label = "Epoch";
    p.spec.legend = true;
    p.spec.smooth = smoothing_window_;
    size_t shown = 0, total = 0;
    const auto add = [&](const MetricSeries& full, const DrawnSeries& drawn, int colour, bool markers) {
        if (full.values.empty()) return;
        p.series.push_back(ToPlotSeries(full, drawn.line, drawn.smooth_x, drawn.smooth_y, colour, markers));
        shown += drawn.line.values.size();
        total += full.values.size();
    };
    if (chart == 0) {
        p.spec.title = "Loss";
        p.spec.y_label = "Loss";
        p.spec.log_y = log_loss_scale_;
        add(train_loss_, d.train_loss, kTrainLossColour, false);
        add(val_loss_, d.val_loss, kValLossColour, true);
    } else if (chart == 1) {
        p.spec.title = "Accuracy";
        p.spec.y_label = "Accuracy (%)";
        add(train_accuracy_, d.train_accuracy, kTrainAccuracyColour, false);
        add(val_accuracy_, d.val_accuracy, kValAccuracyColour, true);
    } else {
        const char* title = nullptr;
        const char* y_label = nullptr;
        ClassifyCustomMetrics(custom_metrics_, &title, &y_label);
        p.spec.title = title ? title : "Metrics";
        p.spec.y_label = y_label ? y_label : "";
        for (size_t i = 0; i < custom_metrics_.size() && i < d.custom.size(); ++i) {
            const MetricSeries& full = custom_metrics_[i];
            if (full.values.empty()) continue;
            p.series.push_back(ToPlotSeries(full, d.custom[i], {}, {}, static_cast<int>(i % 6),
                                            IsValidationMetricName(full.name)));
            shown += d.custom[i].values.size();
            total += full.values.size();
        }
    }
    p.label = shown < total ? plot::DataLabel{plot::DataLabel::State::Reduced, shown, total}
                            : plot::DataLabel{plot::DataLabel::State::Exact, total, total};
    if (p.series.empty()) p.problem = "No values yet.";
    return p;
}

void TrainingPlotPanel::DrawChart(int chart, const ImVec2& size, bool fit, bool in_window) {
    const DrawnCache& d = Drawn();
    ChartSlot& slot = chart_slots_[chart][in_window ? 1 : 0];
    if (!slot.view) {
        static const char* ids[3] = {"training_loss", "training_accuracy", "training_metrics"};
        slot.view = std::make_unique<plot::PlotView>(std::string(ids[chart]) + (in_window ? "_window" : "_card"));
    }
    if (slot.build != d.build || slot.log != log_loss_scale_) {
        slot.view->SetData(ChartData(chart));
        slot.build = d.build;
        slot.log = log_loss_scale_;
    }
    if (fit) slot.view->RequestFit();

    plot::PlotView::Options o;
    static const char* names[3] = {"training_loss", "training_accuracy", "training_metrics"};
    o.export_name = names[chart];
    // The dashboard has its own Log loss axis, Fit (chart window) and
    // pop-out icon.
    o.tool_fit = false;
    o.tool_log = false;
    o.own_window_button = false;
    // In the dashboard the wheel scrolls it; a chart window zooms with it.
    o.zoom_with_ctrl = !in_window;
    // Auto scale and Follow current epoch, as before: the epoch window and
    // the value range of what is visible.
    if (chart == 0 && auto_scale_ && !d.train_loss.line.epochs.empty()) {
        const auto [min_epoch, max_epoch] = CalculateEpochWindow(d.train_loss.line);
        // Following: the window moves with training; otherwise the planned
        // epochs are set once and the user may pan (as before the port).
        o.x_range = {true, min_epoch, max_epoch, !follow_current_epoch_};
        const ValueRange range = CalculateVisibleRange(d.train_loss.line, d.val_loss.line, min_epoch, max_epoch);
        if (log_loss_scale_) {
            double min_positive = std::numeric_limits<double>::max();
            for (const MetricSeries* series : {&d.train_loss.line, &d.val_loss.line}) {
                const size_t count = std::min(series->epochs.size(), series->values.size());
                for (size_t i = 0; i < count; ++i) {
                    if (series->epochs[i] < min_epoch || series->epochs[i] > max_epoch || series->values[i] <= 0.0) continue;
                    min_positive = std::min(min_positive, series->values[i]);
                }
            }
            if (min_positive == std::numeric_limits<double>::max()) min_positive = 1.0e-6;
            const double lower = std::max(1.0e-12, min_positive / 1.25);
            o.y_range = {true, lower, std::max(lower * 10.0, range.max * 1.25)};
        } else {
            double padding = (range.max - range.min) * 0.1;
            if (padding < 0.01) padding = 0.1;
            o.y_range = {true, std::max(0.0, range.min - padding), range.max + padding};
        }
    } else if (chart == 1 && auto_scale_ && follow_current_epoch_ && !d.train_accuracy.line.epochs.empty()) {
        const auto [min_epoch, max_epoch] = CalculateEpochWindow(d.train_accuracy.line);
        o.x_range = {true, min_epoch, max_epoch};
        const ValueRange range = CalculateVisibleRange(d.train_accuracy.line, d.val_accuracy.line, min_epoch, max_epoch);
        double padding = (range.max - range.min) * 0.1;
        if (padding < 1.0) padding = 5.0;
        o.y_range = {true, std::max(0.0, range.min - padding), std::min(100.0, range.max + padding)};
    } else if (chart == 1 && auto_scale_ && !d.train_accuracy.line.epochs.empty()) {
        // Not following: all epochs so far, set once (as before the port);
        // with Auto scale the value range follows the data, like the loss
        // chart (it used to stay at the first fit, so the curve left the top).
        // The planned epochs, as the loss chart (it took the epochs so far,
        // set once: 0..2 while epoch 1 ran, for the rest of the run).
        const double max_epoch = CalculateEpochWindow(d.train_accuracy.line).second;
        o.x_range = {true, 0.0, max_epoch, true};
        const ValueRange range = CalculateVisibleRange(d.train_accuracy.line, d.val_accuracy.line, 0.0, max_epoch);
        double padding = (range.max - range.min) * 0.1;
        if (padding < 1.0) padding = 5.0;
        o.y_range = {true, std::max(0.0, range.min - padding), std::min(100.0, range.max + padding)};
    }
    slot.view->Draw(size, o);
}


void TrainingPlotPanel::RenderChartWindows() {
    const ui::Tokens& c = ui::CurrentTokens();
    const auto chart_window = [&](const char* title, bool* open, int kind) {
        if (!*open) {
            return;
        }
        ImGui::SetNextWindowSize(ImVec2(960, 600), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title, open)) {
            // Toolbar: fit, the shared view options, help.
            bool fit = false;
            if (ui::SecondaryButton(ICON_FA_EXPAND " Fit")) {
                fit = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Fit all data (with Auto scale on, the axes already follow the data).");
            }
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::Checkbox("Auto scale", &auto_scale_);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("When enabled, axes adapt to the live training data.\nDisable to manually zoom/pan.");
            }
            ImGui::SameLine();
            ImGui::Checkbox("Follow current epoch", &follow_current_epoch_);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Keep the epoch axis scrolled to the latest batch/epoch.");
            }
            if (follow_current_epoch_) {
                ImGui::SameLine();
                EpochsField("Epoch window", "##win_epochs", &visible_epoch_window_, 3, 50,
                            "Epochs shown while following live training");
            }
            if (kind == 0) {
                ImGui::SameLine();
                ImGui::Checkbox("Log loss axis", &log_loss_scale_);
            }
            if (kind != 2) {
                ImGui::SameLine();
                ImGui::Checkbox("Smooth", &show_smoothed_curves_);
                if (show_smoothed_curves_) {
                    ImGui::SameLine();
                    EpochsField("Smooth window", "##win_smooth", &smoothing_window_, 2, 50,
                                "Points in the moving average");
                }
            }
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::TextColored(c.text_dim, ICON_FA_CIRCLE_INFO);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Plot Controls:");
                ImGui::BulletText("Scroll wheel: Zoom both axes");
                ImGui::BulletText("Scroll on axis: Zoom that axis only");
                ImGui::BulletText("Drag: Pan view");
                ImGui::BulletText("Right-drag: Zoom to a box");
                ImGui::BulletText("Double-click: Fit data");
                ImGui::BulletText("Right-click: Chart menu (axes, legend, fit)");
                ImGui::BulletText("Click a legend entry: Show or hide that curve");
                ImGui::BulletText("Disable Auto Scale or Follow Current for manual control");
                ImGui::EndTooltip();
            }

            const bool has_series =
                kind == 0 ? !train_loss_.values.empty()
                          : (kind == 1 ? !train_accuracy_.values.empty() : !custom_metrics_.empty());
            if (!has_series) {
                ImGui::Spacing();
                ImGui::TextColored(c.text_dim, "%s",
                                   kind == 0 ? "No loss data yet. Start training to see the curve."
                                   : kind == 1 ? "No accuracy data yet. Start training to see the curve."
                                               : "No custom metrics yet.");
            } else if (kind == 0) {
                DrawLossPlot(ImVec2(-1, -1), fit, true);
            } else if (kind == 1) {
                DrawAccuracyPlot(ImVec2(-1, -1), fit, true);
            } else {
                DrawCustomMetricsPlot(ImVec2(-1, -1), fit, true);
            }
        }
        ImGui::End();
    };

    chart_window(ICON_FA_CHART_LINE " Training Loss###TrainingLossChart", &loss_window_open_, 0);
    chart_window(ICON_FA_CHART_LINE " Training Accuracy###TrainingAccuracyChart", &accuracy_window_open_, 1);
    chart_window(ICON_FA_CHART_LINE " Training Metrics###TrainingCustomChart", &custom_window_open_, 2);
}

void TrainingPlotPanel::RenderControls() {
    const ui::Tokens& c = ui::CurrentTokens();
    ImGui::Spacing();

    // Which charts to show; the parts wrap in a narrow dashboard.
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Charts").x);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c.text_dim, "Charts");
    row.Next(ChipToggleWidth("Loss"));
    ChipToggle("Loss", &show_loss_plot_);
    row.Next(ChipToggleWidth("Accuracy"));
    ChipToggle("Accuracy", &show_accuracy_plot_);
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

        const char* label = has_sequence_metrics && !has_non_sequence_metrics
            ? "Sequence metrics"
            : "Custom metrics";
        row.Next(ChipToggleWidth(label));
        ChipToggle(label, &show_custom_metrics_);
    }

    // Active view options, so a changed view is never a surprise.
    std::string view_state;
    if (log_loss_scale_) {
        view_state = "log loss axis";
    }
    if (show_smoothed_curves_) {
        view_state += (view_state.empty() ? "" : " \xC2\xB7 ") +
                      std::string("smoothed (") + std::to_string(smoothing_window_) + ")";
    }
    if (!auto_scale_) {
        view_state += (view_state.empty() ? "" : " \xC2\xB7 ") + std::string("manual zoom");
    } else if (follow_current_epoch_) {
        view_state += (view_state.empty() ? "" : " \xC2\xB7 ") +
                      std::string("following last ") + std::to_string(visible_epoch_window_) +
                      " epochs";
    }
    if (!view_state.empty()) {
        row.Next(ImGui::CalcTextSize(view_state.c_str()).x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.text_faint, "%s", view_state.c_str());
    }

    // View menu and plot help on the right (on the next line when they do not fit).
    const char* view_label = ICON_FA_SLIDERS " View";
    const float help_width = ImGui::CalcTextSize(ICON_FA_CIRCLE_INFO).x;
    const float needed = ui::ButtonWidth(view_label, ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x + help_width;
    ui::SameLineRight(needed);
    if (ui::SecondaryButton(view_label)) {
        ImGui::OpenPopup("##dash_view");
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c.text_dim, ICON_FA_CIRCLE_INFO);
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Plot Controls:");
        ImGui::BulletText("Wheel: scrolls the dashboard");
        ImGui::BulletText("Ctrl + wheel: zoom (on an axis: that axis only); the value axis with Auto scale off");
        ImGui::BulletText("Drag: Pan view");
        ImGui::BulletText("Right-drag: Zoom to a box");
        ImGui::BulletText("Right-click: Chart menu (axes, legend, fit)");
        ImGui::BulletText("Chart window button: open the chart in its own window (the wheel zooms there)");
        ImGui::BulletText("Double-click: Fit data");
        ImGui::BulletText("Disable Auto Scale or Follow Current for manual control");
        ImGui::EndTooltip();
    }

    if (ImGui::BeginPopup("##dash_view")) {
        ImGui::TextColored(c.text_dim, "Axes");
        ImGui::Checkbox("Auto scale", &auto_scale_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("When enabled, axes adapt to the live training data.\nDisable to manually zoom/pan.");
        }
        ImGui::Checkbox("Follow current epoch", &follow_current_epoch_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Keep the epoch axis scrolled to the latest batch/epoch.");
        }
        EpochsField("Epoch window", "##view_epochs", &visible_epoch_window_, 3, 50,
                    "Epochs shown while following live training");
        ImGui::Checkbox("Log loss axis", &log_loss_scale_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Use a logarithmic loss axis to compare large early losses with "
                "smaller later losses. Metric values are unchanged.");
        }
        ImGui::Spacing();
        ImGui::TextColored(c.text_dim, "Curves");
        ImGui::Checkbox("Smooth curves", &show_smoothed_curves_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display-only moving average overlay. Raw curves are unchanged.");
        }
        if (show_smoothed_curves_) {
            EpochsField("Smooth window", "##view_smooth", &smoothing_window_, 2, 50, "Points in the moving average");
        }
        ImGui::Spacing();
        ImGui::TextColored(c.text_dim, "Open in a window");
        ImGui::MenuItem(ICON_FA_WINDOW_RESTORE " Loss chart", nullptr, &loss_window_open_);
        ImGui::MenuItem(ICON_FA_WINDOW_RESTORE " Accuracy chart", nullptr, &accuracy_window_open_);
        if (!custom_metrics_.empty()) {
            ImGui::MenuItem(ICON_FA_WINDOW_RESTORE " Metrics chart", nullptr, &custom_window_open_);
        }
        ImGui::EndPopup();
    }
    ImGui::Spacing();
}

void TrainingPlotPanel::RenderCurveSummary() {
    const ui::Tokens& c = ui::CurrentTokens();
    namespace td = training_dashboard;
    const td::Insights insights = td::BuildInsights(facts_);
    const auto tone = [&](td::Tone t) {
        switch (t) {
        case td::Tone::Good: return c.success;
        case td::Tone::Bad: return c.caution;
        case td::Tone::Error: return c.error;
        case td::Tone::Faint: return c.text_dim;  // as on the cards
        case td::Tone::Plain: break;
        }
        return c.text;
    };

    ui::BeginCard("##dash_insights");
    // The title, then the suggestion on the same line when it fits.
    ui::CardTitle("How the run is going", "");
    if (!insights.rows.empty()) {
        const float pill_width = ImGui::CalcTextSize(insights.suggestion.c_str()).x + 30.0f;
        ImGui::SameLine();
        if (ImGui::GetContentRegionAvail().x < pill_width) ImGui::NewLine();
        ui::Chip(insights.suggestion.c_str(), tone(insights.suggestion_tone));
    }
    ImGui::Spacing();
    if (insights.rows.empty()) {
        ImGui::TextColored(c.text_faint, "The first epoch is running.");
        ui::EndCard();
        return;
    }
    // Labels as wide as the widest; the values wrap in the rest.
    float label_width = 0.0f;
    for (const auto& row : insights.rows) label_width = std::max(label_width, ImGui::CalcTextSize(row.label.c_str()).x);
    if (ImGui::BeginTable("##curve_summary", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, label_width + 12.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        for (const auto& row : insights.rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(c.text_dim, "%s", row.label.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(tone(row.tone), "%s", row.value.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
    ui::EndCard();
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

    const ui::Tokens& c = ui::CurrentTokens();
    ui::BeginCard("##dash_sequence");
    ui::CardTitle("Sequence Metrics", "latest values");
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ui::WithAlpha(c.border, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
    if (ImGui::BeginTable("##sequence_metrics", 3,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Train", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Val", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, c.text_dim);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();

        const auto value_cell = [&](bool has_value, double value) {
            ImGui::TableNextColumn();
            if (has_value) {
                ImGui::Text("%.2f%%", value);
            } else {
                ImGui::TextColored(c.text_faint, "no data");
            }
        };

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Token Accuracy");
        value_cell(has_train_token_accuracy, train_token_accuracy);
        value_cell(has_val_token_accuracy, val_token_accuracy);

        if (has_train_entity_f1 || has_val_entity_f1) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("Entity F1");
            value_cell(has_train_entity_f1, train_entity_f1);
            value_cell(has_val_entity_f1, val_entity_f1);
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleColor(2);
    ui::EndCard();
}

void TrainingPlotPanel::RenderActiveTaskSummary() {
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
    std::vector<training_dashboard::EngineTask> tasks;
    for (const auto& task : AsyncTaskManager::Instance().GetActiveTasks()) {
        tasks.push_back({task.name, task.progress, task.status_message});
    }
    DetailGroup("Engine tasks", training_dashboard::TaskRows(tasks));
#endif
}

void TrainingPlotPanel::RenderMaterializationSummary() {
    if (materialization_events_.empty() && materialization_notice_.empty()) {
        return;
    }
    const ui::Tokens& c = ui::CurrentTokens();
    const auto now = std::chrono::steady_clock::now();
    const bool has_actions = static_cast<bool>(materialization_action_callback_);
    if (has_actions && materialization_cache_entries_ < 0 &&
        !materialization_cache_refresh_requested_) {
        materialization_cache_refresh_requested_ = true;
        materialization_action_callback_("refresh");
    }

    const std::string& status = materialization_status_;
    const MaterializationProgress* latest =
        materialization_events_.empty() ? nullptr : &materialization_events_.back();
    const bool blocked = latest && latest->status == "blocked";
    const bool complete = !status.empty();
    const bool preparing = !complete && is_preparing_ && !preparation_failed_;

    // Overall state.
    const char* state = "Prepared";
    ImVec4 state_color = c.success;
    if (preparation_failed_ && !complete) {
        state = blocked ? "Blocked" : "Failed";
        state_color = c.error;
    } else if (preparing) {
        state = "Preparing";
        state_color = c.info;
    } else if (!materialization_notice_.empty() && materialization_events_.empty()) {
        state = "Not applied";
        state_color = c.warning;
    } else if (status == "cache_hit") {
        state = "Reused from cache";
    } else if (status == "cache_saved") {
        state = materialization_rebuild_reason_.empty() ? "Prepared and cached"
                                                        : "Rebuilt and cached";
    } else if (status == "cache_save_failed") {
        state = "Prepared, not cached";
        state_color = c.warning;
    } else if (!complete) {
        state = "Stopped";
        state_color = c.text_dim;
    }

    ui::BeginCard("##dash_dataprep");

    // Header: title, state, actions.
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c.text, "Data preparation");
    ImGui::SameLine(0.0f, 10.0f);
    ui::Chip(state, state_color);
    if (has_actions) {
        const char* rebuild_label = materialization_rebuild_pending_
            ? ICON_FA_ARROWS_ROTATE " Rebuild on next run###dash_rebuild"
            : "Rebuild on next run###dash_rebuild";
        const char* actions_label = "Actions " ICON_FA_CHEVRON_DOWN "###dash_prep_actions";
        const float width = ChipToggleWidth(rebuild_label) +
                            ui::ButtonWidth(actions_label, ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x;
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x - width));
        bool pending = materialization_rebuild_pending_;
        if (ChipToggle(rebuild_label, &pending)) {
            materialization_rebuild_pending_ = pending;
            materialization_action_callback_(pending ? "rebuild" : "cancel_rebuild");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(materialization_rebuild_pending_
                                  ? "The next Train rebuilds the prepared data. Click to cancel."
                                  : "The next Train rebuilds the prepared data instead of reusing it.");
        }
        ImGui::SameLine();
        if (ui::SecondaryButton(actions_label)) {
            ImGui::OpenPopup("##dash_prep_actions_menu");
        }
        if (ImGui::BeginPopup("##dash_prep_actions_menu")) {
            const std::string folder = !materialization_cache_directory_.empty()
                ? materialization_cache_directory_
                : ParentDirectoryForPath(ParentDirectoryForPath(
                      materialization_cache_manifest_path_));
            if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open cache folder", nullptr, false,
                                !folder.empty())) {
                if (OpenDirectoryInFileBrowser(folder)) {
                    RecordPanelEvent("TrainingPlotPanel.OpenCacheLocation", folder);
                } else {
                    RecordPanelEvent("TrainingPlotPanel.OpenCacheLocationFailed", folder);
                }
            }
            if (ImGui::MenuItem(ICON_FA_COPY " Copy prepared data path", nullptr, false,
                                !materialization_cache_artifact_path_.empty())) {
                ImGui::SetClipboardText(materialization_cache_artifact_path_.c_str());
            }
            if (ImGui::MenuItem(ICON_FA_COPY " Copy manifest path", nullptr, false,
                                !materialization_cache_manifest_path_.empty())) {
                ImGui::SetClipboardText(materialization_cache_manifest_path_.c_str());
            }
            if (ImGui::MenuItem(ICON_FA_ARROWS_ROTATE " Refresh cache size")) {
                materialization_action_callback_("refresh");
            }
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, c.error);
            const bool clear = ImGui::MenuItem(ICON_FA_TRASH " Clear prepared-data cache...",
                                               nullptr, false, !is_preparing_);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && is_preparing_) {
                ImGui::SetTooltip("Available after the current preparation finishes.");
            }
            if (clear) {
                materialization_clear_confirm_ = true;
            }
            ImGui::EndPopup();
        }
    }

    // The prepared dataset and a one-line summary.
    if (!materialization_output_dataset_.empty()) {
        ImGui::TextUnformatted(materialization_output_dataset_.c_str());
    }
    {
        std::string summary;
        if (materialization_operators_applied_ > 0) {
            summary = std::to_string(materialization_operators_applied_) +
                      " preprocessing node" +
                      (materialization_operators_applied_ == 1 ? "" : "s");
        }
        if (materialization_cache_row_count_ > 0 || materialization_cache_column_count_ > 0) {
            summary += (summary.empty() ? "" : " \xC2\xB7 ") +
                       std::to_string(materialization_cache_row_count_) + " rows x " +
                       std::to_string(materialization_cache_column_count_) + " columns";
        }
        if (!summary.empty()) {
            ImGui::TextColored(c.text_dim, "%s", summary.c_str());
        }
    }
    if (latest && !preparing && !latest->message.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.text_dim, "%s", latest->message.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!materialization_rebuild_reason_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.warning, "Rebuilt: %s", materialization_rebuild_reason_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!materialization_notice_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.warning, ICON_FA_TRIANGLE_EXCLAMATION " %s",
                           materialization_notice_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (preparing && latest) {
        ui::ProgressLine(latest->progress, c.accent);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::Text("%s", latest->message.c_str());
        ImGui::PopTextWrapPos();
    }

    // Steps.
    if (!materialization_events_.empty()) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ui::WithAlpha(c.border, 0.6f));
        if (ImGui::BeginTable("##dash_prep_steps", 3,
                              ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn("icon", ImGuiTableColumnFlags_WidthFixed, 18.0f);
            ImGui::TableSetupColumn("step", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("time", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            for (size_t i = 0; i < materialization_events_.size(); ++i) {
                const auto& event = materialization_events_[i];
                const bool last = i + 1 == materialization_events_.size();
                const bool running = last && preparing;
                const bool stale = event.status == "cache_stale" ||
                                   event.status == "cache_corrupt" ||
                                   event.status == "cache_miss";
                const bool failed_step = event.status == "blocked" ||
                                         (last && preparation_failed_ && !complete);
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (failed_step) {
                    ImGui::TextColored(c.error, ICON_FA_CIRCLE_XMARK);
                } else if (running) {
                    ImGui::TextColored(c.info, ICON_FA_SPINNER);
                } else if (stale) {
                    ImGui::TextColored(c.warning, ICON_FA_ARROWS_ROTATE);
                } else {
                    ImGui::TextColored(c.success, ICON_FA_CHECK);
                }
                ImGui::TableNextColumn();
                const std::string title = event.stage == "MaterializationCache"
                    ? std::string("Cache decision")
                    : event.stage;
                ImGui::TextUnformatted(title.c_str());
                std::string detail;
                const auto add = [&detail](const std::string& part) {
                    if (part.empty()) return;
                    detail += (detail.empty() ? "" : " \xC2\xB7 ") + part;
                };
                if (event.message != event.stage) add(event.message);
                if (!event.node_name.empty()) add("node " + event.node_name);
                if (event.total_items > 0) {
                    add(std::to_string(event.processed_items) + " / " +
                        std::to_string(event.total_items));
                } else if (event.processed_items > 0) {
                    add(std::to_string(event.processed_items) + " processed");
                }
                if (event.estimated_memory_bytes > 0) {
                    add("estimate " + FormatTraceBytes(event.estimated_memory_bytes));
                }
                if (!event.memory_risk_level.empty()) add("risk " + event.memory_risk_level);
                if (event.process_memory_detected) {
                    add("process " + FormatTraceBytes(event.process_resident_memory_bytes) +
                        " (+" + FormatTraceBytes(event.process_resident_growth_bytes) + ")");
                }
                if (event.cache_row_count > 0 || event.cache_column_count > 0) {
                    add(std::to_string(event.cache_row_count) + " x " +
                        std::to_string(event.cache_column_count));
                }
                if (!detail.empty()) {
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextColored(c.text_faint, "%s", detail.c_str());
                    ImGui::PopTextWrapPos();
                }
                ImGui::TableNextColumn();
                const auto end_time = !last
                    ? materialization_events_[i + 1].started_at
                    : (running ? now : event.updated_at);
                const double seconds =
                    std::chrono::duration<double>(end_time - event.started_at).count();
                if (running && event.progress > 0.0f && event.progress < 1.0f) {
                    ImGui::TextColored(c.text_faint, "%.0f%%", event.progress * 100.0f);
                } else if (seconds >= 0.05) {
                    ImGui::TextColored(c.text_faint, "%s",
                                       FormatTrainingDuration(seconds).c_str());
                } else {
                    ImGui::TextColored(c.text_faint, "%s", running ? "..." : "<0.1s");
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleColor();
    }

    // Cache usage.
    ImGui::Spacing();
    if (materialization_cache_entries_ >= 0) {
        ImGui::TextColored(c.text_dim, "Prepared-data cache");
        ImGui::SameLine();
        std::string usage = std::to_string(materialization_cache_entries_) + " dataset" +
                            (materialization_cache_entries_ == 1 ? "" : "s") + " \xC2\xB7 " +
                            FormatTraceBytes(materialization_cache_bytes_);
        if (materialization_cache_limit_bytes_ > 0) {
            usage += " of " + FormatTraceBytes(materialization_cache_limit_bytes_);
        }
        ImGui::TextUnformatted(usage.c_str());
        if (materialization_pruned_entries_ > 0) {
            ImGui::SameLine();
            ImGui::TextColored(c.text_faint, "(removed %d older, %s)",
                               materialization_pruned_entries_,
                               FormatTraceBytes(materialization_pruned_bytes_).c_str());
        }
    }
    if (!materialization_clear_message_.empty()) {
        ImGui::TextColored(c.text_faint, "%s", materialization_clear_message_.c_str());
    }
    if (!PreparationRows().empty()) {
        ImGui::TextColored(c.text_faint, "The file, cache key and memory figures are under Run details.");
    }
    ui::EndCard();

    // Clear confirmation.
    if (materialization_clear_confirm_) {
        ImGui::OpenPopup("Clear prepared-data cache?###dash_clear_cache");
        materialization_clear_confirm_ = false;
    }
    ImGui::SetNextWindowSize(ImVec2(460.0f, 0.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Clear prepared-data cache?###dash_clear_cache", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushTextWrapPos(430.0f);
        if (materialization_cache_entries_ >= 0) {
            ImGui::TextColored(c.text_dim,
                               "Removes %d prepared dataset%s (%s) from this project's cache "
                               "folder. They are rebuilt the next time a graph needs them. "
                               "Your datasets and graphs are not touched.",
                               materialization_cache_entries_,
                               materialization_cache_entries_ == 1 ? "" : "s",
                               FormatTraceBytes(materialization_cache_bytes_).c_str());
        } else {
            ImGui::TextColored(c.text_dim,
                               "Removes every prepared dataset from this project's cache "
                               "folder. They are rebuilt the next time a graph needs them. "
                               "Your datasets and graphs are not touched.");
        }
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        const float buttons = ui::ButtonWidth("Cancel", ui::ButtonSize::Small) + ui::ButtonWidth("Clear cache", ui::ButtonSize::Small) +
                              ImGui::GetStyle().ItemSpacing.x;
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x - buttons));
        if (ui::SecondaryButton("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::DangerButton("Clear cache")) {
            materialization_clear_message_ = "Clearing prepared-data cache...";
            if (materialization_action_callback_) {
                materialization_action_callback_("clear");
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

std::vector<training_dashboard::DetailRow> TrainingPlotPanel::PreparationRows() const {
    std::vector<training_dashboard::DetailRow> rows;
    if (materialization_events_.empty() && materialization_notice_.empty()) return rows;
    const MaterializationProgress* latest =
        materialization_events_.empty() ? nullptr : &materialization_events_.back();
    const auto row = [&rows](const std::string& key, const std::string& value) {
        if (!value.empty()) rows.push_back({key, value});
    };
    if (!materialization_status_.empty()) {
        row("Status", std::string(MaterializationStatusDisplayName(materialization_status_)) + " (" +
                          materialization_status_ + ")");
    }
    if (latest) row("Node", latest->node_name);
    row("Output dataset", materialization_output_dataset_);
    if (materialization_operators_applied_ > 0) {
        row("Operators applied", std::to_string(materialization_operators_applied_));
    }
    row("Cache key", materialization_cache_key_);
    row("Prepared data file", materialization_cache_artifact_path_);
    row("Manifest", materialization_cache_manifest_path_);
    row("Cache folder", materialization_cache_directory_);
    if (latest) {
        if (latest->estimated_memory_bytes > 0) row("Estimated memory", FormatTraceBytes(latest->estimated_memory_bytes));
        if (latest->available_memory_bytes > 0) {
            row("Available RAM / safe budget", FormatTraceBytes(latest->available_memory_bytes) + " / " +
                                                   FormatTraceBytes(latest->safe_memory_budget_bytes));
        }
        if (latest->process_memory_detected) {
            row("Process resident / growth", FormatTraceBytes(latest->process_resident_memory_bytes) + " / +" +
                                                 FormatTraceBytes(latest->process_resident_growth_bytes));
            if (latest->process_private_memory_bytes > 0) {
                row(latest->process_private_memory_name.empty() ? std::string("Process private memory")
                                                                : "Process " + latest->process_private_memory_name,
                    FormatTraceBytes(latest->process_private_memory_bytes));
            }
            row("Memory note", "Process RAM; ArrayFire device memory is reported separately.");
        }
        row("Memory risk", latest->memory_risk_level);
        if (latest->status != "running") row("Decision status", latest->status);
        if (latest->total_items > 0) {
            row("Work", std::to_string(latest->processed_items) + " / " + std::to_string(latest->total_items));
        }
    }
    row("Rebuild reason", materialization_rebuild_reason_);
    if (materialization_cache_limit_bytes_ > 0) {
        row("Cache limit", FormatTraceBytes(materialization_cache_limit_bytes_) +
                               "; least recently used prepared data is removed first");
    }
    return rows;
}

void TrainingPlotPanel::RenderRunComparisonTable() {
    namespace td = training_dashboard;
    const ui::Tokens& c = ui::CurrentTokens();
    const td::RunsTable runs = td::BuildRunsTable(run_comparison_records_, active_checkpoint_loaded_);
    const auto tone = [&c](td::Tone t) {
        switch (t) {
        case td::Tone::Good: return c.success;
        case td::Tone::Bad: return c.caution;
        case td::Tone::Error: return c.error;
        case td::Tone::Faint: return c.text_faint;
        case td::Tone::Plain: break;
        }
        return c.text;
    };

    ui::BeginCard("##dash_runs");
    // The title, then the card's actions at the right (they wrap when narrow).
    ImGui::AlignTextToFramePadding();
    ui::CardTitle("Runs", "this session, best test accuracy first");
    ui::Tooltip("Ranked by test accuracy, then validation accuracy and loss, then time.");
    const bool has_runs = !runs.rows.empty();
    const char* columns_label = runs_all_columns_ ? "Key columns###runs_columns" : "All columns###runs_columns";
    const char* export_label = "Export " ICON_FA_CHEVRON_DOWN "###runs_export";
    const char* clear_label = "Clear runs...";
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float actions_width = ui::ButtonWidth(columns_label, ui::ButtonSize::Small) +
                                ui::ButtonWidth(export_label, ui::ButtonSize::Small) +
                                ui::ButtonWidth(clear_label, ui::ButtonSize::Small) + spacing * 2.0f;
    ui::SameLineRight(actions_width);
    ui::FlowRow buttons;
    buttons.Next(ui::ButtonWidth(columns_label, ui::ButtonSize::Small));
    if (ui::SecondaryButton(columns_label, has_runs, "No run has ended in this session yet.")) {
        runs_all_columns_ = !runs_all_columns_;
    }
    buttons.Next(ui::ButtonWidth(export_label, ui::ButtonSize::Small));
    if (ui::SecondaryButton(export_label)) ImGui::OpenPopup("##runs_export_menu");
    if (ImGui::BeginPopup("##runs_export_menu")) {
        if (ImGui::MenuItem("Metrics (CSV)...")) {
            pending_export_ = PendingExport::Metrics;
            pending_note_at_ = ExportNoteAt::Runs;
        }
        ui::Tooltip("The loss, accuracy and metric points of the run on the charts. Asks where to save.");
        if (ImGui::MenuItem("Runs (CSV)...", nullptr, false, has_runs)) {
            pending_export_ = PendingExport::Runs;
            pending_note_at_ = ExportNoteAt::Runs;
        }
        ui::Tooltip(has_runs ? "Every run in this table, all columns. Asks where to save."
                             : "No run has ended in this session yet.");
        ImGui::EndPopup();
    }
    buttons.Next(ui::ButtonWidth(clear_label, ui::ButtonSize::Small));
    if (ui::DangerButton(clear_label, has_runs, "No run has ended in this session yet.")) open_clear_runs_ = true;
    if (open_clear_runs_) {
        ImGui::OpenPopup("Clear the runs?");
        open_clear_runs_ = false;
    }
    const std::string clear_message = "This removes " + std::to_string(runs.rows.size()) +
                                      (runs.rows.size() == 1 ? " run" : " runs") +
                                      " from the table. Export them first to keep them.";
    if (ui::ConfirmDialog("Clear the runs?", clear_message.c_str(), "Clear runs", true,
                          {"The charts and the trained model stay."}) == ui::DialogResult::Primary) {
        run_comparison_records_.clear();
        runs_all_columns_ = false;
        if (export_note_at_ == ExportNoteAt::Runs) export_note_.clear();
    }

    if (!has_runs) {
        ImGui::TextColored(c.text_faint, "%s", runs.empty.c_str());
    } else {
        // Its own height: up to five rows, then the rows scroll inside; the
        // Run column and the header stay while it scrolls sideways.
        const size_t shown_columns = runs_all_columns_ ? runs.columns.size() : td::RunsTable::kKeyColumns;
        const ImGuiStyle& style = ImGui::GetStyle();
        const float row_height = ImGui::GetTextLineHeight() + style.CellPadding.y * 2.0f;
        const float height = row_height * static_cast<float>(1 + std::min<size_t>(runs.rows.size(), 5)) +
                             style.ScrollbarSize + style.CellPadding.y * 2.0f;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bg_window);
        // ImGui draws a line under the frozen header row whatever the flags.
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, c.bg_window);
        // Room between the columns; each is as wide as its longest value.
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(10.0f, style.CellPadding.y));
        if (ImGui::BeginTable("##dash_runs_table", static_cast<int>(shown_columns),
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX,
                              ImVec2(0.0f, height))) {
            ImGui::TableSetupScrollFreeze(1, 1);
            for (size_t i = 0; i < shown_columns; ++i) ImGui::TableSetupColumn(runs.columns[i].c_str());
            // ImGui's header row, so each column is as wide as its name.
            ImGui::PushStyleColor(ImGuiCol_Text, c.text_dim);
            ImGui::TableHeadersRow();
            ImGui::PopStyleColor();
            for (size_t r = 0; r < runs.rows.size(); ++r) {
                ImGui::TableNextRow();
                for (size_t i = 0; i < shown_columns; ++i) {
                    const td::RunCell& cell = runs.rows[r][i];
                    ImGui::TableSetColumnIndex(static_cast<int>(i));
                    ImGui::TextColored(tone(cell.tone), "%s", cell.text.c_str());
                    if (!cell.tooltip.empty()) ui::Tooltip(cell.tooltip.c_str());
                }
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
        const std::string& note = runs_all_columns_ ? runs.all_note : runs.key_note;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.text_faint, "%s", note.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!runs.loaded_note.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.text_faint, "%s", runs.loaded_note.c_str());
        ImGui::PopTextWrapPos();
    }
    if (export_note_at_ == ExportNoteAt::Runs && !export_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(export_failed_ ? c.error : c.success, "%s", export_note_.c_str());
        ImGui::PopTextWrapPos();
    }
    ui::EndCard();
}

void TrainingPlotPanel::ClearMissingValues() {
    missing_columns_.clear();
    missing_values_total_rows_ = -1;
    missing_values_supplied_role_.clear();
}

// Under a run refused for missing values: the same card as the "Cannot start
// training" report, its steps added by the main window; Train again once
// every group has its step (mockup board 2).
void TrainingPlotPanel::RenderMissingValuesFix(const std::vector<MissingColumn>& columns, int64_t total_rows,
                                               const std::string& input_name) {
    namespace mv = missing_values_fix;
    mv::Input in = mv::MakeInput(columns, total_rows, input_name, missing_values_supplied_role_);
    in.number_fill = missing_values_fill_;
    if (missing_values_actions_.state) missing_values_actions_.state(in);
    const mv::Card card = mv::BuildCard(in);
    const bool can_act = static_cast<bool>(missing_values_actions_.add);
    ImGui::PushID("missing_values_fix");
    for (const bool label : {false, true}) {
        const mv::Group& g = label ? card.label : card.features;
        mv::Group shown = g;
        if (!can_act) shown.can_fill = shown.can_drop = false;  // no canvas to add the steps to
        shown.can_leave_out.clear();  // the column's role is set from the report or Data Studio
        const auto events = ui::MissingValuesGroup(label ? "label" : "features", shown, label, missing_values_fill_);
        if (events.fill && can_act) missing_values_actions_.add(false, g.fill_steps, in);
        if (events.drop && can_act) missing_values_actions_.add(label, g.drop_steps, in);
        if (events.undo && missing_values_actions_.undo) missing_values_actions_.undo(label);
    }
    const ui::Tokens& c = ui::CurrentTokens();
    if (missing_values_actions_.show_input) {
        if (ui::LinkButton("Show on canvas")) missing_values_actions_.show_input();
    }
    if (card.resolved && missing_values_actions_.train_again) {
        ImGui::SameLine();
        if (ui::PrimaryButton("Train again")) missing_values_actions_.train_again();
    } else if (!card.resolved) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.text_faint, "%s",
                           "The same steps as before training. After one is added, Train again is offered here; "
                           "the next run checks every batch the same way.");
        ImGui::PopTextWrapPos();
    }
    ImGui::PopID();
}

training_dashboard::HeaderFacts TrainingPlotPanel::HeaderFactsNow() const {
    training_dashboard::HeaderFacts h;
    h.preparing = is_preparing_;
    h.preparation_failed = preparation_failed_;
    h.preparation_message = preparation_status_message_;
    h.preparation_error = preparation_error_message_;
    h.preparation_progress = preparation_progress_;
    h.terminal_status = terminal_status_;
    h.terminal_reason = terminal_reason_;
    h.missing_columns = missing_columns_;
    h.dataset = run_dataset_;
    h.train_rows = run_train_rows_;
    h.validation_rows = run_validation_rows_;
    h.test_rows = run_test_rows_;
    h.samples_per_batch = samples_per_batch_;
    h.batches_per_update = batches_per_update_;
    h.metric_interval = metric_reporting_interval_;
    h.last_batch_loss = current_batch_loss_;
    h.active_model_provenance = active_model_provenance_;
    h.checkpoint_loaded = active_checkpoint_loaded_;
    h.checkpoint_path = checkpoint_used_;
    h.checkpoint_epoch = checkpoint_epoch_;
    h.checkpoint_step = checkpoint_step_;
    h.has_checkpoint_validation = has_checkpoint_validation_metrics_;
    h.checkpoint_val_loss = checkpoint_val_loss_;
    h.checkpoint_val_accuracy = checkpoint_val_accuracy_;
    // A regression run reports MAE / RMSE and no accuracy.
    for (const auto& metric : custom_metrics_) {
        if (IsRegressionMetricName(metric.name) && !metric.values.empty()) h.classification = false;
    }
    h.has_test = has_test_metrics_;
    h.test_loss = test_loss_;
    h.test_accuracy = test_accuracy_;
    h.test_mae = test_mae_;
    return h;
}

void TrainingPlotPanel::RenderTrainingStatus() {
    namespace td = training_dashboard;
    const ui::Tokens& c = ui::CurrentTokens();
    td::HeaderFacts facts = HeaderFactsNow();
    std::vector<td::DetailRow> where_rows;
    std::vector<std::string> warnings;
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
    auto& tm = TrainingManager::Instance();
    const bool training_active = tm.IsTrainingActive();
    facts.paused = tm.IsPaused();
    if (training_active) facts.waiting_for_gpu = tm.WaitingForGpu();
    const TrainingTraceSummary& trace = LatestTraceCached();
    facts.route = TraceRoute(trace);
    for (const auto& task : AsyncTaskManager::Instance().GetActiveTasks()) {
        facts.tasks.push_back({task.name, task.progress, task.status_message});
    }
#else
    const bool training_active = false;
#endif
    const td::Header header = td::BuildHeader(facts_, facts);
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
    // The trace is the last run's: shown for a run that trains or ended, not
    // while idle, preparing, waiting or with a loaded checkpoint.
    switch (header.state) {
    case td::RunState::Training:
    case td::RunState::Paused:
    case td::RunState::EarlyStopped:
    case td::RunState::Stopped:
    case td::RunState::Failed:
    case td::RunState::Completed:
        where_rows = TraceRows(trace);
        warnings = TraceWarnings(trace);
        break;
    case td::RunState::Idle:
    case td::RunState::WaitingForGpu:
    case td::RunState::PreparationFailed:
    case td::RunState::Preparing:
    case td::RunState::ModelLoaded: break;
    }
#endif

    ImVec4 state_colour = c.text_dim;
    switch (header.state) {
    case td::RunState::WaitingForGpu:
    case td::RunState::Stopped: state_colour = c.caution; break;
    case td::RunState::Training:
    case td::RunState::ModelLoaded: state_colour = c.success; break;
    case td::RunState::Paused:
    case td::RunState::EarlyStopped: state_colour = c.warning; break;
    case td::RunState::PreparationFailed:
    case td::RunState::Failed: state_colour = c.error; break;
    case td::RunState::Preparing:
    case td::RunState::Completed: state_colour = c.info; break;
    case td::RunState::Idle: break;
    }
    const auto tone = [&](td::Tone t) {
        switch (t) {
        case td::Tone::Good: return c.success;
        case td::Tone::Bad: return c.caution;
        case td::Tone::Error: return c.error;
        case td::Tone::Faint: return c.text_dim;
        case td::Tone::Plain: break;
        }
        return c.text;
    };
    const auto wrapped = [](const ImVec4& colour, const std::string& text) {
        if (text.empty()) return;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(colour, "%s", text.c_str());
        ImGui::PopTextWrapPos();
    };

    ui::BeginCard("##dash_header");

    // Row 1: the state, then the run's actions at the right; each part
    // moves to the next line when the slot is narrow (B4-2).
    ui::FlowRow row;
    row.Next(0.0f);
    ui::Chip(header.pill.c_str(), state_colour);
    if (!header.status.empty()) {
        row.Next(ImGui::CalcTextSize(header.status.c_str()).x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.text, "%s", header.status.c_str());
        ui::Tooltip(header.status_tooltip.c_str());
    }
    const char* pause_label = facts.paused ? ICON_FA_PLAY " Resume" : ICON_FA_PAUSE " Pause";
    const char* stop_label = ICON_FA_STOP " Stop";
    const char* more_label = "More " ICON_FA_CHEVRON_DOWN;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    float actions_width = ui::ButtonWidth(more_label, ui::ButtonSize::Small);
    if (training_active) {
        actions_width += ui::ButtonWidth(pause_label, ui::ButtonSize::Small) +
                         ui::ButtonWidth(stop_label, ui::ButtonSize::Small) + spacing * 2.0f;
    }
    ui::SameLineRight(actions_width);
    ui::FlowRow buttons;
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
    if (training_active) {
        buttons.Next(ui::ButtonWidth(pause_label, ui::ButtonSize::Small));
        if (ui::SecondaryButton(pause_label)) {
            if (facts.paused) {
                tm.ResumeTraining();
            } else {
                tm.PauseTraining();
            }
        }
        buttons.Next(ui::ButtonWidth(stop_label, ui::ButtonSize::Small));
        if (ui::DangerButton(stop_label)) tm.StopTraining();
        ui::Tooltip("Stops training at the next batch boundary and keeps the current model state.");
    }
#endif
    buttons.Next(ui::ButtonWidth(more_label, ui::ButtonSize::Small));
    if (ui::SecondaryButton(more_label)) ImGui::OpenPopup("##dash_actions");
    if (ImGui::BeginPopup("##dash_actions")) {
        if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export metrics (CSV)...")) {
            pending_export_ = PendingExport::Metrics;
            pending_note_at_ = ExportNoteAt::Header;
        }
        ui::Tooltip("The loss, accuracy and metric points of this run. Asks where to save.");
        if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export runs (CSV)...", nullptr, false,
                            !run_comparison_records_.empty())) {
            pending_export_ = PendingExport::Runs;
            pending_note_at_ = ExportNoteAt::Header;
        }
        ui::Tooltip(run_comparison_records_.empty() ? "No run has ended in this session yet."
                                                    : "Every run in the Runs table, all columns. Asks where to save.");
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, c.error);
        if (ImGui::MenuItem(ICON_FA_TRASH " Clear all...")) open_clear_all_ = true;
        ImGui::PopStyleColor();
        ui::Tooltip("Clears this run's charts, cards and metrics from the dashboard. Asks first.");
        ImGui::EndPopup();
    }
    if (open_clear_all_) {
        ImGui::OpenPopup("Clear the dashboard?");
        open_clear_all_ = false;
    }
    if (ui::ConfirmDialog("Clear the dashboard?",
                          "This clears the run's charts, cards and metrics from the dashboard.", "Clear all", true,
                          {"The Runs table keeps its rows.", "The trained model and its checkpoint stay."}) ==
        ui::DialogResult::Primary) {
        ClearLocked();
        export_note_.clear();
        facts_ = DashboardFacts();  // the cards below draw this frame
    }
    if (export_note_at_ == ExportNoteAt::Header) wrapped(export_failed_ ? c.error : c.success, export_note_);

    // Row 2: progress.
    if (header.show_progress) {
        const bool ended = header.state == td::RunState::Completed || header.state == td::RunState::EarlyStopped ||
                           header.state == td::RunState::Stopped || header.state == td::RunState::Failed;
        const ImVec4 bar = header.state == td::RunState::Preparing ? c.info : (ended ? state_colour : c.accent);
        ui::ProgressLine(static_cast<float>(header.progress), bar);
        ui::Tooltip(header.progress_tooltip.c_str());
    }
    wrapped(c.text_dim, header.progress_text);

    // Where it ran, on what data, and the trace's warnings.
    wrapped(c.text_dim, header.where);
    ui::Tooltip(header.where_tooltip.c_str());
    wrapped(c.text_dim, header.data);
    for (const std::string& warning : warnings) {
        // A missing-values refusal has its card under the outcome: the
        // task's copy of the raw message is not repeated (Run details keep it).
        if (header.missing_values && warning.find("the data has missing values") != std::string::npos) continue;
        wrapped(c.caution, std::string(ICON_FA_TRIANGLE_EXCLAMATION " ") + warning);
    }

    // While it trains: the latest batches. After: how it ended, the active
    // model and the held-out test.
    LineWithTail(header.now, c.text, header.now_detail, c.text_dim);
    wrapped(tone(header.outcome_tone), header.outcome);
    if (header.missing_values) RenderMissingValuesFix(facts.missing_columns, missing_values_total_rows_, facts.dataset);
    LineWithTail(header.active_model, c.text, header.active_model_detail, c.text_dim);
    LineWithTail(header.test, c.text, header.test_detail, c.text_dim);

    // Run details: the codes, paths and figures behind the lines above.
    const std::vector<td::DetailRow> preparation_rows = PreparationRows();
    if (!header.run_rows.empty() || !where_rows.empty() || !preparation_rows.empty()) {
        std::string what;
        const auto part = [&what](const char* p) {
            if (what.find(p) == std::string::npos) what += (what.empty() ? "" : ", ") + std::string(p);
        };
        for (const auto& r : header.run_rows) {
            if (r.key == "Stop code") part("stop code");
            if (r.key == "Checkpoint" || r.key == "Path") part("checkpoint");
            if (r.key == "Batches" || r.key == "Last batch loss") part("batches");
            if (r.key.rfind("Engine task", 0) == 0) part("engine tasks");
        }
        if (!where_rows.empty()) part("where it ran");
        if (!preparation_rows.empty()) part("data preparation");
        ImGui::Spacing();
        ImGui::SetNextItemOpen(run_details_open_);
        run_details_open_ = ImGui::TreeNodeEx("Run details##dash_run_details",
                                              ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanLabelWidth);
        ImGui::SameLine();
        wrapped(c.text_faint, what);
        if (run_details_open_) {
            struct Group {
                const char* title;
                const std::vector<td::DetailRow>* rows;
            };
            std::vector<Group> groups;
            if (!header.run_rows.empty()) groups.push_back({"The run", &header.run_rows});
            if (!where_rows.empty()) groups.push_back({"Where it ran", &where_rows});
            if (!preparation_rows.empty()) groups.push_back({"Data", &preparation_rows});
            // Side by side when the window is wide, one under the other otherwise.
            const int columns = ImGui::GetContentRegionAvail().x >= 900.0f ? static_cast<int>(groups.size()) : 1;
            if (columns > 1 && ImGui::BeginTable("##dash_run_details_groups", columns,
                                                 ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
                for (const Group& g : groups) {
                    ImGui::TableNextColumn();
                    DetailGroup(g.title, *g.rows);
                }
                ImGui::EndTable();
            } else {
                for (const Group& g : groups) DetailGroup(g.title, *g.rows);
            }
        }
    }

    ui::EndCard();
}

void TrainingPlotPanel::RenderStatistics() {
    const ui::Tokens& c = ui::CurrentTokens();
    ui::BeginCard("##dash_stats");
    ui::CardTitle("Statistics", "mean of the last 10 points; min and max of the run");
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ui::WithAlpha(c.border, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
    if (ImGui::BeginTable("##dash_stats_table", 4,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Series", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, c.text_dim);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();

        const auto series_row = [&](const char* label, const MetricSeries& series) {
            if (series.values.empty()) {
                return;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const float line = ImGui::GetTextLineHeight();
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 4.0f, pos.y + line * 0.5f), 3.5f,
                ImGui::GetColorU32(series.color));
            ImGui::Dummy(ImVec2(8.0f, line));
            ImGui::SameLine(0.0f, 6.0f);
            ImGui::TextUnformatted(label);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", CalculateMean(series.values));
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", CalculateMin(series.values));
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", CalculateMax(series.values));
        };
        series_row("Train loss", train_loss_);
        series_row("Val loss", val_loss_);
        ImGui::EndTable();
    }
    ImGui::PopStyleColor(2);
    ui::EndCard();
}

void TrainingPlotPanel::TrimDataIfNeeded(MetricSeries& series) {
    // Trimmed in chunks of a tenth: erasing the front for every new point
    // moved the whole series each time once it was full (TOFIX134 P0 item 8).
    if (series.epochs.size() > max_points_ + max_points_ / 10) {
        size_t to_remove = series.epochs.size() - max_points_;
        RecordPanelEvent("TrainingPlotPanel.TrimData",
                         series.name + " remove=" + std::to_string(to_remove));
        series.epochs.erase(series.epochs.begin(), series.epochs.begin() + to_remove);
        series.values.erase(series.values.begin(), series.values.begin() + to_remove);
    }
}

void TrainingPlotPanel::RecordPanelEvent(const std::string& action,
                                         const std::string& detail) const {
#ifndef CYXWIZ_DASHBOARD_WITHOUT_TRAINING
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

namespace {
// Enough for a full-screen chart; min/max per bucket keeps spikes.
constexpr size_t kDrawnPoints = 4000;
}

const TrainingPlotPanel::DrawnCache& TrainingPlotPanel::Drawn() {
    if (drawn_.version == data_version_ && drawn_.smoothing == smoothing_window_ &&
        drawn_.smoothed == show_smoothed_curves_) {
        return drawn_;
    }
    const auto reduce = [](const MetricSeries& from) {
        MetricSeries line;
        line.name = from.name;
        line.color = from.color;
        auto d = series::MinMaxDecimate(from.epochs, from.values, kDrawnPoints);
        line.epochs = std::move(d.x);
        line.values = std::move(d.y);
        return line;
    };
    const auto build = [&](const MetricSeries& from, DrawnSeries& to) {
        to.line = reduce(from);
        to.smooth_x.clear();
        to.smooth_y.clear();
        if (show_smoothed_curves_ && static_cast<int>(from.values.size()) >= smoothing_window_) {
            const auto avg = series::MovingAverage(from.values, smoothing_window_);
            auto d = series::MinMaxDecimate(from.epochs, avg, kDrawnPoints);
            to.smooth_x = std::move(d.x);
            to.smooth_y = std::move(d.y);
        }
    };
    build(train_loss_, drawn_.train_loss);
    build(val_loss_, drawn_.val_loss);
    build(train_accuracy_, drawn_.train_accuracy);
    build(val_accuracy_, drawn_.val_accuracy);
    drawn_.custom.clear();
    for (const auto& metric : custom_metrics_) drawn_.custom.push_back(reduce(metric));
    drawn_.version = data_version_;
    ++drawn_.build;
    drawn_.smoothing = smoothing_window_;
    drawn_.smoothed = show_smoothed_curves_;
    return drawn_;
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
        : (is_preparing_ ? preparation_status_message_ : terminal_reason_);
    snapshot.terminal_status = terminal_status_;
    snapshot.terminal_reason = terminal_reason_;
    snapshot.current_epoch = current_epoch_;
    snapshot.last_executed_epoch = last_executed_epoch_;
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
    snapshot.checkpoint_step = checkpoint_step_;
    snapshot.checkpoint_used = checkpoint_used_;
    snapshot.active_model_provenance = active_model_provenance_;
    snapshot.metric_points = train_loss_.values.size();
    snapshot.materialization_status = materialization_status_;
    if (!materialization_events_.empty()) {
        snapshot.materialization_message =
            materialization_events_.back().message;
    }
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

} // namespace cyxwiz
