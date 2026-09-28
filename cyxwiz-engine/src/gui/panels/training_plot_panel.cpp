#include "training_plot_panel.h"
#ifndef CYXWIZ_PLOTTING_MODULE
#include "../../core/async_task_manager.h"
#include "../../core/crash_run_recorder.h"
#include "../../core/training_manager.h"
#include "../../core/training_trace_collector.h"
#endif
#include "../../core/training_run_comparison.h"
#include "../../core/route_qualification_snapshot.h"
#include "../icons.h"
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
// ---- Dashboard look ----------------------------------------------------
// This file is also compiled into test targets and the Python plotting
// module, so the dashboard styles itself from the active ImGui theme here
// instead of depending on the engine's shared button and palette helpers.

struct DashColors {
    ImVec4 text;
    ImVec4 muted;
    ImVec4 faint;
    ImVec4 window;
    ImVec4 card;
    ImVec4 track;
    ImVec4 border;
    ImVec4 accent;
    ImVec4 accent_text;
    ImVec4 success;
    ImVec4 warning;
    ImVec4 caution;
    ImVec4 error;
    ImVec4 info;
    bool light = false;
};

ImVec4 MixColor(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                  a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

ImVec4 WithAlpha(ImVec4 color, float alpha) {
    color.w = alpha;
    return color;
}

DashColors CurrentDashColors() {
    const ImGuiStyle& style = ImGui::GetStyle();
    DashColors c;
    c.window = WithAlpha(style.Colors[ImGuiCol_WindowBg], 1.0f);
    const float luminance =
        0.299f * c.window.x + 0.587f * c.window.y + 0.114f * c.window.z;
    c.light = luminance > 0.5f;
    const ImVec4 lift = c.light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);
    c.text = style.Colors[ImGuiCol_Text];
    c.muted = style.Colors[ImGuiCol_TextDisabled];
    c.faint = MixColor(c.muted, c.window, 0.35f);
    c.card = WithAlpha(MixColor(c.window, lift, c.light ? 0.035f : 0.03f), 1.0f);
    c.track = WithAlpha(MixColor(c.window, lift, c.light ? 0.09f : 0.08f), 1.0f);
    c.border = WithAlpha(style.Colors[ImGuiCol_Border], 1.0f);
    c.accent = WithAlpha(style.Colors[ImGuiCol_SliderGrab], 1.0f);
    c.accent_text = WithAlpha(style.Colors[ImGuiCol_CheckMark], 1.0f);
    if (c.light) {
        c.success = ImVec4(0.10f, 0.55f, 0.32f, 1.0f);
        c.warning = ImVec4(0.70f, 0.47f, 0.02f, 1.0f);
        c.caution = ImVec4(0.80f, 0.38f, 0.10f, 1.0f);
        c.error = ImVec4(0.80f, 0.22f, 0.20f, 1.0f);
        c.info = ImVec4(0.15f, 0.45f, 0.80f, 1.0f);
    } else {
        c.success = ImVec4(0.24f, 0.84f, 0.55f, 1.0f);
        c.warning = ImVec4(0.89f, 0.70f, 0.25f, 1.0f);
        c.caution = ImVec4(0.96f, 0.60f, 0.35f, 1.0f);
        c.error = ImVec4(1.0f, 0.48f, 0.45f, 1.0f);
        c.info = ImVec4(0.45f, 0.72f, 1.0f, 1.0f);
    }
    return c;
}

// Card: a filled, rounded, borderless group sized to its content.
bool BeginDashCard(const char* id, const DashColors& c, float width = 0.0f) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, c.card);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 10.0f));
    const bool open = ImGui::BeginChild(
        id, ImVec2(width, 0.0f),
        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    return open;
}

void EndDashCard() {
    ImGui::EndChild();
}

void DashCardTitle(const char* title, const char* subtitle, const DashColors& c) {
    ImGui::TextColored(c.text, "%s", title);
    if (subtitle && *subtitle) {
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::TextColored(c.muted, "%s", subtitle);
    }
}

// Status pill: tinted background, a dot and the label in the status colour.
void DashPill(const char* label, const ImVec4& color, bool dot = true) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 text = ImGui::CalcTextSize(label);
    const float pad_x = 9.0f;
    const float pad_y = 3.0f;
    const float dot_space = dot ? 12.0f : 0.0f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(text.x + pad_x * 2.0f + dot_space, text.y + pad_y * 2.0f);
    draw->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                        ImGui::GetColorU32(WithAlpha(color, 0.16f)), size.y * 0.5f);
    if (dot) {
        draw->AddCircleFilled(ImVec2(pos.x + pad_x + 3.5f, pos.y + size.y * 0.5f), 3.5f,
                              ImGui::GetColorU32(color));
    }
    draw->AddText(ImVec2(pos.x + pad_x + dot_space, pos.y + pad_y),
                  ImGui::GetColorU32(color), label);
    ImGui::Dummy(size);
}

// Thin rounded progress bar.
void DashProgress(float fraction, const ImVec4& color, const DashColors& c,
                  float width = -1.0f, float height = 6.0f) {
    fraction = std::clamp(fraction, 0.0f, 1.0f);
    if (width <= 0.0f) {
        width = ImGui::GetContentRegionAvail().x;
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float line_h = ImGui::GetTextLineHeight();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float y = pos.y + (line_h - height) * 0.5f;
    draw->AddRectFilled(ImVec2(pos.x, y), ImVec2(pos.x + width, y + height),
                        ImGui::GetColorU32(c.track), height * 0.5f);
    if (fraction > 0.0f) {
        draw->AddRectFilled(ImVec2(pos.x, y),
                            ImVec2(pos.x + std::max(height, width * fraction), y + height),
                            ImGui::GetColorU32(color), height * 0.5f);
    }
    ImGui::Dummy(ImVec2(width, line_h));
}

enum class DashButtonKind { Secondary, Danger };

bool DashButton(const char* label, DashButtonKind kind, const DashColors& c) {
    const ImVec4 fg = kind == DashButtonKind::Danger ? c.error : c.text;
    const ImVec4 hover = kind == DashButtonKind::Danger
        ? WithAlpha(c.error, 0.14f)
        : WithAlpha(c.text, 0.07f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, WithAlpha(hover, hover.w * 1.8f));
    ImGui::PushStyleColor(ImGuiCol_Border,
                          kind == DashButtonKind::Danger ? WithAlpha(c.error, 0.45f) : c.border);
    ImGui::PushStyleColor(ImGuiCol_Text, fg);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 4.0f));
    const bool pressed = ImGui::Button(label);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(5);
    return pressed;
}

float DashButtonWidth(const char* label) {
    return ImGui::CalcTextSize(label, nullptr, true).x + 20.0f;
}

// Toggle chip: filled with the accent tint while on.
bool DashChip(const char* label, bool* value, const DashColors& c) {
    const bool on = *value;
    ImGui::PushStyleColor(ImGuiCol_Button, on ? WithAlpha(c.accent, 0.24f) : ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          on ? WithAlpha(c.accent, 0.34f) : WithAlpha(c.text, 0.07f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, WithAlpha(c.accent, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_Border, on ? WithAlpha(c.accent, 0.55f) : c.border);
    ImGui::PushStyleColor(ImGuiCol_Text, on ? c.accent_text : c.muted);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 11.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 3.0f));
    const bool pressed = ImGui::Button(label);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(5);
    if (pressed) {
        *value = !*value;
    }
    return pressed;
}

// Bold UI font for metric values (falls back to the current font).
ImFont* DashValueFont() {
    ImGuiIO& io = ImGui::GetIO();
    for (ImFont* font : io.Fonts->Fonts) {
        if (!font || !font->Sources || font->SourcesCount <= 0) {
            continue;
        }
        const char* name = font->Sources[0].Name;
        if (std::strstr(name, "Bold") && !std::strstr(name, "Mono")) {
            return font;
        }
    }
    return ImGui::GetFont();
}

void DashBigValue(const char* text, const ImVec4& color) {
    ImFont* font = DashValueFont();
    const float size = ImGui::GetFontSize() * 1.45f;
    const ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
    ImGui::GetWindowDrawList()->AddText(font, size, ImGui::GetCursorScreenPos(),
                                        ImGui::GetColorU32(color), text);
    ImGui::Dummy(extent);
}

std::string FormatEpochValue(double epoch) {
    char buffer[32];
    if (std::abs(epoch - std::round(epoch)) < 1e-6) {
        std::snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(std::round(epoch)));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f", epoch);
    }
    return buffer;
}

// Plot styling: transparent frame and plot area (the card is the frame),
// soft grid, muted axes, a translucent legend inside the plot.
void PushDashPlotStyle(const DashColors& c) {
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0, 0, 0, 0));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0, 0, 0, 0));
    ImPlot::PushStyleColor(ImPlotCol_LegendBg, WithAlpha(c.window, 0.82f));
    ImPlot::PushStyleColor(ImPlotCol_LegendBorder, WithAlpha(c.border, 0.8f));
    ImPlot::PushStyleColor(ImPlotCol_LegendText, c.text);
    ImPlot::PushStyleColor(ImPlotCol_AxisText, c.muted);
    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, WithAlpha(c.muted, c.light ? 0.18f : 0.13f));
    ImPlot::PushStyleColor(ImPlotCol_AxisTick, WithAlpha(c.muted, 0.35f));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(2.0f, 6.0f));
    ImPlot::PushStyleVar(ImPlotStyleVar_LegendPadding, ImVec2(10.0f, 10.0f));
    ImPlot::PushStyleVar(ImPlotStyleVar_LegendInnerPadding, ImVec2(8.0f, 5.0f));
    ImPlot::PushStyleVar(ImPlotStyleVar_LegendSpacing, ImVec2(6.0f, 3.0f));
}

void PopDashPlotStyle() {
    ImPlot::PopStyleVar(4);
    ImPlot::PopStyleColor(9);
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

const TrainingTraceEvent* FindLatestNativeCpuFallbackEvent(
    const TrainingTraceSummary& trace) {
    for (auto it = trace.recent_events.rbegin();
         it != trace.recent_events.rend();
         ++it) {
        if (it->native_cpu_fallback) {
            return &(*it);
        }
    }
    return nullptr;
}

bool IsPinMemoryTransferWarning(const std::string& warning) {
    return warning.rfind("DataLoader.PinMemoryTransfer", 0) == 0;
}

bool HasResidencyVerdict(const TrainingTraceSummary& trace) {
    return !trace.residency_verdict.empty() &&
           trace.residency_verdict != "unavailable" &&
           trace.residency_verdict != "in_progress";
}

const char* ResidencyVerdictDisplayName(const std::string& verdict) {
    if (verdict == "strict_arrayfire_declared_boundaries") {
        return "Strict ArrayFire residency";
    }
    if (verdict == "native_cpu_fallback_observed") {
        return "Native CPU fallback observed";
    }
    if (verdict == "compatibility_no_observed_fallback") {
        return "ArrayFire-first; no fallback observed";
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

ImVec4 ResidencyVerdictColor(const std::string& verdict) {
    if (verdict == "strict_arrayfire_declared_boundaries") {
        return ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
    }
    if (verdict == "native_cpu_fallback_observed" ||
        verdict == "terminal_without_residency_pass") {
        return ImVec4(1.0f, 0.38f, 0.38f, 1.0f);
    }
    return ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
}

void RenderExecutionTruthRow(const char* label, const std::string& value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", label);
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", value.empty() ? "Not recorded" : value.c_str());
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
#endif

} // namespace

TrainingPlotPanel::TrainingPlotPanel()
    : Panel("Training Dashboard") {

    // Initialize metric series
    train_loss_.name = "Training Loss";
    train_loss_.color = ImVec4(1.0f, 0.48f, 0.45f, 1.0f);  // Coral

    val_loss_.name = "Validation Loss";
    val_loss_.color = ImVec4(0.42f, 0.66f, 1.0f, 1.0f);  // Blue

    train_accuracy_.name = "Training Accuracy";
    train_accuracy_.color = ImVec4(0.24f, 0.84f, 0.55f, 1.0f);  // Green

    val_accuracy_.name = "Validation Accuracy";
    val_accuracy_.color = ImVec4(0.95f, 0.76f, 0.30f, 1.0f);  // Amber

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

    // Header: run status, progress and run actions (always visible).
    RenderTrainingStatus();
    RenderActiveTaskSummary();
    RenderMaterializationSummary();
    RenderTrainingWarningSummary();

    // Metric cards: latest values, change and timing.
    RenderKpiCards();

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
    } else {
        RenderEmptyState();
    }

    RenderRunComparisonTable();

    ImGui::End();
}

void TrainingPlotPanel::RenderEmptyState() {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (BeginDashCard("##dash_empty", c)) {
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
                                                ImGui::GetColorU32(c.faint), icon);
            ImGui::Dummy(extent);
        }
        ImGui::Spacing();
        centered("No training data yet", c.text);
        centered("Press Train on the canvas, or run a training script, to see real-time metrics", c.muted);
        ImGui::Spacing();
        centered("Try: scripts/train_xor_simple.py", c.accent_text);
        ImGui::Spacing();
        centered("Loss and accuracy charts will appear here", c.faint);
        ImGui::Dummy(ImVec2(0.0f, 48.0f));
    }
    EndDashCard();
}

void TrainingPlotPanel::RenderKpiCards() {
    const DashColors c = CurrentDashColors();

    struct Card {
        std::string id;
        std::string label;
        ImVec4 dot;
        std::string value;
        std::string tooltip;
        std::string sub1;
        ImVec4 sub1_color;
        std::string sub2;
        ImVec4 sub2_color;
    };
    std::vector<Card> cards;

    const auto count_of = [](const MetricSeries& s) {
        return std::min(s.epochs.size(), s.values.size());
    };
    // Change of the latest value against the point about one epoch earlier
    // (the first point while still in the first epoch).
    const auto change_text = [&](const MetricSeries& s, bool lower_is_better,
                                 const char* value_format, std::string& text,
                                 ImVec4& color) {
        const size_t n = count_of(s);
        if (n < 2) {
            text = "first value";
            color = c.faint;
            return;
        }
        const double target = s.epochs[n - 1] - 1.0;
        size_t reference = 0;
        for (size_t i = n - 1; i-- > 0;) {
            if (s.epochs[i] <= target + 1e-9) {
                reference = i;
                break;
            }
        }
        const double delta = s.values[n - 1] - s.values[reference];
        if (std::abs(delta) < 1e-12) {
            text = "no change since epoch " + FormatEpochValue(s.epochs[reference]);
            color = c.muted;
            return;
        }
        char amount[48];
        std::snprintf(amount, sizeof(amount), value_format, std::abs(delta));
        text = std::string(delta < 0.0 ? ICON_FA_CARET_DOWN : ICON_FA_CARET_UP) + " " +
               amount + " since epoch " + FormatEpochValue(s.epochs[reference]);
        const bool better = lower_is_better ? delta < 0.0 : delta > 0.0;
        color = better ? c.success : c.caution;
    };
    const auto best_text = [&](const MetricSeries& s, bool lower_is_better,
                               const char* value_format) {
        const size_t n = count_of(s);
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) {
            if ((lower_is_better && s.values[i] < s.values[best]) ||
                (!lower_is_better && s.values[i] > s.values[best])) {
                best = i;
            }
        }
        char value[48];
        std::snprintf(value, sizeof(value), value_format, s.values[best]);
        return std::string("best ") + value + " at epoch " + FormatEpochValue(s.epochs[best]);
    };
    const auto add_metric = [&](const char* id, const char* label, const MetricSeries& s,
                                bool lower_is_better, const char* value_format,
                                const char* delta_format, const char* tooltip_format) {
        if (count_of(s) == 0) {
            return;
        }
        Card card;
        card.id = id;
        card.label = label;
        card.dot = s.color;
        char value[48];
        std::snprintf(value, sizeof(value), value_format, s.values.back());
        card.value = value;
        char tooltip[96];
        std::snprintf(tooltip, sizeof(tooltip), tooltip_format, s.values.back());
        card.tooltip = tooltip;
        change_text(s, lower_is_better, delta_format, card.sub1, card.sub1_color);
        card.sub2 = best_text(s, lower_is_better, value_format);
        card.sub2_color = c.muted;
        cards.push_back(std::move(card));
    };

    add_metric("##kpi_loss", "Loss", train_loss_, true, "%.4f", "%.4f", "Training loss: %.6f");
    add_metric("##kpi_acc", "Accuracy", train_accuracy_, false, "%.2f%%", "%.2f%%",
               "Training accuracy: %.2f%%");
    add_metric("##kpi_val_loss", "Val loss", val_loss_, true, "%.4f", "%.4f",
               "Validation loss: %.6f");
    if (!val_loss_.values.empty() && !cards.empty() && cards.back().id == "##kpi_val_loss") {
        // Validation signal: how far validation loss sits above training loss.
        const double recent_val_loss = val_loss_.values.back();
        const double recent_train_loss =
            train_loss_.values.empty() ? recent_val_loss : train_loss_.values.back();
        const double gap = recent_val_loss - recent_train_loss;
        char text[96];
        if (gap > 0.25) {
            std::snprintf(text, sizeof(text), "above train loss by %.4f", gap);
            cards.back().sub2_color = c.caution;
        } else {
            std::snprintf(text, sizeof(text), "gap to train loss controlled (%+.4f)", gap);
            cards.back().sub2_color = c.success;
        }
        cards.back().sub2 = text;
    }
    add_metric("##kpi_val_acc", "Val accuracy", val_accuracy_, false, "%.2f%%", "%.2f%%",
               "Validation accuracy: %.2f%%");

    // Timing card.
    if (is_training_ && (total_batches_ > 0 || avg_epoch_time_ > 0)) {
        Card card;
        card.id = "##kpi_time";
        card.label = "Time remaining";
        card.dot = c.warning;
        // Dynamic estimate: remaining batches over all epochs at the rate of the
        // last two minutes, plus measured epoch-boundary overhead. It follows
        // speed changes and appears after ~10 s, also in single-epoch runs.
        const auto remaining = eta_estimator_.RemainingSeconds();
        if (remaining) {
            card.value = FormatTrainingDuration(*remaining);
            card.tooltip =
                "Remaining batches across all epochs at the recent rate (last 2 minutes),\n"
                "plus " +
                (eta_estimator_.HasMeasuredEpochOverhead()
                     ? FormatTrainingDuration(eta_estimator_.MeanEpochOverheadSeconds())
                     : std::string("not yet measured")) +
                " per remaining epoch boundary (validation, previews, checkpoint).\n"
                "Updates as the training speed changes.";
        } else {
            card.value = "Estimating...";
        }
        char text[96] = "";
        std::string speed;
        if (const double rate = eta_estimator_.BatchesPerSecond(); rate > 0.0) {
            std::snprintf(text, sizeof(text), "%.2f batches/s", rate);
            speed = text;
        }
        if (samples_per_second_ > 0) {
            std::snprintf(text, sizeof(text), "%.0f samples/s", samples_per_second_);
            speed += (speed.empty() ? "" : " \xC2\xB7 ") + std::string(text);
        }
        card.sub1 = speed.empty() ? "measuring speed" : speed;
        card.sub1_color = speed.empty() ? c.faint : c.muted;
        if (avg_epoch_time_ > 0) {
            std::snprintf(text, sizeof(text), "last epoch %.1fs \xC2\xB7 avg %.1fs/epoch",
                          last_epoch_time_, avg_epoch_time_);
            card.sub2 = text;
            card.sub2_color = c.muted;
        } else {
            card.sub2 = "epoch time after the first epoch";
            card.sub2_color = c.faint;
        }
        cards.push_back(std::move(card));
    } else if (!is_training_ && total_training_time_ > 0) {
        Card card;
        card.id = "##kpi_time";
        card.label = "Total time";
        card.dot = c.info;
        card.value = FormatTrainingDuration(total_training_time_);
        char text[96];
        if (total_epochs_ > 0) {
            std::snprintf(text, sizeof(text), "%d / %d epochs run",
                          last_executed_epoch_, total_epochs_);
            card.sub1 = text;
        } else {
            card.sub1 = "run finished";
        }
        card.sub1_color = c.muted;
        if (avg_epoch_time_ > 0) {
            std::snprintf(text, sizeof(text), "avg %.1fs/epoch", avg_epoch_time_);
            card.sub2 = text;
        } else {
            card.sub2 = " ";
        }
        card.sub2_color = c.muted;
        cards.push_back(std::move(card));
    }

    if (cards.empty()) {
        return;
    }

    const float min_card_width = 190.0f;
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(available / min_card_width), 1,
                                   static_cast<int>(cards.size()));
    ImGui::Spacing();
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("##dash_kpis", columns,
                          ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
        for (const auto& card : cards) {
            ImGui::TableNextColumn();
            if (BeginDashCard(card.id.c_str(), c)) {
                const float line = ImGui::GetTextLineHeight();
                const ImVec2 pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddCircleFilled(
                    ImVec2(pos.x + 4.0f, pos.y + line * 0.5f), 4.0f,
                    ImGui::GetColorU32(card.dot));
                ImGui::Dummy(ImVec2(8.0f, line));
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextColored(c.muted, "%s", card.label.c_str());
                DashBigValue(card.value.c_str(), c.text);
                if (!card.tooltip.empty() && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", card.tooltip.c_str());
                }
                ImGui::TextColored(card.sub1_color, "%s", card.sub1.c_str());
                ImGui::TextColored(card.sub2_color, "%s", card.sub2.c_str());
            }
            EndDashCard();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
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
    checkpoint_used_.clear();
    checkpoint_epoch_ = 0;
    checkpoint_step_ = 0;
    active_model_provenance_.clear();
    has_checkpoint_validation_metrics_ = false;
    checkpoint_val_loss_ = 0.0f;
    checkpoint_val_accuracy_ = 0.0f;
    active_checkpoint_loaded_ = false;
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
        last_executed_epoch_ = 0;
        checkpoint_used_.clear();
        checkpoint_epoch_ = 0;
        checkpoint_step_ = 0;
        active_model_provenance_.clear();
        has_checkpoint_validation_metrics_ = false;
        active_checkpoint_loaded_ = false;
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
        last_executed_epoch_ = 0;
        checkpoint_used_.clear();
        checkpoint_epoch_ = 0;
        checkpoint_step_ = 0;
        active_model_provenance_.clear();
        active_checkpoint_loaded_ = false;
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

void TrainingPlotPanel::RenderLossPlot(float plot_height) {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (!BeginDashCard("##dash_loss_card", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle("Loss", log_loss_scale_ ? "training and validation, log axis"
                                          : "training and validation", c);
    PushDashPlotStyle(c);
    if (ImPlot::BeginPlot("Loss", ImVec2(-1, plot_height), ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
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
                ImPlot::SetNextLineStyle(MixColor(train_loss_.color, ImVec4(1, 1, 1, 1), 0.35f), 3.0f);
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
                ImPlot::SetNextLineStyle(MixColor(val_loss_.color, ImVec4(1, 1, 1, 1), 0.35f), 3.0f);
                ImPlot::PlotLine("Validation Loss (smoothed)",
                                 val_loss_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        ImPlot::EndPlot();
    }
    PopDashPlotStyle();
    EndDashCard();
}

void TrainingPlotPanel::RenderAccuracyPlot(float plot_height) {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (!BeginDashCard("##dash_accuracy_card", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle("Accuracy", "training and validation, percent", c);
    PushDashPlotStyle(c);
    if (ImPlot::BeginPlot("Accuracy", ImVec2(-1, plot_height), ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("Epoch", "Accuracy (%)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupLegend(ImPlotLocation_SouthEast);

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
                ImPlot::SetNextLineStyle(MixColor(train_accuracy_.color, ImVec4(1, 1, 1, 1), 0.35f), 3.0f);
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
                ImPlot::SetNextLineStyle(MixColor(val_accuracy_.color, ImVec4(1, 1, 1, 1), 0.35f), 3.0f);
                ImPlot::PlotLine("Validation Accuracy (smoothed)",
                                 val_accuracy_.epochs.data(),
                                 smoothed.data(),
                                 static_cast<int>(smoothed.size()));
            }
        }

        ImPlot::EndPlot();
    }
    PopDashPlotStyle();
    EndDashCard();
}

void TrainingPlotPanel::RenderCustomMetricsPlot(float plot_height) {
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

    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (!BeginDashCard("##dash_custom_card", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle(plot_title, "per epoch", c);
    PushDashPlotStyle(c);
    if (ImPlot::BeginPlot(plot_title, ImVec2(-1, plot_height), ImPlotFlags_NoTitle)) {
        // Enable zoom and pan on both axes
        ImPlot::SetupAxes("Epoch", y_label, ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

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
    PopDashPlotStyle();
    EndDashCard();
}

void TrainingPlotPanel::RenderControls() {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();

    // Which charts to show.
    DashChip("Loss", &show_loss_plot_, c);
    ImGui::SameLine();
    DashChip("Accuracy", &show_accuracy_plot_, c);
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
            ? "Sequence metrics"
            : "Custom metrics";
        DashChip(label, &show_custom_metrics_, c);
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
        ImGui::SameLine(0.0f, 12.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.faint, "%s", view_state.c_str());
    }

    // View menu and plot help on the right.
    const char* view_label = ICON_FA_SLIDERS " View";
    const float help_width = ImGui::CalcTextSize(ICON_FA_CIRCLE_INFO).x;
    const float needed = DashButtonWidth(view_label) + ImGui::GetStyle().ItemSpacing.x + help_width;
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                  ImGui::GetWindowContentRegionMax().x - needed));
    if (DashButton(view_label, DashButtonKind::Secondary, c)) {
        ImGui::OpenPopup("##dash_view");
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c.muted, ICON_FA_CIRCLE_INFO);
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

    if (ImGui::BeginPopup("##dash_view")) {
        ImGui::TextColored(c.muted, "Axes");
        ImGui::Checkbox("Auto scale", &auto_scale_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("When enabled, axes adapt to the live training data.\nDisable to manually zoom/pan.");
        }
        ImGui::Checkbox("Follow current epoch", &follow_current_epoch_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Keep the epoch axis scrolled to the latest batch/epoch.");
        }
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderInt("Epoch window", &visible_epoch_window_, 3, 50);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Number of epochs visible while following live training.");
        }
        ImGui::Checkbox("Log loss axis", &log_loss_scale_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Use a logarithmic loss axis to compare large early losses with "
                "smaller later losses. Metric values are unchanged.");
        }
        ImGui::Spacing();
        ImGui::TextColored(c.muted, "Curves");
        ImGui::Checkbox("Smooth curves", &show_smoothed_curves_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display-only moving average overlay. Raw curves are unchanged.");
        }
        if (show_smoothed_curves_) {
            ImGui::SetNextItemWidth(160.0f);
            ImGui::SliderInt("Smooth window", &smoothing_window_, 2, 50);
        }
        ImGui::EndPopup();
    }
    ImGui::Spacing();
}

void TrainingPlotPanel::RenderCurveSummary() {
    const DashColors c = CurrentDashColors();

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
    ImVec4 recommendation_color = c.success;
    if (has_validation && (gap_large || val_loss_worsening || val_acc_worsening || rough_epoch >= 0.0)) {
        recommendation = "inspect validation";
        recommendation_color = c.warning;
    }
    if (has_validation && rough_epoch >= 0.0 && val_loss_volatility > 0.02) {
        recommendation = "consider early stop";
        recommendation_color = c.caution;
    }

    ImGui::Spacing();
    if (!BeginDashCard("##dash_insights", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle("Insights", "how the curves are behaving", c);
    {
        const std::string pill = std::string("Suggested: ") + recommendation;
        const float pill_width = ImGui::CalcTextSize(pill.c_str()).x + 30.0f;
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x - pill_width));
        DashPill(pill.c_str(), recommendation_color);
    }
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, WithAlpha(c.border, 0.6f));
    if (ImGui::BeginTable("##curve_summary", 2,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        const auto row = [&](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(c.muted, "%s", label);
            ImGui::TableNextColumn();
        };

        row("Train curve");
        ImGui::Text("loss %s, accuracy %s",
            trend_label(train_loss_slope, true),
            trend_label(train_acc_slope, false));

        row("Validation curve");
        if (!val_loss_.values.empty() || !val_accuracy_.values.empty()) {
            ImGui::Text("loss %s, accuracy %s",
                trend_label(val_loss_slope, true),
                trend_label(val_acc_slope, false));
        } else {
            ImGui::TextColored(c.faint, "waiting for validation points");
        }

        row("Best validation");
        if (best_val_loss_idx >= 0) {
            ImGui::Text("loss %.4f at epoch %.2f",
                val_loss_.values[best_val_loss_idx],
                val_loss_.epochs[best_val_loss_idx]);
            if (best_val_acc_idx >= 0) {
                ImGui::Text("accuracy %.2f%% at epoch %.2f",
                    val_accuracy_.values[best_val_acc_idx],
                    val_accuracy_.epochs[best_val_acc_idx]);
            }
        } else {
            ImGui::TextColored(c.faint, "no validation data yet");
        }

        row("Rough point");
        ImGui::PushTextWrapPos(0.0f);
        if (rough_epoch >= 0.0) {
            ImGui::TextColored(c.caution,
                "validation loss starts rising around epoch %.2f", rough_epoch);
        } else if (val_loss_.values.size() >= 3) {
            ImGui::TextColored(c.success, "no sustained validation rise detected");
        } else {
            ImGui::TextColored(c.faint, "need more validation points");
        }
        ImGui::PopTextWrapPos();

        row("Generalization gap");
        if (!val_loss_.values.empty() && !train_loss_.values.empty()) {
            const double epoch = val_loss_.epochs.back();
            const double train_near_val = closest_value(train_loss_, epoch);
            const double gap = val_loss_.values.back() - train_near_val;
            ImGui::Text("val_loss - train_loss = %.4f", gap);
            ImGui::SameLine(0.0f, 10.0f);
            if (gap > 0.25) {
                ImGui::TextColored(c.caution, "possible overfit");
            } else {
                ImGui::TextColored(c.success, "controlled");
            }
        } else {
            ImGui::TextColored(c.faint, "waiting for train/validation loss");
        }

        row("Validation roughness");
        if (val_loss_.values.size() >= 3) {
            ImGui::Text("recent avg delta %.4f", val_loss_volatility);
        } else {
            ImGui::TextColored(c.faint, "need more validation points");
        }

        row("Suggested action");
        ImGui::TextColored(recommendation_color, "%s", recommendation);

        ImGui::EndTable();
    }
    ImGui::PopStyleColor();
    EndDashCard();
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

    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (!BeginDashCard("##dash_sequence", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle("Sequence Metrics", "latest values", c);
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, WithAlpha(c.border, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
    if (ImGui::BeginTable("##sequence_metrics", 3,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Train", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Val", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, c.muted);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();

        const auto value_cell = [&](bool has_value, double value) {
            ImGui::TableNextColumn();
            if (has_value) {
                ImGui::Text("%.2f%%", value);
            } else {
                ImGui::TextColored(c.faint, "no data");
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
    EndDashCard();
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
        const bool successful = materialization_status_ == "completed" ||
                                materialization_status_ == "cache_hit" ||
                                materialization_status_ == "cache_saved";
        const ImVec4 color = successful
            ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
            : ImVec4(1.0f, 0.75f, 0.25f, 1.0f);
        ImGui::TextColored(
            color,
            "Status: %s",
            MaterializationStatusDisplayName(materialization_status_));
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
        if (latest.available_memory_bytes > 0) {
            ImGui::Text("Available RAM / safe budget: %s / %s",
                        FormatTraceBytes(latest.available_memory_bytes).c_str(),
                        FormatTraceBytes(latest.safe_memory_budget_bytes).c_str());
        }
        if (latest.process_memory_detected) {
            ImGui::Text("Process resident / growth: %s / +%s",
                        FormatTraceBytes(
                            latest.process_resident_memory_bytes).c_str(),
                        FormatTraceBytes(
                            latest.process_resident_growth_bytes).c_str());
            if (latest.process_private_memory_bytes > 0) {
                ImGui::Text("Process %s: %s",
                            latest.process_private_memory_name.empty()
                                ? "private memory"
                                : latest.process_private_memory_name.c_str(),
                            FormatTraceBytes(
                                latest.process_private_memory_bytes).c_str());
            }
            ImGui::TextDisabled(
                "Process RAM; ArrayFire device memory is reported separately.");
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
        if (event.process_memory_detected) {
            ImGui::Text("Process resident: %s (+%s)",
                        FormatTraceBytes(
                            event.process_resident_memory_bytes).c_str(),
                        FormatTraceBytes(
                            event.process_resident_growth_bytes).c_str());
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
    const auto trace = TrainingTraceCollector::LatestTrace();
    if (!trace.available && trace.run_id.empty()) {
        return;
    }

    const TrainingTraceEvent* transfer = FindLatestPinMemoryTransferEvent(trace);
    const TrainingTraceEvent* fallback =
        FindLatestNativeCpuFallbackEvent(trace);
    const bool has_execution_truth =
        !trace.requested_backend.empty() ||
        !trace.effective_backend.empty() ||
        !trace.placement_fingerprint.empty();
    if (!transfer && !fallback && trace.warnings.empty() &&
        !HasResidencyVerdict(trace) && !has_execution_truth) {
        return;
    }

    ImGui::Spacing();
    if (has_execution_truth || !trace.residency_verdict.empty()) {
        ImGui::SeparatorText("Execution Truth");
        if (ImGui::BeginTable(
                "##training_execution_truth",
                2,
                ImGuiTableFlags_SizingStretchProp |
                    ImGuiTableFlags_BordersInnerH |
                    ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn(
                "Field", ImGuiTableColumnFlags_WidthFixed, 155.0f);
            ImGui::TableSetupColumn(
                "Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Residency");
            ImGui::TableNextColumn();
            ImGui::TextColored(
                ResidencyVerdictColor(trace.residency_verdict),
                "%s",
                ResidencyVerdictDisplayName(trace.residency_verdict));
            if (!trace.residency_verdict.empty() &&
                trace.residency_verdict != "unavailable") {
                RenderExecutionTruthRow("Verdict code",
                                        trace.residency_verdict);
            }

            RenderExecutionTruthRow("Requested backend",
                                    trace.requested_backend);
            RenderExecutionTruthRow(
                "Requested qualification",
                FormatRouteQualification(
                    trace.requested_qualification_evidence_available,
                    trace.requested_route_qualified,
                    trace.requested_qualification_matrix_id,
                    trace.requested_qualification_message));

            std::string effective = trace.effective_backend;
            if (!trace.effective_device_name.empty()) {
                effective += " | " + trace.effective_device_name;
            }
            if (!trace.effective_backend.empty()) {
                effective += " | device " +
                    std::to_string(trace.effective_device_id);
            }
            RenderExecutionTruthRow("Effective backend", effective);
            RenderExecutionTruthRow(
                "Effective qualification",
                FormatRouteQualification(
                    trace.effective_qualification_evidence_available,
                    trace.effective_route_qualified,
                    trace.effective_qualification_matrix_id,
                    trace.effective_qualification_message));
            std::string identity = trace.identity_confidence;
            if (!trace.physical_fingerprint.empty()) {
                if (!identity.empty()) {
                    identity += " | ";
                }
                identity += trace.physical_fingerprint;
            }
            RenderExecutionTruthRow("Physical identity", identity);
            RenderExecutionTruthRow(
                "Execution preflight",
                trace.execution_validated
                    ? "Validated"
                    : (trace.preflight_stage.empty()
                           ? "Not recorded"
                           : "Not validated | " + trace.preflight_stage));
            if (trace.selection_fallback_applied) {
                RenderExecutionTruthRow(
                    "Selection fallback",
                    "Applied to ArrayFire CPU");
            }
            RenderExecutionTruthRow("Fallback policy",
                                    trace.fallback_policy);

            std::string placement = trace.placement_fingerprint;
            if (!placement.empty()) {
                placement += " | " +
                    std::to_string(trace.placement_entry_count) +
                    " entries";
            }
            RenderExecutionTruthRow("Placement", placement);
            RenderExecutionTruthRow(
                "Host boundaries",
                std::to_string(trace.declared_output_boundary_count));
            RenderExecutionTruthRow(
                "Transfers",
                std::to_string(trace.transfer_event_count) + " events | " +
                    FormatTraceBytes(trace.transfer_known_bytes));
            RenderExecutionTruthRow(
                "Transfer reasons",
                trace.transfer_summary.empty()
                    ? "None recorded"
                    : trace.transfer_summary);
            RenderExecutionTruthRow(
                "Synchronizations",
                std::to_string(trace.synchronization_event_count) +
                    " events | " +
                    FormatTraceBytes(trace.synchronization_known_bytes));
            RenderExecutionTruthRow(
                "Sync reasons",
                trace.synchronization_summary.empty()
                    ? "None recorded"
                    : trace.synchronization_summary);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Native CPU fallback");
            ImGui::TableNextColumn();
            const ImVec4 fallback_count_color =
                trace.native_cpu_fallback_count == 0
                    ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
                    : ImVec4(1.0f, 0.38f, 0.38f, 1.0f);
            ImGui::TextColored(
                fallback_count_color,
                "%llu event%s",
                static_cast<unsigned long long>(
                    trace.native_cpu_fallback_count),
                trace.native_cpu_fallback_count == 1 ? "" : "s");
            ImGui::EndTable();
        }

        if (!trace.placement_summary.empty() &&
            ImGui::TreeNodeEx(
                "Placement stages",
                ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
            ImGui::TextWrapped("%s", trace.placement_summary.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TreePop();
        }
    }

    if (transfer) {
        ImGui::SeparatorText("Transfer Detail");
        const ImVec4 color = transfer->status == "warning"
            ? ImVec4(1.0f, 0.82f, 0.35f, 1.0f)
            : ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
        ImGui::TextColored(color, "Pin-memory transfer");
        if (ImGui::BeginTable(
                "##pin_memory_transfer_detail",
                2,
                ImGuiTableFlags_SizingStretchProp |
                    ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn(
                "Field", ImGuiTableColumnFlags_WidthFixed, 155.0f);
            ImGui::TableSetupColumn(
                "Value", ImGuiTableColumnFlags_WidthStretch);
            RenderExecutionTruthRow("Mode", transfer->transfer_mode);
            RenderExecutionTruthRow("Reason", transfer->transfer_reason);
            RenderExecutionTruthRow("Backend", transfer->transfer_backend);
            RenderExecutionTruthRow(
                "Batch size",
                std::to_string(transfer->transfer_batch_size));
            ImGui::EndTable();
        }
    }

    if (fallback) {
        ImGui::SeparatorText("Fallback Detail");
        const ImVec4 color = fallback->status == "error"
            ? ImVec4(1.0f, 0.35f, 0.35f, 1.0f)
            : ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
        ImGui::TextColored(color, "Native CPU fallback");
        if (ImGui::BeginTable(
                "##native_cpu_fallback_detail",
                2,
                ImGuiTableFlags_SizingStretchProp |
                    ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn(
                "Field", ImGuiTableColumnFlags_WidthFixed, 155.0f);
            ImGui::TableSetupColumn(
                "Value", ImGuiTableColumnFlags_WidthStretch);
            RenderExecutionTruthRow(
                "Count",
                std::to_string(trace.native_cpu_fallback_count));
            RenderExecutionTruthRow("Selected backend",
                                    fallback->compute_backend);
            RenderExecutionTruthRow("Operation",
                                    fallback->fallback_operation);
            RenderExecutionTruthRow("Reason", fallback->fallback_reason);
            RenderExecutionTruthRow("Policy", fallback->fallback_policy);
            ImGui::EndTable();
        }
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
                ImGui::SeparatorText("Warnings");
                rendered_warning_header = true;
            }
            ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.35f, 1.0f),
                               "%s", ClassifyTrainingWarning(warning));
            ImGui::SameLine(155.0f);
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
    const DashColors c = CurrentDashColors();
#ifndef CYXWIZ_PLOTTING_MODULE
    auto& tm = TrainingManager::Instance();
    const bool training_active = tm.IsTrainingActive();
    const bool training_paused = tm.IsPaused();
#else
    const bool training_active = false;
    const bool training_paused = false;
#endif
    const bool finished = !is_training_ && total_training_time_ > 0;
    const bool early_stopped = terminal_status_ == "early_stopped";
    const bool cancelled = terminal_status_ == "cancelled" ||
                           terminal_status_ == "stopped";
    const bool failed = terminal_status_ == "failed";

    const char* status = "Idle";
    ImVec4 status_color = c.muted;
    if (is_training_) {
        status = training_paused ? "Paused" : "Training";
        status_color = training_paused ? c.warning : c.success;
    } else if (preparation_failed_) {
        status = "Preparation failed";
        status_color = c.error;
    } else if (is_preparing_) {
        status = "Preparing";
        status_color = c.info;
    } else if (total_training_time_ > 0) {
        if (early_stopped) {
            status = "Early stopped";
            status_color = c.warning;
        } else if (cancelled) {
            status = "Cancelled";
            status_color = c.caution;
        } else if (failed) {
            status = "Failed";
            status_color = c.error;
        } else {
            status = "Completed";
            status_color = c.info;
        }
    } else if (active_checkpoint_loaded_) {
        status = "Model loaded";
        status_color = c.success;
    }

    if (!BeginDashCard("##dash_header", c)) {
        EndDashCard();
        return;
    }

    // Row 1: status, epoch and the run actions on the right.
    const float row_top = ImGui::GetCursorPosY();
    ImGui::SetCursorPosY(row_top + 1.0f);
    DashPill(status, status_color);
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::SetCursorPosY(row_top + 4.0f);
    if (preparation_failed_ || is_preparing_) {
        ImGui::TextColored(c.muted, "Getting the data and model ready");
    } else if (total_epochs_ > 0) {
        if (finished) {
            ImGui::Text("Executed epochs %d / %d", last_executed_epoch_, total_epochs_);
            if (current_epoch_ > last_executed_epoch_) {
                ImGui::SameLine(0.0f, 8.0f);
                ImGui::TextColored(c.muted, "stopped during epoch %d", current_epoch_);
            }
        } else {
            const int display_epoch = std::max(1, current_epoch_);
            const int remaining_epochs = std::max(0, total_epochs_ - display_epoch);
            ImGui::Text("Epoch %d / %d", display_epoch, total_epochs_);
            ImGui::SameLine(0.0f, 8.0f);
            ImGui::TextColored(c.muted, "%d remaining", remaining_epochs);
        }
    } else if (is_training_) {
        ImGui::Text("Epoch %d / ?", std::max(1, current_epoch_));
    } else if (!active_checkpoint_loaded_) {
        ImGui::TextColored(c.muted, "No training run yet");
    }

    const char* pause_label = training_paused ? ICON_FA_PLAY " Continue" : ICON_FA_PAUSE " Pause";
    const char* stop_label = ICON_FA_STOP " Early stop";
    const char* actions_label = "Actions " ICON_FA_CHEVRON_DOWN;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    float actions_width = DashButtonWidth(actions_label);
    if (training_active) {
        actions_width += DashButtonWidth(pause_label) + DashButtonWidth(stop_label) + spacing * 2.0f;
    }
    ImGui::SameLine();
    const float right_edge = ImGui::GetWindowContentRegionMax().x;
    ImGui::SetCursorPos(ImVec2(std::max(ImGui::GetCursorPosX(), right_edge - actions_width),
                               row_top));
#ifndef CYXWIZ_PLOTTING_MODULE
    if (training_active) {
        if (DashButton(pause_label, DashButtonKind::Secondary, c)) {
            if (training_paused) {
                tm.ResumeTraining();
            } else {
                tm.PauseTraining();
            }
        }
        ImGui::SameLine();
        if (DashButton(stop_label, DashButtonKind::Danger, c)) {
            tm.StopTraining();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stops training at the next batch boundary and keeps the current model state.");
        }
        ImGui::SameLine();
    }
#endif
    if (DashButton(actions_label, DashButtonKind::Secondary, c)) {
        ImGui::OpenPopup("##dash_actions");
    }
    if (ImGui::BeginPopup("##dash_actions")) {
        if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export metrics (CSV)")) {
            ExportToCSVLocked("training_metrics.csv");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Writes training_metrics.csv in the working folder.");
        }
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, c.error);
        const bool clear = ImGui::MenuItem(ICON_FA_TRASH " Clear all");
        ImGui::PopStyleColor();
        if (clear) {
            ClearLocked();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Clears the charts and metrics of this run from the dashboard.");
        }
        ImGui::EndPopup();
    }

    // Row 2: progress.
    if (preparation_failed_) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.error, "%s",
                           preparation_error_message_.empty()
                               ? "Training preparation failed."
                               : preparation_error_message_.c_str());
        ImGui::PopTextWrapPos();
    } else if (is_preparing_) {
        DashProgress(preparation_progress_, c.info, c);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.muted, "%s",
                           preparation_status_message_.empty()
                               ? "Preparing training..."
                               : preparation_status_message_.c_str());
        ImGui::PopTextWrapPos();
    } else if (total_epochs_ > 0) {
        // During training, count the finished part of the epoch in progress
        // (epoch counter moves to the new epoch at its first batch).
        const float progress = is_training_ && total_batches_ > 0
            ? static_cast<float>(TrainingFractionComplete(
                  current_epoch_, current_batch_, total_batches_, total_epochs_))
            : static_cast<float>(current_epoch_) / std::max(1, total_epochs_);
        DashProgress(progress, finished ? status_color : c.accent, c);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%.0f%% of all epochs", progress * 100.0f);
        }
    }

    // Why the run ended.
    if (finished) {
        ImGui::PushTextWrapPos(0.0f);
        if (early_stopped) {
            ImGui::TextColored(c.warning, "Stop reason: early stopping triggered.");
        } else if (cancelled) {
            ImGui::TextColored(c.caution, "Stop reason: training was cancelled.");
        } else if (failed) {
            ImGui::TextColored(c.error, "Stop reason: training failed.");
        } else {
            ImGui::TextColored(c.success, "Stop reason: training reached its terminal epoch.");
        }
        if (!terminal_reason_.empty()) {
            ImGui::TextColored(c.muted, "%s", terminal_reason_.c_str());
        }
        ImGui::PopTextWrapPos();
    }

    // Batch-level progress within the current epoch (live feedback during training)
    if (is_training_ && total_batches_ > 0) {
        const float batch_progress = static_cast<float>(current_batch_) /
                                     std::max(1, total_batches_);
        ImGui::Spacing();
        ImGui::TextColored(c.muted, "Batch");
        ImGui::SameLine();
        ImGui::Text("%d / %d", current_batch_, total_batches_);
        ImGui::SameLine(0.0f, 12.0f);
        DashProgress(batch_progress, c.accent, c, 160.0f, 5.0f);
        ImGui::SameLine(0.0f, 16.0f);
        ImGui::TextColored(c.muted, "Running loss");
        ImGui::SameLine();
        ImGui::Text("%.4f", current_batch_loss_);

        std::string detail;
        char part[96];
        if (samples_per_batch_ > 0) {
            std::snprintf(part, sizeof(part), "%d sample%s each", samples_per_batch_,
                          samples_per_batch_ == 1 ? "" : "s");
            detail = part;
            if (batches_per_update_ > 1) {
                std::snprintf(part, sizeof(part), "update every %d batches", batches_per_update_);
                detail += std::string(" \xC2\xB7 ") + part;
            }
        }
        if (metric_reporting_interval_ > 0) {
            std::snprintf(part, sizeof(part), "metrics every %d batches", metric_reporting_interval_);
        } else {
            std::snprintf(part, sizeof(part), "metrics at first/final batch");
        }
        detail += (detail.empty() ? "" : " \xC2\xB7 ") + std::string(part);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(c.faint, "%s", detail.c_str());
        ImGui::PopTextWrapPos();
    }

    // The model the Engine holds now (after a run or a loaded checkpoint).
    if (!is_training_ &&
        (!active_model_provenance_.empty() || !checkpoint_used_.empty())) {
        ImGui::Spacing();
        ImGui::TextColored(c.muted, "Active model");
        ImGui::PushStyleColor(ImGuiCol_TableBorderLight, WithAlpha(c.border, 0.6f));
        if (ImGui::BeginTable("##dash_active_model", 2,
                              ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 150.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            const auto row = [&](const char* label) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(c.muted, "%s", label);
                ImGui::TableNextColumn();
            };
            const bool restored_checkpoint = checkpoint_epoch_ > 0 &&
                !checkpoint_used_.empty();
            row("State");
            ImGui::TextColored(
                c.success, "%s",
                active_checkpoint_loaded_
                    ? "Checkpoint loaded for testing"
                    : (restored_checkpoint
                        ? "Best validation checkpoint restored"
                        : "Final state from executed run"));
            if (!active_model_provenance_.empty()) {
                row("Provenance");
                ImGui::TextWrapped("%s", active_model_provenance_.c_str());
            }
            if (checkpoint_epoch_ > 0) {
                row("Checkpoint epoch");
                ImGui::Text("%d", checkpoint_epoch_);
            }
            if (checkpoint_epoch_ > 0 && has_checkpoint_validation_metrics_) {
                row("Checkpoint validation");
                ImGui::Text("loss %.4f, accuracy %.2f%%",
                            checkpoint_val_loss_,
                            checkpoint_val_accuracy_ * 100.0f);
            }
            if (checkpoint_step_ > 0) {
                row("Checkpoint step");
                ImGui::Text("%d", checkpoint_step_);
            }
            if (!checkpoint_used_.empty()) {
                row("Path");
                ImGui::TextWrapped("%s", checkpoint_used_.c_str());
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleColor();
    }

    EndDashCard();
}

void TrainingPlotPanel::RenderStatistics() {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (!BeginDashCard("##dash_stats", c)) {
        EndDashCard();
        return;
    }
    DashCardTitle("Statistics", "mean of the last 10 points; min and max of the run", c);
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, WithAlpha(c.border, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
    if (ImGui::BeginTable("##dash_stats_table", 4,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Series", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, c.muted);
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
    EndDashCard();
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
