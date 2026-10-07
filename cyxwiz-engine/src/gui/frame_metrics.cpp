#include "frame_metrics.h"

#include "ui_tokens.h"
#include "separate_windows.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace gui {

namespace {
constexpr size_t kHistory = 180;
constexpr double kSmoothing = 0.1;  // per-panel exponential average
}  // namespace

FrameMetrics& FrameMetrics::Instance() {
    static FrameMetrics instance;
    return instance;
}

void FrameMetrics::BeginFrame() {
    if (!enabled_) return;
    frame_start_ = std::chrono::steady_clock::now();
    for (auto& p : panels_) p.milliseconds *= (1.0 - kSmoothing);
    panels_total_ms_ = 0.0;
}

void FrameMetrics::EndFrame() {
    if (!enabled_) return;
    if (frame_history_.size() < kHistory) frame_history_.resize(kHistory, 0.0f);
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - frame_start_).count();
    frame_history_[history_pos_] = ms;
    history_pos_ = (history_pos_ + 1) % kHistory;
}

void FrameMetrics::AddPanelTime(const char* name, double milliseconds) {
    if (!enabled_) return;
    panels_total_ms_ += milliseconds;
    for (auto& p : panels_) {
        if (p.name == name) {
            p.milliseconds += kSmoothing * milliseconds;
            return;
        }
    }
    panels_.push_back({name, milliseconds});
}

void FrameMetrics::Render() {
    if (!enabled_) return;
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    const ImGuiIO& io = ImGui::GetIO();

    float avg = 0.0f, worst = 0.0f;
    int counted = 0;
    for (float ms : frame_history_) {
        if (ms <= 0.0f) continue;
        avg += ms;
        worst = std::max(worst, ms);
        ++counted;
    }
    if (counted > 0) avg /= static_cast<float>(counted);

    std::vector<PanelCost> top = panels_;
    std::sort(top.begin(), top.end(), [](const PanelCost& a, const PanelCost& b) { return a.milliseconds > b.milliseconds; });
    if (top.size() > 10) top.resize(10);

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ::gui::NextWindowStaysInMain();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 12.0f, viewport->WorkPos.y + 44.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.92f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
    if (ImGui::Begin("##frame_metrics", nullptr, flags)) {
        ImGui::TextColored(t.text_bright, "Frame time");
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(t.text_dim, "%.0f fps", io.Framerate);
        ImGui::Text("%.2f ms now   %.2f ms avg   %.2f ms worst (last %d)", 1000.0f / std::max(io.Framerate, 1.0f), avg, worst, counted);
        if (counted > 1) {
            // Rotate the ring so the plot reads left to right in time.
            std::vector<float> ordered(frame_history_.size());
            for (size_t i = 0; i < frame_history_.size(); ++i)
                ordered[i] = frame_history_[(history_pos_ + i) % frame_history_.size()];
            ImGui::PlotLines("##frames", ordered.data(), static_cast<int>(ordered.size()), 0, nullptr, 0.0f,
                             std::max(worst, 16.7f), ImVec2(300.0f, 44.0f));
        }
        ImGui::Spacing();
        ImGui::TextColored(t.text_bright, "Panels this frame");
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(t.text_dim, "%.2f ms in %zu panels", panels_total_ms_, panels_.size());
        if (ImGui::BeginTable("##panels", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableSetupColumn("panel", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            for (const auto& p : top) {
                if (p.milliseconds < 0.005) continue;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(p.name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(p.milliseconds > 2.0 ? t.caution : t.text, "%.2f", p.milliseconds);
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "Draw: %d vertices, %d indices, %d windows", io.MetricsRenderVertices,
                           io.MetricsRenderIndices, io.MetricsRenderWindows);
        if (io.Fonts && io.Fonts->TexWidth > 0) {
            const double atlas_mb = static_cast<double>(io.Fonts->TexWidth) * io.Fonts->TexHeight * 4.0 / (1024.0 * 1024.0);
            ImGui::TextColored(t.text_dim, "Font atlas: %d x %d (%.0f MB as RGBA)", io.Fonts->TexWidth, io.Fonts->TexHeight, atlas_mb);
        }
    }
    ImGui::End();
}

PanelTimer::PanelTimer(const char* name) : name_(name), active_(FrameMetrics::Instance().enabled()) {
    if (active_) start_ = std::chrono::steady_clock::now();
}

PanelTimer::~PanelTimer() {
    if (!active_) return;
    const auto now = std::chrono::steady_clock::now();
    FrameMetrics::Instance().AddPanelTime(name_, std::chrono::duration<double, std::milli>(now - start_).count());
}

}  // namespace gui
