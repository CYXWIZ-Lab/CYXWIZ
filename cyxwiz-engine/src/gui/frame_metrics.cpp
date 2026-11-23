#include "frame_metrics.h"

#include "separate_windows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace gui {

FrameMetrics& FrameMetrics::Instance() {
    static FrameMetrics instance;
    return instance;
}

void FrameMetrics::BeginFrame() {
    const bool wanted = enabled_ || watched_frames_ > 0;
    if (watched_frames_ > 0) --watched_frames_;
    // Recording starts with an empty history: frames from the last time it
    // was on say nothing about now.
    if (wanted && !recording_) times_.Reset();
    recording_ = wanted;
    if (!recording_) return;
    frame_start_ = std::chrono::steady_clock::now();
    times_.BeginFrame();
}

void FrameMetrics::EndFrame() {
    if (!recording_) return;
    const auto now = std::chrono::steady_clock::now();
    times_.EndFrame(std::chrono::duration<double, std::milli>(now - frame_start_).count());
}

void FrameMetrics::AddPanelTime(const char* name, double milliseconds) {
    if (!recording_) return;
    times_.AddWindow(name, milliseconds);
}

void FrameMetrics::Render() {
    if (!enabled_ || !recording_) return;
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    const ImGuiIO& io = ImGui::GetIO();
    const cyxwiz::FrameTimes::Summary summary = times_.Summarize();

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ::gui::NextWindowStaysInMain();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 12.0f, viewport->WorkPos.y + 44.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.92f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
    if (ImGui::Begin("##frame_metrics", nullptr, flags)) {
        ImGui::TextColored(t.text_bright, "Whole frame");
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::Text("%.1f ms", 1000.0f / std::max(io.Framerate, 1.0f));
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(t.text_dim, "%.0f fps", io.Framerate);

        ImGui::TextColored(t.text_bright, "Building the interface");
        ImGui::Text("%.2f ms now", summary.build_now);
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(t.text_dim, "%.2f average   %.2f worst (last %zu)", summary.build_average, summary.build_worst,
                           summary.frames);
        const std::vector<float> history = times_.History();
        if (history.size() > 1)
            ImGui::PlotLines("##frames", history.data(), static_cast<int>(history.size()), 0, nullptr, 0.0f,
                             std::max(static_cast<float>(summary.build_worst), 4.0f), ImVec2(300.0f, 44.0f));

        // The three costliest windows of this frame.
        size_t shown = 0;
        for (const auto& window : summary.windows) {
            if (shown == 3 || window.now < 0.005) break;
            if (shown > 0) {
                ImGui::SameLine(0.0f, t.space_md);
                ImGui::TextColored(t.text_faint, "\xC2\xB7");
                ImGui::SameLine(0.0f, t.space_md);
            }
            ImGui::TextColored(window.now > 2.0 ? t.caution : t.text_dim, "%s %.2f", window.name.c_str(), window.now);
            ++shown;
        }
        if (open_monitor_ && cyxwiz::ui::LinkButton("More in Monitor > Frames")) open_monitor_();
    }
    ImGui::End();
}

PanelTimer::PanelTimer(const char* name) : name_(name), active_(FrameMetrics::Instance().recording()) {
    if (active_) start_ = std::chrono::steady_clock::now();
}

PanelTimer::~PanelTimer() {
    if (!active_) return;
    const auto now = std::chrono::steady_clock::now();
    FrameMetrics::Instance().AddPanelTime(name_, std::chrono::duration<double, std::milli>(now - start_).count());
}

}  // namespace gui
