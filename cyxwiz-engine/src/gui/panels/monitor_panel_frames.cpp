// Monitor > Frames: the whole frame and the Engine's own interface-building
// time, kept apart, with this frame's windows (gui/frame_metrics). Frames are
// recorded while this tab is open, with or without the overlay.

#include "monitor_panel.h"

#include "../frame_metrics.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

std::string Milliseconds(double ms) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", ms);
    return text;
}

std::string Thousands(long long n) {
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

}  // namespace

void MonitorPanel::RenderFrames() {
    const auto& t = ui::CurrentTokens();
    auto& metrics = ::gui::FrameMetrics::Instance();
    metrics.Watch();
    const ImGuiIO& io = ImGui::GetIO();
    const FrameTimes::Summary summary = metrics.times().Summarize();

    const int columns = ImGui::GetContentRegionAvail().x >= 620.0f ? 2 : 1;
    if (ImGui::BeginTable("##frame_cards", columns, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        ui::BeginCard("whole_frame");
        ui::CardHeader("Whole frame", "what the screen shows");
        {
            ui::FontScope heading(ui::Font::Heading);
            ImGui::Text("%.1f", 1000.0f / std::max(io.Framerate, 1.0f));
        }
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "ms \xC2\xB7 %.0f fps", io.Framerate);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "Includes waiting for the display and the pauses the Engine takes when idle.");
        ImGui::PopTextWrapPos();
        ui::EndCard();

        ImGui::TableNextColumn();
        ui::BeginCard("build");
        ui::CardHeader("Building the interface", "the Engine's own work");
        {
            ui::FontScope heading(ui::Font::Heading);
            ImGui::TextUnformatted(Milliseconds(summary.build_now).c_str());
        }
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "ms now \xC2\xB7 %s average \xC2\xB7 %s worst", Milliseconds(summary.build_average).c_str(),
                           Milliseconds(summary.build_worst).c_str());
        ImGui::TextColored(t.text_faint, "Last %zu frames, counted since this tab or the overlay was opened.", summary.frames);
        ui::EndCard();
        ImGui::EndTable();
    }

    const std::vector<float> history = metrics.times().History();
    if (history.size() > 1) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, t.plot_bg);
        ImGui::PushStyleColor(ImGuiCol_PlotLines, t.series[0]);
        ImGui::PlotLines("##build_history", history.data(), static_cast<int>(history.size()), 0, nullptr, 0.0f,
                         std::max(static_cast<float>(summary.build_worst), 4.0f), ImVec2(ImGui::GetContentRegionAvail().x, 56.0f));
        ImGui::PopStyleColor(2);
    }

    if (ImGui::BeginTable("##frame_windows", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Window, this frame", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("Average", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableHeadersRow();
        size_t shown = 0;
        for (const auto& window : summary.windows) {
            if (shown == 12) break;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(window.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(window.now > 2.0 ? t.caution : t.text, "%s", Milliseconds(window.now).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", Milliseconds(window.average).c_str());
            ++shown;
        }
        ImGui::EndTable();
    }

    std::vector<std::pair<std::string, std::string>> rows;
    rows.push_back({"Windows timed this frame",
                    std::to_string(summary.windows.size()) + " (" + Milliseconds(summary.windows_total) + " ms together)"});
    rows.push_back({"Drawn", Thousands(io.MetricsRenderVertices) + " vertices \xC2\xB7 " + Thousands(io.MetricsRenderIndices) +
                                 " indices \xC2\xB7 " + std::to_string(io.MetricsRenderWindows) + " windows"});
    if (io.Fonts && io.Fonts->TexWidth > 0) {
        // The atlas is uploaded as one RGBA texture: four bytes a pixel.
        const double megabytes = static_cast<double>(io.Fonts->TexWidth) * io.Fonts->TexHeight * 4.0 / (1024.0 * 1024.0);
        char atlas[96];
        std::snprintf(atlas, sizeof(atlas), "%d x %d (%.1f MB as an RGBA texture)", io.Fonts->TexWidth, io.Fonts->TexHeight, megabytes);
        rows.push_back({"Font atlas", atlas});
    }
    MonitorRows("##frame_facts", rows);

    bool overlay = metrics.enabled();
    if (ImGui::Checkbox("Show the overlay on screen", &overlay)) metrics.set_enabled(overlay);
}

}  // namespace cyxwiz
