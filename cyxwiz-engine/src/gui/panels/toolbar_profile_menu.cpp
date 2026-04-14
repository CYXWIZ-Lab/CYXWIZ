#include "toolbar.h"
#include "../theme.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include "../icons.h"

namespace cyxwiz {

void ToolbarPanel::RenderProfileMenu() {
    if (ImGui::BeginMenu("Profile")) {
        // Increase padding for menu items
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // Header text
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
        ImGui::Text("Performance Analysis");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // ========== Performance Profiler (ProfilingPanel) ==========
        if (ImGui::MenuItem(ICON_FA_GAUGE_HIGH " Performance Profiler")) {
            if (open_profiler_callback_) {
                open_profiler_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Per-layer timing analysis:\n"
                              "- Forward/backward pass timing\n"
                              "- Layer breakdown by execution time\n"
                              "- Timeline view of training steps\n"
                              "- History charts and statistics");
        }

        // ========== Memory Visualization (MemoryPanel) ==========
        if (ImGui::MenuItem(ICON_FA_CHART_PIE " Memory Visualization")) {
            if (open_memory_panel_callback_) {
                open_memory_panel_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Detailed memory analysis:\n"
                              "- CPU heap and tensor memory\n"
                              "- GPU allocated and cached memory\n"
                              "- Per-layer memory breakdown\n"
                              "- Memory timeline charts");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Header text
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
        ImGui::Text("System Monitoring");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // ========== System Monitor (MemoryMonitor) ==========
        if (ImGui::MenuItem(ICON_FA_MICROCHIP " System Monitor")) {
            if (open_memory_monitor_callback_) {
                open_memory_monitor_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Quick system metrics overlay:\n"
                              "- CPU utilization sparkline\n"
                              "- RAM usage sparkline\n"
                              "- GPU usage sparkline\n"
                              "- Process memory info");
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
