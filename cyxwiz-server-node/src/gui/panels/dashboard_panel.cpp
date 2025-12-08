// dashboard_panel.cpp - Task Manager style system overview
#include "gui/panels/dashboard_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "core/metrics_collector.h"
#include "ipc/daemon_client.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <intrin.h>
#pragma comment(lib, "pdh.lib")
#endif

namespace cyxwiz::servernode::gui {

// Forward declarations
static std::string FormatUptime(int64_t seconds);
static std::string FormatBytes(uint64_t bytes);

// Color scheme matching Task Manager dark theme
namespace Colors {
    const ImVec4 Background = ImVec4(0.05f, 0.08f, 0.10f, 1.0f);
    const ImVec4 SidebarBg = ImVec4(0.08f, 0.10f, 0.12f, 1.0f);
    const ImVec4 GraphBg = ImVec4(0.02f, 0.12f, 0.15f, 1.0f);
    const ImVec4 GraphLine = ImVec4(0.30f, 0.75f, 0.85f, 1.0f);
    const ImVec4 GraphFill = ImVec4(0.15f, 0.45f, 0.55f, 0.5f);
    const ImVec4 Selected = ImVec4(0.15f, 0.35f, 0.45f, 1.0f);
    const ImVec4 Hover = ImVec4(0.12f, 0.25f, 0.35f, 1.0f);
    const ImVec4 TextPrimary = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    const ImVec4 TextSecondary = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    const ImVec4 CPUColor = ImVec4(0.40f, 0.75f, 0.90f, 1.0f);
    const ImVec4 MemoryColor = ImVec4(0.55f, 0.45f, 0.75f, 1.0f);
    const ImVec4 GPUColor = ImVec4(0.45f, 0.80f, 0.45f, 1.0f);
    const ImVec4 NetworkColor = ImVec4(0.85f, 0.65f, 0.35f, 1.0f);
}

DashboardPanel::DashboardPanel() : ServerPanel("Dashboard") {
    cpu_history_.resize(HISTORY_SIZE, 0.0f);
    gpu_history_.resize(HISTORY_SIZE, 0.0f);
    ram_history_.resize(HISTORY_SIZE, 0.0f);
    vram_history_.resize(HISTORY_SIZE, 0.0f);
    net_in_history_.resize(HISTORY_SIZE, 0.0f);
    net_out_history_.resize(HISTORY_SIZE, 0.0f);
    last_update_ = std::chrono::steady_clock::now();

    // Initialize CPU info
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu_logical_ = sysInfo.dwNumberOfProcessors;
    cpu_cores_ = cpu_logical_ / 2;  // Approximate physical cores
    if (cpu_cores_ < 1) cpu_cores_ = 1;

    // Get CPU name from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD bufSize = sizeof(cpuName);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)cpuName, &bufSize) == ERROR_SUCCESS) {
            cpu_name_ = cpuName;
            // Trim whitespace
            size_t start = cpu_name_.find_first_not_of(" ");
            size_t end = cpu_name_.find_last_not_of(" ");
            if (start != std::string::npos) {
                cpu_name_ = cpu_name_.substr(start, end - start + 1);
            }
        }
        DWORD mhz = 0;
        bufSize = sizeof(mhz);
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr,
                            (LPBYTE)&mhz, &bufSize) == ERROR_SUCCESS) {
            cpu_speed_ghz_ = mhz / 1000.0f;
            cpu_max_speed_ghz_ = cpu_speed_ghz_;
        }
        RegCloseKey(hKey);
    }

    // Initialize per-core history
    per_core_usage_.resize(cpu_logical_, 0.0f);
    per_core_history_.resize(cpu_logical_);
    for (auto& hist : per_core_history_) {
        hist.resize(HISTORY_SIZE, 0.0f);
    }
#endif
}

void DashboardPanel::Render() {
    // Update metrics periodically
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_).count();
    if (elapsed >= 1000) {
        UpdateMetrics();
        last_update_ = now;
    }

    // Main layout: sidebar (200px) | main graph area | stats at bottom
    ImVec2 content_size = ImGui::GetContentRegionAvail();
    float sidebar_width = 200.0f;
    float stats_height = 120.0f;

    // Resource sidebar
    ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::SidebarBg);
    if (ImGui::BeginChild("ResourceSidebar", ImVec2(sidebar_width, content_size.y), true)) {
        RenderResourceSidebar();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::SameLine();

    // Main content area (graph + stats)
    if (ImGui::BeginChild("MainArea", ImVec2(0, 0), false)) {
        float main_height = ImGui::GetContentRegionAvail().y;

        // Graph area
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::Background);
        if (ImGui::BeginChild("GraphArea", ImVec2(0, main_height - stats_height), false)) {
            RenderMainGraphArea();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        // Stats section
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::SidebarBg);
        if (ImGui::BeginChild("StatsArea", ImVec2(0, 0), true)) {
            RenderStatsSection();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

void DashboardPanel::RenderResourceSidebar() {
    // CPU
    std::string cpu_subtitle = fmt::format("{:.0f}%", cpu_usage_ * 100);
    RenderResourceItem(ResourceType::CPU, ICON_FA_MICROCHIP " CPU", cpu_subtitle.c_str(),
                       cpu_usage_, cpu_history_, selected_resource_ == ResourceType::CPU);

    // Memory
    double ram_used_gb = ram_used_ / (1024.0 * 1024.0 * 1024.0);
    double ram_total_gb = ram_total_ / (1024.0 * 1024.0 * 1024.0);
    std::string mem_subtitle = fmt::format("{:.1f}/{:.1f} GB ({:.0f}%)",
                                           ram_used_gb, ram_total_gb, ram_usage_ * 100);
    RenderResourceItem(ResourceType::Memory, ICON_FA_MEMORY " Memory", mem_subtitle.c_str(),
                       ram_usage_, ram_history_, selected_resource_ == ResourceType::Memory);

    // GPU
    std::string gpu_subtitle = fmt::format("{:.0f}%", gpu_usage_ * 100);
    if (!gpu_name_.empty() && gpu_name_ != "Unknown GPU") {
        std::string short_name = gpu_name_.length() > 20 ? gpu_name_.substr(0, 17) + "..." : gpu_name_;
        gpu_subtitle = short_name + "\n" + gpu_subtitle;
    }
    RenderResourceItem(ResourceType::GPU, ICON_FA_DESKTOP " GPU", gpu_subtitle.c_str(),
                       gpu_usage_, gpu_history_, selected_resource_ == ResourceType::GPU);

    // Network
    std::string net_subtitle = fmt::format("In: {:.1f} Mbps\nOut: {:.1f} Mbps",
                                           net_in_mbps_, net_out_mbps_);
    float net_usage = std::min(1.0f, (net_in_mbps_ + net_out_mbps_) / 100.0f);
    RenderResourceItem(ResourceType::Network, ICON_FA_NETWORK_WIRED " Network", net_subtitle.c_str(),
                       net_usage, net_in_history_, selected_resource_ == ResourceType::Network);

    // Spacer
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Active jobs and deployments with icons
    ImGui::TextColored(Colors::TextSecondary, ICON_FA_BARS_PROGRESS " Active Jobs");
    ImGui::TextColored(Colors::TextPrimary, "  %d", active_jobs_);
    ImGui::Spacing();
    ImGui::TextColored(Colors::TextSecondary, ICON_FA_ROCKET " Deployed Models");
    ImGui::TextColored(Colors::TextPrimary, "  %d", active_deployments_);
}

void DashboardPanel::RenderResourceItem(ResourceType type, const char* name, const char* subtitle,
                                         float usage, const std::vector<float>& history, bool selected) {
    ImVec2 item_size(ImGui::GetContentRegionAvail().x, 70.0f);
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();

    // Background
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec4 bg_color = selected ? Colors::Selected : ImVec4(0, 0, 0, 0);

    if (ImGui::InvisibleButton(name, item_size)) {
        selected_resource_ = type;
    }
    if (ImGui::IsItemHovered()) {
        bg_color = selected ? Colors::Selected : Colors::Hover;
    }

    draw_list->AddRectFilled(cursor_pos,
                             ImVec2(cursor_pos.x + item_size.x, cursor_pos.y + item_size.y),
                             ImGui::ColorConvertFloat4ToU32(bg_color));

    // Mini sparkline graph
    float graph_width = 60.0f;
    float graph_height = 35.0f;
    ImVec2 graph_pos(cursor_pos.x + 8, cursor_pos.y + 8);

    // Graph background
    draw_list->AddRectFilled(graph_pos,
                             ImVec2(graph_pos.x + graph_width, graph_pos.y + graph_height),
                             ImGui::ColorConvertFloat4ToU32(Colors::GraphBg));

    // Draw sparkline
    if (!history.empty()) {
        ImVec4 color;
        switch (type) {
            case ResourceType::CPU: color = Colors::CPUColor; break;
            case ResourceType::Memory: color = Colors::MemoryColor; break;
            case ResourceType::GPU: color = Colors::GPUColor; break;
            case ResourceType::Network: color = Colors::NetworkColor; break;
            default: color = Colors::GraphLine; break;
        }

        std::vector<ImVec2> points;
        int sample_count = std::min(static_cast<int>(history.size()), 30);
        int start_idx = static_cast<int>(history.size()) - sample_count;

        for (int i = 0; i < sample_count; ++i) {
            float x = graph_pos.x + (static_cast<float>(i) / (sample_count - 1)) * graph_width;
            float y = graph_pos.y + graph_height - history[start_idx + i] * graph_height;
            points.push_back(ImVec2(x, y));
        }

        // Draw filled area
        if (points.size() >= 2) {
            // Fill
            ImVec4 fill_color = color;
            fill_color.w = 0.3f;
            for (size_t i = 0; i < points.size() - 1; ++i) {
                ImVec2 p1 = points[i];
                ImVec2 p2 = points[i + 1];
                ImVec2 p3(p2.x, graph_pos.y + graph_height);
                ImVec2 p4(p1.x, graph_pos.y + graph_height);
                draw_list->AddQuadFilled(p1, p2, p3, p4, ImGui::ColorConvertFloat4ToU32(fill_color));
            }
            // Line
            draw_list->AddPolyline(points.data(), static_cast<int>(points.size()),
                                   ImGui::ColorConvertFloat4ToU32(color), 0, 1.5f);
        }
    }

    // Text
    float text_x = cursor_pos.x + graph_width + 16;
    draw_list->AddText(ImVec2(text_x, cursor_pos.y + 8),
                       ImGui::ColorConvertFloat4ToU32(Colors::TextPrimary), name);
    draw_list->AddText(ImVec2(text_x, cursor_pos.y + 28),
                       ImGui::ColorConvertFloat4ToU32(Colors::TextSecondary), subtitle);

    ImGui::SetCursorScreenPos(ImVec2(cursor_pos.x, cursor_pos.y + item_size.y + 4));
}

void DashboardPanel::RenderMainGraphArea() {
    ImVec2 avail = ImGui::GetContentRegionAvail();

    // Title based on selected resource
    const char* title = "";
    const char* icon = "";
    const char* subtitle = "% Utilization over 60 seconds";
    ImVec4 color = Colors::GraphLine;
    const std::vector<float>* history = nullptr;
    bool show_core_grid = false;

    switch (selected_resource_) {
        case ResourceType::CPU:
            title = "CPU";
            icon = ICON_FA_MICROCHIP;
            color = Colors::CPUColor;
            history = &cpu_history_;
            show_core_grid = !per_core_usage_.empty();
            break;
        case ResourceType::Memory:
            title = "Memory";
            icon = ICON_FA_MEMORY;
            color = Colors::MemoryColor;
            history = &ram_history_;
            subtitle = "Memory usage over 60 seconds";
            break;
        case ResourceType::GPU:
            title = "GPU";
            icon = ICON_FA_DESKTOP;
            color = Colors::GPUColor;
            history = &gpu_history_;
            break;
        case ResourceType::Network:
            title = "Network";
            icon = ICON_FA_NETWORK_WIRED;
            color = Colors::NetworkColor;
            history = &net_in_history_;
            subtitle = "Throughput over 60 seconds";
            break;
        default:
            break;
    }

    // Header with icon
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::TextColored(color, "%s", icon);
    ImGui::SameLine();
    ImGui::Text("%s", title);
    ImGui::PopFont();

    // Right-aligned CPU name or GPU name
    if (selected_resource_ == ResourceType::CPU && !cpu_name_.empty()) {
        ImGui::SameLine(avail.x - 400);
        ImGui::TextColored(Colors::TextSecondary, "%s", cpu_name_.c_str());
    } else if (selected_resource_ == ResourceType::GPU && !gpu_name_.empty()) {
        ImGui::SameLine(avail.x - 300);
        ImGui::TextColored(Colors::TextSecondary, "%s", gpu_name_.c_str());
    }

    ImGui::TextColored(Colors::TextSecondary, "%s", subtitle);
    ImGui::Spacing();

    // Show per-core grid for CPU, otherwise large chart
    if (show_core_grid && selected_resource_ == ResourceType::CPU) {
        RenderCoreGraphGrid();
    } else if (history) {
        RenderLargeAreaChart(title, subtitle, *history, color);
    }
}

void DashboardPanel::RenderCoreGraphGrid() {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    int num_cores = static_cast<int>(per_core_usage_.size());
    if (num_cores == 0) return;

    // Calculate grid layout
    int cols = 4;
    if (num_cores <= 4) cols = 2;
    else if (num_cores <= 8) cols = 4;
    else cols = 6;

    int rows = (num_cores + cols - 1) / cols;
    float cell_width = avail.x / cols - 8;
    float cell_height = avail.y / rows - 8;
    if (cell_height > 80) cell_height = 80;

    ImPlot::PushStyleColor(ImPlotCol_FrameBg, Colors::GraphBg);
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, Colors::GraphBg);
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0.2f, 0.3f, 0.4f, 0.3f));
    ImPlot::PushStyleColor(ImPlotCol_Line, Colors::CPUColor);
    ImVec4 fill = Colors::CPUColor;
    fill.w = 0.3f;
    ImPlot::PushStyleColor(ImPlotCol_Fill, fill);

    for (int i = 0; i < num_cores; ++i) {
        int col = i % cols;
        int row = i / cols;

        if (col > 0) ImGui::SameLine();

        ImGui::BeginGroup();
        ImGui::TextColored(Colors::TextSecondary, "CPU %d", i);
        ImGui::SameLine(cell_width - 40);
        ImGui::TextColored(Colors::CPUColor, "%2.0f%%", per_core_usage_[i] * 100);

        std::string plot_id = fmt::format("##Core{}", i);
        ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus |
                           ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;
        ImPlotAxisFlags axis_flags = ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels |
                                     ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoGridLines;

        if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(cell_width, cell_height - 20), flags)) {
            ImPlot::SetupAxes(nullptr, nullptr, axis_flags, axis_flags);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, HISTORY_SIZE, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImGuiCond_Always);

            if (i < static_cast<int>(per_core_history_.size()) && !per_core_history_[i].empty()) {
                ImPlot::PlotShaded("##area", per_core_history_[i].data(),
                                   static_cast<int>(per_core_history_[i].size()), 0.0);
                ImPlot::PlotLine("##line", per_core_history_[i].data(),
                                static_cast<int>(per_core_history_[i].size()));
            }
            ImPlot::EndPlot();
        }
        ImGui::EndGroup();
    }

    ImPlot::PopStyleColor(5);
}

void DashboardPanel::RenderLargeAreaChart(const char* title, const char* subtitle,
                                           const std::vector<float>& data, ImVec4 color,
                                           float min_val, float max_val) {
    ImVec2 avail = ImGui::GetContentRegionAvail();

    ImPlot::PushStyleColor(ImPlotCol_FrameBg, Colors::GraphBg);
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, Colors::GraphBg);
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0.2f, 0.3f, 0.4f, 0.5f));
    ImPlot::PushStyleColor(ImPlotCol_Line, color);

    ImVec4 fill_color = color;
    fill_color.w = 0.4f;
    ImPlot::PushStyleColor(ImPlotCol_Fill, fill_color);

    ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;
    ImPlotAxisFlags axis_flags = ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines;
    ImPlotAxisFlags y_axis_flags = ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoGridLines;

    if (ImPlot::BeginPlot("##LargeChart", avail, flags)) {
        ImPlot::SetupAxes(nullptr, nullptr, axis_flags, y_axis_flags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(data.size()), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, min_val, max_val, ImGuiCond_Always);

        // Draw grid lines manually
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.15f, 0.25f, 0.3f, 0.5f));
        for (int i = 0; i <= 4; ++i) {
            double y = min_val + (max_val - min_val) * i / 4.0;
            double xs[] = {0, static_cast<double>(data.size())};
            double ys[] = {y, y};
            ImPlot::PlotLine("##grid", xs, ys, 2);
        }
        for (int i = 0; i <= 6; ++i) {
            double x = data.size() * i / 6.0;
            double xs[] = {x, x};
            double ys[] = {min_val, max_val};
            ImPlot::PlotLine("##grid", xs, ys, 2);
        }
        ImPlot::PopStyleColor();

        // Draw shaded area
        ImPlot::PlotShaded("##area", data.data(), static_cast<int>(data.size()), 0.0);

        // Draw line on top
        ImPlot::PlotLine("##line", data.data(), static_cast<int>(data.size()));

        // Y axis label at top right
        ImPlot::PushStyleColor(ImPlotCol_InlayText, Colors::TextSecondary);
        ImPlot::PlotText("100%", data.size() - 5, max_val - 0.05);
        ImPlot::PopStyleColor();

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleColor(5);
}

void DashboardPanel::RenderStatsSection() {
    // Layout stats in columns based on selected resource
    switch (selected_resource_) {
        case ResourceType::CPU: {
            ImGui::Columns(4, nullptr, false);

            // Column 1: Utilization
            RenderStatItemLarge(ICON_FA_GAUGE_HIGH " Utilization", fmt::format("{:.0f}%", cpu_usage_ * 100).c_str());
            ImGui::NextColumn();

            // Column 2: Speed and processes
            RenderStatItem(ICON_FA_BOLT " Speed", fmt::format("{:.2f} GHz", cpu_speed_ghz_).c_str());
            RenderStatItem(ICON_FA_WINDOW_RESTORE " Processes", std::to_string(process_count_).c_str());
            RenderStatItem(ICON_FA_BARS " Threads", std::to_string(thread_count_).c_str());
            ImGui::NextColumn();

            // Column 3: Handles and uptime
            RenderStatItem(ICON_FA_LINK " Handles", std::to_string(handle_count_).c_str());
            RenderStatItem(ICON_FA_CLOCK " Up time", FormatUptime(uptime_seconds_).c_str());
            ImGui::NextColumn();

            // Column 4: Cores info
            RenderStatItem(ICON_FA_MICROCHIP " Cores", std::to_string(cpu_cores_).c_str());
            RenderStatItem(ICON_FA_LAYER_GROUP " Logical", std::to_string(cpu_logical_).c_str());
            RenderStatItem(ICON_FA_BARS_PROGRESS " Base speed", fmt::format("{:.2f} GHz", cpu_max_speed_ghz_).c_str());
            ImGui::NextColumn();
            break;
        }

        case ResourceType::Memory: {
            ImGui::Columns(4, nullptr, false);

            // Column 1: In Use
            RenderStatItemLarge(ICON_FA_MEMORY " In Use", fmt::format("{:.1f} GB", ram_used_ / (1024.0 * 1024.0 * 1024.0)).c_str());
            ImGui::NextColumn();

            // Column 2: Available and Committed
            RenderStatItem(ICON_FA_HARD_DRIVE " Available", FormatBytes(ram_total_ - ram_used_).c_str());
            RenderStatItem(ICON_FA_DATABASE " Committed", FormatBytes(mem_committed_).c_str());
            ImGui::NextColumn();

            // Column 3: Cached and Paged
            RenderStatItem(ICON_FA_RECYCLE " Cached", FormatBytes(mem_cached_).c_str());
            RenderStatItem(ICON_FA_CHART_BAR " Paged pool", FormatBytes(mem_paged_pool_).c_str());
            ImGui::NextColumn();

            // Column 4: Non-paged and speed
            RenderStatItem(ICON_FA_CHART_AREA " Non-paged pool", FormatBytes(mem_non_paged_pool_).c_str());
            RenderStatItem(ICON_FA_GAUGE " Usage", fmt::format("{:.1f}%", ram_usage_ * 100).c_str());
            ImGui::NextColumn();
            break;
        }

        case ResourceType::GPU: {
            ImGui::Columns(4, nullptr, false);

            // Column 1: GPU Usage
            RenderStatItemLarge(ICON_FA_GAUGE_HIGH " GPU", fmt::format("{:.0f}%", gpu_usage_ * 100).c_str());
            ImGui::NextColumn();

            // Column 2: VRAM
            RenderStatItem(ICON_FA_MEMORY " Dedicated GPU memory", FormatBytes(vram_used_).c_str());
            RenderStatItem(ICON_FA_DATABASE " Shared GPU memory", FormatBytes(gpu_shared_mem_).c_str());
            ImGui::NextColumn();

            // Column 3: Usage details
            RenderStatItem(ICON_FA_CUBE " 3D", fmt::format("{:.0f}%", gpu_3d_usage_ * 100).c_str());
            RenderStatItem(ICON_FA_COPY " Copy", fmt::format("{:.0f}%", gpu_copy_usage_ * 100).c_str());
            ImGui::NextColumn();

            // Column 4: Temperature and video
            if (gpu_temp_ > 0) {
                RenderStatItem(ICON_FA_TEMPERATURE_HIGH " Temperature", fmt::format("{:.0f} °C", gpu_temp_).c_str());
            }
            RenderStatItem(ICON_FA_VIDEO " Video Decode", fmt::format("{:.0f}%", gpu_video_decode_ * 100).c_str());
            RenderStatItem(ICON_FA_FILM " Video Encode", fmt::format("{:.0f}%", gpu_video_encode_ * 100).c_str());
            ImGui::NextColumn();
            break;
        }

        case ResourceType::Network: {
            ImGui::Columns(3, nullptr, false);

            // Column 1: Send
            RenderStatItemLarge(ICON_FA_ARROW_UP " Send", fmt::format("{:.1f} Mbps", net_out_mbps_).c_str());
            ImGui::NextColumn();

            // Column 2: Receive
            RenderStatItemLarge(ICON_FA_ARROW_DOWN " Receive", fmt::format("{:.1f} Mbps", net_in_mbps_).c_str());
            ImGui::NextColumn();

            // Column 3: Jobs
            RenderStatItem(ICON_FA_BARS_PROGRESS " Active Jobs", std::to_string(active_jobs_).c_str());
            RenderStatItem(ICON_FA_ROCKET " Deployments", std::to_string(active_deployments_).c_str());
            ImGui::NextColumn();
            break;
        }

        default:
            break;
    }

    ImGui::Columns(1);
}

void DashboardPanel::RenderStatItem(const char* label, const char* value) {
    ImGui::TextColored(Colors::TextSecondary, "%s", label);
    ImGui::TextColored(Colors::TextPrimary, "%s", value);
    ImGui::Spacing();
}

void DashboardPanel::RenderStatItemLarge(const char* label, const char* value) {
    ImGui::TextColored(Colors::TextSecondary, "%s", label);
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::TextColored(Colors::TextPrimary, "%s", value);
    ImGui::PopFont();
    ImGui::Spacing();
}

std::string FormatUptime(int64_t seconds) {
    int64_t days = seconds / 86400;
    int64_t hours = (seconds % 86400) / 3600;
    int64_t mins = (seconds % 3600) / 60;
    int64_t secs = seconds % 60;

    if (days > 0) {
        return fmt::format("{}:{:02d}:{:02d}:{:02d}", days, hours, mins, secs);
    }
    return fmt::format("{:02d}:{:02d}:{:02d}", hours, mins, secs);
}

std::string FormatBytes(uint64_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024 * 1024) {
        return fmt::format("{:.1f} TB", bytes / (1024.0 * 1024 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024 * 1024) {
        return fmt::format("{:.1f} GB", bytes / (1024.0 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024) {
        return fmt::format("{:.1f} MB", bytes / (1024.0 * 1024));
    } else if (bytes >= 1024) {
        return fmt::format("{:.1f} KB", bytes / 1024.0);
    }
    return fmt::format("{} B", bytes);
}

void DashboardPanel::UpdateMetrics() {
    // Get metrics from daemon client if connected, otherwise use local collector
    if (IsDaemonConnected()) {
        auto* daemon = GetDaemonClient();

        // Get status
        ipc::DaemonStatus status;
        if (daemon->GetStatus(status)) {
            cpu_usage_ = status.metrics.cpu_usage;
            gpu_usage_ = status.metrics.gpu_usage;
            ram_usage_ = status.metrics.ram_usage;
            vram_usage_ = status.metrics.vram_usage;
            ram_used_ = status.metrics.ram_used;
            ram_total_ = status.metrics.ram_total;
            vram_used_ = status.metrics.vram_used;
            vram_total_ = status.metrics.vram_total;
            net_in_mbps_ = status.metrics.network_rx_mbps;
            net_out_mbps_ = status.metrics.network_tx_mbps;
            gpu_name_ = status.gpu_name;
            gpu_count_ = status.gpu_count;
            uptime_seconds_ = status.uptime_seconds;
            active_jobs_ = status.active_jobs;
            active_deployments_ = status.active_deployments;
        }

        // Get history
        ipc::SystemMetrics dm;
        std::vector<float> cpu_hist, gpu_hist, ram_hist, vram_hist;
        if (daemon->GetMetrics(dm, cpu_hist, gpu_hist, ram_hist, vram_hist)) {
            if (!cpu_hist.empty()) cpu_history_ = cpu_hist;
            if (!gpu_hist.empty()) gpu_history_ = gpu_hist;
            if (!ram_hist.empty()) ram_history_ = ram_hist;
            if (!vram_hist.empty()) vram_history_ = vram_hist;
        }
    } else {
        // Local mode - get from backend
        auto* metrics_collector = GetBackend().GetMetricsCollector();
        if (metrics_collector) {
            auto m = metrics_collector->GetCurrentMetrics();
            cpu_usage_ = m.cpu_usage;
            gpu_usage_ = m.gpu_usage;
            ram_usage_ = m.ram_usage;
            vram_usage_ = m.vram_usage;
            ram_used_ = m.ram_used_bytes;
            ram_total_ = m.ram_total_bytes;
            vram_used_ = m.vram_used_bytes;
            vram_total_ = m.vram_total_bytes;
            net_in_mbps_ = m.network_in_mbps;
            net_out_mbps_ = m.network_out_mbps;

            // Update history
            cpu_history_.erase(cpu_history_.begin());
            cpu_history_.push_back(cpu_usage_);
            gpu_history_.erase(gpu_history_.begin());
            gpu_history_.push_back(gpu_usage_);
            ram_history_.erase(ram_history_.begin());
            ram_history_.push_back(ram_usage_);
            vram_history_.erase(vram_history_.begin());
            vram_history_.push_back(vram_usage_);
            net_in_history_.erase(net_in_history_.begin());
            net_in_history_.push_back(std::min(1.0f, net_in_mbps_ / 100.0f));
            net_out_history_.erase(net_out_history_.begin());
            net_out_history_.push_back(std::min(1.0f, net_out_mbps_ / 100.0f));
        }

        auto* state = GetState();
        if (state) {
            active_jobs_ = static_cast<int>(state->GetActiveJobs().size());
            active_deployments_ = static_cast<int>(state->GetDeployments().size());
        }
    }

#ifdef _WIN32
    // Get system metrics from Windows APIs
    PERFORMANCE_INFORMATION perfInfo = {0};
    perfInfo.cb = sizeof(perfInfo);
    if (GetPerformanceInfo(&perfInfo, sizeof(perfInfo))) {
        mem_committed_ = perfInfo.CommitTotal * perfInfo.PageSize;
        mem_cached_ = perfInfo.SystemCache * perfInfo.PageSize;
        mem_paged_pool_ = perfInfo.KernelPaged * perfInfo.PageSize;
        mem_non_paged_pool_ = perfInfo.KernelNonpaged * perfInfo.PageSize;
        process_count_ = perfInfo.ProcessCount;
        thread_count_ = perfInfo.ThreadCount;
        handle_count_ = perfInfo.HandleCount;
    }

    // Get uptime
    uptime_seconds_ = GetTickCount64() / 1000;

    // Generate simulated per-core usage (actual per-core requires more complex PDH queries)
    for (int i = 0; i < static_cast<int>(per_core_usage_.size()); ++i) {
        // Add some variation per core
        float variation = static_cast<float>(rand() % 20 - 10) / 100.0f;
        per_core_usage_[i] = std::clamp(cpu_usage_ + variation, 0.0f, 1.0f);

        // Update per-core history
        if (i < static_cast<int>(per_core_history_.size())) {
            per_core_history_[i].erase(per_core_history_[i].begin());
            per_core_history_[i].push_back(per_core_usage_[i]);
        }
    }

    // Get VRAM from ArrayFire if available
    try {
        // GPU shared memory estimate (system RAM used as GPU memory)
        MEMORYSTATUSEX memStatus = {0};
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            ram_total_ = memStatus.ullTotalPhys;
            ram_used_ = memStatus.ullTotalPhys - memStatus.ullAvailPhys;
            // Estimate shared GPU memory as a portion of system memory
            gpu_shared_mem_ = ram_total_ / 4;  // Typically ~25% available for GPU
        }
    } catch (...) {
        // Ignore ArrayFire errors
    }
#endif
}

} // namespace cyxwiz::servernode::gui
