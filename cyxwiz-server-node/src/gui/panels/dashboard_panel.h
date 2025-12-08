// dashboard_panel.h - System overview panel (Task Manager style)
#pragma once

#include "gui/server_panel.h"
#include <vector>
#include <string>
#include <chrono>

namespace cyxwiz::servernode::gui {

// Resource types that can be selected in the sidebar
enum class ResourceType {
    CPU = 0,
    Memory,
    GPU,
    Network,
    COUNT
};

class DashboardPanel : public ServerPanel {
public:
    DashboardPanel();
    void Render() override;

private:
    // Main layout sections
    void RenderResourceSidebar();
    void RenderMainGraphArea();
    void RenderStatsSection();

    // Sidebar resource items with mini sparklines
    void RenderResourceItem(ResourceType type, const char* name, const char* subtitle,
                            float usage, const std::vector<float>& history, bool selected);

    // Large graph rendering
    void RenderLargeAreaChart(const char* title, const char* subtitle,
                              const std::vector<float>& data, ImVec4 color,
                              float min_val = 0.0f, float max_val = 1.0f);
    void RenderMultiGraphGrid(const std::vector<std::vector<float>>& data_sets,
                              const std::vector<std::string>& labels, ImVec4 color);
    void RenderCoreGraphGrid();  // Per-core CPU graphs

    // Stats rendering
    void RenderStatItem(const char* label, const char* value);
    void RenderStatItemLarge(const char* label, const char* value);

    // Data fetching
    void UpdateMetrics();

    // Selected resource
    ResourceType selected_resource_ = ResourceType::CPU;

    // History buffers (60 seconds of data)
    static constexpr int HISTORY_SIZE = 60;
    std::vector<float> cpu_history_;
    std::vector<float> gpu_history_;
    std::vector<float> ram_history_;
    std::vector<float> vram_history_;
    std::vector<float> net_in_history_;
    std::vector<float> net_out_history_;

    // Current metrics
    float cpu_usage_ = 0.0f;
    float gpu_usage_ = 0.0f;
    float ram_usage_ = 0.0f;
    float vram_usage_ = 0.0f;
    uint64_t ram_used_ = 0;
    uint64_t ram_total_ = 0;
    uint64_t vram_used_ = 0;
    uint64_t vram_total_ = 0;
    float net_in_mbps_ = 0.0f;
    float net_out_mbps_ = 0.0f;

    // CPU info
    std::string cpu_name_ = "Unknown CPU";
    int cpu_cores_ = 0;           // Physical cores
    int cpu_logical_ = 0;         // Logical processors (threads)
    float cpu_speed_ghz_ = 0.0f;  // Current/base speed
    float cpu_max_speed_ghz_ = 0.0f;
    int process_count_ = 0;
    int thread_count_ = 0;
    int handle_count_ = 0;
    std::vector<float> per_core_usage_;  // Per-core utilization
    std::vector<std::vector<float>> per_core_history_;  // Per-core history

    // GPU info
    std::string gpu_name_ = "Unknown GPU";
    int gpu_count_ = 0;
    float gpu_temp_ = 0.0f;       // Temperature in Celsius
    uint64_t gpu_shared_mem_ = 0; // Shared GPU memory
    float gpu_copy_usage_ = 0.0f;
    float gpu_3d_usage_ = 0.0f;
    float gpu_video_decode_ = 0.0f;
    float gpu_video_encode_ = 0.0f;

    // Memory info
    uint64_t mem_committed_ = 0;
    uint64_t mem_cached_ = 0;
    uint64_t mem_paged_pool_ = 0;
    uint64_t mem_non_paged_pool_ = 0;

    // System info
    int64_t uptime_seconds_ = 0;
    int active_jobs_ = 0;
    int active_deployments_ = 0;

    // Update timing
    std::chrono::steady_clock::time_point last_update_;
};

} // namespace cyxwiz::servernode::gui
