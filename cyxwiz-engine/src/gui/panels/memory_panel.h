#pragma once

#include "../panel.h"
#include <imgui.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <chrono>
#include <functional>

namespace cyxwiz {

/**
 * Memory snapshot at a specific point in time
 */
struct MemorySnapshot {
    double timestamp = 0.0;          // Seconds since start
    size_t cpu_heap_bytes = 0;       // Total CPU heap allocation
    size_t cpu_tensors_bytes = 0;    // CPU tensor memory
    size_t gpu_allocated_bytes = 0;  // GPU memory allocated
    size_t gpu_cached_bytes = 0;     // GPU memory cached
    std::map<std::string, size_t> per_layer_memory;  // Memory per layer
    int epoch = 0;
    int step = 0;
};

/**
 * Memory Visualization Panel
 * Displays CPU and GPU memory usage over time with per-layer breakdown.
 * Uses ImPlot for charts and provides real-time monitoring.
 */
class MemoryPanel : public Panel {
public:
    MemoryPanel();
    ~MemoryPanel() override = default;

    void Render() override;

    // Data collection interface (called from TrainingExecutor or backend)
    void BeginMonitoring();
    void EndMonitoring();
    void RecordSnapshot(size_t cpu_heap, size_t cpu_tensors, size_t gpu_allocated, size_t gpu_cached);
    void RecordLayerMemory(const std::string& layer_name, size_t bytes);
    void FinalizeSnapshot(int epoch, int step);

    // Query current GPU status (if ArrayFire is available)
    void UpdateGPUStatus();

    // Clear all data
    void Clear();

    // Export data
    bool ExportToCSV(const std::string& path);

    // Check if monitoring is active
    bool IsMonitoringActive() const { return is_monitoring_; }

    // Set callback for highlighting a layer in the editor when clicked
    using LayerClickCallback = std::function<void(const std::string& layer_name)>;
    void SetLayerClickCallback(LayerClickCallback callback) { layer_click_callback_ = callback; }

private:
    // Rendering functions
    void RenderToolbar();
    void RenderOverviewChart();
    void RenderLayerBreakdown();
    void RenderGPUDetails();
    void RenderMemoryStats();

    // Helper functions
    std::string FormatBytes(size_t bytes) const;
    ImVec4 GetMemoryColor(double usage_percentage) const;

    // Monitoring state
    bool is_monitoring_ = false;
    std::chrono::steady_clock::time_point monitoring_start_time_;

    // Current snapshot being collected
    MemorySnapshot current_snapshot_;
    mutable std::mutex data_mutex_;

    // Historical data
    std::vector<MemorySnapshot> history_;
    static constexpr size_t kMaxHistorySize = 2000;

    // GPU status info
    struct GPUInfo {
        std::string name;
        size_t total_memory = 0;
        size_t free_memory = 0;
        std::string backend;  // "CUDA", "OpenCL", "CPU", etc.
        int device_id = -1;
        float utilization = 0.0f;  // 0-100%
        float temperature = 0.0f;  // Celsius
    };
    GPUInfo gpu_info_;
    std::chrono::steady_clock::time_point last_gpu_update_;

    // UI state
    int view_mode_ = 0;  // 0=overview, 1=layers, 2=gpu
    bool auto_refresh_ = true;
    float refresh_interval_ms_ = 100.0f;
    std::chrono::steady_clock::time_point last_refresh_;

    // Chart options
    bool show_cpu_heap_ = true;
    bool show_cpu_tensors_ = true;
    bool show_gpu_allocated_ = true;
    bool show_gpu_cached_ = false;
    float chart_time_window_ = 60.0f;  // Show last N seconds

    // Layer breakdown options
    int breakdown_mode_ = 0;  // 0=bar chart, 1=pie chart
    int top_n_layers_ = 10;   // Show top N layers by memory

    // Colors
    ImVec4 cpu_heap_color_ = ImVec4(0.2f, 0.6f, 0.2f, 1.0f);     // Green
    ImVec4 cpu_tensor_color_ = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);   // Light green
    ImVec4 gpu_allocated_color_ = ImVec4(0.2f, 0.4f, 0.8f, 1.0f); // Blue
    ImVec4 gpu_cached_color_ = ImVec4(0.4f, 0.6f, 0.9f, 1.0f);   // Light blue

    // Callback
    LayerClickCallback layer_click_callback_;
};

} // namespace cyxwiz
