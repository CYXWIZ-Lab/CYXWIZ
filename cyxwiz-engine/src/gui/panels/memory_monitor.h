#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <imgui.h>

#ifdef _WIN32
#include <pdh.h>
#endif

namespace cyxwiz {

// System metrics collected each update
struct SystemMetrics {
    // CPU
    float cpu_usage_percent = 0.0f;
    float cpu_frequency_ghz = 0.0f;
    int cpu_core_count = 0;

    // RAM (System-wide)
    size_t ram_used_bytes = 0;
    size_t ram_total_bytes = 0;
    float ram_usage_percent = 0.0f;

    // GPU
    bool gpu_available = false;
    std::string gpu_name;
    size_t gpu_vram_used = 0;
    size_t gpu_vram_total = 0;
    float gpu_vram_usage_percent = 0.0f;

    // Process
    size_t process_memory_bytes = 0;
    float process_cpu_percent = 0.0f;
    int process_thread_count = 0;
};

class MemoryMonitor {
public:
    MemoryMonitor();
    ~MemoryMonitor();

    void Render();
    void Update();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    void Toggle() { visible_ = !visible_; }

private:
    bool visible_ = false;

    // Current metrics
    SystemMetrics metrics_;

    // History for sparklines (60 points = 30 seconds at 500ms interval)
    static constexpr size_t SPARKLINE_POINTS = 60;
    std::vector<float> cpu_history_;
    std::vector<float> ram_history_;
    std::vector<float> gpu_vram_history_;
    std::vector<float> process_mem_history_;

    // Update timing
    std::chrono::steady_clock::time_point last_update_;
    float update_interval_ms_ = 500.0f;

    // Windows PDH for CPU usage
#ifdef _WIN32
    PDH_HQUERY cpu_query_ = nullptr;
    PDH_HCOUNTER cpu_counter_ = nullptr;
    bool pdh_initialized_ = false;

    // For process CPU calculation
    ULONGLONG last_process_time_ = 0;
    ULONGLONG last_system_time_ = 0;
#endif

    // Data collection methods
    void InitializePDH();
    void CleanupPDH();
    void UpdateCPUUsage();
    void UpdateRAMUsage();
    void UpdateGPUStatus();
    void UpdateProcessInfo();
    void ShiftHistory();

    // UI rendering helpers
    void RenderMetricBar(const char* label, float percent, const char* value_text);
    void RenderSparkline(const char* id, const std::vector<float>& data, float height, ImU32 color);
    ImU32 ColorFromPercent(float percent) const;
    std::string FormatBytes(size_t bytes) const;
};

} // namespace cyxwiz
