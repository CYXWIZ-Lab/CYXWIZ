#pragma once
#include <string>
#include <vector>
#include <chrono>

namespace cyxwiz {

class MemoryMonitor {
public:
    MemoryMonitor();
    ~MemoryMonitor() = default;

    void Render();
    void Update();  // Call each frame to update stats

    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    void Toggle() { visible_ = !visible_; }

private:
    bool visible_ = false;

    // CPU Memory
    float cpu_used_mb_ = 0.0f;
    float cpu_total_mb_ = 0.0f;

    // GPU Memory (if available)
    float gpu_used_mb_ = 0.0f;
    float gpu_total_mb_ = 0.0f;
    bool gpu_available_ = false;

    // History for graphs
    static constexpr size_t HISTORY_SIZE = 120;
    std::vector<float> cpu_history_;
    std::vector<float> gpu_history_;

    // Update timing
    std::chrono::steady_clock::time_point last_update_;
    float update_interval_ms_ = 500.0f;

    void UpdateCPUMemory();
    void UpdateGPUMemory();
};

} // namespace cyxwiz
