// hardware_panel.h - Hardware detection display panel
#pragma once

#include "gui/server_panel.h"
#include "core/state_manager.h"
#include <vector>
#include <string>
#include <chrono>

namespace cyxwiz::servernode::gui {

class HardwarePanel : public ServerPanel {
public:
    HardwarePanel();
    void Render() override;
    void Update() override;

private:
    // Main render sections
    void RenderGpuSection();
    void RenderCpuSection();
    void RenderMemorySection();
    void RenderSummarySection();

    // Helper functions
    void RenderSectionHeader(const char* title, const char* icon = nullptr);
    void RenderGpuCard(int index, const struct core::GPUMetrics& gpu);
    std::string FormatBytes(size_t bytes) const;
    std::string FormatSpeed(float ghz) const;

    // Cached hardware info (updated periodically)
    struct CpuInfo {
        std::string name;
        std::string vendor;
        int physical_cores = 0;
        int logical_cores = 0;
        float base_speed_ghz = 0.0f;
        float current_speed_ghz = 0.0f;
        std::string architecture;
    };
    CpuInfo cpu_info_;

    struct MemoryInfo {
        size_t total_bytes = 0;
        size_t used_bytes = 0;
        size_t available_bytes = 0;
        int num_slots = 0;
        std::string speed;
    };
    MemoryInfo memory_info_;

    // Update timing
    std::chrono::steady_clock::time_point last_update_;
    static constexpr int UPDATE_INTERVAL_MS = 1000;

    // Refresh hardware detection
    void RefreshHardwareInfo();
    bool hardware_initialized_ = false;
};

} // namespace cyxwiz::servernode::gui
