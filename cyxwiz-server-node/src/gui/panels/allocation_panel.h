// allocation_panel.h - Resource allocation panel for compute sharing
#pragma once

#include "gui/server_panel.h"
#include <vector>
#include <string>
#include <chrono>

namespace cyxwiz::servernode::gui {

// Resource allocation for a single device
struct ResourceAllocation {
    enum class DeviceType { Cpu, Gpu };

    DeviceType device_type = DeviceType::Gpu;
    int device_id = 0;
    std::string device_name;

    bool is_enabled = false;          // Share this device?

    // GPU-specific (in MB)
    size_t vram_total_mb = 0;
    size_t vram_allocated_mb = 0;
    size_t vram_reserved_mb = 2048;   // Keep 2GB for system by default

    // CPU-specific
    int cores_total = 0;
    int cores_allocated = 0;
    int cores_reserved = 2;           // Keep 2 cores for system

    // Common settings
    int priority = 1;                 // 0=low, 1=normal, 2=high
    int max_power_percent = 100;
};

class AllocationPanel : public ServerPanel {
public:
    AllocationPanel();
    void Render() override;
    void Update() override;

    // Get current allocations for registration
    const std::vector<ResourceAllocation>& GetAllocations() const { return allocations_; }

    // Check if any resources are allocated
    bool HasAllocations() const;

    // Get total allocated resources for summary
    size_t GetTotalAllocatedVramMb() const;
    int GetTotalAllocatedCores() const;
    int GetEnabledGpuCount() const;

private:
    // Main sections
    void RenderHeader();
    void RenderGpuAllocations();
    void RenderCpuAllocation();
    void RenderSummary();
    void RenderActionButtons();

    // Individual device allocation
    void RenderGpuAllocationCard(int index, ResourceAllocation& alloc);
    void RenderCpuAllocationCard(ResourceAllocation& alloc);

    // Helpers
    void RefreshDeviceList();
    void LoadAllocations();
    void SaveAllocations();
    void ApplyAllocations();
    void RetryConnection();
    std::string FormatMB(size_t mb) const;

    // Allocation data
    std::vector<ResourceAllocation> allocations_;
    bool allocations_dirty_ = false;

    // UI state
    bool show_advanced_ = false;
    std::string status_message_;
    bool is_applying_ = false;
    bool show_retry_button_ = false;
    bool connection_failed_ = false;

    // Update timing
    std::chrono::steady_clock::time_point last_refresh_;
    bool devices_initialized_ = false;
};

} // namespace cyxwiz::servernode::gui
