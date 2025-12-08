// device_pool.h - Multi-GPU device pool management
#pragma once

#include <cyxwiz/device.h>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <map>
#include <optional>
#include <memory>
#include <functional>

namespace cyxwiz::servernode::core {

// State of a single device in the pool
struct DeviceState {
    int device_id = -1;
    cyxwiz::DeviceType type = cyxwiz::DeviceType::CPU;
    std::string name;

    // Memory info (bytes)
    size_t total_memory = 0;
    size_t available_memory = 0;
    size_t used_memory = 0;

    // Utilization (0.0 - 1.0)
    float utilization = 0.0f;
    float temperature = 0.0f;    // Celsius
    float power_watts = 0.0f;

    // Assignment state
    bool in_use = false;
    std::string assigned_job_id;
    int64_t assignment_time = 0;  // Unix timestamp

    // Cumulative stats
    int64_t total_jobs_completed = 0;
    int64_t total_compute_time_ms = 0;
};

// Device selection strategy
enum class DeviceSelectionStrategy {
    LeastUtilized,      // Pick device with lowest utilization
    MostMemory,         // Pick device with most free memory
    RoundRobin,         // Cycle through devices
    Affinity,           // Try to use same device as previous job from same user
    FirstAvailable      // Just pick first free device
};

// Configuration for device pool
struct DevicePoolConfig {
    // Which device types to include
    bool include_cpu = false;       // Usually false for GPU compute
    bool include_cuda = true;
    bool include_opencl = true;
    bool include_metal = true;

    // Selection strategy
    DeviceSelectionStrategy strategy = DeviceSelectionStrategy::LeastUtilized;

    // Memory threshold (don't assign if less than this available)
    size_t min_available_memory_mb = 512;

    // Max concurrent jobs per device (0 = 1 job per device)
    int max_jobs_per_device = 1;

    // Refresh interval for metrics
    int metrics_refresh_ms = 1000;
};

// Callback for device state changes
using DeviceStateCallback = std::function<void(int device_id, const DeviceState& state)>;

/**
 * DevicePool - Manages multiple compute devices for job assignment
 *
 * Features:
 * - Enumerates and tracks all available GPUs
 * - Load balancing across multiple devices
 * - Memory-aware job assignment
 * - Per-device metrics tracking
 * - Job queuing when all devices busy
 */
class DevicePool {
public:
    DevicePool();
    explicit DevicePool(const DevicePoolConfig& config);
    ~DevicePool();

    // Initialize the pool by enumerating devices
    bool Initialize();

    // Re-enumerate devices (call if hardware changes)
    void Refresh();

    // Device queries
    int GetDeviceCount() const;
    int GetAvailableDeviceCount() const;
    std::vector<DeviceState> GetAllDeviceStates() const;
    std::optional<DeviceState> GetDeviceState(int device_id) const;

    // Job assignment
    // Returns device_id if acquired, -1 if no device available
    int AcquireDevice(const std::string& job_id,
                      size_t required_memory_mb = 0,
                      const std::string& preferred_device_hint = "");

    // Release device back to pool
    void ReleaseDevice(int device_id, bool job_completed = true);

    // Check if a specific device is available
    bool IsDeviceAvailable(int device_id) const;

    // Get device assigned to a job
    int GetDeviceForJob(const std::string& job_id) const;

    // Update device metrics (call periodically)
    void UpdateMetrics();

    // Get pool statistics
    struct PoolStats {
        int total_devices = 0;
        int available_devices = 0;
        int busy_devices = 0;
        int64_t total_jobs_completed = 0;
        size_t total_memory = 0;
        size_t available_memory = 0;
        float avg_utilization = 0.0f;
    };
    PoolStats GetStats() const;

    // Configuration
    void SetConfig(const DevicePoolConfig& config);
    const DevicePoolConfig& GetConfig() const { return config_; }

    // Callbacks
    void SetDeviceStateCallback(DeviceStateCallback callback);

private:
    // Device selection based on strategy
    int SelectDevice(size_t required_memory_mb);
    int SelectLeastUtilized(size_t required_memory_mb);
    int SelectMostMemory(size_t required_memory_mb);
    int SelectRoundRobin(size_t required_memory_mb);
    int SelectFirstAvailable(size_t required_memory_mb);

    // Check if device meets requirements
    bool MeetsRequirements(const DeviceState& state, size_t required_memory_mb) const;

    // Update a single device's metrics
    void UpdateDeviceMetrics(int device_id);

    // Configuration
    DevicePoolConfig config_;

    // Device states
    std::vector<DeviceState> devices_;
    mutable std::mutex devices_mutex_;

    // Job -> device mapping
    std::map<std::string, int> job_device_map_;

    // Round robin counter
    int round_robin_index_ = 0;

    // User affinity (user_id -> last_device_id)
    std::map<std::string, int> user_affinity_;

    // Callback for state changes
    DeviceStateCallback state_callback_;

    // Initialized flag
    bool initialized_ = false;
};

/**
 * ScopedDeviceContext - RAII helper for device switching
 *
 * Automatically switches to the specified device on construction
 * and restores the previous device on destruction.
 */
class ScopedDeviceContext {
public:
    explicit ScopedDeviceContext(int device_id);
    ~ScopedDeviceContext();

    // Non-copyable, non-movable
    ScopedDeviceContext(const ScopedDeviceContext&) = delete;
    ScopedDeviceContext& operator=(const ScopedDeviceContext&) = delete;
    ScopedDeviceContext(ScopedDeviceContext&&) = delete;
    ScopedDeviceContext& operator=(ScopedDeviceContext&&) = delete;

    bool IsValid() const { return valid_; }
    int GetDeviceId() const { return device_id_; }

private:
    int device_id_;
    int previous_device_id_;
    bool valid_ = false;
};

} // namespace cyxwiz::servernode::core
