// device_pool.cpp - Multi-GPU device pool management implementation
#include "core/device_pool.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <algorithm>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::servernode::core {

DevicePool::DevicePool()
    : config_{} {
}

DevicePool::DevicePool(const DevicePoolConfig& config)
    : config_(config) {
}

DevicePool::~DevicePool() {
    spdlog::info("DevicePool destroyed");
}

bool DevicePool::Initialize() {
    spdlog::info("Initializing DevicePool...");

    Refresh();

    if (devices_.empty()) {
        spdlog::warn("No compute devices found during initialization");
        return false;
    }

    initialized_ = true;
    spdlog::info("DevicePool initialized with {} devices", devices_.size());
    return true;
}

void DevicePool::Refresh() {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Get list of available devices from backend
    auto available_devices = cyxwiz::Device::GetAvailableDevices();

    spdlog::info("Found {} devices during enumeration", available_devices.size());

    // Preserve existing job assignments when refreshing
    std::map<int, std::string> existing_assignments;
    for (const auto& dev : devices_) {
        if (dev.in_use) {
            existing_assignments[dev.device_id] = dev.assigned_job_id;
        }
    }

    devices_.clear();

    for (const auto& info : available_devices) {
        // Filter by device type based on config
        switch (info.type) {
            case cyxwiz::DeviceType::CPU:
                if (!config_.include_cpu) continue;
                break;
            case cyxwiz::DeviceType::CUDA:
                if (!config_.include_cuda) continue;
                break;
            case cyxwiz::DeviceType::OPENCL:
                if (!config_.include_opencl) continue;
                break;
            case cyxwiz::DeviceType::METAL:
                if (!config_.include_metal) continue;
                break;
            default:
                continue;
        }

        DeviceState state;
        state.device_id = info.device_id;
        state.type = info.type;
        state.name = info.name;
        state.total_memory = info.memory_total;
        state.available_memory = info.memory_available;
        state.used_memory = info.memory_total - info.memory_available;
        state.utilization = 0.0f;
        state.in_use = false;

        // Restore existing assignment if any
        auto it = existing_assignments.find(info.device_id);
        if (it != existing_assignments.end()) {
            state.in_use = true;
            state.assigned_job_id = it->second;
        }

        devices_.push_back(state);

        spdlog::debug("Device {}: {} ({} MB total, {} MB available)",
                      info.device_id, info.name,
                      info.memory_total / (1024 * 1024),
                      info.memory_available / (1024 * 1024));
    }
}

int DevicePool::GetDeviceCount() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return static_cast<int>(devices_.size());
}

int DevicePool::GetAvailableDeviceCount() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    int count = 0;
    for (const auto& dev : devices_) {
        if (!dev.in_use) {
            count++;
        }
    }
    return count;
}

std::vector<DeviceState> DevicePool::GetAllDeviceStates() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return devices_;
}

std::optional<DeviceState> DevicePool::GetDeviceState(int device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    for (const auto& dev : devices_) {
        if (dev.device_id == device_id) {
            return dev;
        }
    }
    return std::nullopt;
}

int DevicePool::AcquireDevice(const std::string& job_id,
                               size_t required_memory_mb,
                               const std::string& preferred_device_hint) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Check if job already has a device
    auto it = job_device_map_.find(job_id);
    if (it != job_device_map_.end()) {
        spdlog::warn("Job {} already has device {} assigned", job_id, it->second);
        return it->second;
    }

    // Try to parse preferred device hint as device ID
    if (!preferred_device_hint.empty()) {
        try {
            int preferred_id = std::stoi(preferred_device_hint);
            for (auto& dev : devices_) {
                if (dev.device_id == preferred_id &&
                    !dev.in_use &&
                    MeetsRequirements(dev, required_memory_mb)) {
                    dev.in_use = true;
                    dev.assigned_job_id = job_id;
                    dev.assignment_time = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    job_device_map_[job_id] = dev.device_id;

                    spdlog::info("Acquired preferred device {} for job {}", dev.device_id, job_id);
                    return dev.device_id;
                }
            }
        } catch (...) {
            // Not a valid integer, continue with strategy selection
        }
    }

    // Select device based on strategy
    int device_id = SelectDevice(required_memory_mb);

    if (device_id < 0) {
        spdlog::warn("No available device for job {} (required {} MB)", job_id, required_memory_mb);
        return -1;
    }

    // Mark device as in use
    for (auto& dev : devices_) {
        if (dev.device_id == device_id) {
            dev.in_use = true;
            dev.assigned_job_id = job_id;
            dev.assignment_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            job_device_map_[job_id] = device_id;

            spdlog::info("Acquired device {} ({}) for job {}", device_id, dev.name, job_id);

            if (state_callback_) {
                state_callback_(device_id, dev);
            }

            return device_id;
        }
    }

    return -1;
}

void DevicePool::ReleaseDevice(int device_id, bool job_completed) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    for (auto& dev : devices_) {
        if (dev.device_id == device_id) {
            std::string job_id = dev.assigned_job_id;

            if (job_completed) {
                dev.total_jobs_completed++;

                // Calculate compute time
                int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                int64_t duration_s = now - dev.assignment_time;
                dev.total_compute_time_ms += duration_s * 1000;
            }

            dev.in_use = false;
            dev.assigned_job_id.clear();
            dev.assignment_time = 0;

            // Remove from job map
            job_device_map_.erase(job_id);

            spdlog::info("Released device {} from job {} (completed: {})",
                         device_id, job_id, job_completed);

            if (state_callback_) {
                state_callback_(device_id, dev);
            }

            return;
        }
    }

    spdlog::warn("Attempted to release unknown device {}", device_id);
}

bool DevicePool::IsDeviceAvailable(int device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    for (const auto& dev : devices_) {
        if (dev.device_id == device_id) {
            return !dev.in_use;
        }
    }
    return false;
}

int DevicePool::GetDeviceForJob(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    auto it = job_device_map_.find(job_id);
    if (it != job_device_map_.end()) {
        return it->second;
    }
    return -1;
}

void DevicePool::UpdateMetrics() {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    for (auto& dev : devices_) {
        UpdateDeviceMetrics(dev.device_id);
    }
}

void DevicePool::UpdateDeviceMetrics(int device_id) {
    // Note: This requires backend-level access to device metrics
    // For now, we'll use ArrayFire directly if available

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Temporarily switch to this device to get metrics
        int current_device = af::getDevice();
        af::setDevice(device_id);

        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        // Update device state
        for (auto& dev : devices_) {
            if (dev.device_id == device_id) {
                dev.used_memory = alloc_bytes;
                dev.available_memory = dev.total_memory - alloc_bytes;

                // Estimate utilization based on lock ratio
                if (dev.total_memory > 0) {
                    dev.utilization = static_cast<float>(lock_bytes) /
                                     static_cast<float>(dev.total_memory);
                }
                break;
            }
        }

        // Restore previous device
        af::setDevice(current_device);
    } catch (const af::exception& e) {
        spdlog::warn("Failed to update metrics for device {}: {}", device_id, e.what());
    }
#endif
}

DevicePool::PoolStats DevicePool::GetStats() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    PoolStats stats;
    stats.total_devices = static_cast<int>(devices_.size());

    for (const auto& dev : devices_) {
        if (dev.in_use) {
            stats.busy_devices++;
        } else {
            stats.available_devices++;
        }
        stats.total_jobs_completed += dev.total_jobs_completed;
        stats.total_memory += dev.total_memory;
        stats.available_memory += dev.available_memory;
        stats.avg_utilization += dev.utilization;
    }

    if (!devices_.empty()) {
        stats.avg_utilization /= static_cast<float>(devices_.size());
    }

    return stats;
}

void DevicePool::SetConfig(const DevicePoolConfig& config) {
    config_ = config;

    // Re-filter devices if config changes
    if (initialized_) {
        Refresh();
    }
}

void DevicePool::SetDeviceStateCallback(DeviceStateCallback callback) {
    state_callback_ = std::move(callback);
}

int DevicePool::SelectDevice(size_t required_memory_mb) {
    switch (config_.strategy) {
        case DeviceSelectionStrategy::LeastUtilized:
            return SelectLeastUtilized(required_memory_mb);
        case DeviceSelectionStrategy::MostMemory:
            return SelectMostMemory(required_memory_mb);
        case DeviceSelectionStrategy::RoundRobin:
            return SelectRoundRobin(required_memory_mb);
        case DeviceSelectionStrategy::FirstAvailable:
        default:
            return SelectFirstAvailable(required_memory_mb);
    }
}

int DevicePool::SelectLeastUtilized(size_t required_memory_mb) {
    int best_device = -1;
    float lowest_utilization = 2.0f;  // Higher than max (1.0)

    for (const auto& dev : devices_) {
        if (!dev.in_use && MeetsRequirements(dev, required_memory_mb)) {
            if (dev.utilization < lowest_utilization) {
                lowest_utilization = dev.utilization;
                best_device = dev.device_id;
            }
        }
    }

    return best_device;
}

int DevicePool::SelectMostMemory(size_t required_memory_mb) {
    int best_device = -1;
    size_t most_memory = 0;

    for (const auto& dev : devices_) {
        if (!dev.in_use && MeetsRequirements(dev, required_memory_mb)) {
            if (dev.available_memory > most_memory) {
                most_memory = dev.available_memory;
                best_device = dev.device_id;
            }
        }
    }

    return best_device;
}

int DevicePool::SelectRoundRobin(size_t required_memory_mb) {
    if (devices_.empty()) return -1;

    int num_devices = static_cast<int>(devices_.size());
    int start_index = round_robin_index_;

    for (int i = 0; i < num_devices; ++i) {
        int idx = (start_index + i) % num_devices;
        const auto& dev = devices_[idx];

        if (!dev.in_use && MeetsRequirements(dev, required_memory_mb)) {
            round_robin_index_ = (idx + 1) % num_devices;
            return dev.device_id;
        }
    }

    return -1;
}

int DevicePool::SelectFirstAvailable(size_t required_memory_mb) {
    for (const auto& dev : devices_) {
        if (!dev.in_use && MeetsRequirements(dev, required_memory_mb)) {
            return dev.device_id;
        }
    }
    return -1;
}

bool DevicePool::MeetsRequirements(const DeviceState& state, size_t required_memory_mb) const {
    // Check minimum available memory config
    size_t min_available_bytes = config_.min_available_memory_mb * 1024 * 1024;
    if (state.available_memory < min_available_bytes) {
        return false;
    }

    // Check job-specific memory requirement
    if (required_memory_mb > 0) {
        size_t required_bytes = required_memory_mb * 1024 * 1024;
        if (state.available_memory < required_bytes) {
            return false;
        }
    }

    return true;
}

// ============================================================
// ScopedDeviceContext Implementation
// ============================================================

ScopedDeviceContext::ScopedDeviceContext(int device_id)
    : device_id_(device_id)
    , previous_device_id_(-1)
    , valid_(false) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Save current device
        previous_device_id_ = af::getDevice();

        // Switch to requested device
        if (device_id_ >= 0 && device_id_ < af::getDeviceCount()) {
            af::setDevice(device_id_);
            valid_ = true;
            spdlog::debug("ScopedDeviceContext: switched to device {}", device_id_);
        } else {
            spdlog::warn("ScopedDeviceContext: invalid device_id {}", device_id_);
        }
    } catch (const af::exception& e) {
        spdlog::error("ScopedDeviceContext: failed to set device {}: {}", device_id_, e.what());
    }
#else
    // Without ArrayFire, just mark as valid (CPU-only mode)
    valid_ = true;
    previous_device_id_ = 0;
#endif
}

ScopedDeviceContext::~ScopedDeviceContext() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (valid_ && previous_device_id_ >= 0) {
        try {
            af::setDevice(previous_device_id_);
            spdlog::debug("ScopedDeviceContext: restored to device {}", previous_device_id_);
        } catch (const af::exception& e) {
            spdlog::error("ScopedDeviceContext: failed to restore device {}: {}",
                          previous_device_id_, e.what());
        }
    }
#endif
}

} // namespace cyxwiz::servernode::core
