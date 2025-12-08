// metrics_collector.h - System metrics collection (CPU, GPU, RAM, Network)
#pragma once

#include "core/state_manager.h"
#include <thread>
#include <atomic>
#include <deque>
#include <mutex>
#include <chrono>

namespace cyxwiz::servernode::core {

enum class MetricType {
    CPU,
    GPU,
    RAM,
    VRAM,
    NetworkIn,
    NetworkOut,
    Temperature,
    Power
};

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();

    // Start/stop background collection
    void StartCollection(int interval_ms = 1000);
    void StopCollection();
    bool IsCollecting() const { return running_.load(); }

    // Get current snapshot
    SystemMetrics GetCurrentMetrics() const;

    // Get history (for graphs)
    std::vector<float> GetHistory(MetricType type, int samples) const;

    // Set collection interval
    void SetInterval(int interval_ms);

    // Get max history size
    static constexpr int MAX_HISTORY_SIZE = 300;  // 5 minutes at 1 second intervals

private:
    void CollectionLoop();

    // Platform-specific collection
    float CollectCPUUsage();
    float CollectGPUUsage();
    size_t CollectRAMUsed();
    size_t CollectRAMTotal();
    size_t CollectVRAMUsed();
    size_t CollectVRAMTotal();
    float CollectNetworkIn();
    float CollectNetworkOut();
    float CollectTemperature();
    float CollectPowerUsage();

    std::thread collection_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> interval_ms_{1000};

    mutable std::mutex mutex_;
    SystemMetrics current_metrics_;

    // History buffers
    std::deque<float> cpu_history_;
    std::deque<float> gpu_history_;
    std::deque<float> ram_history_;
    std::deque<float> vram_history_;
    std::deque<float> net_in_history_;
    std::deque<float> net_out_history_;

    // For calculating rates
    std::chrono::steady_clock::time_point last_sample_time_;
    uint64_t last_net_bytes_in_ = 0;
    uint64_t last_net_bytes_out_ = 0;

#ifdef _WIN32
    // Windows-specific handles
    void* cpu_query_ = nullptr;
    void* cpu_counter_ = nullptr;
#else
    // Linux-specific state
    long last_cpu_idle_ = 0;
    long last_cpu_total_ = 0;
#endif
};

} // namespace cyxwiz::servernode::core
