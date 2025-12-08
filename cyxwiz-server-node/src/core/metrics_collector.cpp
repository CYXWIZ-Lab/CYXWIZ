// metrics_collector.cpp - System metrics collection implementation
#include "core/metrics_collector.h"
#include <spdlog/spdlog.h>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#else
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <fstream>
#include <sstream>
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::servernode::core {

MetricsCollector::MetricsCollector() {
    spdlog::debug("MetricsCollector created");

#ifdef _WIN32
    // Initialize PDH for CPU monitoring
    PDH_STATUS status = PdhOpenQuery(NULL, 0, reinterpret_cast<PDH_HQUERY*>(&cpu_query_));
    if (status == ERROR_SUCCESS) {
        PdhAddEnglishCounter(
            reinterpret_cast<PDH_HQUERY>(cpu_query_),
            "\\Processor(_Total)\\% Processor Time",
            0,
            reinterpret_cast<PDH_HCOUNTER*>(&cpu_counter_)
        );
        PdhCollectQueryData(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    }
#endif

    last_sample_time_ = std::chrono::steady_clock::now();
}

MetricsCollector::~MetricsCollector() {
    StopCollection();

#ifdef _WIN32
    if (cpu_query_) {
        PdhCloseQuery(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    }
#endif
}

void MetricsCollector::StartCollection(int interval_ms) {
    if (running_.load()) {
        return;
    }

    interval_ms_.store(interval_ms);
    running_.store(true);
    collection_thread_ = std::thread(&MetricsCollector::CollectionLoop, this);
    spdlog::info("MetricsCollector started with {}ms interval", interval_ms);
}

void MetricsCollector::StopCollection() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
    spdlog::info("MetricsCollector stopped");
}

void MetricsCollector::SetInterval(int interval_ms) {
    interval_ms_.store(interval_ms);
}

SystemMetrics MetricsCollector::GetCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_metrics_;
}

std::vector<float> MetricsCollector::GetHistory(MetricType type, int samples) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const std::deque<float>* source = nullptr;
    switch (type) {
        case MetricType::CPU: source = &cpu_history_; break;
        case MetricType::GPU: source = &gpu_history_; break;
        case MetricType::RAM: source = &ram_history_; break;
        case MetricType::VRAM: source = &vram_history_; break;
        case MetricType::NetworkIn: source = &net_in_history_; break;
        case MetricType::NetworkOut: source = &net_out_history_; break;
        default: return {};
    }

    if (!source) return {};

    int count = std::min(samples, static_cast<int>(source->size()));
    std::vector<float> result(count);
    auto start = source->end() - count;
    std::copy(start, source->end(), result.begin());
    return result;
}

void MetricsCollector::CollectionLoop() {
    while (running_.load()) {
        auto start = std::chrono::steady_clock::now();

        // Collect all metrics
        SystemMetrics metrics;
        metrics.cpu_usage = CollectCPUUsage();
        metrics.gpu_usage = CollectGPUUsage();
        metrics.ram_used_bytes = CollectRAMUsed();
        metrics.ram_total_bytes = CollectRAMTotal();
        metrics.vram_used_bytes = CollectVRAMUsed();
        metrics.vram_total_bytes = CollectVRAMTotal();
        metrics.network_in_mbps = CollectNetworkIn();
        metrics.network_out_mbps = CollectNetworkOut();
        metrics.temperature_celsius = CollectTemperature();
        metrics.power_watts = CollectPowerUsage();

        // Calculate percentages
        if (metrics.ram_total_bytes > 0) {
            metrics.ram_usage = static_cast<float>(metrics.ram_used_bytes) / metrics.ram_total_bytes;
        }
        if (metrics.vram_total_bytes > 0) {
            metrics.vram_usage = static_cast<float>(metrics.vram_used_bytes) / metrics.vram_total_bytes;
        }

        // Update state and history
        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_metrics_ = metrics;

            // Update history buffers
            auto addToHistory = [](std::deque<float>& history, float value) {
                history.push_back(value);
                if (history.size() > MAX_HISTORY_SIZE) {
                    history.pop_front();
                }
            };

            addToHistory(cpu_history_, metrics.cpu_usage);
            addToHistory(gpu_history_, metrics.gpu_usage);
            addToHistory(ram_history_, metrics.ram_usage);
            addToHistory(vram_history_, metrics.vram_usage);
            addToHistory(net_in_history_, metrics.network_in_mbps);
            addToHistory(net_out_history_, metrics.network_out_mbps);
        }

        // Sleep for remaining interval
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_time = std::chrono::milliseconds(interval_ms_.load()) - elapsed;
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

// ========== Platform-specific implementations ==========

#ifdef _WIN32

float MetricsCollector::CollectCPUUsage() {
    if (!cpu_query_ || !cpu_counter_) return 0.0f;

    PDH_FMT_COUNTERVALUE value;
    PdhCollectQueryData(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    PdhGetFormattedCounterValue(
        reinterpret_cast<PDH_HCOUNTER>(cpu_counter_),
        PDH_FMT_DOUBLE,
        NULL,
        &value
    );
    return static_cast<float>(value.doubleValue) / 100.0f;
}

size_t MetricsCollector::CollectRAMUsed() {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys - memInfo.ullAvailPhys;
}

size_t MetricsCollector::CollectRAMTotal() {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys;
}

float MetricsCollector::CollectNetworkIn() {
    // Simplified: would need to track actual network bytes
    // For now, return placeholder
    return 0.0f;
}

float MetricsCollector::CollectNetworkOut() {
    return 0.0f;
}

#else  // Linux/macOS

float MetricsCollector::CollectCPUUsage() {
    std::ifstream stat("/proc/stat");
    if (!stat.is_open()) return 0.0f;

    std::string cpu;
    long user, nice, system, idle, iowait, irq, softirq;
    stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;

    long total = user + nice + system + idle + iowait + irq + softirq;
    long idle_time = idle + iowait;

    float usage = 0.0f;
    if (last_cpu_total_ > 0) {
        long total_diff = total - last_cpu_total_;
        long idle_diff = idle_time - last_cpu_idle_;
        if (total_diff > 0) {
            usage = 1.0f - (static_cast<float>(idle_diff) / total_diff);
        }
    }

    last_cpu_total_ = total;
    last_cpu_idle_ = idle_time;

    return usage;
}

size_t MetricsCollector::CollectRAMUsed() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (info.totalram - info.freeram) * info.mem_unit;
    }
    return 0;
}

size_t MetricsCollector::CollectRAMTotal() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
}

float MetricsCollector::CollectNetworkIn() {
    std::ifstream net("/proc/net/dev");
    if (!net.is_open()) return 0.0f;

    std::string line;
    uint64_t total_bytes = 0;

    while (std::getline(net, line)) {
        if (line.find(':') == std::string::npos) continue;
        if (line.find("lo:") != std::string::npos) continue;  // Skip loopback

        std::istringstream iss(line.substr(line.find(':') + 1));
        uint64_t bytes_in;
        iss >> bytes_in;
        total_bytes += bytes_in;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_in_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_in_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);  // Convert to Mbps
    }

    last_net_bytes_in_ = total_bytes;
    return mbps;
}

float MetricsCollector::CollectNetworkOut() {
    std::ifstream net("/proc/net/dev");
    if (!net.is_open()) return 0.0f;

    std::string line;
    uint64_t total_bytes = 0;

    while (std::getline(net, line)) {
        if (line.find(':') == std::string::npos) continue;
        if (line.find("lo:") != std::string::npos) continue;

        std::istringstream iss(line.substr(line.find(':') + 1));
        uint64_t bytes_in, packets_in, errin, dropin, fifoin, framein, compressedin, multicastin;
        uint64_t bytes_out;
        iss >> bytes_in >> packets_in >> errin >> dropin >> fifoin >> framein >> compressedin >> multicastin >> bytes_out;
        total_bytes += bytes_out;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_out_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_out_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);
    }

    last_net_bytes_out_ = total_bytes;
    last_sample_time_ = now;
    return mbps;
}

#endif  // Platform-specific

// ========== GPU metrics (cross-platform via ArrayFire) ==========

float MetricsCollector::CollectGPUUsage() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // ArrayFire doesn't provide direct GPU usage, estimate from memory
        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        // Rough estimate based on locked memory
        if (lock_bytes > 0) {
            return 0.5f;  // Active operations
        }
        return 0.1f;  // Idle
    } catch (...) {
        return 0.0f;
    }
#else
    return 0.0f;
#endif
}

size_t MetricsCollector::CollectVRAMUsed() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
        return alloc_bytes;
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}

size_t MetricsCollector::CollectVRAMTotal() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Use deviceMemInfo to get memory statistics
        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
        // Return a reasonable estimate for total VRAM (allocated + available)
        // Note: ArrayFire doesn't expose total VRAM directly
        return alloc_bytes * 2;  // Rough estimate
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}

float MetricsCollector::CollectTemperature() {
    // Would require NVML on NVIDIA or platform-specific APIs
    return 0.0f;
}

float MetricsCollector::CollectPowerUsage() {
    // Would require NVML on NVIDIA or platform-specific APIs
    return 0.0f;
}

} // namespace cyxwiz::servernode::core
