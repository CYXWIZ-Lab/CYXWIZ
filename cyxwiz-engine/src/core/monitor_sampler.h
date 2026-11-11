#pragma once

// The Monitor window's history: one reading of the machine, the process and
// the Engine's memory every interval, taken on its own thread and kept for a
// set time. The interface thread only copies readings out.

#include "system_metrics.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cyxwiz {

struct MonitorSample {
    double seconds = 0.0;  // since the sampler started
    system_metrics::Sample system;

    // Filled by the reader the Engine gives the sampler (it must not call
    // ArrayFire: the sampler thread owns no device).
    bool device_held_known = false;
    std::uint64_t device_allocated_bytes = 0;
    std::uint64_t device_in_use_bytes = 0;
    bool device_free_known = false;
    std::uint64_t device_total_bytes = 0;
    std::uint64_t device_free_bytes = 0;
    bool tables_known = false;
    std::uint64_t tables_bytes = 0;  // Arrow's memory pool
    bool tensors_known = false;
    std::uint64_t tensors_bytes = 0;  // host tensors (the tensor allocator)
};

class MonitorSampler {
public:
    using Reader = std::function<void(MonitorSample&)>;

    // Limits of the two settings, in seconds.
    static constexpr double kMinInterval = 0.25;
    static constexpr double kMaxInterval = 60.0;
    static constexpr double kMinKeep = 60.0;
    static constexpr double kMaxKeep = 3600.0;

    explicit MonitorSampler(Reader reader = {});
    ~MonitorSampler();
    MonitorSampler(const MonitorSampler&) = delete;
    MonitorSampler& operator=(const MonitorSampler&) = delete;

    // Starts the thread (a second Start only changes the settings).
    void Start(double interval_seconds, double keep_seconds);
    void Stop();
    bool running() const { return running_.load(); }

    void SetInterval(double seconds);
    void SetKeep(double seconds);
    double interval() const;
    double keep() const;

    // Oldest first.
    std::vector<MonitorSample> History() const;
    bool Latest(MonitorSample& sample) const;
    size_t size() const;
    void Clear();
    // Grows by one with every reading and every Clear: a cheap "did anything
    // change" for the window.
    std::uint64_t generation() const { return generation_.load(); }

private:
    void Run();
    void TrimLocked();

    Reader reader_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<MonitorSample> history_;
    double interval_ = 1.0;
    double keep_ = 900.0;
    bool stop_ = false;
    std::chrono::steady_clock::time_point next_{};  // when the next reading is due
    // `seconds` of every reading counts from here, also across Stop and Start.
    const std::chrono::steady_clock::time_point epoch_ = std::chrono::steady_clock::now();
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> generation_{0};
    std::thread thread_;
};

}  // namespace cyxwiz
