#include "monitor_sampler.h"

#include <algorithm>
#include <chrono>

namespace cyxwiz {

MonitorSampler::MonitorSampler(Reader reader) : reader_(std::move(reader)) {}

MonitorSampler::~MonitorSampler() { Stop(); }

void MonitorSampler::Start(double interval_seconds, double keep_seconds) {
    SetInterval(interval_seconds);
    SetKeep(keep_seconds);
    if (running_.exchange(true)) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = false;
    }
    thread_ = std::thread([this] { Run(); });
}

void MonitorSampler::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

void MonitorSampler::SetInterval(double seconds) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        interval_ = std::clamp(seconds, kMinInterval, kMaxInterval);
        // A shorter interval takes effect now, not after the old wait.
        const auto soonest = std::chrono::steady_clock::now() +
                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(interval_));
        if (soonest < next_) next_ = soonest;
    }
    wake_.notify_all();
}

void MonitorSampler::SetKeep(double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    keep_ = std::clamp(seconds, kMinKeep, kMaxKeep);
    TrimLocked();
}

double MonitorSampler::interval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return interval_;
}

double MonitorSampler::keep() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keep_;
}

std::vector<MonitorSample> MonitorSampler::History() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {history_.begin(), history_.end()};
}

bool MonitorSampler::Latest(MonitorSample& sample) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.empty()) return false;
    sample = history_.back();
    return true;
}

size_t MonitorSampler::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_.size();
}

void MonitorSampler::Clear() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
    }
    generation_.fetch_add(1);
}

void MonitorSampler::TrimLocked() {
    if (history_.empty()) return;
    const double oldest = history_.back().seconds - keep_;
    while (history_.size() > 1 && history_.front().seconds < oldest) history_.pop_front();
}

void MonitorSampler::Run() {
    using Clock = std::chrono::steady_clock;
    system_metrics::Probe probe;
    probe.Read();  // the first reading only gives the CPU counters a start
    {
        // The first kept reading comes a full interval later: CPU figures
        // over a few milliseconds would be noise.
        std::lock_guard<std::mutex> lock(mutex_);
        next_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(interval_));
    }
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Sleeps until the next reading is due; a changed interval or a
            // stop wakes it.
            while (!stop_ && Clock::now() < next_) wake_.wait_until(lock, next_);
            if (stop_) return;
        }
        MonitorSample sample;
        sample.seconds = std::chrono::duration<double>(Clock::now() - epoch_).count();
        sample.system = probe.Read();
        if (reader_) {
            try {
                reader_(sample);
            } catch (...) {
                // A reader that fails leaves its figures "not known".
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            history_.push_back(sample);
            TrimLocked();
            next_ = Clock::now() +
                    std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(interval_));
        }
        generation_.fetch_add(1);
    }
}

}  // namespace cyxwiz
