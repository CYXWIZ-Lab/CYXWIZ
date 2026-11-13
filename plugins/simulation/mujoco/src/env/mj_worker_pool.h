#pragma once

// A few threads that share out the copies of a batch. Run(count, fn) calls
// fn(i) for every i in [0, count), split into contiguous ranges, the calling
// thread taking the first, and returns when all are done. Each copy is
// independent, so the results do not depend on the number of threads.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cyxwiz::plugin::mujoco {

class MjWorkerPool {
public:
    // threads = how many work at once, the caller included (1: no extra thread)
    explicit MjWorkerPool(int threads) {
        for (int i = 1; i < threads; ++i) threads_.emplace_back([this, i]() { Loop(i); });
    }

    ~MjWorkerPool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    MjWorkerPool(const MjWorkerPool&) = delete;
    MjWorkerPool& operator=(const MjWorkerPool&) = delete;

    int Threads() const { return static_cast<int>(threads_.size()) + 1; }

    void Run(int count, const std::function<void(int)>& fn) {
        if (threads_.empty() || count <= 1) {
            for (int i = 0; i < count; ++i) fn(i);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            fn_ = &fn;
            count_ = count;
            pending_ = static_cast<int>(threads_.size());
            ++generation_;
        }
        wake_.notify_all();
        RunPart(0, fn, count);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this]() { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    void RunPart(int part, const std::function<void(int)>& fn, int count) const {
        const int parts = Threads();
        const int begin = static_cast<int>(static_cast<int64_t>(count) * part / parts);
        const int end = static_cast<int>(static_cast<int64_t>(count) * (part + 1) / parts);
        for (int i = begin; i < end; ++i) fn(i);
    }

    void Loop(int part) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* fn = nullptr;
            int count = 0;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&]() { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                fn = fn_;
                count = count_;
            }
            RunPart(part, *fn, count);
            {
                std::lock_guard lock(mutex_);
                --pending_;
            }
            done_.notify_one();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(int)>* fn_ = nullptr;
    int count_ = 0;
    int pending_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

}  // namespace cyxwiz::plugin::mujoco
