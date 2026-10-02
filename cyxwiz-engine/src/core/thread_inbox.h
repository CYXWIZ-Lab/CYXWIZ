#pragma once

#include <mutex>
#include <vector>

namespace cyxwiz {

// Items a worker thread hands to the UI thread: the worker publishes, the UI
// takes every frame, each item is taken once and in order (TOFIX134 P0
// items 6 and 7: script figures, RL metrics).
template <typename T>
class ThreadInbox {
public:
    void Publish(std::vector<T> items) {
        if (items.empty()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : items) items_.push_back(std::move(item));
    }

    std::vector<T> Take() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> taken;
        taken.swap(items_);
        return taken;
    }

private:
    std::mutex mutex_;
    std::vector<T> items_;
};

}  // namespace cyxwiz
