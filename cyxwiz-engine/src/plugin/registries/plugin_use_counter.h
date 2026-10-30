#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace cyxwiz::plugin {

// Counts the calls into a plugin that are still running on other threads (a
// dataset stream being read, an analytic computing), so the registry can wait
// for them before the plugin's library is freed.
class PluginUseCounter {
public:
    // Held while the call runs; releasing it ends the use.
    std::shared_ptr<void> Acquire(const std::string& plugin_id) {
        {
            std::lock_guard lock(mutex_);
            ++uses_[plugin_id];
        }
        return std::shared_ptr<void>(nullptr, [this, plugin_id](void*) {
            std::lock_guard lock(mutex_);
            if (--uses_[plugin_id] == 0) {
                uses_.erase(plugin_id);
                released_.notify_all();
            }
        });
    }

    void WaitUntilUnused(const std::string& plugin_id) {
        std::unique_lock lock(mutex_);
        released_.wait(lock, [&] { return uses_.find(plugin_id) == uses_.end(); });
    }

    bool InUse(const std::string& plugin_id) const {
        std::lock_guard lock(mutex_);
        return uses_.find(plugin_id) != uses_.end();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable released_;
    std::map<std::string, int> uses_;
};

} // namespace cyxwiz::plugin
