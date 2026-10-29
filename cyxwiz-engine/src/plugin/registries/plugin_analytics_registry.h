#pragma once

#include "../interfaces/i_analytics_provider.h"
#include "plugin_use_counter.h"

#include <arrow/record_batch.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// The dataset analytics plugins add (IAnalyticsProvider, TOFIX181), run from
// Data Studio's Explore > Plugins tab.
class PluginAnalyticsRegistry {
public:
    static PluginAnalyticsRegistry& Instance();

    PluginAnalyticsRegistry(const PluginAnalyticsRegistry&) = delete;
    PluginAnalyticsRegistry& operator=(const PluginAnalyticsRegistry&) = delete;

    void Register(const std::string& plugin_id, IAnalyticsProvider* provider);
    // Waits until no analytic of the plugin is computing any more.
    void RemoveByPlugin(const std::string& plugin_id);

    struct Analytic {
        std::string plugin_id;
        PluginAnalyticsInfo info;
    };
    std::vector<Analytic> Analytics() const;  // ordered by category, then name

    // Runs the analytic on `data` (the chosen columns), on the caller's thread.
    AnalyticsResult Compute(const std::string& analytics_id,
                            std::shared_ptr<arrow::RecordBatchReader> data) const;

    // Changes whenever an analytic is added or removed.
    uint64_t Generation() const { return generation_.load(); }

private:
    PluginAnalyticsRegistry() = default;

    struct Entry {
        Analytic analytic;
        IAnalyticsProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, Entry> analytics_;  // analytics_id -> entry
    std::atomic<uint64_t> generation_{0};
    mutable PluginUseCounter uses_;
};

} // namespace cyxwiz::plugin
