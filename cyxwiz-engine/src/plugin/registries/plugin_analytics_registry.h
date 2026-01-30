#pragma once

#include "../interfaces/i_analytics_provider.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

namespace cyxwiz::plugin {

class PluginAnalyticsRegistry {
public:
    static PluginAnalyticsRegistry& Instance();

    PluginAnalyticsRegistry(const PluginAnalyticsRegistry&) = delete;
    PluginAnalyticsRegistry& operator=(const PluginAnalyticsRegistry&) = delete;

    void Register(const std::string& plugin_id, IAnalyticsProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    // Query
    std::vector<PluginAnalyticsInfo> GetAllAnalytics() const;
    std::vector<PluginAnalyticsInfo> GetAnalyticsByCategory(const std::string& category) const;
    bool HasAnalytics(const std::string& analytics_id) const;

    // Computation
    AnalyticsResult Compute(
        const std::string& analytics_id,
        std::shared_ptr<PluginDataset> dataset
    ) const;

    size_t GetCount() const;

private:
    PluginAnalyticsRegistry() = default;

    struct AnalyticsEntry {
        std::string plugin_id;
        PluginAnalyticsInfo info;
        IAnalyticsProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, AnalyticsEntry> analytics_;  // analytics_id -> entry
};

} // namespace cyxwiz::plugin
