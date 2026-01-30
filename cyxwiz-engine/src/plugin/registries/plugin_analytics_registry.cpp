#include "plugin_analytics_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

PluginAnalyticsRegistry& PluginAnalyticsRegistry::Instance() {
    static PluginAnalyticsRegistry instance;
    return instance;
}

void PluginAnalyticsRegistry::Register(const std::string& plugin_id, IAnalyticsProvider* provider) {
    if (!provider) return;

    auto analytics = provider->GetAnalytics();
    std::lock_guard lock(mutex_);

    for (auto& info : analytics) {
        if (analytics_.count(info.analytics_id)) {
            spdlog::warn("PluginAnalyticsRegistry: Duplicate analytics {}, skipping", info.analytics_id);
            continue;
        }
        spdlog::info("PluginAnalyticsRegistry: Registered analytics '{}' ({})",
                     info.name, info.analytics_id);
        analytics_[info.analytics_id] = AnalyticsEntry{plugin_id, std::move(info), provider};
    }
}

void PluginAnalyticsRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = analytics_.begin(); it != analytics_.end();) {
        if (it->second.plugin_id == plugin_id)
            it = analytics_.erase(it);
        else
            ++it;
    }
}

std::vector<PluginAnalyticsInfo> PluginAnalyticsRegistry::GetAllAnalytics() const {
    std::lock_guard lock(mutex_);
    std::vector<PluginAnalyticsInfo> result;
    result.reserve(analytics_.size());
    for (const auto& [_, entry] : analytics_)
        result.push_back(entry.info);
    return result;
}

std::vector<PluginAnalyticsInfo> PluginAnalyticsRegistry::GetAnalyticsByCategory(const std::string& category) const {
    std::lock_guard lock(mutex_);
    std::vector<PluginAnalyticsInfo> result;
    for (const auto& [_, entry] : analytics_) {
        if (entry.info.category == category)
            result.push_back(entry.info);
    }
    return result;
}

bool PluginAnalyticsRegistry::HasAnalytics(const std::string& analytics_id) const {
    std::lock_guard lock(mutex_);
    return analytics_.count(analytics_id) > 0;
}

AnalyticsResult PluginAnalyticsRegistry::Compute(
    const std::string& analytics_id,
    std::shared_ptr<PluginDataset> dataset
) const {
    std::lock_guard lock(mutex_);
    auto it = analytics_.find(analytics_id);
    if (it == analytics_.end() || !it->second.provider) {
        AnalyticsResult r;
        r.success = false;
        r.error_message = "Analytics not found: " + analytics_id;
        return r;
    }

    try {
        return it->second.provider->Compute(analytics_id, dataset);
    } catch (const std::exception& e) {
        spdlog::error("PluginAnalyticsRegistry: Compute({}) threw: {}", analytics_id, e.what());
        AnalyticsResult r;
        r.success = false;
        r.error_message = e.what();
        return r;
    }
}

size_t PluginAnalyticsRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return analytics_.size();
}

} // namespace cyxwiz::plugin
