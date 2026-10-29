#include "plugin_analytics_registry.h"

#include "../security/safe_execute.h"

#include <arrow/c/bridge.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <tuple>

namespace cyxwiz::plugin {

namespace {

AnalyticsResult Failure(const std::string& analytics_id, std::string message) {
    AnalyticsResult result;
    result.name = analytics_id;
    result.success = false;
    result.error_message = std::move(message);
    return result;
}

}  // namespace

PluginAnalyticsRegistry& PluginAnalyticsRegistry::Instance() {
    static PluginAnalyticsRegistry instance;
    return instance;
}

void PluginAnalyticsRegistry::Register(const std::string& plugin_id, const std::string& plugin_version,
                                       IAnalyticsProvider* provider) {
    if (!provider) return;

    auto analytics = provider->GetAnalytics();
    std::lock_guard lock(mutex_);
    for (auto& info : analytics) {
        if (info.analytics_id.empty() || info.name.empty()) {
            spdlog::warn("PluginAnalyticsRegistry: {} declares an analytic without an id or a name; skipped",
                         plugin_id);
            continue;
        }
        if (const auto it = analytics_.find(info.analytics_id); it != analytics_.end()) {
            if (it->second.analytic.plugin_id != plugin_id || it->second.provider != provider)
                spdlog::warn("PluginAnalyticsRegistry: analytic {} of {} is already registered by {}; skipped",
                             info.analytics_id, plugin_id, it->second.analytic.plugin_id);
            continue;
        }
        if (info.min_columns < 1) info.min_columns = 1;
        if (info.max_columns != 0 && info.max_columns < info.min_columns) info.max_columns = info.min_columns;
        spdlog::info("PluginAnalyticsRegistry: registered '{}' ({}) from {}",
                     info.name, info.analytics_id, plugin_id);
        // The key first: the right side (which moves info) is evaluated
        // before the subscript (C++17), which would read an empty id.
        const std::string analytics_id = info.analytics_id;
        analytics_[analytics_id] = Entry{Analytic{plugin_id, plugin_version, std::move(info)}, provider};
        ++generation_;
    }
}

void PluginAnalyticsRegistry::RemoveByPlugin(const std::string& plugin_id) {
    {
        std::lock_guard lock(mutex_);
        for (auto it = analytics_.begin(); it != analytics_.end();) {
            if (it->second.analytic.plugin_id == plugin_id) {
                it = analytics_.erase(it);
                ++generation_;
            } else {
                ++it;
            }
        }
    }
    if (uses_.InUse(plugin_id))
        spdlog::info("PluginAnalyticsRegistry: waiting for an analytic of {} to finish", plugin_id);
    uses_.WaitUntilUnused(plugin_id);
}

std::vector<PluginAnalyticsRegistry::Analytic> PluginAnalyticsRegistry::Analytics() const {
    std::vector<Analytic> result;
    {
        std::lock_guard lock(mutex_);
        result.reserve(analytics_.size());
        for (const auto& [_, entry] : analytics_) result.push_back(entry.analytic);
    }
    std::sort(result.begin(), result.end(), [](const Analytic& a, const Analytic& b) {
        return std::tie(a.info.category, a.info.name) < std::tie(b.info.category, b.info.name);
    });
    return result;
}

AnalyticsResult PluginAnalyticsRegistry::Compute(const std::string& analytics_id,
                                                 std::shared_ptr<arrow::RecordBatchReader> data) const {
    IAnalyticsProvider* provider = nullptr;
    Analytic analytic;
    std::shared_ptr<void> use;
    {
        std::lock_guard lock(mutex_);
        const auto it = analytics_.find(analytics_id);
        if (it == analytics_.end())
            return Failure(analytics_id, "The analytic '" + analytics_id + "' is not installed (its plugin is not loaded)");
        provider = it->second.provider;
        analytic = it->second.analytic;
        use = uses_.Acquire(analytic.plugin_id);
    }
    if (!data) return Failure(analytics_id, "No data to analyse");

    ArrowArrayStream stream{};
    const auto exported = arrow::ExportRecordBatchReader(std::move(data), &stream);
    if (!exported.ok()) return Failure(analytics_id, "Cannot hand the data over: " + exported.message());

    AnalyticsResult result;
    const auto run = security::SafeExecute(analytic.plugin_id, "Compute", [&] {
        try {
            result = provider->Compute(analytics_id, &stream);
        } catch (const std::exception& e) {
            result = Failure(analytics_id, std::string("The plugin threw: ") + e.what());
        } catch (...) {
            result = Failure(analytics_id, "The plugin threw an unknown exception");
        }
    });
    if (!run.success) result = Failure(analytics_id, run.error_message + "; the Engine kept running");
    if (stream.release) stream.release(&stream);
    if (result.name.empty()) result.name = analytic.info.name;
    if (result.category.empty()) result.category = analytic.info.category;
    if (!result.success)
        spdlog::warn("PluginAnalyticsRegistry: {} failed: {}", analytics_id, result.error_message);
    return result;
}

}  // namespace cyxwiz::plugin
