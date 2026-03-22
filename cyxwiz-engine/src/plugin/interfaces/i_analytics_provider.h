#pragma once

#include <string>
#include <vector>
#include <memory>

namespace cyxwiz::plugin {

struct AnalyticsResult {
    std::string name;
    std::string category;

    float scalar_value = 0.0f;
    bool has_scalar = false;

    std::vector<float> vector_value;
    std::vector<std::string> vector_labels;

    std::string text_value;

    bool success = true;
    std::string error_message;
};

struct PluginAnalyticsInfo {
    std::string analytics_id;           // Unique ID
    std::string name;                   // Display name
    std::string description;
    std::string category;               // e.g. "Quality", "Distribution"
    bool supports_parallel = true;
};

// Forward-declare opaque dataset for analytics
struct PluginDataset;

class IAnalyticsProvider {
public:
    virtual ~IAnalyticsProvider() = default;

    virtual std::vector<PluginAnalyticsInfo> GetAnalytics() = 0;

    virtual AnalyticsResult Compute(
        const std::string& analytics_id,
        std::shared_ptr<PluginDataset> dataset
    ) = 0;
};

} // namespace cyxwiz::plugin
