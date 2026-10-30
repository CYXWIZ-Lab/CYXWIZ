// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
//
// A plugin's dataset analytics (plugin API 1.3): Data Studio's Explore >
// Plugins tab lists them, the user picks a dataset and columns, and the Engine
// calls Compute on a worker thread with those columns as an Arrow C stream.
#pragma once

#include "arrow_c_stream.h"

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct AnalyticsResult {
    std::string name;
    std::string category;

    float scalar_value = 0.0f;
    bool has_scalar = false;

    std::vector<float> vector_value;          // Shown as labelled bars
    std::vector<std::string> vector_labels;

    std::string text_value;

    bool success = true;
    std::string error_message;
};

// The column types an analytic accepts.
enum class AnalyticsColumns {
    Any,      // every column
    Numeric,  // integer and floating-point columns
    Text,     // string columns
};

struct PluginAnalyticsInfo {
    std::string analytics_id;           // Unique across plugins
    std::string name;                   // Display name
    std::string description;
    std::string category;               // e.g. "Quality", "Distribution"
    AnalyticsColumns columns = AnalyticsColumns::Any;
    int min_columns = 1;                // The user picks at least this many
    int max_columns = 0;                // 0: no limit
};

class IAnalyticsProvider {
public:
    virtual ~IAnalyticsProvider() = default;

    virtual std::vector<PluginAnalyticsInfo> GetAnalytics() = 0;

    // `data` streams the chosen columns of the dataset, every row. The Engine
    // owns it and releases it after Compute returns: read it here, do not keep it.
    virtual AnalyticsResult Compute(const std::string& analytics_id, ArrowArrayStream* data) = 0;
};

} // namespace cyxwiz::plugin
