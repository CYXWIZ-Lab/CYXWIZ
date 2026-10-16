#pragma once

#include "debug_session.h"
#include "graph_compiler.h"
#include "graph_model.h"
#include "materialization_cache.h"
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct SmokeRunResult {
    bool supported = false;
    bool success = false;
    std::string summary;
    int requested_samples = 100;
    int samples_seen = 0;
    int batches_seen = 0;
    float average_loss = 0.0f;
    float last_accuracy = 0.0f;
    std::vector<ValidationIssue> issues;
    std::vector<DebugTraceRecord> traces;
    // The run was stopped by the caller before it finished.
    bool stopped = false;
};

struct SmokeRunOptions {
    // Same prepared-data cache as Train, so a Smoke Run reuses (or seeds)
    // the cached materialization instead of rebuilding it every time.
    MaterializationCacheConfig cache_config;
    // Checked during materialization and between batches.
    std::function<bool()> should_stop;
};

class SmokeRunExecutor {
public:
    SmokeRunResult RunTextSmoke(
        TrainingConfiguration config,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::string& run_id,
        int max_samples = 100,
        const SmokeRunOptions& options = {}) const;
};

} // namespace cyxwiz
