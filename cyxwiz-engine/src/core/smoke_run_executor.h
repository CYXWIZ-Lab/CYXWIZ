#pragma once

#include "debug_session.h"
#include "graph_compiler.h"
#include "../gui/node_editor.h"
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
};

class SmokeRunExecutor {
public:
    SmokeRunResult RunTextSmoke(
        TrainingConfiguration config,
        const std::vector<gui::MLNode>& nodes,
        const std::string& run_id,
        int max_samples = 100) const;
};

} // namespace cyxwiz
