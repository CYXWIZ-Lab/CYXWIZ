#pragma once

#include "debug_trace_record.h"
#include "training_trace_collector.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct DebugTrainingStallConfigEvidence {
    bool learning_rate_available = false;
    double learning_rate = 0.0;
};

class DebugTrainingStallDetector {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.training_stall.v1";

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const std::vector<DebugTraceRecord>& traces,
        const TrainingTraceSummary& training_trace,
        const DebugTrainingStallConfigEvidence& config = {}) const;
};

} // namespace cyxwiz
