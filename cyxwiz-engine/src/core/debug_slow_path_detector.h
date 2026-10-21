#pragma once

#include "debug_trace_record.h"
#include "training_trace_collector.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct DebugSlowPathLocalTimings {
    bool forward_available = false;
    float forward_ms = 0.0f;
    bool backward_available = false;
    float backward_ms = 0.0f;
    bool optimizer_available = false;
    float optimizer_ms = 0.0f;
};

class DebugSlowPathDetector {
public:
    static constexpr const char* kSchema = "cyxwiz.debug.slow_path.v1";

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const std::vector<DebugTraceRecord>& traces,
        const TrainingTraceSummary& training_trace,
        const DebugSlowPathLocalTimings& local_timings = {}) const;
};

} // namespace cyxwiz
