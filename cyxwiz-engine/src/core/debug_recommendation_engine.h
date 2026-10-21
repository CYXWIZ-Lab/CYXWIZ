#pragma once

#include "crash_run_recorder.h"
#include "debug_trace_record.h"
#include "smoke_run_executor.h"
#include "training_trace_collector.h"
#include <string>
#include <vector>

namespace cyxwiz {

enum class DebugRecommendationSeverity {
    Info,
    Warning,
    Critical
};

struct DebugRecommendation {
    DebugRecommendationSeverity severity = DebugRecommendationSeverity::Info;
    int node_id = -1;
    std::string category;
    std::string title;
    std::string detail;
    std::string action;
};

class DebugRecommendationEngine {
public:
    std::vector<DebugRecommendation> Build(
        const std::vector<DebugTraceRecord>& traces,
        const std::vector<ValidationIssue>& issues,
        const SmokeRunResult& smoke_result,
        const CrashRunSummary& last_run,
        const TrainingTraceSummary& training_trace = {}) const;
};

const char* DebugRecommendationSeverityName(DebugRecommendationSeverity severity);

} // namespace cyxwiz
