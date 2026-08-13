#pragma once

#include "crash_run_recorder.h"
#include "debug_trace_record.h"
#include "runtime_log_event.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class DebugErrorCodeTimeline {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.error_code_timeline.v1";
    static constexpr size_t kMaxEntries = 128;

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const std::vector<DebugTraceRecord>& traces,
        const std::vector<RuntimeLogEvent>& runtime_events = {},
        const std::optional<CrashRunSummary>& last_run = std::nullopt) const;
};

} // namespace cyxwiz
