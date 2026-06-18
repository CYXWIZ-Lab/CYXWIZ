#pragma once

#include "crash_run_recorder.h"
#include "debug_trace_record.h"

#include <string>

namespace cyxwiz {

struct DebugWindowsCrashReport {
    bool available = false;
    std::string process_name;
    std::string fault_module;
    std::string exception_code;
    std::string crash_time;
    std::string report_id;
    std::string report_path;
    std::string raw_excerpt;
};

struct DebugWindowsCrashCorrelation {
    bool matched = false;
    std::string match_reason;
};

class DebugWindowsCrashImporter {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.windows_crash_import.v1";
    static constexpr const char* kCrashErrorCode = "CW-R-0501";

    DebugWindowsCrashReport ParseWerText(
        const std::string& content,
        const std::string& report_path = "") const;

    DebugWindowsCrashCorrelation Correlate(
        const CrashRunSummary& run,
        const DebugWindowsCrashReport& report) const;

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const CrashRunSummary& run,
        const DebugWindowsCrashReport& report) const;
};

} // namespace cyxwiz
