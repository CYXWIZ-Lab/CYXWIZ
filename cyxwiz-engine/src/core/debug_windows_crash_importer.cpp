#include "debug_windows_crash_importer.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>

namespace cyxwiz {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::map<std::string, std::string> ParseKeyValues(const std::string& content) {
    std::map<std::string, std::string> values;
    std::istringstream in(content);
    std::string line;
    std::string pending_sig_name;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        values[key] = value;

        if (key.find("Sig[") == 0 &&
            key.find("].Name") != std::string::npos) {
            pending_sig_name = value;
        } else if (!pending_sig_name.empty() &&
                   key.find("Sig[") == 0 &&
                   key.find("].Value") != std::string::npos) {
            values[pending_sig_name] = value;
            pending_sig_name.clear();
        }
    }

    return values;
}

std::string GetFirst(const std::map<std::string, std::string>& values,
                     std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        const auto it = values.find(key);
        if (it != values.end() && !it->second.empty()) {
            return it->second;
        }
    }
    return {};
}

std::string Excerpt(const std::string& content) {
    constexpr size_t kMaxExcerpt = 512;
    if (content.size() <= kMaxExcerpt) {
        return content;
    }
    return content.substr(0, kMaxExcerpt);
}

bool ContainsLower(const std::string& haystack, const std::string& needle) {
    return !needle.empty() &&
           ToLower(haystack).find(ToLower(needle)) != std::string::npos;
}

} // namespace

DebugWindowsCrashReport DebugWindowsCrashImporter::ParseWerText(
    const std::string& content,
    const std::string& report_path) const {
    const auto values = ParseKeyValues(content);

    DebugWindowsCrashReport report;
    report.process_name = GetFirst(values, {
        "AppName",
        "Application Name",
        "OriginalFilename",
        "Faulting Application Name",
        "Sig[0].Value"
    });
    report.fault_module = GetFirst(values, {
        "Fault Module Name",
        "Fault Module",
        "Faulting Module Name"
    });
    report.exception_code = GetFirst(values, {
        "Exception Code",
        "ExceptionCode"
    });
    report.crash_time = GetFirst(values, {
        "EventTime",
        "Event Time"
    });
    report.report_id = GetFirst(values, {
        "ReportIdentifier",
        "Report Id",
        "ReportID"
    });
    report.report_path = report_path;
    report.raw_excerpt = Excerpt(content);
    report.available = !report.process_name.empty() ||
                       !report.fault_module.empty() ||
                       !report.exception_code.empty() ||
                       !report.report_id.empty();
    return report;
}

DebugWindowsCrashCorrelation DebugWindowsCrashImporter::Correlate(
    const CrashRunSummary& run,
    const DebugWindowsCrashReport& report) const {
    DebugWindowsCrashCorrelation correlation;
    if (!run.available || !report.available) {
        correlation.match_reason = "run or crash report unavailable";
        return correlation;
    }

    if (!run.run_id.empty() && ContainsLower(report.raw_excerpt, run.run_id)) {
        correlation.matched = true;
        correlation.match_reason = "run id found in crash report";
        return correlation;
    }

    if (ContainsLower(report.process_name, "cyxwiz")) {
        correlation.matched = true;
        correlation.match_reason = "WER process name matches CyxWiz";
        return correlation;
    }

    if (!run.windows_report_id.empty() &&
        run.windows_report_id == report.report_id) {
        correlation.matched = true;
        correlation.match_reason = "report id matches run summary";
        return correlation;
    }

    correlation.match_reason = "no stable match";
    return correlation;
}

DebugTraceRecord DebugWindowsCrashImporter::BuildTrace(
    const std::string& run_id,
    const CrashRunSummary& run,
    const DebugWindowsCrashReport& report) const {
    const auto correlation = Correlate(run, report);

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "WindowsCrashImport",
        "WER",
        "WindowsCrashImport",
        DebugTraceRole::Error,
        {},
        {},
        "wer",
        "WindowsCrashImport",
        report.available ? "captured" : "missing");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "windows_crash_import",
        "DebugWindowsCrashImporter",
        "cyxwiz-engine/src/core/debug_windows_crash_importer.cpp",
        "cyxwiz::DebugWindowsCrashImporter::BuildTrace");
    trace.payload["schema"] = kSchema;
    trace.payload["error_code"] = kCrashErrorCode;
    trace.payload["report_available"] = report.available;
    trace.payload["matched"] = correlation.matched;
    trace.payload["match_reason"] = correlation.match_reason;
    trace.payload["run_status"] = run.status;
    trace.payload["run_last_stage"] = run.last_stage;
    trace.payload["run_last_event_time"] = run.last_event_time;
    trace.payload["process_name"] = report.process_name;
    trace.payload["fault_module"] = report.fault_module;
    trace.payload["exception_code"] = report.exception_code;
    trace.payload["crash_time"] = report.crash_time;
    trace.payload["report_id"] = report.report_id;
    trace.payload["report_path"] = report.report_path;

    if (!report.available) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "No Windows crash report was available for this debug run.",
            kCrashErrorCode);
    } else if (!correlation.matched) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Windows crash report was imported but not confidently matched.",
            kCrashErrorCode);
    }
    trace.payload["success"] = report.available && correlation.matched;

    return trace;
}

} // namespace cyxwiz
