#include "debug_error_code_timeline.h"

#include "error_codes.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <string_view>

namespace cyxwiz {

namespace {

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool Contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

std::string ClassifyPhase(const std::string& phase,
                          const std::string& diagnostic_phase = {}) {
    const std::string value = Lower(
        diagnostic_phase.empty() ? phase : diagnostic_phase);
    if (Contains(value, "compile") || Contains(value, "placement")) {
        return "compile";
    }
    if (Contains(value, "preflight")) {
        return "preflight";
    }
    if (Contains(value, "material") || Contains(value, "data") ||
        Contains(value, "operator") || Contains(value, "preprocess") ||
        Contains(value, "batch")) {
        return "data_materialization";
    }
    if (Contains(value, "buildmodel") || Contains(value, "model_build") ||
        Contains(value, "modelconstruction") ||
        Contains(value, "model_construction")) {
        return "model_build";
    }
    if (Contains(value, "forward")) {
        return "forward";
    }
    if (Contains(value, "loss")) {
        return "loss";
    }
    if (Contains(value, "backward") || Contains(value, "gradient")) {
        return "backward";
    }
    if (Contains(value, "optimizer") ||
        Contains(value, "updateparameters")) {
        return "optimizer";
    }
    if (Contains(value, "export") || Contains(value, "import") ||
        Contains(value, "serial")) {
        return "export_import";
    }
    return "crash_runtime";
}

const char* SeverityName(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error: return "error";
        case IssueLevel::Warning: return "warning";
        case IssueLevel::Info: return "info";
    }
    return "info";
}

const char* SeverityName(RuntimeLogLevel level) {
    if (level >= RuntimeLogLevel::Error) {
        return "error";
    }
    if (level == RuntimeLogLevel::Warning) {
        return "warning";
    }
    return "info";
}

std::string PayloadString(const nlohmann::json& payload, const char* key) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_string()
        ? it->get<std::string>()
        : std::string{};
}

nlohmann::json MakeEntry(const std::string& code,
                         const std::string& severity,
                         const std::string& phase,
                         const std::string& message,
                         const std::string& source,
                         const std::string& source_run_id,
                         int node_id,
                         const std::string& node_name,
                         const std::string& component) {
    nlohmann::json entry = {
        {"code", code},
        {"severity", severity},
        {"phase", phase},
        {"message", message},
        {"source", source},
        {"source_run_id", source_run_id},
        {"node_id", node_id},
        {"node_name", node_name},
        {"component", component},
    };
    if (const auto* descriptor = errors::FindDiagnosticCode(code)) {
        entry["symbolic_name"] = descriptor->symbolic_name;
    } else {
        entry["symbolic_name"] = "unknown";
    }
    entry["subsystem"] = code.size() > 3
        ? std::string(errors::DiagnosticFamilyName(code[3]))
        : "unknown";
    return entry;
}

class TimelineBuilder {
public:
    void Add(nlohmann::json entry) {
        const std::string key = entry.value("source", "") + "|" +
            entry.value("source_run_id", "") + "|" +
            entry.value("phase", "") + "|" +
            entry.value("code", "") + "|" +
            std::to_string(entry.value("node_id", -1)) + "|" +
            entry.value("message", "");
        if (std::find(keys_.begin(), keys_.end(), key) != keys_.end()) {
            return;
        }
        keys_.push_back(key);
        ++observed_count_;
        if (entries_.size() >= DebugErrorCodeTimeline::kMaxEntries) {
            truncated_ = true;
            return;
        }
        entry["ordinal"] = entries_.size() + 1;
        entries_.push_back(std::move(entry));
    }

    nlohmann::json&& Entries() { return std::move(entries_); }
    size_t ObservedCount() const { return observed_count_; }
    bool Truncated() const { return truncated_; }

private:
    nlohmann::json entries_ = nlohmann::json::array();
    std::vector<std::string> keys_;
    size_t observed_count_ = 0;
    bool truncated_ = false;
};

void AddTraceIssues(TimelineBuilder& builder,
                    const DebugTraceRecord& trace) {
    const std::string diagnostic_phase =
        PayloadString(trace.payload, "diagnostic_phase");
    const std::string component = PayloadString(trace.payload, "component");
    for (const auto& issue : trace.issues) {
        if (issue.error_code.empty()) {
            continue;
        }
        builder.Add(MakeEntry(
            issue.error_code,
            SeverityName(issue.level),
            ClassifyPhase(trace.phase, diagnostic_phase),
            issue.message,
            "canonical_trace_issue",
            trace.run_id,
            issue.node_id >= 0 ? issue.node_id : trace.node_id,
            issue.node_name.empty() ? trace.node_name : issue.node_name,
            component.empty() ? trace.node_type : component));
    }
}

void AddPayloadCode(TimelineBuilder& builder,
                    const DebugTraceRecord& trace,
                    const char* code_key,
                    const char* severity) {
    const std::string code = PayloadString(trace.payload, code_key);
    if (code.empty()) {
        return;
    }
    const bool represented_by_issue = std::any_of(
        trace.issues.begin(), trace.issues.end(),
        [&code](const ValidationIssue& issue) {
            return issue.error_code == code;
        });
    if (represented_by_issue) {
        return;
    }
    std::string message = PayloadString(trace.payload, "message");
    if (message.empty()) {
        message = PayloadString(trace.payload, "summary");
    }
    builder.Add(MakeEntry(
        code,
        severity,
        ClassifyPhase(trace.phase,
                      PayloadString(trace.payload, "diagnostic_phase")),
        message,
        "canonical_trace_payload",
        trace.run_id,
        trace.node_id,
        trace.node_name,
        PayloadString(trace.payload, "component")));
}

} // namespace

DebugTraceRecord DebugErrorCodeTimeline::BuildTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces,
    const std::vector<RuntimeLogEvent>& runtime_events,
    const std::optional<CrashRunSummary>& last_run) const {
    TimelineBuilder builder;
    for (const auto& trace : traces) {
        if (trace.phase == "ErrorCodeTimeline") {
            continue;
        }
        AddTraceIssues(builder, trace);
        AddPayloadCode(builder, trace, "primary_error_code", "error");
        AddPayloadCode(builder, trace, "primary_warning_code", "warning");
        AddPayloadCode(builder, trace, "error_code",
                       trace.status == "warning" ? "warning" : "error");
    }

    for (const auto& event : runtime_events) {
        if (event.primary_error_code.empty() && event.issue_codes.empty()) {
            continue;
        }
        const std::string phase = ClassifyPhase(
            event.event_name, event.diagnostic_phase);
        const std::string source_run_id = event.run_id;
        if (!event.primary_error_code.empty()) {
            auto entry = MakeEntry(
                event.primary_error_code,
                SeverityName(event.level),
                phase,
                event.message,
                "runtime_log_event",
                source_run_id,
                event.node_id,
                "",
                event.component.empty() ? event.source : event.component);
            entry["runtime_sequence"] = event.sequence;
            entry["timestamp_unix_ms"] =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    event.timestamp_utc.time_since_epoch()).count();
            builder.Add(std::move(entry));
        }
        for (const auto& code : event.issue_codes) {
            auto entry = MakeEntry(
                code,
                SeverityName(event.level),
                phase,
                event.message,
                "runtime_log_event",
                source_run_id,
                event.node_id,
                "",
                event.component.empty() ? event.source : event.component);
            entry["runtime_sequence"] = event.sequence;
            entry["timestamp_unix_ms"] =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    event.timestamp_utc.time_since_epoch()).count();
            builder.Add(std::move(entry));
        }
    }

    if (last_run && last_run->available &&
        (last_run->suspected_crash || last_run->status == "failed")) {
        std::string message = last_run->failure_reason;
        if (message.empty()) {
            message = last_run->windows_crash_available
                ? "Windows reported a crash for the recorded training run."
                : "The recorded training run ended without a clean terminal marker.";
        }
        auto entry = MakeEntry(
            errors::Runtime::ExecutionFailed,
            "error",
            "crash_runtime",
            message,
            "crash_run_summary",
            last_run->run_id,
            -1,
            "",
            "CrashRunRecorder");
        entry["last_stage"] = last_run->last_stage;
        entry["windows_crash_available"] = last_run->windows_crash_available;
        entry["windows_exception_code"] = last_run->windows_exception_code;
        builder.Add(std::move(entry));
    }

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "Error Code Timeline",
        "DiagnosticTimeline",
        "ErrorCodeTimeline",
        DebugTraceRole::StudioEvent,
        {},
        {},
        "diagnostic",
        "structured_diagnostics",
        "captured");
    trace.payload["error_code_timeline_schema"] = kSchema;
    trace.payload["error_code_timeline"] = true;
    trace.payload["trace_producer"] = "DebugErrorCodeTimeline";
    trace.payload["success"] = true;
    trace.payload["ordering_semantics"] =
        "canonical_trace_capture_order_then_runtime_sequence";
    trace.payload["wall_clock_complete"] = false;
    trace.payload["entry_limit"] = kMaxEntries;
    trace.payload["observed_entry_count"] = builder.ObservedCount();
    trace.payload["timeline_truncated"] = builder.Truncated();
    trace.payload["entries"] = builder.Entries();
    trace.payload["entry_count"] = trace.payload["entries"].size();
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "diagnostic_timeline",
        "DebugErrorCodeTimeline",
        "cyxwiz-engine/src/core/debug_error_code_timeline.cpp",
        "cyxwiz::DebugErrorCodeTimeline::BuildTrace");
    return trace;
}

} // namespace cyxwiz
