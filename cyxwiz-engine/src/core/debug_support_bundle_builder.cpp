#include "debug_support_bundle_builder.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool IsSensitiveKey(const std::string& key) {
    const std::string lower = ToLower(key);
    return lower.find("path") != std::string::npos ||
           lower.find("file") != std::string::npos ||
           lower.find("dataset") != std::string::npos ||
           lower.find("raw") != std::string::npos ||
           lower.find("preview") != std::string::npos ||
           lower.find("token") != std::string::npos ||
           lower.find("password") != std::string::npos ||
           lower.find("secret") != std::string::npos ||
           lower.find("credential") != std::string::npos;
}

std::string IssueLevelName(IssueLevel level) {
    switch (level) {
        case IssueLevel::Info: return "Info";
        case IssueLevel::Warning: return "Warning";
        case IssueLevel::Error: return "Error";
    }
    return "Unknown";
}

nlohmann::json IssueToJson(const ValidationIssue& issue) {
    return {
        {"level", IssueLevelName(issue.level)},
        {"node_id", issue.node_id},
        {"node_name", issue.node_name},
        {"error_code", issue.error_code},
        {"message", issue.message}
    };
}

nlohmann::json TraceToJson(const DebugTraceRecord& trace) {
    return {
        {"run_id", trace.run_id},
        {"node_id", trace.node_id},
        {"node_name", trace.node_name},
        {"node_type", trace.node_type},
        {"phase", trace.phase},
        {"role", DebugTraceRoleName(trace.role)},
        {"input_shape", trace.input_shape},
        {"output_shape", trace.output_shape},
        {"dtype", trace.dtype},
        {"duration_ms", trace.duration_ms},
        {"status", trace.status},
        {"issues", nlohmann::json::array()},
        {"payload", DebugSupportBundleBuilder::RedactJson(trace.payload)}
    };
}

nlohmann::json EventToJson(const StudioEventRecord& event) {
    return {
        {"run_id", event.run_id},
        {"timestamp", event.timestamp},
        {"graph_hash", event.graph_hash},
        {"selected_node_id", event.selected_node_id},
        {"action", event.action},
        {"status", event.status},
        {"message", event.message}
    };
}

nlohmann::json RecommendationToJson(const DebugRecommendation& recommendation) {
    return {
        {"node_id", recommendation.node_id},
        {"category", recommendation.category},
        {"title", recommendation.title},
        {"detail", recommendation.detail},
        {"action", recommendation.action}
    };
}

} // namespace

nlohmann::json DebugSupportBundleBuilder::Build(
    const DebugSupportBundleInput& input) const {
    nlohmann::json environment = nlohmann::json::object();
    for (const auto& [key, value] : input.environment) {
        environment[key] = IsSensitiveKey(key) ? "[REDACTED]" : RedactString(value);
    }

    nlohmann::json logs = nlohmann::json::array();
    for (const auto& log : input.recent_logs) {
        logs.push_back(RedactString(log));
    }

    return {
        {"schema", kSchema},
        {"request_id", input.request_id},
        {"reason", input.reason},
        {"local_first", true},
        {"hq_upload_allowed", input.allow_hq_upload},
        {"hq_upload_performed", false},
        {"redaction_applied", true},
        {"debug_run", DebugRunToJson(input.debug_run)},
        {"crash_run", CrashRunToJson(input.crash_run)},
        {"training_trace", TrainingTraceToJson(input.training_trace)},
        {"placement_observations",
         PlacementObservationsToJson(input.placement_observations)},
        {"environment", environment},
        {"recent_logs", logs}
    };
}

nlohmann::json DebugSupportBundleBuilder::RedactJson(
    const nlohmann::json& value) {
    if (value.is_object()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (IsSensitiveKey(it.key())) {
                out[it.key()] = "[REDACTED]";
            } else {
                out[it.key()] = RedactJson(it.value());
            }
        }
        return out;
    }

    if (value.is_array()) {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& item : value) {
            out.push_back(RedactJson(item));
        }
        return out;
    }

    if (value.is_string()) {
        return RedactString(value.get<std::string>());
    }

    return value;
}

std::string DebugSupportBundleBuilder::RedactString(
    const std::string& value) {
    std::string out = value;
    const std::vector<std::string> markers = {
        "token=",
        "password=",
        "secret=",
        "credential="
    };

    for (const auto& marker : markers) {
        const auto lower = ToLower(out);
        const auto pos = lower.find(marker);
        if (pos != std::string::npos) {
            out = out.substr(0, pos + marker.size()) + "[REDACTED]";
        }
    }

    return out;
}

nlohmann::json DebugSupportBundleBuilder::DebugRunToJson(
    const DebugRunStoreRecord& record) {
    nlohmann::json issues = nlohmann::json::array();
    for (const auto& issue : record.issues) {
        issues.push_back(IssueToJson(issue));
    }

    nlohmann::json traces = nlohmann::json::array();
    for (const auto& trace : record.traces) {
        nlohmann::json trace_json = TraceToJson(trace);
        for (const auto& issue : trace.issues) {
            trace_json["issues"].push_back(IssueToJson(issue));
        }
        traces.push_back(std::move(trace_json));
    }

    nlohmann::json events = nlohmann::json::array();
    for (const auto& event : record.studio_events) {
        events.push_back(EventToJson(event));
    }

    nlohmann::json recommendations = nlohmann::json::array();
    for (const auto& recommendation : record.recommendations) {
        recommendations.push_back(RecommendationToJson(recommendation));
    }

    return {
        {"summary", {
            {"run_id", record.summary.run_id},
            {"timestamp", record.summary.timestamp},
            {"graph_hash", record.summary.graph_hash},
            {"success", record.summary.success},
            {"issue_count", record.summary.issue_count},
            {"trace_count", record.summary.trace_count},
            {"event_count", record.summary.event_count},
            {"recommendation_count", record.summary.recommendation_count},
            {"summary", record.summary.summary},
            {"file_path", "[REDACTED]"}
        }},
        {"issues", issues},
        {"traces", traces},
        {"studio_events", events},
        {"recommendations", recommendations}
    };
}

nlohmann::json DebugSupportBundleBuilder::CrashRunToJson(
    const CrashRunSummary& summary) {
    return {
        {"available", summary.available},
        {"suspected_crash", summary.suspected_crash},
        {"run_id", summary.run_id},
        {"status", summary.status},
        {"dataset_name", "[REDACTED]"},
        {"backend", summary.backend},
        {"last_stage", summary.last_stage},
        {"last_event_time", summary.last_event_time},
        {"epoch", summary.epoch},
        {"batch", summary.batch},
        {"total_batches", summary.total_batches},
        {"epochs", summary.epochs},
        {"batch_size", summary.batch_size},
        {"sample_count", summary.sample_count},
        {"loss", summary.loss},
        {"accuracy", summary.accuracy},
        {"file_path", "[REDACTED]"},
        {"warning", summary.warning},
        {"panel_events", summary.panel_events},
        {"windows_crash_available", summary.windows_crash_available},
        {"windows_fault_module", summary.windows_fault_module},
        {"windows_exception_code", summary.windows_exception_code},
        {"windows_crash_time", summary.windows_crash_time},
        {"windows_report_id", summary.windows_report_id},
        {"windows_report_path", "[REDACTED]"}
    };
}

nlohmann::json DebugSupportBundleBuilder::TrainingTraceToJson(
    const TrainingTraceSummary& summary) {
    nlohmann::json events = nlohmann::json::array();
    for (const auto& event : summary.recent_events) {
        events.push_back({
            {"timestamp", event.timestamp},
            {"run_id", event.run_id},
            {"stage", event.stage},
            {"thread_id", event.thread_id},
            {"node_id", event.node_id},
            {"node_name", RedactString(event.node_name)},
            {"epoch", event.epoch},
            {"batch", event.batch},
            {"total_batches", event.total_batches},
            {"loss", event.loss},
            {"accuracy", event.accuracy},
            {"duration_ms", event.duration_ms},
            {"cpu_allocated_bytes", event.cpu_allocated_bytes},
            {"cpu_peak_bytes", event.cpu_peak_bytes},
            {"af_allocated_bytes", event.af_allocated_bytes},
            {"af_locked_bytes", event.af_locked_bytes},
            {"af_alloc_buffers", event.af_alloc_buffers},
            {"af_lock_buffers", event.af_lock_buffers},
            {"status", event.status},
            {"message", RedactString(event.message)},
            {"pin_memory_requested", event.pin_memory_requested},
            {"transfer_mode", event.transfer_mode},
            {"transfer_reason", event.transfer_reason},
            {"transfer_backend", event.transfer_backend},
            {"transfer_batch_size", event.transfer_batch_size}
        });
    }

    return {
        {"available", summary.available},
        {"run_id", summary.run_id},
        {"status", summary.status},
        {"latest_stage", summary.latest_stage},
        {"latest_timestamp", summary.latest_timestamp},
        {"latest_epoch", summary.latest_epoch},
        {"latest_batch", summary.latest_batch},
        {"latest_total_batches", summary.latest_total_batches},
        {"latest_loss", summary.latest_loss},
        {"latest_accuracy", summary.latest_accuracy},
        {"recent_events", events},
        {"warnings", summary.warnings}
    };
}

nlohmann::json DebugSupportBundleBuilder::PlacementObservationsToJson(
    const std::vector<BackendPlacementObservation>& observations) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& observation : observations) {
        out.push_back(PlacementObservationToJson(observation));
    }
    return out;
}

nlohmann::json DebugSupportBundleBuilder::PlacementObservationToJson(
    const BackendPlacementObservation& observation) {
    return {
        {"op_type", observation.op_type},
        {"backend", observation.backend},
        {"device", observation.device},
        {"dtype", observation.dtype},
        {"shape_signature", observation.shape_signature},
        {"reason_code", observation.reason_code},
        {"source", observation.source},
        {"detail", RedactString(observation.detail)},
        {"timestamp", observation.timestamp},
        {"probe_outcome", observation.probe_outcome},
        {"probe_scope", observation.probe_scope}
    };
}

} // namespace cyxwiz
