#include "debug_support_bundle_builder.h"
#include "support_redaction.h"

namespace cyxwiz {

namespace {

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
        {"node_name", DebugSupportBundleBuilder::RedactString(issue.node_name)},
        {"error_code", issue.error_code},
        {"message", DebugSupportBundleBuilder::RedactString(issue.message)}
    };
}

nlohmann::json TraceToJson(const DebugTraceRecord& trace) {
    return {
        {"run_id", trace.run_id},
        {"node_id", trace.node_id},
        {"node_name", DebugSupportBundleBuilder::RedactString(trace.node_name)},
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
        {"message", DebugSupportBundleBuilder::RedactString(event.message)}
    };
}

nlohmann::json RecommendationToJson(const DebugRecommendation& recommendation) {
    return {
        {"node_id", recommendation.node_id},
        {"category", recommendation.category},
        {"title", DebugSupportBundleBuilder::RedactString(recommendation.title)},
        {"detail", DebugSupportBundleBuilder::RedactString(recommendation.detail)},
        {"action", DebugSupportBundleBuilder::RedactString(recommendation.action)}
    };
}

} // namespace

nlohmann::json DebugSupportBundleBuilder::Build(
    const DebugSupportBundleInput& input) const {
    nlohmann::json environment = nlohmann::json::object();
    for (const auto& [key, value] : input.environment) {
        environment[key] = SupportRedaction::IsSensitiveKey(key)
            ? "[REDACTED]"
            : RedactString(value);
    }

    nlohmann::json logs = nlohmann::json::array();
    for (const auto& log : input.recent_logs) {
        logs.push_back(RedactString(log));
    }

    nlohmann::json runtime_log_slice = {
        {"included", false},
        {"reason", "no explicit runtime-log slice supplied"}};
    if (input.runtime_log_slice) {
        const RuntimeLogRedactionOptions shareable_redaction;
        runtime_log_slice = RuntimeLogExportService::SnapshotToJson(
            *input.runtime_log_slice, shareable_redaction);
        runtime_log_slice["included"] = true;
    }

    return {
        {"schema", kSchema},
        {"request_id", input.request_id},
        {"reason", RedactString(input.reason)},
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
        {"recent_logs", logs},
        {"runtime_log_slice", std::move(runtime_log_slice)}
    };
}

nlohmann::json DebugSupportBundleBuilder::RedactJson(
    const nlohmann::json& value) {
    return SupportRedaction::RedactJson(value);
}

std::string DebugSupportBundleBuilder::RedactString(
    const std::string& value) {
    return SupportRedaction::RedactString(value);
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
            {"summary", RedactString(record.summary.summary)},
            {"file_path", "[REDACTED]"},
            {"execution", {
                {"available", record.summary.execution.available},
                {"correlated", record.summary.execution.correlated},
                {"evidence_scope", record.summary.execution.evidence_scope},
                {"training_run_id", record.summary.execution.training_run_id},
                {"status", record.summary.execution.status},
                {"requested_backend", record.summary.execution.requested_backend},
                {"requested_device_id", record.summary.execution.requested_device_id},
                {"effective_backend", record.summary.execution.effective_backend},
                {"effective_device_id", record.summary.execution.effective_device_id},
                {"effective_device_name", record.summary.execution.effective_device_name},
                {"execution_context_id", record.summary.execution.execution_context_id},
                {"placement_fingerprint", record.summary.execution.placement_fingerprint},
                {"residency_verdict", record.summary.execution.residency_verdict},
                {"native_cpu_fallback_count", record.summary.execution.native_cpu_fallback_count},
                {"transfer_event_count", record.summary.execution.transfer_event_count},
                {"transfer_known_bytes", record.summary.execution.transfer_known_bytes},
                {"synchronization_event_count", record.summary.execution.synchronization_event_count},
                {"synchronization_known_bytes", record.summary.execution.synchronization_known_bytes}
            }}
        }},
        {"replay_capsule", RedactJson(
            DebugRunReplayCapsuleToJson(record.replay_capsule))},
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
            {"estimated_memory_bytes", event.estimated_memory_bytes},
            {"available_memory_bytes", event.available_memory_bytes},
            {"safe_memory_budget_bytes", event.safe_memory_budget_bytes},
            {"memory_risk_level", event.memory_risk_level},
            {"process_memory_detected", event.process_memory_detected},
            {"process_resident_memory_bytes", event.process_resident_memory_bytes},
            {"process_private_memory_bytes", event.process_private_memory_bytes},
            {"process_resident_growth_bytes", event.process_resident_growth_bytes},
            {"process_private_memory_name", event.process_private_memory_name},
            {"process_memory_source", event.process_memory_source},
            {"pin_memory_requested", event.pin_memory_requested},
            {"transfer_mode", event.transfer_mode},
            {"transfer_reason", event.transfer_reason},
            {"transfer_backend", event.transfer_backend},
            {"transfer_batch_size", event.transfer_batch_size},
            {"compute_backend", event.compute_backend},
            {"stage_backend", event.stage_backend},
            {"stage_device_id", event.stage_device_id},
            {"stage_device_name", event.stage_device_name},
            {"requested_backend", event.requested_backend},
            {"requested_device_id", event.requested_device_id},
            {"effective_backend", event.effective_backend},
            {"effective_device_id", event.effective_device_id},
            {"effective_device_name", event.effective_device_name},
            {"execution_platform", event.execution_platform},
            {"execution_context_id", event.execution_context_id},
            {"fallback_target", event.fallback_target},
            {"fallback_operation", event.fallback_operation},
            {"fallback_reason", event.fallback_reason},
            {"fallback_policy", event.fallback_policy},
            {"native_cpu_fallback", event.native_cpu_fallback},
            {"arrayfire_host_sync_bytes", event.arrayfire_host_sync_bytes},
            {"arrayfire_host_sync_reason", event.arrayfire_host_sync_reason},
            {"placement_fingerprint", event.placement_fingerprint},
            {"placement_entry_count", event.placement_entry_count},
            {"placement_summary", event.placement_summary}
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
        {"native_cpu_fallback_count", summary.native_cpu_fallback_count},
        {"transfer_event_count", summary.transfer_event_count},
        {"transfer_known_bytes", summary.transfer_known_bytes},
        {"transfer_summary", summary.transfer_summary},
        {"synchronization_event_count", summary.synchronization_event_count},
        {"synchronization_known_bytes", summary.synchronization_known_bytes},
        {"synchronization_summary", summary.synchronization_summary},
        {"arrayfire_host_sync_count", summary.arrayfire_host_sync_count},
        {"arrayfire_host_sync_bytes", summary.arrayfire_host_sync_bytes},
        {"placement_fingerprint", summary.placement_fingerprint},
        {"placement_entry_count", summary.placement_entry_count},
        {"placement_summary", summary.placement_summary},
        {"execution_platform", summary.execution_platform},
        {"requested_backend", summary.requested_backend},
        {"requested_device_id", summary.requested_device_id},
        {"effective_backend", summary.effective_backend},
        {"effective_device_id", summary.effective_device_id},
        {"effective_device_name", summary.effective_device_name},
        {"execution_context_id", summary.execution_context_id},
        {"fallback_policy", summary.fallback_policy},
        {"declared_output_boundary_count",
         summary.declared_output_boundary_count},
        {"residency_verdict", summary.residency_verdict},
        {"residency_reason", summary.residency_reason},
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
