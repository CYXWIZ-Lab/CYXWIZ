#include "debug_runtime_backend_classifier.h"

#include "debug_run_store.h"
#include "cyxwiz/backend_placement_observation.h"

#include <algorithm>

namespace cyxwiz {

namespace {

bool IsProvenPlacementStatus(const std::string& status) {
    return status == BackendPlacementStatus::Gpu ||
           status == BackendPlacementStatus::Cpu ||
           status == BackendPlacementStatus::Mixed ||
           status == BackendPlacementStatus::Unsupported;
}

bool HasFallbackPath(const BackendPlacementEntry& placement) {
    return !placement.fallback_backend.empty() &&
           placement.status != BackendPlacementStatus::Unsupported;
}

std::string EvidenceScope(const BackendPlacementEntry& placement) {
    if (placement.observation_source ==
        BackendPlacementObservationSource::RuntimeFallback) {
        return "prior_runtime_fallback";
    }
    if (placement.observation_source ==
        BackendPlacementObservationSource::PreflightProbe) {
        return "preflight_probe";
    }
    if (placement.HasObservationMetadata()) {
        return "external_observation";
    }
    return "compiler_capability";
}

std::string PayloadString(const nlohmann::json& payload,
                          const char* key,
                          const std::string& fallback = {}) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_string()
        ? it->get<std::string>()
        : fallback;
}

bool PayloadBool(const nlohmann::json& payload,
                 const char* key,
                 bool fallback = false) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_boolean()
        ? it->get<bool>()
        : fallback;
}

bool IsPlacementTrace(const DebugTraceRecord& trace,
                      const std::string& run_id) {
    return trace.run_id == run_id && trace.phase == "BackendPlacement";
}

} // namespace

DebugRuntimeBackendClassification DebugRuntimeBackendClassifier::Classify(
    const BackendPlacementEntry& placement) const {
    DebugRuntimeBackendClassification classification;
    classification.requested_backend = placement.requested_backend;
    classification.expected_backend = placement.expected_backend;
    classification.fallback_backend = placement.fallback_backend;
    classification.status = placement.status;
    classification.reason_code = placement.reason_code;
    classification.explanation = placement.explanation;
    classification.suggested_action = placement.suggested_action;
    classification.observation_source = placement.observation_source;
    classification.observation_device = placement.observation_device;
    classification.observation_dtype = placement.observation_dtype;
    classification.observation_shape_signature =
        placement.observation_shape_signature;
    classification.observation_detail = placement.observation_detail;
    classification.observation_timestamp = placement.observation_timestamp;
    classification.observation_probe_outcome =
        placement.observation_probe_outcome;
    classification.observation_probe_scope = placement.observation_probe_scope;
    classification.evidence_scope = EvidenceScope(placement);
    classification.proven = IsProvenPlacementStatus(placement.status);
    classification.fallback_possible = HasFallbackPath(placement);
    classification.prior_runtime_fallback_observed =
        placement.observation_source ==
        BackendPlacementObservationSource::RuntimeFallback;
    classification.needs_attention = placement.NeedsUserAttention();
    return classification;
}

void DebugRuntimeBackendClassifier::AttachToTrace(
    DebugTraceRecord& trace,
    const BackendPlacementEntry& placement) const {
    const auto classification = Classify(placement);
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "backend_placement",
        "DebugRuntimeBackendClassifier",
        "cyxwiz-engine/src/core/debug_runtime_backend_classifier.cpp",
        "cyxwiz::DebugRuntimeBackendClassifier::AttachToTrace");

    trace.payload["backend_requested"] = classification.requested_backend;
    trace.payload["backend_intended"] = classification.expected_backend;
    trace.payload["backend_expected"] = classification.expected_backend;
    trace.payload["backend_actual"] = classification.actual_backend;
    trace.payload["backend_actual_observed"] =
        classification.actual_backend_observed;
    trace.payload["backend_evidence_scope"] =
        classification.evidence_scope;
    trace.payload["backend_fallback"] = classification.fallback_backend;
    trace.payload["backend_status"] = classification.status;
    trace.payload["backend_reason_code"] = classification.reason_code;
    trace.payload["backend_explanation"] = classification.explanation;
    trace.payload["backend_suggested_action"] = classification.suggested_action;
    trace.payload["backend_proven"] = classification.proven;
    trace.payload["backend_proven_scope"] = "placement_classification";
    trace.payload["backend_placement_classified"] = classification.proven;
    trace.payload["backend_fallback_possible"] =
        classification.fallback_possible;
    trace.payload["backend_prior_runtime_fallback_observed"] =
        classification.prior_runtime_fallback_observed;
    trace.payload["backend_fallback_observed_this_run"] = false;
    trace.payload["backend_cost_estimate_available"] =
        classification.cost_estimate_available;
    trace.payload["backend_cost_estimate_reason"] =
        "No same-run node timing or cost model is attached to compiler "
        "placement evidence.";
    trace.payload["backend_needs_attention"] =
        classification.needs_attention;
    if (classification.status == BackendPlacementStatus::Unsupported) {
        trace.payload["backend_unsupported_reason"] =
            classification.reason_code;
    }
    if (placement.HasObservationMetadata()) {
        trace.payload["backend_observation_source"] =
            classification.observation_source;
        trace.payload["backend_observation_device"] =
            classification.observation_device;
        trace.payload["backend_observation_dtype"] =
            classification.observation_dtype;
        trace.payload["backend_observation_shape_signature"] =
            classification.observation_shape_signature;
        trace.payload["backend_observation_detail"] =
            classification.observation_detail;
        trace.payload["backend_observation_timestamp"] =
            classification.observation_timestamp;
        trace.payload["backend_observation_probe_outcome"] =
            classification.observation_probe_outcome;
        trace.payload["backend_observation_probe_scope"] =
            classification.observation_probe_scope;
    }

    if (!trace.payload.contains("backend")) {
        trace.payload["backend"] = classification.expected_backend;
    }

    if (classification.needs_attention) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Backend placement requires attention: " +
                classification.reason_code);
    }
    trace.payload["success"] = trace.status == "ok";
}

DebugTraceRecord DebugRuntimeBackendClassifier::BuildPlacementTrace(
    const std::string& run_id,
    uint64_t graph_hash,
    const BackendPlacementEntry& placement) const {
    const bool needs_attention = placement.NeedsUserAttention();
    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        placement.node_id,
        placement.node_name,
        placement.node_type,
        "BackendPlacement",
        needs_attention ? DebugTraceRole::Warning
                        : DebugTraceRole::CompileArtifact,
        {},
        {},
        placement.observation_dtype.empty()
            ? "placement"
            : placement.observation_dtype,
        placement.expected_backend,
        needs_attention ? "warning" : "ok");
    trace.payload["graph_hash"] = graph_hash;
    trace.payload["trace_producer"] = "DebugRuntimeBackendClassifier";
    AttachToTrace(trace, placement);
    return trace;
}

DebugTraceRecord DebugRuntimeBackendClassifier::BuildAuditTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces,
    const DebugRunExecutionSummary& execution) const {
    nlohmann::json rows = nlohmann::json::array();
    size_t placement_count = 0;
    size_t attention_count = 0;
    size_t actual_observed_count = 0;
    size_t same_run_fallback_count = 0;
    size_t prior_fallback_count = 0;
    size_t unsupported_count = 0;
    size_t unknown_count = 0;
    size_t observed_duration_count = 0;

    for (const auto& placement : traces) {
        if (!IsPlacementTrace(placement, run_id)) {
            continue;
        }
        ++placement_count;

        std::string actual_backend = PayloadString(
            placement.payload, "backend_actual", "unobserved");
        bool actual_observed = PayloadBool(
            placement.payload, "backend_actual_observed", false);
        bool fallback_this_run = PayloadBool(
            placement.payload, "backend_fallback_observed_this_run", false);
        float observed_duration_ms = 0.0f;
        size_t duration_trace_count = 0;
        std::string actual_evidence_phase;

        for (const auto& candidate : traces) {
            if (candidate.run_id != run_id ||
                candidate.node_id != placement.node_id ||
                candidate.phase == "BackendPlacement" ||
                candidate.phase == "BackendDecisionAudit") {
                continue;
            }
            if (candidate.duration_ms > 0.0f) {
                observed_duration_ms += candidate.duration_ms;
                ++duration_trace_count;
            }
            fallback_this_run = fallback_this_run || PayloadBool(
                candidate.payload,
                "backend_fallback_observed_this_run",
                false);
            if (PayloadBool(candidate.payload,
                            "backend_actual_observed", false)) {
                const std::string candidate_actual = PayloadString(
                    candidate.payload, "backend_actual");
                if (!candidate_actual.empty() &&
                    candidate_actual != "unobserved") {
                    actual_backend = candidate_actual;
                    actual_observed = true;
                    actual_evidence_phase = candidate.phase;
                }
            }
        }

        const bool prior_fallback = PayloadBool(
            placement.payload,
            "backend_prior_runtime_fallback_observed",
            false);
        const bool needs_attention = PayloadBool(
            placement.payload, "backend_needs_attention", false);
        const std::string status = PayloadString(
            placement.payload, "backend_status", "Unknown");
        if (needs_attention) {
            ++attention_count;
        }
        if (actual_observed) {
            ++actual_observed_count;
        }
        if (fallback_this_run) {
            ++same_run_fallback_count;
        }
        if (prior_fallback) {
            ++prior_fallback_count;
        }
        if (status == BackendPlacementStatus::Unsupported) {
            ++unsupported_count;
        }
        if (status == BackendPlacementStatus::Unknown) {
            ++unknown_count;
        }
        if (duration_trace_count > 0) {
            ++observed_duration_count;
        }

        if (rows.size() >= kMaxAuditRows) {
            continue;
        }
        nlohmann::json row = {
            {"node_id", placement.node_id},
            {"node_name", placement.node_name},
            {"node_type", placement.node_type},
            {"requested_backend", PayloadString(
                placement.payload, "backend_requested", "auto")},
            {"intended_backend", PayloadString(
                placement.payload, "backend_intended", "unobserved")},
            {"placement_status", status},
            {"actual_backend", actual_observed
                ? actual_backend
                : "unobserved"},
            {"actual_backend_observed", actual_observed},
            {"actual_evidence_scope", actual_observed
                ? "same_debug_run_node_trace"
                : "unobserved"},
            {"actual_evidence_phase", actual_evidence_phase},
            {"fallback_target", PayloadString(
                placement.payload, "backend_fallback")},
            {"fallback_possible", PayloadBool(
                placement.payload, "backend_fallback_possible", false)},
            {"fallback_observed_this_run", fallback_this_run},
            {"prior_runtime_fallback_observed", prior_fallback},
            {"reason_code", PayloadString(
                placement.payload, "backend_reason_code")},
            {"explanation", PayloadString(
                placement.payload, "backend_explanation")},
            {"suggested_fix", PayloadString(
                placement.payload, "backend_suggested_action")},
            {"unsupported_reason", PayloadString(
                placement.payload, "backend_unsupported_reason")},
            {"needs_attention", needs_attention},
            {"cost_estimate_available", false},
            {"cost_estimate", "unavailable"},
            {"cost_estimate_reason",
             "No calibrated per-node backend cost model is attached."},
            {"observed_duration_available", duration_trace_count > 0},
            {"observed_duration_trace_count", duration_trace_count},
        };
        if (duration_trace_count > 0) {
            row["observed_duration_ms"] = observed_duration_ms;
            row["observed_duration_scope"] =
                "same_debug_run_trace_timing_not_cost_estimate";
        }
        rows.push_back(std::move(row));
    }

    const bool has_attention = attention_count > 0 ||
        unsupported_count > 0 || unknown_count > 0 ||
        same_run_fallback_count > 0;
    DebugTraceRecord result = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "Backend Decision Audit",
        "TrainingDiagnostics",
        "BackendDecisionAudit",
        has_attention ? DebugTraceRole::Warning
                      : DebugTraceRole::CompileArtifact,
        {}, {}, "backend_decision_metadata", "canonical_debug_evidence",
        placement_count == 0
            ? "unobserved"
            : (has_attention ? "needs_attention" : "captured"));
    auto& payload = result.payload;
    payload["backend_decision_audit_schema"] = kAuditSchema;
    payload["trace_producer"] = "DebugRuntimeBackendClassifier";
    payload["observation_scope"] =
        "compiler_placement_plus_same_debug_run_node_evidence";
    payload["placement_count"] = placement_count;
    payload["retained_row_count"] = rows.size();
    payload["row_limit"] = kMaxAuditRows;
    payload["rows_truncated"] = placement_count > rows.size();
    payload["attention_count"] = attention_count;
    payload["actual_backend_observed_count"] = actual_observed_count;
    payload["same_run_fallback_count"] = same_run_fallback_count;
    payload["prior_runtime_fallback_count"] = prior_fallback_count;
    payload["unsupported_count"] = unsupported_count;
    payload["unknown_count"] = unknown_count;
    payload["observed_duration_count"] = observed_duration_count;
    payload["cost_estimate_available"] = false;
    payload["cost_estimate_reason"] =
        "No calibrated per-node backend cost model is attached.";
    payload["linked_execution_context_available"] = execution.available;
    payload["linked_execution_scope"] = !execution.available
        ? "unobserved"
        : (execution.training_run_id == run_id
            ? "same_run"
            : "linked_training_run");
    payload["linked_training_run_id"] = execution.training_run_id;
    payload["linked_requested_backend"] = execution.requested_backend;
    payload["linked_effective_backend"] = execution.effective_backend;
    payload["linked_effective_device_id"] = execution.effective_device_id;
    payload["linked_effective_device_name"] =
        execution.effective_device_name;
    payload["linked_residency_verdict"] = execution.residency_verdict;
    payload["linked_native_cpu_fallback_count"] =
        execution.native_cpu_fallback_count;
    payload["linked_context_is_node_actual_evidence"] = false;
    payload["tensor_reads_added"] = false;
    payload["raw_tensor_values_included"] = false;
    payload["rows"] = std::move(rows);
    payload["scope_note"] =
        "Placement is intended execution. Actual backend is populated only "
        "from a same-debug-run node trace that explicitly marks it observed. "
        "A linked training context is shown separately and is not promoted "
        "to per-node actual evidence.";
    DebugNodeTraceContract::AttachDiagnosticContext(
        result,
        "backend_decision_audit",
        "DebugRuntimeBackendClassifier",
        "cyxwiz-engine/src/core/debug_runtime_backend_classifier.cpp",
        "cyxwiz::DebugRuntimeBackendClassifier::BuildAuditTrace");
    return result;
}

} // namespace cyxwiz
