#include "debug_runtime_backend_classifier.h"

#include "cyxwiz/backend_placement_observation.h"

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

} // namespace cyxwiz
