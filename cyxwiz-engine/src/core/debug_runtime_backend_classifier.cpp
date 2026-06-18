#include "debug_runtime_backend_classifier.h"

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
    classification.proven = IsProvenPlacementStatus(placement.status);
    classification.fallback_possible = HasFallbackPath(placement);
    classification.needs_attention = placement.NeedsUserAttention();
    return classification;
}

void DebugRuntimeBackendClassifier::AttachToTrace(
    DebugTraceRecord& trace,
    const BackendPlacementEntry& placement) const {
    const auto classification = Classify(placement);

    trace.payload["backend_requested"] = classification.requested_backend;
    trace.payload["backend_expected"] = classification.expected_backend;
    trace.payload["backend_fallback"] = classification.fallback_backend;
    trace.payload["backend_status"] = classification.status;
    trace.payload["backend_reason_code"] = classification.reason_code;
    trace.payload["backend_explanation"] = classification.explanation;
    trace.payload["backend_suggested_action"] = classification.suggested_action;
    trace.payload["backend_proven"] = classification.proven;
    trace.payload["backend_fallback_possible"] =
        classification.fallback_possible;
    trace.payload["backend_needs_attention"] =
        classification.needs_attention;

    if (!trace.payload.contains("backend")) {
        trace.payload["backend"] = classification.expected_backend;
    }

    if (classification.needs_attention) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Backend placement requires attention: " +
                classification.reason_code);
    }
}

} // namespace cyxwiz
