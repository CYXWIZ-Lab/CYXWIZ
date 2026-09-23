#pragma once

#include "backend_fallback_reason.h"
#include "backend_placement_observation.h"

#include <optional>
#include <string>
#include <string_view>

// Typed GPU-resilience contracts (tofix67 slice 3).
//
// These are thin typed views over the existing placement-evidence spine —
// the observation store key (op|backend|device|dtype|shape), the
// BackendPlacementObservation record, and the compiler's placement decision.
// Every string spelling here is FAITHFUL to what is already persisted in
// placement caches and support bundles; this layer adds type safety, not a
// second schema. The engine/backend version the ticket lists as part of the
// key is carried by the cache-file envelope, not per entry.

namespace cyxwiz {

// ---------------------------------------------------------------------------
// GpuExecutionKey — the exact identity the observation store is keyed by.
// ---------------------------------------------------------------------------

struct GpuExecutionKey {
    std::string op_type;
    std::string backend;
    std::string device_signature;
    std::string dtype;
    std::string shape_signature;

    std::string ToStoreKey() const {
        return BuildBackendPlacementObservationKey(
            op_type, backend, device_signature, dtype, shape_signature);
    }

    static GpuExecutionKey FromObservation(
        const BackendPlacementObservation& observation) {
        return GpuExecutionKey{observation.op_type,
                               observation.backend,
                               observation.device,
                               observation.dtype,
                               observation.shape_signature};
    }

    bool operator==(const GpuExecutionKey& other) const {
        return op_type == other.op_type && backend == other.backend &&
               device_signature == other.device_signature &&
               dtype == other.dtype &&
               shape_signature == other.shape_signature;
    }
};

// ---------------------------------------------------------------------------
// GpuExecutionEvidence — where a supported/unsafe claim comes from.
// ---------------------------------------------------------------------------

enum class GpuExecutionEvidenceSource {
    StaticPolicy,
    PreflightProbe,
    RuntimeSuccess,
    RuntimeFallback,
    Test,
};

constexpr const char* GpuExecutionEvidenceSourceName(
    GpuExecutionEvidenceSource source) {
    switch (source) {
    case GpuExecutionEvidenceSource::StaticPolicy:
        return "static_policy";
    case GpuExecutionEvidenceSource::PreflightProbe:
        return "preflight_probe";
    case GpuExecutionEvidenceSource::RuntimeSuccess:
        return "runtime_success";
    case GpuExecutionEvidenceSource::RuntimeFallback:
        return "runtime_fallback";
    case GpuExecutionEvidenceSource::Test:
        return "test";
    }
    return "static_policy";
}

inline std::optional<GpuExecutionEvidenceSource>
TryParseGpuExecutionEvidenceSource(std::string_view name) {
    for (const auto source : {GpuExecutionEvidenceSource::StaticPolicy,
                              GpuExecutionEvidenceSource::PreflightProbe,
                              GpuExecutionEvidenceSource::RuntimeSuccess,
                              GpuExecutionEvidenceSource::RuntimeFallback,
                              GpuExecutionEvidenceSource::Test}) {
        if (name == GpuExecutionEvidenceSourceName(source)) {
            return source;
        }
    }
    return std::nullopt;
}

// Outcome of a probe/policy evaluation. "Supported" is spelled "safe" in
// persisted records — the existing BackendPlacementProbeOutcome contract —
// and that spelling is kept; renaming is a cache-schema migration decision.
enum class GpuExecutionOutcome {
    Supported,
    Unsafe,
    Unsupported,
    Timeout,
    Inconclusive,
};

constexpr const char* GpuExecutionOutcomeName(GpuExecutionOutcome outcome) {
    switch (outcome) {
    case GpuExecutionOutcome::Supported:
        return "safe";
    case GpuExecutionOutcome::Unsafe:
        return "unsafe";
    case GpuExecutionOutcome::Unsupported:
        return "unsupported";
    case GpuExecutionOutcome::Timeout:
        return "timeout";
    case GpuExecutionOutcome::Inconclusive:
        return "inconclusive";
    }
    return "inconclusive";
}

inline std::optional<GpuExecutionOutcome> TryParseGpuExecutionOutcome(
    std::string_view name) {
    for (const auto outcome : {GpuExecutionOutcome::Supported,
                               GpuExecutionOutcome::Unsafe,
                               GpuExecutionOutcome::Unsupported,
                               GpuExecutionOutcome::Timeout,
                               GpuExecutionOutcome::Inconclusive}) {
        if (name == GpuExecutionOutcomeName(outcome)) {
            return outcome;
        }
    }
    return std::nullopt;
}

struct GpuExecutionEvidence {
    GpuExecutionEvidenceSource source =
        GpuExecutionEvidenceSource::StaticPolicy;
    // Empty for runtime fallbacks (which carry a reason, not a probe
    // verdict); set for probe/policy evaluations.
    std::optional<GpuExecutionOutcome> outcome;
    // A BackendFallbackReasonName string, or a placement-policy reason code
    // (e.g. RecurrentCudaPlacementReason). Kept as a string for fidelity
    // with persisted records.
    std::string reason_code;
    std::string detail;
    std::string timestamp;
    std::string probe_scope;

    static GpuExecutionEvidence FromObservation(
        const BackendPlacementObservation& observation) {
        GpuExecutionEvidence evidence;
        evidence.source =
            TryParseGpuExecutionEvidenceSource(observation.source)
                .value_or(GpuExecutionEvidenceSource::RuntimeFallback);
        evidence.outcome = TryParseGpuExecutionOutcome(
            observation.probe_outcome);
        evidence.reason_code = observation.reason_code;
        evidence.detail = observation.detail;
        evidence.timestamp = observation.timestamp;
        evidence.probe_scope = observation.probe_scope;
        return evidence;
    }

    BackendPlacementObservation ToObservation(
        const GpuExecutionKey& key) const {
        BackendPlacementObservation observation;
        observation.op_type = key.op_type;
        observation.backend = key.backend;
        observation.device = key.device_signature;
        observation.dtype = key.dtype;
        observation.shape_signature = key.shape_signature;
        observation.source = GpuExecutionEvidenceSourceName(source);
        observation.probe_outcome =
            outcome ? GpuExecutionOutcomeName(*outcome) : "";
        observation.reason_code = reason_code;
        observation.detail = detail;
        observation.timestamp = timestamp;
        observation.probe_scope = probe_scope;
        return observation;
    }
};

// ---------------------------------------------------------------------------
// GpuExecutionDecision — what the engine decided to do for a key, and why.
// ---------------------------------------------------------------------------

enum class GpuExecutionDecisionAction {
    AttemptGpu,
    StagedArrayFire,
    NativeProvider,
    Cpu,
    FailClosed,
};

constexpr const char* GpuExecutionDecisionActionName(
    GpuExecutionDecisionAction action) {
    switch (action) {
    case GpuExecutionDecisionAction::AttemptGpu:
        return "attempt_gpu";
    case GpuExecutionDecisionAction::StagedArrayFire:
        return "staged_arrayfire";
    case GpuExecutionDecisionAction::NativeProvider:
        return "native_provider";
    case GpuExecutionDecisionAction::Cpu:
        return "cpu";
    case GpuExecutionDecisionAction::FailClosed:
        return "fail_closed";
    }
    return "fail_closed";
}

struct GpuExecutionDecision {
    GpuExecutionDecisionAction action = GpuExecutionDecisionAction::FailClosed;
    GpuExecutionKey key;
    // User-facing explanation; must never describe a JIT parameter-space
    // overflow as a VRAM condition.
    std::string explanation;
    std::optional<GpuExecutionEvidence> evidence;
};

} // namespace cyxwiz
