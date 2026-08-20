#pragma once

#include "debug_trace_record.h"
#include "graph_compiler.h"

#include <cstdint>
#include <string>

namespace cyxwiz {

struct DebugRunExecutionSummary;

struct DebugRuntimeBackendClassification {
    std::string requested_backend = "auto";
    std::string expected_backend = BackendPlacementStatus::Unknown;
    std::string fallback_backend;
    std::string status = BackendPlacementStatus::Unknown;
    std::string reason_code;
    std::string explanation;
    std::string suggested_action;
    std::string observation_source;
    std::string observation_device;
    std::string observation_dtype;
    std::string observation_shape_signature;
    std::string observation_detail;
    std::string observation_timestamp;
    std::string observation_probe_outcome;
    std::string observation_probe_scope;
    std::string evidence_scope = "compiler_capability";
    std::string actual_backend = "unobserved";
    bool proven = false;
    bool fallback_possible = false;
    bool prior_runtime_fallback_observed = false;
    bool actual_backend_observed = false;
    bool cost_estimate_available = false;
    bool needs_attention = false;
};

class DebugRuntimeBackendClassifier {
public:
    static constexpr const char* kAuditSchema =
        "cyxwiz.debug.backend_decision_audit.v1";
    static constexpr size_t kMaxAuditRows = 256;

    DebugRuntimeBackendClassification Classify(
        const BackendPlacementEntry& placement) const;

    void AttachToTrace(DebugTraceRecord& trace,
                       const BackendPlacementEntry& placement) const;

    DebugTraceRecord BuildPlacementTrace(
        const std::string& run_id,
        uint64_t graph_hash,
        const BackendPlacementEntry& placement) const;

    DebugTraceRecord BuildAuditTrace(
        const std::string& run_id,
        const std::vector<DebugTraceRecord>& traces,
        const DebugRunExecutionSummary& execution) const;
};

} // namespace cyxwiz
