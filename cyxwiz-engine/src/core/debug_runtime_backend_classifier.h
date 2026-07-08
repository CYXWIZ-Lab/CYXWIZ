#pragma once

#include "debug_trace_record.h"
#include "graph_compiler.h"

#include <string>

namespace cyxwiz {

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
    bool proven = false;
    bool fallback_possible = false;
    bool needs_attention = false;
};

class DebugRuntimeBackendClassifier {
public:
    DebugRuntimeBackendClassification Classify(
        const BackendPlacementEntry& placement) const;

    void AttachToTrace(DebugTraceRecord& trace,
                       const BackendPlacementEntry& placement) const;
};

} // namespace cyxwiz
