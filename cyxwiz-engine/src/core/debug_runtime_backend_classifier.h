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
