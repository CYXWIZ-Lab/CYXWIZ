#pragma once

#include "debug_trace_record.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

class DebugGradientHealthAnalyzer {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.gradient_health.v1";
    static constexpr size_t kMaxLayerRows = 128;
    static constexpr size_t kMaxParameterNamesPerLayer = 16;
    static constexpr size_t kMaxMissingReasonsPerLayer = 8;

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const std::vector<DebugTraceRecord>& traces) const;
};

} // namespace cyxwiz
