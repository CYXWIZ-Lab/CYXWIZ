#pragma once

#include "debug_trace_record.h"
#include "training_trace_collector.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugMemoryOwnershipInput {
    int node_id = -1;
    std::string node_name;
    std::string node_type;
    std::string phase = "MemoryOwnership";
    DebugTraceRole role = DebugTraceRole::Activation;
    std::vector<size_t> output_shape;
    std::string dtype = "float32";
    std::string backend = "unknown";
    uint64_t bytes_per_element = 4;
    uint64_t host_budget_bytes = 0;
    uint64_t device_budget_bytes = 0;
    TrainingTraceEvent before;
    TrainingTraceEvent after;
};

class DebugMemoryOwnershipTracer {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.memory_ownership.v1";

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const DebugMemoryOwnershipInput& input) const;

    static uint64_t EstimateTensorBytes(
        const std::vector<size_t>& shape,
        uint64_t bytes_per_element);
};

} // namespace cyxwiz
