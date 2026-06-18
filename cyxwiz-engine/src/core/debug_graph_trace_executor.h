#pragma once

#include "debug_trace_record.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugGraphTraceStep {
    int node_id = -1;
    std::string node_name;
    std::string node_type;
    std::string phase;
    DebugTraceRole role = DebugTraceRole::Activation;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::string dtype;
    std::string backend = "unknown";
    std::string status = "ok";
    float duration_ms = 0.0f;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    nlohmann::json payload = nlohmann::json::object();
};

class DebugGraphTraceExecutor {
public:
    std::vector<DebugTraceRecord> TraceSteps(
        const std::string& run_id,
        const std::vector<DebugGraphTraceStep>& steps) const;
};

} // namespace cyxwiz
