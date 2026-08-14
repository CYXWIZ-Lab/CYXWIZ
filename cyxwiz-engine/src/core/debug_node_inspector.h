#pragma once

#include "debug_recommendation_engine.h"
#include "debug_trace_record.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugNodeInspectorSummary {
    bool available = false;
    int node_id = -1;
    std::string node_name;
    std::string node_type;
    std::string phase;
    std::string role;
    std::string status;
    std::string dtype;
    std::string backend;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    size_t input_rank = 0;
    size_t output_rank = 0;
    size_t input_numel = 0;
    size_t output_numel = 0;
    float duration_ms = 0.0f;
    std::vector<ValidationIssue> issues;
    std::vector<DebugRecommendation> recommendations;
};

class DebugNodeInspector {
public:
    DebugNodeInspectorSummary BuildSummary(
        const DebugTraceRecord& trace,
        const std::vector<DebugRecommendation>& recommendations) const;

    DebugTraceRecord BuildExplanationTrace(
        const std::string& run_id,
        uint64_t graph_hash,
        const gui::MLNode& node,
        const std::string& node_type,
        const TrainingConfiguration* configuration,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::vector<DebugTraceRecord>& traces,
        const std::vector<ValidationIssue>& issues,
        const std::vector<DebugRecommendation>& recommendations) const;
};

} // namespace cyxwiz
