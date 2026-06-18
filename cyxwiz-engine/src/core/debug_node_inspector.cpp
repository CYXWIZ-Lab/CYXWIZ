#include "debug_node_inspector.h"

namespace cyxwiz {

namespace {

size_t ShapeNumElements(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t n = 1;
    for (size_t dim : shape) {
        n *= dim;
    }
    return n;
}

size_t PayloadSize(const nlohmann::json& payload,
                   const char* key,
                   size_t fallback) {
    auto it = payload.find(key);
    if (it == payload.end() || !it->is_number_unsigned()) {
        return fallback;
    }
    return it->get<size_t>();
}

std::string PayloadString(const nlohmann::json& payload,
                          const char* key) {
    auto it = payload.find(key);
    if (it == payload.end() || !it->is_string()) {
        return "";
    }
    return it->get<std::string>();
}

bool RecommendationApplies(const DebugRecommendation& rec,
                           const DebugTraceRecord& trace) {
    if (rec.node_id == trace.node_id) {
        return true;
    }
    return rec.node_id < 0 &&
           trace.status != "ok" &&
           trace.status != "passed" &&
           trace.status != "captured";
}

} // namespace

DebugNodeInspectorSummary DebugNodeInspector::BuildSummary(
    const DebugTraceRecord& trace,
    const std::vector<DebugRecommendation>& recommendations) const {
    DebugNodeInspectorSummary summary;
    summary.available = true;
    summary.node_id = trace.node_id;
    summary.node_name = trace.node_name;
    summary.node_type = trace.node_type;
    summary.phase = trace.phase;
    summary.role = DebugTraceRoleName(trace.role);
    summary.status = trace.status;
    summary.dtype = trace.dtype;
    summary.backend = PayloadString(trace.payload, "backend");
    summary.input_shape = trace.input_shape;
    summary.output_shape = trace.output_shape;
    summary.input_rank = PayloadSize(
        trace.payload, "input_rank", trace.input_shape.size());
    summary.output_rank = PayloadSize(
        trace.payload, "output_rank", trace.output_shape.size());
    summary.input_numel = PayloadSize(
        trace.payload, "input_numel", ShapeNumElements(trace.input_shape));
    summary.output_numel = PayloadSize(
        trace.payload, "output_numel", ShapeNumElements(trace.output_shape));
    summary.duration_ms = trace.duration_ms;
    summary.issues = trace.issues;

    for (const auto& rec : recommendations) {
        if (RecommendationApplies(rec, trace)) {
            summary.recommendations.push_back(rec);
        }
    }

    return summary;
}

} // namespace cyxwiz
