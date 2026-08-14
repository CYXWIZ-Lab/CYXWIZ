#include "debug_node_inspector.h"

#include <algorithm>
#include <set>

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

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes,
                            int node_id) {
    const auto it = std::find_if(
        nodes.begin(), nodes.end(),
        [node_id](const gui::MLNode& candidate) {
            return candidate.id == node_id;
        });
    return it == nodes.end() ? nullptr : &*it;
}

bool ContainsNode(const std::vector<CompiledGraphNode>& nodes, int node_id) {
    return std::any_of(
        nodes.begin(), nodes.end(),
        [node_id](const CompiledGraphNode& candidate) {
            return candidate.node_id == node_id;
        });
}

const CompiledLayer* FindLayer(const TrainingConfiguration& configuration,
                               int node_id) {
    const auto it = std::find_if(
        configuration.layers.begin(), configuration.layers.end(),
        [node_id](const CompiledLayer& layer) {
            return layer.node_id == node_id;
        });
    return it == configuration.layers.end() ? nullptr : &*it;
}

const DebugTraceRecord* FindDataTrace(
    const std::vector<DebugTraceRecord>& traces,
    int node_id) {
    const DebugTraceRecord* best = nullptr;
    for (const auto& trace : traces) {
        if (trace.node_id != node_id || trace.phase == "NodeExplanation" ||
            trace.phase == "BackendPlacement") {
            continue;
        }
        if (!trace.input_shape.empty() || !trace.output_shape.empty() ||
            !trace.dtype.empty()) {
            best = &trace;
        }
    }
    return best;
}

const DebugTraceRecord* FindPlacementTrace(
    const std::vector<DebugTraceRecord>& traces,
    int node_id) {
    const auto it = std::find_if(
        traces.begin(), traces.end(),
        [node_id](const DebugTraceRecord& trace) {
            return trace.node_id == node_id &&
                   trace.phase == "BackendPlacement";
        });
    return it == traces.end() ? nullptr : &*it;
}

void CopyPayloadValue(const nlohmann::json& source,
                      nlohmann::json& destination,
                      const char* key) {
    const auto it = source.find(key);
    if (it != source.end()) {
        destination[key] = *it;
    }
}

struct NodeGraphContext {
    std::string role = "unavailable";
    std::string path_status = "unavailable";
};

NodeGraphContext ResolveNodeGraphContext(
    const TrainingConfiguration* configuration,
    int node_id) {
    if (!configuration) {
        return {};
    }
    if (node_id == configuration->data_source_node_id) {
        return {"data_source", "selected"};
    }
    if (node_id == configuration->loss_node_id) {
        return {"loss", "selected"};
    }
    if (node_id == configuration->optimizer_node_id) {
        return {"optimizer", "selected"};
    }
    if (FindLayer(*configuration, node_id)) {
        return {"model_layer", "selected"};
    }
    if (!configuration->graph_plan.available) {
        return {};
    }
    return ContainsNode(configuration->graph_plan.nodes, node_id)
        ? NodeGraphContext{"training_path_node", "selected"}
        : NodeGraphContext{"outside_selected_training_path", "outside"};
}

DebugTraceRole ExplanationRole(const std::string& role) {
    if (role == "data_source") {
        return DebugTraceRole::RawInput;
    }
    if (role == "loss") {
        return DebugTraceRole::Loss;
    }
    if (role == "optimizer") {
        return DebugTraceRole::OptimizerStep;
    }
    return DebugTraceRole::Activation;
}

std::string IssueKey(const ValidationIssue& issue) {
    return std::to_string(static_cast<int>(issue.level)) + "\n" +
           issue.error_code + "\n" + issue.message;
}

void AppendIssueIfRelevant(std::vector<ValidationIssue>& destination,
                           std::set<std::string>& seen,
                           const ValidationIssue& issue,
                           int node_id,
                           bool allow_unscoped) {
    if (issue.node_id != node_id &&
        !(allow_unscoped && issue.node_id < 0)) {
        return;
    }
    const std::string key = IssueKey(issue);
    if (seen.insert(key).second) {
        destination.push_back(issue);
    }
}

std::string NextInspectionAction(
    const std::string& path_status,
    const std::string& data_evidence_scope,
    bool backend_actual_observed,
    const std::vector<ValidationIssue>& issues,
    const std::vector<DebugRecommendation>& recommendations) {
    for (const auto& recommendation : recommendations) {
        if (!recommendation.action.empty()) {
            return recommendation.action;
        }
    }
    if (!issues.empty()) {
        return "Inspect this node's attached issue codes and fix the earliest blocking issue.";
    }
    if (path_status == "outside") {
        return "Inspect graph wiring if this node should connect the selected data, loss, and optimizer path.";
    }
    if (data_evidence_scope != "same_run_trace") {
        return "Run Local Debug or Smoke Run to capture the tensor that reaches this node.";
    }
    if (!backend_actual_observed) {
        return "Capture a same-run runtime trace before treating the expected backend as actual execution.";
    }
    return "Inspect the upstream and downstream nodes next to continue the data-path diagnosis.";
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

DebugTraceRecord DebugNodeInspector::BuildExplanationTrace(
    const std::string& run_id,
    uint64_t graph_hash,
    const gui::MLNode& node,
    const std::string& node_type,
    const TrainingConfiguration* configuration,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<DebugTraceRecord>& traces,
    const std::vector<ValidationIssue>& issues,
    const std::vector<DebugRecommendation>& recommendations) const {
    const NodeGraphContext graph_context =
        ResolveNodeGraphContext(configuration, node.id);
    const CompiledLayer* layer = configuration
        ? FindLayer(*configuration, node.id)
        : nullptr;
    const DebugTraceRecord* data_trace = FindDataTrace(traces, node.id);
    const DebugTraceRecord* placement_trace =
        FindPlacementTrace(traces, node.id);

    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::string dtype = "unobserved";
    std::string data_evidence_scope = "unobserved";
    std::string data_evidence_phase;
    if (data_trace) {
        input_shape = data_trace->input_shape;
        output_shape = data_trace->output_shape;
        if (!data_trace->dtype.empty()) {
            dtype = data_trace->dtype;
        }
        data_evidence_scope = "same_run_trace";
        data_evidence_phase = data_trace->phase;
    } else if (layer) {
        input_shape = layer->input_shape;
        output_shape = layer->output_shape;
        data_evidence_scope = "compiler_prediction";
        data_evidence_phase = "Compile";
    }

    std::string expected_backend = "unobserved";
    std::string actual_backend = "unobserved";
    bool actual_backend_observed = false;
    if (placement_trace) {
        expected_backend = PayloadString(
            placement_trace->payload, "backend_expected");
        if (expected_backend.empty()) {
            expected_backend = PayloadString(
                placement_trace->payload, "backend");
        }
    }
    for (const auto& candidate : traces) {
        if (candidate.node_id != node.id ||
            candidate.phase == "NodeExplanation") {
            continue;
        }
        const auto observed =
            candidate.payload.find("backend_actual_observed");
        if (observed != candidate.payload.end() && observed->is_boolean() &&
            observed->get<bool>()) {
            const std::string value =
                PayloadString(candidate.payload, "backend_actual");
            if (!value.empty() && value != "unobserved") {
                actual_backend = value;
                actual_backend_observed = true;
            }
        }
    }

    DebugTraceRecord explanation = DebugNodeTraceContract::Make(
        run_id,
        node.id,
        node.name,
        node_type,
        "NodeExplanation",
        ExplanationRole(graph_context.role),
        input_shape,
        output_shape,
        dtype,
        actual_backend_observed ? actual_backend : expected_backend,
        "captured");
    explanation.payload["explanation_schema"] =
        "cyxwiz.debug.node_explanation.v1";
    explanation.payload["graph_hash"] = graph_hash;
    explanation.payload["trace_producer"] = "DebugNodeInspector";
    explanation.payload["graph_role"] = graph_context.role;
    explanation.payload["training_path_status"] =
        graph_context.path_status;
    explanation.payload["data_evidence_scope"] = data_evidence_scope;
    explanation.payload["data_evidence_phase"] = data_evidence_phase;
    explanation.payload["backend_expected"] = expected_backend;
    explanation.payload["backend_actual"] = actual_backend;
    explanation.payload["backend_actual_observed"] =
        actual_backend_observed;
    explanation.payload["backend_evidence_scope"] =
        actual_backend_observed ? "same_run_runtime" :
        (placement_trace ? "compiler_placement" : "unobserved");

    if (placement_trace) {
        constexpr const char* placement_keys[] = {
            "backend_requested",
            "backend_status",
            "backend_reason_code",
            "backend_explanation",
            "backend_suggested_action",
            "backend_fallback",
            "backend_prior_runtime_fallback_observed",
            "backend_fallback_observed_this_run"
        };
        for (const char* key : placement_keys) {
            CopyPayloadValue(
                placement_trace->payload, explanation.payload, key);
        }
    }

    nlohmann::json upstream = nlohmann::json::array();
    nlohmann::json downstream = nlohmann::json::array();
    std::set<int> seen_upstream;
    std::set<int> seen_downstream;
    for (const auto& link : links) {
        if (link.to_node == node.id && seen_upstream.insert(link.from_node).second) {
            const gui::MLNode* neighbor = FindNode(nodes, link.from_node);
            upstream.push_back({
                {"node_id", link.from_node},
                {"node_name", neighbor ? neighbor->name : ""}
            });
        }
        if (link.from_node == node.id && seen_downstream.insert(link.to_node).second) {
            const gui::MLNode* neighbor = FindNode(nodes, link.to_node);
            downstream.push_back({
                {"node_id", link.to_node},
                {"node_name", neighbor ? neighbor->name : ""}
            });
        }
    }
    explanation.payload["upstream_nodes"] = std::move(upstream);
    explanation.payload["downstream_nodes"] = std::move(downstream);

    std::set<std::string> seen_issues;
    for (const auto& issue : issues) {
        AppendIssueIfRelevant(
            explanation.issues, seen_issues, issue, node.id, false);
    }
    for (const auto& trace : traces) {
        if (trace.node_id != node.id) {
            continue;
        }
        for (const auto& issue : trace.issues) {
            AppendIssueIfRelevant(
                explanation.issues, seen_issues, issue, node.id, true);
        }
    }
    DebugNodeTraceContract::AttachIssueSummary(
        explanation, explanation.issues);

    std::vector<DebugRecommendation> relevant_recommendations;
    nlohmann::json recommendation_payload = nlohmann::json::array();
    for (const auto& recommendation : recommendations) {
        if (recommendation.node_id != node.id) {
            continue;
        }
        relevant_recommendations.push_back(recommendation);
        recommendation_payload.push_back({
            {"severity", DebugRecommendationSeverityName(
                recommendation.severity)},
            {"category", recommendation.category},
            {"title", recommendation.title},
            {"detail", recommendation.detail},
            {"action", recommendation.action}
        });
    }
    explanation.payload["recommendations"] =
        std::move(recommendation_payload);
    explanation.payload["next_inspection_action"] = NextInspectionAction(
        graph_context.path_status,
        data_evidence_scope,
        actual_backend_observed,
        explanation.issues,
        relevant_recommendations);
    DebugNodeTraceContract::AttachDiagnosticContext(
        explanation,
        "node_explanation",
        "DebugNodeInspector",
        "cyxwiz-engine/src/core/debug_node_inspector.cpp",
        "cyxwiz::DebugNodeInspector::BuildExplanationTrace");
    explanation.payload["success"] = true;
    return explanation;
}

} // namespace cyxwiz
