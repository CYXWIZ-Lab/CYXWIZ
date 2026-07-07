#include "data_studio_execution_plan.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <queue>
#include <sstream>
#include <utility>

namespace cyxwiz {
namespace {

std::string Trim(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string NodeLabel(int node_id, const std::string& type) {
    std::ostringstream out;
    out << "node " << node_id << " (" << type << ")";
    return out.str();
}

bool HasNonEmptyParameter(const std::map<std::string, std::string>& parameters,
                          const std::string& name) {
    const auto it = parameters.find(name);
    return it != parameters.end() && !Trim(it->second).empty();
}

bool IsSinkNode(gui::NodeType node_type) {
    switch (node_type) {
    case gui::NodeType::DataOutput:
    case gui::NodeType::DeployToNodeEditorNode:
    case gui::NodeType::ExportCSV:
    case gui::NodeType::ExportJSON:
    case gui::NodeType::ExportParquet:
        return true;
    default:
        return false;
    }
}

struct ResolvedPlanNodeRuntime {
    PipelineRuntimeSupport support;
    std::string canonical_type_name;
    bool compatibility_alias = false;
    PipelineLegacyAliasDecision alias_decision =
        PipelineLegacyAliasDecision::Unknown;
};

ResolvedPlanNodeRuntime ResolvePlanNodeRuntime(
    const std::string& legacy_type_name) {
    ResolvedPlanNodeRuntime resolved;

    if (const auto* alias =
            ResolvePipelineLegacyAliasDecision(legacy_type_name)) {
        resolved.compatibility_alias = true;
        resolved.alias_decision = alias->decision;
        if (alias->decision ==
            PipelineLegacyAliasDecision::NormalizeToCanonical) {
            resolved.canonical_type_name = alias->canonical_type_name;
            resolved.support =
                ResolvePipelineRuntimeSupport(alias->canonical_node_type);
            if (resolved.support.mode != PipelineRuntimeSupportMode::Unknown) {
                return resolved;
            }
        }
    }

    resolved.canonical_type_name = legacy_type_name;
    resolved.support = ResolvePipelineRuntimeSupport(legacy_type_name);
    return resolved;
}

DataStudioExecutionStepKind ClassifyStep(
    const DataStudioPlanNodeInput& node,
    const PipelineRuntimeSupport& support,
    const DataStudioExecutionPlanOptions& options) {
    if (options.training_launch_node_ids.find(node.id) !=
        options.training_launch_node_ids.end()) {
        return DataStudioExecutionStepKind::TrainingLaunch;
    }
    if (support.source_node) {
        return DataStudioExecutionStepKind::Source;
    }
    if (support.node_type.has_value() && IsSinkNode(*support.node_type)) {
        return DataStudioExecutionStepKind::Sink;
    }
    return DataStudioExecutionStepKind::Transform;
}

bool ValidateRuntimeSupport(const DataStudioPlanNodeInput& node,
                            const PipelineRuntimeSupport& support,
                            std::string& error) {
    if (support.mode == PipelineRuntimeSupportMode::Unknown) {
        error = "Unsupported Data Studio node: " + NodeLabel(node.id, node.type);
        return false;
    }
    if (support.mode == PipelineRuntimeSupportMode::FailClosed) {
        error = "Unsupported Data Studio node: " + NodeLabel(node.id, node.type);
        if (support.fail_closed_reason != nullptr) {
            error += ": ";
            error += support.fail_closed_reason;
        }
        return false;
    }
    return true;
}

bool ValidateParameters(const DataStudioPlanNodeInput& node,
                        const PipelineRuntimeSupport& support,
                        std::string& error) {
    for (const char* required : support.required_parameters) {
        if (!HasNonEmptyParameter(node.parameters, required)) {
            error = NodeLabel(node.id, node.type) +
                    " is missing required parameter '" + required + "'";
            return false;
        }
    }

    if (!ValidatePipelineRuntimeParameterCapabilities(
            NodeLabel(node.id, node.type),
            node.parameters,
            support.allowed_parameter_values,
            support.integer_parameters,
            support.float_parameters,
            "Data Studio execution plan",
            error)) {
        return false;
    }

    return true;
}

bool ValidateInputArity(const DataStudioPlanNodeInput& node,
                        const PipelineRuntimeSupport& support,
                        int input_count,
                        DataStudioExecutionStepKind kind,
                        std::string& error) {
    if (support.source_node && input_count > 0) {
        error = NodeLabel(node.id, node.type) + " is a source but has inputs";
        return false;
    }

    if (kind == DataStudioExecutionStepKind::TrainingLaunch) {
        if (input_count == 0) {
            error = NodeLabel(node.id, node.type) +
                    " is a training launch handoff but has no input";
            return false;
        }
        return true;
    }

    if (!support.source_node && input_count == 0) {
        error = NodeLabel(node.id, node.type) + " requires an input";
        return false;
    }

    if (support.required_input_count.has_value() &&
        input_count != *support.required_input_count) {
        std::ostringstream out;
        out << NodeLabel(node.id, node.type) << " requires "
            << *support.required_input_count << " input(s), got "
            << input_count;
        error = out.str();
        return false;
    }

    return true;
}

} // namespace

const char* DataStudioExecutionStepKindName(
    DataStudioExecutionStepKind kind) {
    switch (kind) {
    case DataStudioExecutionStepKind::Source:
        return "source";
    case DataStudioExecutionStepKind::Transform:
        return "transform";
    case DataStudioExecutionStepKind::Sink:
        return "sink";
    case DataStudioExecutionStepKind::TrainingLaunch:
        return "training_launch";
    case DataStudioExecutionStepKind::Unknown:
        return "unknown";
    }
    return "unknown";
}

DataStudioExecutionPlan BuildDataStudioExecutionPlan(
    const std::vector<DataStudioPlanNodeInput>& nodes,
    const std::vector<DataStudioPlanLinkInput>& links,
    const DataStudioExecutionPlanOptions& options) {
    DataStudioExecutionPlan plan;
    if (nodes.empty()) {
        plan.error_message = "Data Studio execution plan requires at least one node";
        return plan;
    }

    std::map<int, const DataStudioPlanNodeInput*> nodes_by_id;
    std::map<int, std::vector<int>> input_nodes;
    std::map<int, std::vector<int>> output_nodes;
    std::map<int, int> indegree;

    for (const auto& node : nodes) {
        if (node.id < 0) {
            plan.error_message = "Data Studio execution plan contains a negative node id";
            return plan;
        }
        if (node.type.empty()) {
            plan.error_message = "Data Studio execution plan contains a node with an empty type";
            return plan;
        }
        if (!nodes_by_id.emplace(node.id, &node).second) {
            plan.error_message = "Data Studio execution plan contains duplicate node id " +
                                 std::to_string(node.id);
            return plan;
        }
        indegree[node.id] = 0;
    }

    for (const auto& link : links) {
        if (nodes_by_id.find(link.from_node_id) == nodes_by_id.end()) {
            plan.error_message = "Data Studio execution plan link starts at unknown node " +
                                 std::to_string(link.from_node_id);
            return plan;
        }
        if (nodes_by_id.find(link.to_node_id) == nodes_by_id.end()) {
            plan.error_message = "Data Studio execution plan link ends at unknown node " +
                                 std::to_string(link.to_node_id);
            return plan;
        }
        if (link.from_node_id == link.to_node_id) {
            plan.error_message = "Data Studio execution plan contains a self link at node " +
                                 std::to_string(link.from_node_id);
            return plan;
        }
        output_nodes[link.from_node_id].push_back(link.to_node_id);
        input_nodes[link.to_node_id].push_back(link.from_node_id);
        ++indegree[link.to_node_id];
    }

    std::queue<int> ready;
    for (const auto& [node_id, degree] : indegree) {
        if (degree == 0) {
            ready.push(node_id);
        }
    }

    std::vector<int> ordered_node_ids;
    while (!ready.empty()) {
        const int node_id = ready.front();
        ready.pop();
        ordered_node_ids.push_back(node_id);

        for (const int output_id : output_nodes[node_id]) {
            auto degree_it = indegree.find(output_id);
            --degree_it->second;
            if (degree_it->second == 0) {
                ready.push(output_id);
            }
        }
    }

    if (ordered_node_ids.size() != nodes.size()) {
        plan.error_message = "Data Studio execution plan graph contains a cycle";
        return plan;
    }

    for (const int node_id : ordered_node_ids) {
        const auto& node = *nodes_by_id[node_id];
        auto runtime = ResolvePlanNodeRuntime(node.type);
        if (!ValidateRuntimeSupport(node, runtime.support, plan.error_message)) {
            return plan;
        }

        const auto kind = ClassifyStep(node, runtime.support, options);
        if (!ValidateInputArity(node,
                                runtime.support,
                                static_cast<int>(input_nodes[node.id].size()),
                                kind,
                                plan.error_message)) {
            return plan;
        }
        if (!ValidateParameters(node, runtime.support, plan.error_message)) {
            return plan;
        }

        DataStudioExecutionPlanStep step;
        step.kind = kind;
        step.node_id = node.id;
        step.legacy_type_name = node.type;
        step.canonical_type_name = runtime.canonical_type_name;
        step.name = node.name;
        step.compatibility_alias = runtime.compatibility_alias;
        step.alias_decision = runtime.alias_decision;
        step.node_type = runtime.support.node_type;
        step.support = std::move(runtime.support);
        step.input_node_ids = input_nodes[node.id];
        step.output_node_ids = output_nodes[node.id];
        step.parameters = node.parameters;

        if (kind == DataStudioExecutionStepKind::TrainingLaunch) {
            if (plan.has_training_launch) {
                plan.error_message =
                    "Data Studio execution plan contains multiple training launch handoffs";
                return DataStudioExecutionPlan{false, plan.error_message};
            }
            plan.has_training_launch = true;
            plan.training_launch_node_id = node.id;
        }

        plan.steps.push_back(std::move(step));
    }

    plan.valid = true;
    return plan;
}

} // namespace cyxwiz
