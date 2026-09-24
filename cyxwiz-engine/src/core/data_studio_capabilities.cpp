#include "data_studio_capabilities.h"
#include "data_studio_execution_plan.h"
#include "node_metadata.h"
#include <algorithm>
#include <cctype>
#include <limits>

namespace cyxwiz {
const char* DataStudioActionStateName(DataStudioActionState state) {
    switch (state) {
    case DataStudioActionState::RuntimeBacked: return "Runtime backed";
    case DataStudioActionState::ExploreOnly: return "Explore only";
    default: return "Unavailable";
    }
}

DataStudioCapability ResolveDataStudioCapability(
    const DataStudioCapabilityRequest& request, const NodeMetadata* metadata) {
    DataStudioCapability result;
    if (!metadata || metadata->type != request.node_type) {
        result.reason = "Matching node metadata is unavailable";
        return result;
    }
    result.display_name = metadata->name;
    const auto support = ResolvePipelineRuntimeSupport(request.node_type);
    // SQL exploration has an existing owner but no PipelineExecutor route.
    // Never infer exploration support for other nodes from metadata status.
    if (request.node_type == gui::NodeType::SQLQuery) {
        if (request.input_storage.size() != 1) {
            result.reason = "Select one dataset to explore with SQL";
            return result;
        }
        if (request.input_storage.front() != PipelineStorageBackend::ArrowTable) {
            result.reason = "SQL exploration currently requires an in-memory Arrow table";
            return result;
        }
        const auto query = request.parameters.find("query");
        if (query == request.parameters.end() ||
            std::all_of(query->second.begin(), query->second.end(),
                        [](unsigned char c) { return std::isspace(c) != 0; })) {
            result.reason = "Enter a SQL query";
            return result;
        }
        result.state = DataStudioActionState::ExploreOnly;
        result.reason = "Scratch queries run in this editor. To save a query in a pipeline or recipe, use a SQL Query step.";
        return result;
    }
    if (!support.pipeline_executor_supported ||
        support.fail_mode != PipelineRuntimeFailMode::Real ||
        (support.implementation_owner != PipelineRuntimeImplementationOwner::PipelineExecutor &&
         support.implementation_owner != PipelineRuntimeImplementationOwner::PipelineOperatorFactory)) {
        result.reason = support.fail_closed_reason ? support.fail_closed_reason :
            "No real pipeline execution implementation is registered for this operation";
        return result;
    }
    if (IsNodeSupportBlocked(*metadata)) {
        result.reason = "This operation is marked blocked in the shared node metadata";
        return result;
    }
    const auto* canonical_name = ResolvePipelineRuntimeLegacyTypeName(request.node_type);
    if (!canonical_name || request.input_storage.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        result.reason = "Operation identity or input count is invalid";
        return result;
    }
    DataStudioPlanNodeInput node{0, canonical_name, result.display_name, request.parameters};
    CanonicalizePipelineParameterAliases(request.node_type, node.parameters);
    if (!ValidateDataStudioOperationConfiguration(
            node, static_cast<int>(request.input_storage.size()), result.reason)) {
        return result;
    }
    // The planner owns explicit arity; metadata supplies the remaining port bound.
    size_t maximum = 0;
    bool unbounded = false;
    for (const auto& port : metadata->inputs) {
        if (port.variadic && port.max_connections <= 0) unbounded = true;
        else maximum += static_cast<size_t>(std::max(1, port.max_connections));
    }
    if (!unbounded && request.input_storage.size() > maximum) {
        result.reason = "More dataset inputs were supplied than the operation's declared ports accept";
        return result;
    }
    for (size_t i = 0; i < request.input_storage.size(); ++i) {
        const auto storage = ResolvePipelineExecutorInputStorageSupport(
            request.node_type, request.input_storage[i]);
        if (!storage.supported) {
            result.reason = "Input " + std::to_string(i + 1) + " (" +
                PipelineStorageBackendName(request.input_storage[i]) + "): " +
                (storage.reason ? storage.reason : "Unsupported storage");
            return result;
        }
    }
    result.state = DataStudioActionState::RuntimeBacked;
    result.reason = "Operation configuration is supported; schema and data checks still run during execution.";
    return result;
}
} // namespace cyxwiz
