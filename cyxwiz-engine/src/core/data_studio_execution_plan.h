#pragma once

#include "pipeline_runtime_capabilities.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

struct DataStudioPlanNodeInput {
    int id = -1;
    std::string type;
    std::string name;
    std::map<std::string, std::string> parameters;
};

struct DataStudioPlanLinkInput {
    int from_node_id = -1;
    int to_node_id = -1;
};

struct DataStudioExecutionPlanOptions {
    std::set<int> training_launch_node_ids;
};

enum class DataStudioExecutionStepKind {
    Unknown,
    Source,
    Transform,
    Sink,
    TrainingLaunch,
};

struct DataStudioExecutionPlanStep {
    DataStudioExecutionStepKind kind = DataStudioExecutionStepKind::Unknown;
    int node_id = -1;
    std::string legacy_type_name;
    std::string canonical_type_name;
    std::string name;
    bool compatibility_alias = false;
    PipelineLegacyAliasDecision alias_decision =
        PipelineLegacyAliasDecision::Unknown;
    std::optional<gui::NodeType> node_type = std::nullopt;
    PipelineRuntimeSupport support;
    std::vector<int> input_node_ids;
    std::vector<int> output_node_ids;
    std::map<std::string, std::string> parameters;
};

struct DataStudioExecutionPlan {
    bool valid = false;
    std::string error_message;
    bool has_training_launch = false;
    int training_launch_node_id = -1;
    std::vector<DataStudioExecutionPlanStep> steps;
};

DataStudioExecutionPlan BuildDataStudioExecutionPlan(
    const std::vector<DataStudioPlanNodeInput>& nodes,
    const std::vector<DataStudioPlanLinkInput>& links,
    const DataStudioExecutionPlanOptions& options = {});

const char* DataStudioExecutionStepKindName(
    DataStudioExecutionStepKind kind);

} // namespace cyxwiz
