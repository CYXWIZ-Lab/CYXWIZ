#include "../src/core/data_studio_execution_plan.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckValid(const cyxwiz::DataStudioExecutionPlan& plan,
                const std::string& context) {
    Check(plan.valid, context + ": expected valid plan, got '" +
                          plan.error_message + "'");
}

void CheckInvalid(const cyxwiz::DataStudioExecutionPlan& plan,
                  const std::string& context) {
    Check(!plan.valid, context + ": expected invalid plan");
    Check(!plan.error_message.empty(),
          context + ": invalid plan should explain the failure");
}

void CheckSourceTransformSinkPlan() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TextTokenizer", "Tokenizer", {{"text_col", "body"}}},
        {3, "DataOutput", "Save", {{"file_path", "out.csv"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
        {2, 3},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckValid(plan, "source transform sink");
    Check(plan.steps.size() == 3, "source transform sink: expected three steps");

    Check(plan.steps[0].kind == cyxwiz::DataStudioExecutionStepKind::Source,
          "DataInput should classify as source");
    Check(plan.steps[0].canonical_type_name == "DataInput",
          "DataInput canonical type should be preserved");
    Check(plan.steps[0].support.implementation_owner ==
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
          "DataInput should remain PipelineExecutor-owned");

    Check(plan.steps[1].kind == cyxwiz::DataStudioExecutionStepKind::Transform,
          "TextTokenizer should classify as transform");
    Check(plan.steps[1].canonical_type_name == "TextTokenizer",
          "TextTokenizer canonical type should be preserved");
    Check(plan.steps[1].support.implementation_owner ==
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
          "TextTokenizer should be PipelineOperatorFactory-owned");

    Check(plan.steps[2].kind == cyxwiz::DataStudioExecutionStepKind::Sink,
          "DataOutput should classify as sink");
    Check(plan.steps[2].support.implementation_owner ==
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
          "DataOutput should remain PipelineExecutor-owned");
}

void CheckAliasNormalizationPlan() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TSLag", "Old lag", {{"columns", "sales"}, {"lag_periods", "1"}}},
        {3, "DataOutput", "Save", {{"file_path", "out.csv"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
        {2, 3},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckValid(plan, "alias normalization");
    Check(plan.steps[1].legacy_type_name == "TSLag",
          "alias normalization should preserve original type");
    Check(plan.steps[1].canonical_type_name == "TimeSeriesLag",
          "TSLag should normalize to TimeSeriesLag before planning");
    Check(plan.steps[1].compatibility_alias,
          "TSLag should be marked as a compatibility alias");
    Check(plan.steps[1].alias_decision ==
              cyxwiz::PipelineLegacyAliasDecision::NormalizeToCanonical,
          "TSLag should carry the central normalize decision");
    Check(plan.steps[1].support.implementation_owner ==
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
          "normalized TSLag plan step should use canonical legacy runtime owner");
}

void CheckHiddenCompatibilityAliasPlan() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TextTokenize", "Old tokenizer", {{"text_column", "body"}}},
        {3, "DataOutput", "Save", {{"file_path", "out.csv"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
        {2, 3},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckValid(plan, "hidden compatibility alias");
    Check(plan.steps[1].legacy_type_name == "TextTokenize",
          "hidden alias should preserve original type");
    Check(plan.steps[1].canonical_type_name == "TextTokenize",
          "hidden alias should not pretend to be behavior-equivalent");
    Check(plan.steps[1].compatibility_alias,
          "TextTokenize should be marked as a compatibility alias");
    Check(plan.steps[1].alias_decision ==
              cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias,
          "TextTokenize should carry the central hidden-alias decision");
    Check(plan.steps[1].support.implementation_owner ==
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
          "hidden TextTokenize plan step should keep legacy runtime owner");
}

void CheckTrainingLaunchPlan() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TextTokenizer", "Tokenizer", {{"text_col", "body"}}},
        {3, "DataOutput", "Train", {{"file_path", "training.csv"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
        {2, 3},
    };
    cyxwiz::DataStudioExecutionPlanOptions options;
    options.training_launch_node_ids.insert(3);

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links, options);
    CheckValid(plan, "training launch handoff");
    Check(plan.has_training_launch,
          "training launch handoff should be explicit in the plan");
    Check(plan.training_launch_node_id == 3,
          "training launch node id should be recorded");
    Check(plan.steps[2].kind ==
              cyxwiz::DataStudioExecutionStepKind::TrainingLaunch,
          "marked node should classify as training launch");
    Check(plan.steps[2].canonical_type_name == "DataOutput",
          "training launch handoff should not create a second graph plan type");
}

void CheckUnsupportedNodeFailsClosed() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TSNENode", "Blocked", {}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckInvalid(plan, "fail closed node");
    Check(plan.error_message.find("legacy t-SNE graph execution is not implemented") !=
              std::string::npos,
          "fail closed node should report the central capability reason");
}

void CheckCentralIntegerBoundsFailBeforeExecution() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TextTokenizer", "Tokenizer",
         {{"text_col", "body"}, {"max_length", "0"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckInvalid(plan, "central integer bounds");
    Check(plan.error_message.find("max_length") != std::string::npos,
          "integer bound failure should name the invalid parameter");
    Check(plan.error_message.find("integer >= 1") != std::string::npos,
          "integer bound failure should come from central runtime capability");
}

void CheckCentralFloatBoundsFailBeforeExecution() {
    const std::vector<cyxwiz::DataStudioPlanNodeInput> nodes = {
        {1, "DataInput", "Input", {{"source_type", "file"}, {"type", "csv"}}},
        {2, "TargetEncoder", "Encode",
         {{"columns", "category"}, {"target_col", "label"}, {"smoothing", "-0.1"}}},
    };
    const std::vector<cyxwiz::DataStudioPlanLinkInput> links = {
        {1, 2},
    };

    const auto plan = cyxwiz::BuildDataStudioExecutionPlan(nodes, links);
    CheckInvalid(plan, "central float bounds");
    Check(plan.error_message.find("smoothing") != std::string::npos,
          "float bound failure should name the invalid parameter");
    Check(plan.error_message.find("greater than or equal to 0.000000") !=
              std::string::npos,
          "float bound failure should come from central runtime capability");
}

} // namespace

int main() {
    CheckSourceTransformSinkPlan();
    CheckAliasNormalizationPlan();
    CheckHiddenCompatibilityAliasPlan();
    CheckTrainingLaunchPlan();
    CheckUnsupportedNodeFailsClosed();
    CheckCentralIntegerBoundsFailBeforeExecution();
    CheckCentralFloatBoundsFailBeforeExecution();
    std::cout << "test_data_studio_execution_plan passed\n";
    return 0;
}
