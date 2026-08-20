#pragma once

#include "executable_model.h"
#include "graph_compiler.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizer.h>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// Exact origin of a SequentialModel module created from one compiled graph
// layer. A missing module_index means the compiled layer did not create a
// module. The builder owns this mapping so callers do not infer it again.
struct BuiltModuleProvenance {
    size_t compiled_layer_index = 0;
    std::optional<size_t> module_index;
    int node_id = -1;
    std::string node_name;
    gui::NodeType node_type = gui::NodeType::Dense;
    std::string module_name;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::map<std::string, std::string> configured_parameters;

    bool created() const { return module_index.has_value(); }
};

// Result of assembling a SequentialModel + Loss + Optimizer from a
// compiled TrainingConfiguration. On failure, `model` is nullptr and the
// builder has already logged the reason.
struct BuiltModel {
    std::unique_ptr<SequentialModel> model;
    std::unique_ptr<Loss>            loss;
    std::unique_ptr<Optimizer>       optimizer;
    std::string                      error_message;
    std::vector<BuiltModuleProvenance> module_provenance;

    bool ok() const { return model != nullptr; }
};

struct BuiltExecutableModel {
    std::unique_ptr<IExecutableModel> model;
    std::unique_ptr<Loss>             loss;
    std::unique_ptr<Optimizer>        optimizer;
    std::string                       error_message;

    bool ok() const { return model != nullptr; }
};

// Effective loss settings resolved by the same code path used to construct
// the runtime Loss. Debugging and UI surfaces consume this value instead of
// re-parsing graph strings or copying backend defaults.
struct ResolvedLossConfiguration {
    gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss;
    std::string loss_name;
    Reduction reduction = Reduction::Mean;
    bool ignore_index_applicable = false;
    int ignore_index = -100;
    bool class_weights_applicable = false;
    std::vector<float> class_weights;
    bool label_smoothing_applicable = false;
    float label_smoothing = 0.0f;
    std::optional<float> pos_weight;
    std::optional<float> alpha;
    std::optional<float> beta;
    std::optional<float> gamma;
    std::optional<float> smooth;
};

// Build SequentialModel + Loss + Optimizer from a TrainingConfiguration.
// Pure function — no side effects beyond logging. Shared between
// TrainingExecutor (real training) and DebugExecutor (one-step local
// debug sanity check).
BuiltModel BuildSequentialFromConfig(const TrainingConfiguration& config);

// Build only the configured loss. Test and training execution share this so
// reduction, class weights, label smoothing, and BCE pos_weight cannot drift.
std::unique_ptr<Loss> BuildLossFromConfig(
    const TrainingConfiguration& config);

ResolvedLossConfiguration ResolveLossConfiguration(
    const TrainingConfiguration& config);

// Build the narrow executable model interface. Sequential configs wrap the
// existing SequentialModel; configs with graph_op_node_ids use the graph-plan
// executable path.
BuiltExecutableModel BuildExecutableFromConfig(const TrainingConfiguration& config);

// Build a graph-plan-backed executable. Used directly by focused graph-runtime
// tests and indirectly by BuildExecutableFromConfig for graph-op configs.
BuiltExecutableModel BuildGraphExecutableFromConfig(const TrainingConfiguration& config);

} // namespace cyxwiz
