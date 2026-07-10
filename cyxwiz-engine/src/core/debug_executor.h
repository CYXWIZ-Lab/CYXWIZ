#pragma once

#include "graph_compiler.h"
#include "../gui/node_editor.h"
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Stage reached inside DebugExecutor::Run. Used for failure localization —
// e.g. a crash in Backward surfaces as `reached == DebugStage::Backward`
// with a non-empty `failure_summary`.
enum class DebugStage {
    NotRun,
    BuildModel,
    Forward,
    Loss,
    Backward,
    OptimizerStep,
    Complete,
};

// Per-layer trace emitted during a DebugExecutor forward pass.
struct LayerTrace {
    int node_id = -1;
    std::string name;                     // e.g. "Dense_3"
    gui::NodeType type = gui::NodeType::Dense;
    std::vector<size_t> predicted_shape;  // from CompiledLayer::output_shape
    std::vector<size_t> actual_shape;     // observed at runtime
    float forward_ms = 0.0f;
    bool shape_matches = false;
    bool has_nan = false;
    bool has_inf = false;
};

// Gradient norm for one learnable parameter tensor. Used to detect dead
// subgraphs (is_zero for every param in a layer) and silent NaN explosions.
struct GradNormEntry {
    std::string param_name;
    int layer_index = -1;
    float l2_norm = 0.0f;
    bool has_gradient = false;
    bool is_nan = false;
    bool is_zero = false;
};

// Full result of a single Local Debug run: one forward + one backward +
// one optimizer step on a synthetic batch. The UI renders this through the
// existing compile-result popup infrastructure by mapping `issues` onto
// the same ValidationIssue list Compile uses.
struct DebugResult {
    bool success = false;
    DebugStage reached = DebugStage::NotRun;
    std::string failure_summary;

    std::vector<LayerTrace> layer_traces;
    float forward_total_ms = 0.0f;
    float backward_total_ms = 0.0f;

    float loss_value = std::numeric_limits<float>::quiet_NaN();
    bool loss_finite = false;

    std::vector<GradNormEntry> grad_norms;
    size_t params_with_grad = 0;
    size_t params_missing_grad = 0;

    std::vector<ValidationIssue> issues;

    std::chrono::steady_clock::time_point timestamp{};
};

// Runs one forward + one backward + one optimizer step on a synthetic
// batch, producing a DebugResult. Does NOT touch DataRegistry — tensors
// live on the stack of Run(). Does NOT trigger a reservation / Central
// Server call. Synchronous (UI thread, ~200ms target).
//
// Hand over ownership of `config` — DebugExecutor keeps it alive so grad
// norms can cross-reference config.layers when building trace names.
class DebugExecutor {
public:
    explicit DebugExecutor(TrainingConfiguration config);
    ~DebugExecutor();

    DebugExecutor(const DebugExecutor&) = delete;
    DebugExecutor& operator=(const DebugExecutor&) = delete;

    // Execute the debug pass. Catches exceptions at every stage — a
    // throw leaves `result.reached` at the stage where it failed and
    // `result.failure_summary` set to the exception message.
    DebugResult Run();

private:
    TrainingConfiguration config_;
    std::unique_ptr<SequentialModel> model_;
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<Optimizer> optimizer_;
};

} // namespace cyxwiz
