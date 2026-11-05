#pragma once

#include "debug_graph_trace_executor.h"
#include "debug_numerics.h"
#include "executable_model.h"
#include "graph_compiler.h"
#include "graph_model.h"
#include "layer_capture.h"
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
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

enum class ShapeMismatchKind {
    None,
    Input,
    Output,
    InputAndOutput,
};

const char* ShapeMismatchKindName(ShapeMismatchKind kind);

inline constexpr float kDebugNormDenominatorFloor = 1.0e-12f;

// Per-layer trace emitted during a DebugExecutor forward pass.
struct LayerTrace {
    int node_id = -1;
    std::string name;                     // graph node name when available
    std::string module_name;              // actual constructed module name
    gui::NodeType type = gui::NodeType::Dense;
    size_t module_index = 0;
    size_t compiled_layer_index = 0;
    std::vector<size_t> predicted_input_shape;
    std::vector<size_t> actual_input_shape;
    std::vector<size_t> predicted_shape;  // from CompiledLayer::output_shape
    std::vector<size_t> actual_shape;     // observed at runtime
    float forward_ms = 0.0f;
    bool input_shape_matches = false;
    bool shape_matches = false;           // output shape comparison
    ShapeMismatchKind shape_mismatch = ShapeMismatchKind::None;
    bool is_first_shape_mismatch = false;
    int upstream_node_id = -1;
    std::string upstream_node_name;
    bool has_nan = false;
    bool has_inf = false;
    DebugTensorNumericSummary numerics;

    bool has_shape_mismatch() const {
        return shape_mismatch != ShapeMismatchKind::None;
    }
};

// Gradient norm for one learnable parameter tensor. Used to detect dead
// subgraphs (is_zero for every param in a layer) and silent NaN explosions.
struct GradNormEntry {
    std::string param_name;
    int layer_index = -1;                 // actual SequentialModel module index
    size_t compiled_layer_index = 0;
    int node_id = -1;
    std::string node_name;
    std::vector<size_t> parameter_shape;
    std::vector<size_t> gradient_shape;
    float parameter_l2_norm = 0.0f;
    float l2_norm = 0.0f;
    float grad_parameter_ratio = 0.0f;
    float update_l2_norm = 0.0f;
    float update_parameter_ratio = 0.0f;
    bool update_observed = false;
    bool has_gradient = false;
    bool is_nan = false;
    bool is_inf = false;
    bool is_zero = false;
    DebugTensorNumericSummary numerics;
    std::string missing_gradient_reason;
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
    std::vector<DebugGraphTraceStep> model_build_traces;
    float forward_total_ms = 0.0f;
    float backward_total_ms = 0.0f;
    float optimizer_step_ms = 0.0f;
    // What ran: the picked item ("image 120 of cat_dog, validation") or
    // "a synthetic sample (seed 1337)".
    std::string input_description;

    float loss_value = std::numeric_limits<float>::quiet_NaN();
    bool loss_finite = false;
    std::vector<size_t> loss_prediction_shape;
    std::vector<size_t> loss_target_shape;
    DebugPredictionTargetCompatibility loss_target_compatibility;

    std::vector<GradNormEntry> grad_norms;
    size_t params_with_grad = 0;
    size_t params_missing_grad = 0;
    // State the layers keep beside their weights (BatchNorm's running
    // statistics): no gradient is expected, so these are not in grad_norms.
    size_t state_tensors = 0;

    // What every layer held (SetLayerCapture); null when capture is off.
    std::shared_ptr<const LayerCapture> layer_capture;

    std::vector<ValidationIssue> issues;

    std::chrono::steady_clock::time_point timestamp{};
};

// Runs one forward + one backward + one optimizer step on the picked sample
// (SetInputSample) or a synthetic batch, producing a DebugResult. Does NOT touch DataRegistry — tensors
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

    // Run this sample instead of a synthetic one (TOFIX178 6.0b: the picked
    // item, see debug_sample_source.h). `description` names it in the result.
    void SetInputSample(Tensor features, Tensor labels, std::string description);

    // Keep what every layer held during the run in DebugResult::layer_capture
    // (TOFIX178 6.1a). Off unless set.
    void SetLayerCapture(LayerCaptureOptions options);

    // Execute the debug pass. Catches exceptions at every stage — a
    // throw leaves `result.reached` at the stage where it failed and
    // `result.failure_summary` set to the exception message.
    DebugResult Run();

private:
    TrainingConfiguration config_;
    std::unique_ptr<IExecutableModel> model_;
    std::unique_ptr<Loss> loss_;
    std::unique_ptr<Optimizer> optimizer_;
    Tensor input_features_;
    Tensor input_labels_;
    std::string input_description_;
    std::optional<LayerCaptureOptions> capture_options_;
};

} // namespace cyxwiz
