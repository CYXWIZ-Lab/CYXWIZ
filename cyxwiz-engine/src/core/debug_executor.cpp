#include "debug_executor.h"
#include "error_codes.h"
#include "model_builder.h"
#include "synthetic_batch.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace cyxwiz {

const char* ShapeMismatchKindName(ShapeMismatchKind kind) {
    switch (kind) {
        case ShapeMismatchKind::None:           return "none";
        case ShapeMismatchKind::Input:          return "input";
        case ShapeMismatchKind::Output:         return "output";
        case ShapeMismatchKind::InputAndOutput: return "input_and_output";
    }
    return "none";
}

namespace {

const char* StageName(DebugStage s) {
    switch (s) {
        case DebugStage::NotRun:        return "NotRun";
        case DebugStage::BuildModel:    return "BuildModel";
        case DebugStage::Forward:       return "Forward";
        case DebugStage::Loss:          return "Loss";
        case DebugStage::Backward:      return "Backward";
        case DebugStage::OptimizerStep: return "OptimizerStep";
        case DebugStage::Complete:      return "Complete";
    }
    return "?";
}

// NaN + Inf scan. Only Float32 can carry non-finite values; int tensors
// (text token IDs, classification labels) are always finite by construction.
// Returns {has_nan, has_inf}.
std::pair<bool, bool> ScanFinite(const Tensor& t) {
    if (t.GetDataType() != DataType::Float32) return {false, false};
    bool has_nan = false;
    bool has_inf = false;
    const float* p = t.ReadData<float>();
    const size_t n = t.NumElements();
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(p[i])) { has_nan = true; }
        else if (std::isinf(p[i])) { has_inf = true; }
        if (has_nan && has_inf) break;
    }
    return {has_nan, has_inf};
}

// L2 norm at the bounded Local Debug reporting boundary. The elementwise
// square and reduction stay on the selected Tensor backend; only the scalar
// reduction result crosses to the host.
float L2Norm(const Tensor& t) {
    if (t.GetDataType() != DataType::Float32) return 0.0f;
    Tensor sum_of_squares = (t * t).Sum();
    if (sum_of_squares.NumElements() != 1 ||
        sum_of_squares.GetDataType() != DataType::Float32) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float value = sum_of_squares.ReadData<float>()[0];
    if (!std::isfinite(value)) {
        return value;
    }
    return std::sqrt(std::max(0.0f, value));
}

float RelativeNorm(float numerator, float denominator) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return numerator /
        std::max(std::abs(denominator), kDebugNormDenominatorFloor);
}

// Extract a scalar from a reduction-output loss tensor. Loss forward
// returns a shape-[1] Float32 tensor after Mean / Sum reduction.
float ExtractLossScalar(const Tensor& t) {
    if (t.NumElements() == 0 || t.GetDataType() != DataType::Float32) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return t.ReadData<float>()[0];
}

bool ShapesMatchForTrace(const std::vector<size_t>& predicted,
                         const std::vector<size_t>& actual) {
    if (predicted.empty()) {
        return false;
    }
    if (actual == predicted) {
        return true;
    }

    // CompiledLayer::output_shape is per-sample, while DebugExecutor runs a
    // synthetic batch. Runtime tensors therefore often carry a leading batch
    // dimension: predicted [512,96] vs actual [1,512,96].
    if (actual.size() == predicted.size() + 1) {
        return std::equal(predicted.begin(), predicted.end(),
                          actual.begin() + 1);
    }

    return false;
}

const BuiltModuleProvenance* FindModuleProvenance(
    const std::vector<BuiltModuleProvenance>& provenance,
    size_t module_index) {
    for (const auto& entry : provenance) {
        if (entry.module_index && *entry.module_index == module_index) {
            return &entry;
        }
    }
    return nullptr;
}

struct ModuleParameterSummary {
    size_t tensor_count = 0;
    size_t element_count = 0;
    bool available = false;
    bool overflow = false;
};

ModuleParameterSummary SummarizeModuleParameters(Module* module) {
    ModuleParameterSummary summary;
    if (!module) {
        return summary;
    }

    try {
        const auto parameters = module->GetParameters();
        summary.available = true;
        summary.tensor_count = parameters.size();
        for (const auto& [name, tensor] : parameters) {
            (void)name;
            const size_t count = tensor.NumElements();
            if (count > std::numeric_limits<size_t>::max() -
                            summary.element_count) {
                summary.element_count = std::numeric_limits<size_t>::max();
                summary.overflow = true;
                break;
            }
            summary.element_count += count;
        }
    } catch (...) {
        // Parameter summaries are bounded debugger metadata. Failure to inspect
        // them must not change whether a correctly built model can execute.
        summary = {};
    }
    return summary;
}

DebugGraphTraceStep BuildModelConstructionTraceStep(
    const BuiltModuleProvenance& provenance,
    SequentialModel* model) {
    DebugGraphTraceStep step;
    step.node_id = provenance.node_id;
    step.node_name = !provenance.node_name.empty()
        ? provenance.node_name
        : (!provenance.module_name.empty()
            ? provenance.module_name
            : "Compiled layer " +
                std::to_string(provenance.compiled_layer_index));
    step.node_type = std::to_string(static_cast<int>(provenance.node_type));
    step.phase = "BuildModel";
    step.role = provenance.created()
        ? DebugTraceRole::CompileArtifact
        : DebugTraceRole::Warning;
    step.input_shape = provenance.input_shape;
    step.output_shape = provenance.output_shape;
    step.dtype = "model";
    step.backend = "not_observed";
    step.status = provenance.created() ? "ok" : "skipped";
    step.payload["trace_producer"] = "DebugExecutor";
    step.payload["diagnostic_phase"] = "local_debug_build_model";
    step.payload["component"] = "ModelBuilder";
    step.payload["source_file"] =
        "cyxwiz-engine/src/core/model_builder.cpp";
    step.payload["source_symbol"] =
        "cyxwiz::BuildSequentialFromConfig";
    step.payload["compiled_layer_index"] =
        provenance.compiled_layer_index;
    step.payload["module_created"] = provenance.created();
    step.payload["module_name"] = provenance.module_name;
    step.payload["configured_parameters"] =
        provenance.configured_parameters;

    if (provenance.module_index) {
        const size_t module_index = *provenance.module_index;
        step.payload["module_index"] = module_index;
        const ModuleParameterSummary parameters = SummarizeModuleParameters(
            model ? model->GetModule(module_index) : nullptr);
        step.payload["parameter_summary_available"] = parameters.available;
        step.payload["parameter_tensor_count"] = parameters.tensor_count;
        step.payload["parameter_numel"] = parameters.element_count;
        step.payload["parameter_numel_overflow"] = parameters.overflow;
    } else {
        step.payload["module_index"] = nullptr;
        step.payload["parameter_summary_available"] = false;
        step.warnings.push_back(
            "Compiled layer did not create a SequentialModel module.");
    }
    return step;
}

} // namespace

DebugExecutor::DebugExecutor(TrainingConfiguration config)
    : config_(std::move(config)) {}

DebugExecutor::~DebugExecutor() = default;

DebugResult DebugExecutor::Run() {
    DebugResult result;
    result.timestamp = std::chrono::steady_clock::now();
    result.reached = DebugStage::NotRun;

    try {
        // ---- Stage: BuildModel --------------------------------------------
        result.reached = DebugStage::BuildModel;
        spdlog::info("DebugExecutor: building model from config ({} layers)",
                     config_.layers.size());
        auto built = BuildSequentialFromConfig(config_);
        result.model_build_traces.reserve(built.module_provenance.size());
        for (const auto& provenance : built.module_provenance) {
            result.model_build_traces.push_back(
                BuildModelConstructionTraceStep(
                    provenance, built.model.get()));
        }
        if (!built.ok()) {
            result.failure_summary = built.error_message.empty()
                ? "Model build failed (no layers produced or invalid config)"
                : built.error_message;
            result.issues.push_back({IssueLevel::Error, -1, "",
                                     result.failure_summary,
                                     errors::Training::ModelBuildFailed});
            return result;
        }
        model_     = std::move(built.model);
        loss_      = std::move(built.loss);
        optimizer_ = std::move(built.optimizer);
        model_->SetTraining(true);
        const auto module_provenance = std::move(built.module_provenance);

        // ---- Synthetic batch ----------------------------------------------
        auto batch = MakeSyntheticBatch(config_, /*seed=*/1337);
        spdlog::info("DebugExecutor: synthetic batch features[{}] labels[{}]",
                     batch.features.NumElements(), batch.labels.NumElements());

        // ---- Stage: Forward (per-layer shape capture) ---------------------
        result.reached = DebugStage::Forward;
        Tensor current = batch.features.Clone();
        bool found_shape_mismatch = false;
        const BuiltModuleProvenance* previous_provenance = nullptr;

        auto fstart = std::chrono::steady_clock::now();
        for (size_t i = 0; i < model_->Size(); ++i) {
            Module* m = model_->GetModule(i);
            LayerTrace trace;
            trace.module_index = i;
            trace.module_name = m->GetName();
            trace.name = trace.module_name;
            const auto* provenance = FindModuleProvenance(
                module_provenance, i);
            if (provenance) {
                trace.compiled_layer_index =
                    provenance->compiled_layer_index;
                trace.type = provenance->node_type;
                trace.node_id = provenance->node_id;
                trace.name = provenance->node_name.empty()
                    ? trace.module_name
                    : provenance->node_name;
                trace.predicted_input_shape = provenance->input_shape;
                trace.predicted_shape = provenance->output_shape;
            }
            if (previous_provenance) {
                trace.upstream_node_id = previous_provenance->node_id;
                trace.upstream_node_name = previous_provenance->node_name;
            }

            trace.actual_input_shape = current.Shape();
            trace.input_shape_matches = ShapesMatchForTrace(
                trace.predicted_input_shape, trace.actual_input_shape);
            auto lstart = std::chrono::steady_clock::now();
            current = m->Forward(current);
            auto lend = std::chrono::steady_clock::now();
            trace.forward_ms =
                std::chrono::duration<float, std::milli>(lend - lstart).count();

            trace.actual_shape = current.Shape();
            trace.shape_matches =
                ShapesMatchForTrace(trace.predicted_shape, trace.actual_shape);

            const bool input_mismatch =
                !trace.predicted_input_shape.empty() &&
                !trace.input_shape_matches;
            const bool output_mismatch = !trace.predicted_shape.empty() &&
                !trace.shape_matches;
            if (input_mismatch && output_mismatch) {
                trace.shape_mismatch = ShapeMismatchKind::InputAndOutput;
            } else if (input_mismatch) {
                trace.shape_mismatch = ShapeMismatchKind::Input;
            } else if (output_mismatch) {
                trace.shape_mismatch = ShapeMismatchKind::Output;
            }
            if (trace.has_shape_mismatch() && !found_shape_mismatch) {
                trace.is_first_shape_mismatch = true;
                found_shape_mismatch = true;
            }

            auto [has_nan, has_inf] = ScanFinite(current);
            trace.has_nan = has_nan;
            trace.has_inf = has_inf;
            if (has_nan) {
                result.issues.push_back({IssueLevel::Error,
                                         trace.node_id, trace.name,
                                         "Forward output contains NaN",
                                         errors::Training::TrainingExecutionFailed});
            } else if (has_inf) {
                result.issues.push_back({IssueLevel::Error,
                                         trace.node_id, trace.name,
                                         "Forward output contains Inf",
                                         errors::Training::TrainingExecutionFailed});
            }

            result.layer_traces.push_back(std::move(trace));
            previous_provenance = provenance;
        }
        auto fend = std::chrono::steady_clock::now();
        result.forward_total_ms =
            std::chrono::duration<float, std::milli>(fend - fstart).count();

        // ---- Stage: Loss --------------------------------------------------
        result.reached = DebugStage::Loss;
        Tensor loss_tensor = loss_->Forward(current, batch.labels);
        result.loss_value = ExtractLossScalar(loss_tensor);
        result.loss_finite = std::isfinite(result.loss_value);
        if (!result.loss_finite) {
            result.failure_summary = "Loss is not finite";
            result.issues.push_back({IssueLevel::Error, -1, "",
                                     "Loss is not finite (value=" +
                                     std::to_string(result.loss_value) + ")",
                                     errors::Training::TrainingExecutionFailed});
            return result;
        }

        // ---- Stage: Backward ----------------------------------------------
        result.reached = DebugStage::Backward;
        auto bstart = std::chrono::steady_clock::now();
        Tensor loss_grad = loss_->Backward(current, batch.labels);
        model_->Backward(loss_grad);
        auto bend = std::chrono::steady_clock::now();
        result.backward_total_ms =
            std::chrono::duration<float, std::milli>(bend - bstart).count();

        // ---- Grad norms ---------------------------------------------------
        auto params = model_->GetParameters();
        auto grads  = model_->GetGradients();

        // Track which layers produced any grad so we can warn on
        // dead subgraphs (learnable layer with every grad == 0).
        std::map<int, bool> layer_any_nonzero;

        for (const auto& [param_name, param_tensor] : params) {
            GradNormEntry entry;
            entry.param_name = param_name;
            entry.parameter_l2_norm = L2Norm(param_tensor);

            // Parse layerN from "layerN.<field>"
            int layer_idx = -1;
            if (param_name.rfind("layer", 0) == 0) {
                size_t dot = param_name.find('.');
                if (dot != std::string::npos) {
                    try {
                        layer_idx = std::stoi(
                            param_name.substr(5, dot - 5));
                    } catch (...) { layer_idx = -1; }
                }
            }
            entry.layer_index = layer_idx;
            if (layer_idx >= 0) {
                if (const auto* provenance = FindModuleProvenance(
                        module_provenance,
                        static_cast<size_t>(layer_idx))) {
                    entry.compiled_layer_index =
                        provenance->compiled_layer_index;
                    entry.node_id = provenance->node_id;
                    entry.node_name = provenance->node_name;
                }
            }

            auto grad_it = grads.find(param_name);
            if (grad_it == grads.end()) {
                entry.has_gradient = false;
                entry.is_zero = true;
                entry.l2_norm = 0.0f;
                entry.missing_gradient_reason =
                    "No gradient tensor matched this trainable parameter "
                    "after backward.";
                ++result.params_missing_grad;
            } else {
                entry.has_gradient = true;
                const Tensor& g = grad_it->second;
                auto [gnan, ginf] = ScanFinite(g);
                entry.is_nan  = gnan;
                entry.l2_norm = L2Norm(g);
                entry.grad_parameter_ratio = RelativeNorm(
                    entry.l2_norm, entry.parameter_l2_norm);
                entry.is_zero = (entry.l2_norm == 0.0f);
                ++result.params_with_grad;
                if (gnan) {
                    result.issues.push_back({IssueLevel::Error, -1,
                                             param_name,
                                             "Gradient contains NaN",
                                             errors::Training::TrainingExecutionFailed});
                } else if (ginf) {
                    result.issues.push_back({IssueLevel::Error, -1,
                                             param_name,
                                             "Gradient contains Inf",
                                             errors::Training::TrainingExecutionFailed});
                }
                if (!entry.is_zero) {
                    layer_any_nonzero[layer_idx] = true;
                } else if (layer_any_nonzero.find(layer_idx) ==
                           layer_any_nonzero.end()) {
                    // Record as dead-so-far; later nonzero grads flip it.
                    layer_any_nonzero[layer_idx] = false;
                }
            }

            result.grad_norms.push_back(std::move(entry));
        }

        // Dead-subgraph detector: a learnable layer where every grad
        // came back zero is almost always a wiring bug (disconnected
        // subgraph or vanishing-at-init). Emit one Warning per dead
        // layer — grouping keeps the issue list readable.
        for (const auto& [layer_idx, any_nonzero] : layer_any_nonzero) {
            if (!any_nonzero) {
                const BuiltModuleProvenance* provenance = layer_idx >= 0
                    ? FindModuleProvenance(
                        module_provenance,
                        static_cast<size_t>(layer_idx))
                    : nullptr;
                result.issues.push_back({IssueLevel::Warning,
                                         provenance ? provenance->node_id : -1,
                                         provenance && !provenance->node_name.empty()
                                             ? provenance->node_name
                                             : "layer" + std::to_string(layer_idx),
                                         "All gradients are zero for this "
                                         "layer (possible dead subgraph or "
                                         "vanishing gradient at init)",
                                         errors::Training::TrainingExecutionFailed});
            }
        }

        // ---- Stage: OptimizerStep -----------------------------------------
        result.reached = DebugStage::OptimizerStep;
        const auto optimizer_start = std::chrono::steady_clock::now();
        model_->UpdateParameters(optimizer_.get());
        result.optimizer_step_ms =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - optimizer_start).count();

        const auto updated_params = model_->GetParameters();
        for (auto& entry : result.grad_norms) {
            const auto before = params.find(entry.param_name);
            const auto after = updated_params.find(entry.param_name);
            if (before == params.end() || after == updated_params.end() ||
                before->second.Shape() != after->second.Shape() ||
                before->second.GetDataType() != after->second.GetDataType()) {
                continue;
            }
            entry.update_l2_norm = L2Norm(after->second - before->second);
            entry.update_parameter_ratio = RelativeNorm(
                entry.update_l2_norm, entry.parameter_l2_norm);
            entry.update_observed = true;
        }

        // ---- Complete -----------------------------------------------------
        result.reached = DebugStage::Complete;

        // Success = reached Complete with zero Error-level issues.
        bool has_error = std::any_of(result.issues.begin(),
                                     result.issues.end(),
                                     [](const ValidationIssue& i) {
                                         return i.level == IssueLevel::Error;
                                     });
        result.success = !has_error;
        spdlog::info("DebugExecutor: reached={} success={} forward={:.2f}ms "
                     "backward={:.2f}ms loss={:.4f} "
                     "params_with_grad={} params_missing_grad={} issues={}",
                     StageName(result.reached), result.success,
                     result.forward_total_ms, result.backward_total_ms,
                     result.loss_value,
                     result.params_with_grad, result.params_missing_grad,
                     result.issues.size());
    } catch (const std::exception& e) {
        result.failure_summary =
            std::string("Exception during ") + StageName(result.reached) +
            ": " + e.what();
        result.issues.push_back({IssueLevel::Error, -1, "",
                                 result.failure_summary,
                                 errors::Training::TrainingExecutionFailed});
        spdlog::error("DebugExecutor: {}", result.failure_summary);
    } catch (...) {
        result.failure_summary =
            std::string("Unknown exception during ") +
            StageName(result.reached);
        result.issues.push_back({IssueLevel::Error, -1, "",
                                 result.failure_summary,
                                 errors::Training::TrainingExecutionFailed});
        spdlog::error("DebugExecutor: {}", result.failure_summary);
    }

    return result;
}

} // namespace cyxwiz
