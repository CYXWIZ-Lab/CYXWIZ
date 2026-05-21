#include "debug_executor.h"
#include "model_builder.h"
#include "synthetic_batch.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <utility>

namespace cyxwiz {

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
    const float* p = t.Data<float>();
    const size_t n = t.NumElements();
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(p[i])) { has_nan = true; }
        else if (std::isinf(p[i])) { has_inf = true; }
        if (has_nan && has_inf) break;
    }
    return {has_nan, has_inf};
}

// L2 norm of a Float32 tensor. Int tensors have no grad by construction;
// callers that see them should skip norm collection.
float L2Norm(const Tensor& t) {
    if (t.GetDataType() != DataType::Float32) return 0.0f;
    const float* p = t.Data<float>();
    const size_t n = t.NumElements();
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += static_cast<double>(p[i]) * p[i];
    return static_cast<float>(std::sqrt(acc));
}

// Extract a scalar from a reduction-output loss tensor. Loss forward
// returns a shape-[1] Float32 tensor after Mean / Sum reduction.
float ExtractLossScalar(const Tensor& t) {
    if (t.NumElements() == 0 || t.GetDataType() != DataType::Float32) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return t.Data<float>()[0];
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
        if (!built.ok()) {
            result.failure_summary = "Model build failed (no layers "
                                     "produced or invalid config)";
            result.issues.push_back({IssueLevel::Error, -1, "",
                                     result.failure_summary});
            return result;
        }
        model_     = std::move(built.model);
        loss_      = std::move(built.loss);
        optimizer_ = std::move(built.optimizer);
        model_->SetTraining(true);

        // ---- Synthetic batch ----------------------------------------------
        auto batch = MakeSyntheticBatch(config_, /*seed=*/1337);
        spdlog::info("DebugExecutor: synthetic batch features[{}] labels[{}]",
                     batch.features.NumElements(), batch.labels.NumElements());

        // ---- Stage: Forward (per-layer shape capture) ---------------------
        result.reached = DebugStage::Forward;
        Tensor current = batch.features.Clone();

        auto fstart = std::chrono::steady_clock::now();
        for (size_t i = 0; i < model_->Size(); ++i) {
            Module* m = model_->GetModule(i);
            LayerTrace trace;
            trace.name = m->GetName();
            // Pair the trace with the i-th config layer when possible.
            // The builder skips non-layer nodes (losses, optimizers,
            // preprocessing), so the config-layer index does not always
            // match the module index. Best-effort match by iterating.
            // Commit 2 v1 ships with config_.layers[i] as a best guess;
            // mismatches are cosmetic (the name / type fields may drift).
            if (i < config_.layers.size()) {
                trace.type    = config_.layers[i].type;
                trace.node_id = config_.layers[i].node_id;
                trace.predicted_shape = config_.layers[i].output_shape;
            }

            auto lstart = std::chrono::steady_clock::now();
            current = m->Forward(current);
            auto lend = std::chrono::steady_clock::now();
            trace.forward_ms =
                std::chrono::duration<float, std::milli>(lend - lstart).count();

            trace.actual_shape = current.Shape();
            trace.shape_matches =
                ShapesMatchForTrace(trace.predicted_shape, trace.actual_shape);

            auto [has_nan, has_inf] = ScanFinite(current);
            trace.has_nan = has_nan;
            trace.has_inf = has_inf;
            if (has_nan) {
                result.issues.push_back({IssueLevel::Error,
                                         trace.node_id, trace.name,
                                         "Forward output contains NaN"});
            } else if (has_inf) {
                result.issues.push_back({IssueLevel::Error,
                                         trace.node_id, trace.name,
                                         "Forward output contains Inf"});
            }

            result.layer_traces.push_back(std::move(trace));
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
                                     std::to_string(result.loss_value) + ")"});
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

            auto grad_it = grads.find(param_name);
            if (grad_it == grads.end()) {
                entry.is_zero = true;
                entry.l2_norm = 0.0f;
                ++result.params_missing_grad;
            } else {
                const Tensor& g = grad_it->second;
                auto [gnan, ginf] = ScanFinite(g);
                entry.is_nan  = gnan;
                entry.l2_norm = L2Norm(g);
                entry.is_zero = (entry.l2_norm == 0.0f);
                ++result.params_with_grad;
                if (gnan) {
                    result.issues.push_back({IssueLevel::Error, -1,
                                             param_name,
                                             "Gradient contains NaN"});
                } else if (ginf) {
                    result.issues.push_back({IssueLevel::Error, -1,
                                             param_name,
                                             "Gradient contains Inf"});
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
                result.issues.push_back({IssueLevel::Warning,
                                         layer_idx,
                                         "layer" + std::to_string(layer_idx),
                                         "All gradients are zero for this "
                                         "layer (possible dead subgraph or "
                                         "vanishing gradient at init)"});
            }
        }

        // ---- Stage: OptimizerStep -----------------------------------------
        result.reached = DebugStage::OptimizerStep;
        model_->UpdateParameters(optimizer_.get());

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
                                 result.failure_summary});
        spdlog::error("DebugExecutor: {}", result.failure_summary);
    } catch (...) {
        result.failure_summary =
            std::string("Unknown exception during ") +
            StageName(result.reached);
        result.issues.push_back({IssueLevel::Error, -1, "",
                                 result.failure_summary});
        spdlog::error("DebugExecutor: {}", result.failure_summary);
    }

    return result;
}

} // namespace cyxwiz
