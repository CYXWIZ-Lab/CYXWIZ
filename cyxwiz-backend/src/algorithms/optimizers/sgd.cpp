#include "cyxwiz/optimizers/sgd.h"
#include "cyxwiz/tensor.h"
#include "../arrayfire_backend_utils.h"
#include "optimizer_utils.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// SGD Optimizer
// ============================================================================

SGDOptimizer::SGDOptimizer(double learning_rate, double momentum)
    : momentum_(momentum) {
    if (!std::isfinite(learning_rate) || learning_rate < 0.0) {
        throw std::invalid_argument(
            "SGD optimizer requires a finite non-negative learning rate");
    }
    if (!std::isfinite(momentum_) || momentum_ < 0.0) {
        throw std::invalid_argument(
            "SGD optimizer requires finite non-negative momentum");
    }
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void SGDOptimizer::Step(std::map<std::string, Tensor>& parameters,
                        const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "SGDOptimizer::Step";
    for (const auto& [name, param] : parameters) {
        const auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, param, grad_it->second);
        const auto velocity = velocity_.find(name);
        if (velocity != velocity_.end() &&
            velocity->second.Shape() != param.Shape()) {
            throw std::invalid_argument(
                "SGD momentum state shape does not match parameter '" + name +
                "'.");
        }
    }

    const bool arrayfire_available =
        optimizer_detail::OptimizerArrayFireAvailable();
    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        const bool use_native_cpu =
            optimizer_detail::PrepareOptimizerNativeCpuFallback(
                kOperation, name, param, arrayfire_available);

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                const std::vector<size_t> parameter_shape = param.Shape();
                af::array param_gpu = param.GetSemanticArray();
                af::array grad_gpu = grad.GetSemanticArray();

                if (momentum_ > 0.0) {
                    // Initialize velocity if needed
                    if (velocity_.find(name) == velocity_.end()) {
                        velocity_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
                    }

                    const std::vector<size_t> velocity_shape =
                        velocity_[name].Shape();
                    af::array v_gpu = velocity_[name].GetSemanticArray();

                    // v = momentum * v + grad
                    v_gpu = static_cast<float>(momentum_) * v_gpu + grad_gpu;
                    // param = param - lr * v
                    param_gpu = param_gpu - static_cast<float>(learning_rate_) * v_gpu;
                    v_gpu.eval();
                    param_gpu.eval();

                    velocity_[name].SetFromSemanticArray(
                        v_gpu, velocity_shape);
                } else {
                    // Simple SGD: param = param - lr * grad
                    param_gpu = param_gpu - static_cast<float>(learning_rate_) * grad_gpu;
                    param_gpu.eval();
                }

                param.SetFromSemanticArray(param_gpu, parameter_shape);
                continue;
            } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    kOperation, name, param, e.what());
            }
        }
#endif

        // CPU fallback
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::OptimizerCpuPath,
            kOperation);
        float* param_data = param.MutableData<float>();
        const float* grad_data = grad.ReadData<float>();
        if (momentum_ > 0.0) {
            if (velocity_.find(name) == velocity_.end()) {
                velocity_[name] =
                    Tensor::Zeros(param.Shape(), DataType::Float32);
            }
            float* velocity_data = velocity_.at(name).MutableData<float>();
            for (size_t i = 0; i < num_elements; i++) {
                velocity_data[i] =
                    static_cast<float>(momentum_) * velocity_data[i] +
                    grad_data[i];
                param_data[i] -= static_cast<float>(learning_rate_) *
                                 velocity_data[i];
            }
        } else {
            for (size_t i = 0; i < num_elements; i++) {
                param_data[i] -= static_cast<float>(learning_rate_) *
                                 grad_data[i];
            }
        }
    }

    step_count_++;
}

void SGDOptimizer::ZeroGrad() {
    // Gradients are caller-owned input maps, so there is no optimizer-owned
    // gradient storage to clear. Momentum is persistent optimizer state.
}

bool SGDOptimizer::ExportState(OptimizerState& state, std::string& error) const {
    if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0 ||
        step_count_ < 0) {
        error = "SGD optimizer contains invalid runtime state.";
        return false;
    }

    OptimizerState exported;
    exported.optimizer_type = "SGD";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {{"momentum", momentum_}};
    for (const auto& [name, velocity] : velocity_) {
        exported.tensors.emplace("velocity/" + name, velocity);
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool SGDOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error)
{
    constexpr const char* kVelocityPrefix = "velocity/";
    constexpr std::size_t kVelocityPrefixLength = 9;

    if (state.schema_version != 1) {
        error = "SGD optimizer state schema version is unsupported.";
        return false;
    }
    if (state.optimizer_type != "SGD") {
        error = "Optimizer state type '" + state.optimizer_type +
                "' is incompatible with SGD.";
        return false;
    }
    if (!std::isfinite(state.learning_rate) || state.learning_rate < 0.0) {
        error = "SGD optimizer state has an invalid learning rate.";
        return false;
    }
    if (state.step_count < 0) {
        error = "SGD optimizer state has a negative step count.";
        return false;
    }
    if (state.hyperparameters !=
        std::map<std::string, double>{{"momentum", momentum_}}) {
        error =
            "SGD optimizer state configuration does not match the active "
            "optimizer.";
        return false;
    }

    std::map<std::string, Tensor> imported_velocity;
    for (const auto& [key, tensor] : state.tensors) {
        if (key.rfind(kVelocityPrefix, 0) != 0 ||
            key.size() == kVelocityPrefixLength) {
            error = "SGD optimizer state contains unknown tensor '" + key +
                    "'.";
            return false;
        }
        if (momentum_ <= 0.0) {
            error = "SGD optimizer without momentum cannot restore velocity.";
            return false;
        }
        if (tensor.GetDataType() != DataType::Float32) {
            error = "SGD optimizer state tensor '" + key +
                    "' must use Float32.";
            return false;
        }
        imported_velocity.emplace(
            key.substr(kVelocityPrefixLength), tensor);
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    velocity_ = std::move(imported_velocity);
    error.clear();
    return true;
}

} // namespace cyxwiz

