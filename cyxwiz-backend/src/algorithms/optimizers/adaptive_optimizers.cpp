#include "cyxwiz/optimizers/adaptive.h"
#include "cyxwiz/tensor.h"
#include "../arrayfire_backend_utils.h"
#include "optimizer_utils.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

void ValidateNonNegativeFinite(
    double value,
    const char* optimizer_name,
    const char* field_name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string(optimizer_name) + " requires finite non-negative " +
            field_name);
    }
}

void ValidateUnitInterval(
    double value,
    const char* optimizer_name,
    const char* field_name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(optimizer_name) + " requires " + field_name +
            " in [0, 1]");
    }
}

bool ValidateAdaptiveStateHeader(
    const OptimizerState& state,
    const char* optimizer_type,
    const std::map<std::string, double>& expected_hyperparameters,
    std::string& error) {
    if (state.schema_version != 1) {
        error = std::string(optimizer_type) +
                " optimizer state schema version is unsupported.";
        return false;
    }
    if (state.optimizer_type != optimizer_type) {
        error = "Optimizer state type '" + state.optimizer_type +
                "' is incompatible with " + optimizer_type + ".";
        return false;
    }
    if (!std::isfinite(state.learning_rate) || state.learning_rate < 0.0) {
        error = std::string(optimizer_type) +
                " optimizer state has an invalid learning rate.";
        return false;
    }
    if (state.step_count < 0) {
        error = std::string(optimizer_type) +
                " optimizer state has a negative step count.";
        return false;
    }
    if (state.hyperparameters != expected_hyperparameters) {
        error = std::string(optimizer_type) +
                " optimizer state configuration does not match the active optimizer.";
        return false;
    }
    return true;
}

bool SplitAdaptiveStateTensors(
    const OptimizerState& state,
    const char* optimizer_type,
    const std::string& first_prefix,
    const std::string& second_prefix,
    bool require_second,
    std::map<std::string, Tensor>& first,
    std::map<std::string, Tensor>& second,
    std::string& error) {
    for (const auto& [key, tensor] : state.tensors) {
        std::string name;
        std::map<std::string, Tensor>* destination = nullptr;
        if (key.rfind(first_prefix, 0) == 0 && key.size() > first_prefix.size()) {
            name = key.substr(first_prefix.size());
            destination = &first;
        } else if (!second_prefix.empty() &&
                   key.rfind(second_prefix, 0) == 0 &&
                   key.size() > second_prefix.size()) {
            name = key.substr(second_prefix.size());
            destination = &second;
        } else {
            error = std::string(optimizer_type) +
                    " optimizer state contains unknown tensor '" + key + "'.";
            return false;
        }
        if (tensor.GetDataType() != DataType::Float32) {
            error = std::string(optimizer_type) + " optimizer state tensor '" +
                    key + "' must use Float32.";
            return false;
        }
        destination->emplace(std::move(name), tensor);
    }

    if (require_second) {
        if (first.size() != second.size()) {
            error = std::string(optimizer_type) +
                    " optimizer state contains incomplete tensor pairs.";
            return false;
        }
        for (const auto& [name, first_tensor] : first) {
            const auto second_it = second.find(name);
            if (second_it == second.end() ||
                second_it->second.Shape() != first_tensor.Shape()) {
                error = std::string(optimizer_type) +
                        " optimizer state contains incomplete or shape-mismatched tensor pairs.";
                return false;
            }
        }
    } else if (!second.empty()) {
        error = std::string(optimizer_type) +
                " optimizer state contains unsupported secondary tensors.";
        return false;
    }
    return true;
}

void ValidateAdaptiveRuntimeStateTensor(
    const Tensor& state,
    const Tensor& parameter,
    const char* optimizer_type,
    const std::string& parameter_name) {
    if (state.GetDataType() != DataType::Float32 ||
        state.Shape() != parameter.Shape()) {
        throw std::invalid_argument(
            std::string(optimizer_type) +
            " state does not match parameter '" + parameter_name + "'.");
    }
}

} // namespace

// ============================================================================
// RMSprop Optimizer
// ============================================================================

RMSpropOptimizer::RMSpropOptimizer(double learning_rate, double alpha, double epsilon, double momentum)
    : alpha_(alpha), epsilon_(epsilon), momentum_(momentum) {
    ValidateNonNegativeFinite(learning_rate, "RMSprop", "learning rate");
    ValidateUnitInterval(alpha_, "RMSprop", "alpha");
    ValidateNonNegativeFinite(epsilon_, "RMSprop", "epsilon");
    ValidateNonNegativeFinite(momentum_, "RMSprop", "momentum");
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void RMSpropOptimizer::Step(std::map<std::string, Tensor>& parameters,
                            const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "RMSpropOptimizer::Step";
    for (const auto& [name, param] : parameters) {
        const auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, param, grad_it->second);
        const auto average = v_.find(name);
        if (average != v_.end()) {
            ValidateAdaptiveRuntimeStateTensor(
                average->second, param, "RMSprop", name);
        }
        const auto buffer = buffer_.find(name);
        if (momentum_ > 0.0 &&
            ((average == v_.end()) != (buffer == buffer_.end()))) {
            throw std::invalid_argument(
                "RMSprop momentum state is incomplete for parameter '" + name +
                "'.");
        }
        if (buffer != buffer_.end()) {
            if (momentum_ <= 0.0) {
                throw std::invalid_argument(
                    "RMSprop without momentum contains a momentum buffer for parameter '" +
                    name + "'.");
            }
            ValidateAdaptiveRuntimeStateTensor(
                buffer->second, param, "RMSprop", name);
        }
    }

    float lr = static_cast<float>(learning_rate_);
    float alpha = static_cast<float>(alpha_);
    float eps = static_cast<float>(epsilon_);
    float mom = static_cast<float>(momentum_);
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

        // Initialize running average only after fallback policy authorizes work.
        if (v_.find(name) == v_.end()) {
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            if (momentum_ > 0) {
                buffer_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            }
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetArray();
                af::array grad_gpu = grad.GetArray();
                af::array v_gpu = v_[name].GetArray();

                // v = alpha * v + (1 - alpha) * grad^2
                v_gpu = alpha * v_gpu + (1.0f - alpha) * grad_gpu * grad_gpu;
                v_gpu.eval();

                if (momentum_ > 0) {
                    af::array buf_gpu = buffer_[name].GetArray();
                    // buf = mom * buf + grad / sqrt(v + eps)
                    buf_gpu = mom * buf_gpu + grad_gpu / (af::sqrt(v_gpu) + eps);
                    buf_gpu.eval();
                    param_gpu = param_gpu - lr * buf_gpu;
                    param_gpu.eval();
                    buffer_[name].SetFromArray(buf_gpu);
                } else {
                    param_gpu = param_gpu - lr * grad_gpu / (af::sqrt(v_gpu) + eps);
                    param_gpu.eval();
                }

                param.SetFromArray(param_gpu);
                v_[name].SetFromArray(v_gpu);
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
        float* v_data = v_[name].MutableData<float>();

        if (momentum_ > 0) {
            float* buf_data = buffer_[name].MutableData<float>();
            for (size_t i = 0; i < num_elements; ++i) {
                v_data[i] = alpha * v_data[i] + (1.0f - alpha) * grad_data[i] * grad_data[i];
                buf_data[i] = mom * buf_data[i] + grad_data[i] / (std::sqrt(v_data[i]) + eps);
                param_data[i] -= lr * buf_data[i];
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                v_data[i] = alpha * v_data[i] + (1.0f - alpha) * grad_data[i] * grad_data[i];
                param_data[i] -= lr * grad_data[i] / (std::sqrt(v_data[i]) + eps);
            }
        }
    }
    step_count_++;
}

void RMSpropOptimizer::ZeroGrad() {
    // Gradients are caller-owned; running averages and momentum are persistent
    // optimizer state and must survive zero_grad between steps.
}

bool RMSpropOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const {
    OptimizerState exported;
    exported.optimizer_type = "RMSprop";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {
        {"alpha", alpha_},
        {"epsilon", epsilon_},
        {"momentum", momentum_},
    };
    for (const auto& [name, average] : v_) {
        exported.tensors.emplace("square_average/" + name, average);
    }
    for (const auto& [name, buffer] : buffer_) {
        exported.tensors.emplace("momentum_buffer/" + name, buffer);
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool RMSpropOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error) {
    if (!ValidateAdaptiveStateHeader(
            state,
            "RMSprop",
            {{"alpha", alpha_}, {"epsilon", epsilon_}, {"momentum", momentum_}},
            error)) {
        return false;
    }
    std::map<std::string, Tensor> imported_average;
    std::map<std::string, Tensor> imported_buffer;
    if (!SplitAdaptiveStateTensors(
            state,
            "RMSprop",
            "square_average/",
            "momentum_buffer/",
            momentum_ > 0.0,
            imported_average,
            imported_buffer,
            error)) {
        return false;
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    v_ = std::move(imported_average);
    buffer_ = std::move(imported_buffer);
    error.clear();
    return true;
}

// ============================================================================
// AdaGrad Optimizer
// ============================================================================

AdaGradOptimizer::AdaGradOptimizer(double learning_rate, double epsilon)
    : epsilon_(epsilon) {
    ValidateNonNegativeFinite(learning_rate, "AdaGrad", "learning rate");
    ValidateNonNegativeFinite(epsilon_, "AdaGrad", "epsilon");
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void AdaGradOptimizer::Step(std::map<std::string, Tensor>& parameters,
                            const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "AdaGradOptimizer::Step";
    for (const auto& [name, param] : parameters) {
        const auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, param, grad_it->second);
        const auto cache = cache_.find(name);
        if (cache != cache_.end()) {
            ValidateAdaptiveRuntimeStateTensor(
                cache->second, param, "AdaGrad", name);
        }
    }

    float lr = static_cast<float>(learning_rate_);
    float eps = static_cast<float>(epsilon_);
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

        // Initialize cache only after fallback policy authorizes work.
        if (cache_.find(name) == cache_.end()) {
            cache_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetArray();
                af::array grad_gpu = grad.GetArray();
                af::array cache_gpu = cache_[name].GetArray();

                // cache += grad^2
                cache_gpu = cache_gpu + grad_gpu * grad_gpu;
                cache_gpu.eval();
                // param -= lr * grad / sqrt(cache + eps)
                param_gpu = param_gpu - lr * grad_gpu / (af::sqrt(cache_gpu) + eps);
                param_gpu.eval();

                param.SetFromArray(param_gpu);
                cache_[name].SetFromArray(cache_gpu);
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
        float* cache_data = cache_[name].MutableData<float>();

        for (size_t i = 0; i < num_elements; ++i) {
            cache_data[i] += grad_data[i] * grad_data[i];
            param_data[i] -= lr * grad_data[i] / (std::sqrt(cache_data[i]) + eps);
        }
    }
    step_count_++;
}

void AdaGradOptimizer::ZeroGrad() {
    // Gradients are caller-owned; accumulated squared gradients persist.
}

bool AdaGradOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const {
    OptimizerState exported;
    exported.optimizer_type = "AdaGrad";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {{"epsilon", epsilon_}};
    for (const auto& [name, cache] : cache_) {
        exported.tensors.emplace("sum/" + name, cache);
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool AdaGradOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error) {
    if (!ValidateAdaptiveStateHeader(
            state,
            "AdaGrad",
            {{"epsilon", epsilon_}},
            error)) {
        return false;
    }
    std::map<std::string, Tensor> imported_cache;
    std::map<std::string, Tensor> unused;
    if (!SplitAdaptiveStateTensors(
            state,
            "AdaGrad",
            "sum/",
            {},
            false,
            imported_cache,
            unused,
            error)) {
        return false;
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    cache_ = std::move(imported_cache);
    error.clear();
    return true;
}

// ============================================================================
// Adadelta Optimizer
// ============================================================================

AdadeltaOptimizer::AdadeltaOptimizer(double rho, double epsilon)
    : rho_(rho), epsilon_(epsilon) {
    ValidateUnitInterval(rho_, "Adadelta", "rho");
    ValidateNonNegativeFinite(epsilon_, "Adadelta", "epsilon");
    // Preserve the public constructor's historical default while honoring the
    // inherited learning-rate contract during updates and checkpoint resume.
    learning_rate_ = 1.0;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void AdadeltaOptimizer::Step(std::map<std::string, Tensor>& parameters,
                              const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "AdadeltaOptimizer::Step";
    for (const auto& [name, param] : parameters) {
        const auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, param, grad_it->second);
        const auto average = acc_grad_.find(name);
        const auto delta = acc_delta_.find(name);
        if ((average == acc_grad_.end()) != (delta == acc_delta_.end())) {
            throw std::invalid_argument(
                "Adadelta state is incomplete for parameter '" + name + "'.");
        }
        if (average != acc_grad_.end()) {
            ValidateAdaptiveRuntimeStateTensor(
                average->second, param, "Adadelta", name);
            ValidateAdaptiveRuntimeStateTensor(
                delta->second, param, "Adadelta", name);
        }
    }

    float rho = static_cast<float>(rho_);
    float eps = static_cast<float>(epsilon_);
    float lr = static_cast<float>(learning_rate_);
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

        // Initialize accumulators only after fallback policy authorizes work.
        if (acc_grad_.find(name) == acc_grad_.end()) {
            acc_grad_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            acc_delta_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetArray();
                af::array grad_gpu = grad.GetArray();
                af::array acc_grad_gpu = acc_grad_[name].GetArray();
                af::array acc_delta_gpu = acc_delta_[name].GetArray();

                // Accumulate squared gradient.
                acc_grad_gpu = rho * acc_grad_gpu + (1.0f - rho) * grad_gpu * grad_gpu;
                acc_grad_gpu.eval();

                // Compute update.
                af::array delta = -af::sqrt(acc_delta_gpu + eps) / af::sqrt(acc_grad_gpu + eps) * grad_gpu;
                delta.eval();

                // Accumulate squared update.
                acc_delta_gpu = rho * acc_delta_gpu + (1.0f - rho) * delta * delta;
                acc_delta_gpu.eval();

                // Apply update.
                param_gpu = param_gpu + lr * delta;
                param_gpu.eval();

                param.SetFromArray(param_gpu);
                acc_grad_[name].SetFromArray(acc_grad_gpu);
                acc_delta_[name].SetFromArray(acc_delta_gpu);
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
        float* acc_grad_data = acc_grad_[name].MutableData<float>();
        float* acc_delta_data = acc_delta_[name].MutableData<float>();

        for (size_t i = 0; i < num_elements; ++i) {
            // Accumulate squared gradient
            acc_grad_data[i] = rho * acc_grad_data[i] + (1.0f - rho) * grad_data[i] * grad_data[i];

            // Compute update
            float delta = -std::sqrt(acc_delta_data[i] + eps) / std::sqrt(acc_grad_data[i] + eps) * grad_data[i];

            // Accumulate squared update
            acc_delta_data[i] = rho * acc_delta_data[i] + (1.0f - rho) * delta * delta;

            // Apply update
            param_data[i] += lr * delta;
        }
    }
    step_count_++;
}

void AdadeltaOptimizer::ZeroGrad() {
    // Gradients are caller-owned; both running accumulators persist.
}

bool AdadeltaOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const {
    OptimizerState exported;
    exported.optimizer_type = "Adadelta";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {
        {"epsilon", epsilon_},
        {"rho", rho_},
    };
    for (const auto& [name, average] : acc_grad_) {
        exported.tensors.emplace("square_average/" + name, average);
    }
    for (const auto& [name, delta] : acc_delta_) {
        exported.tensors.emplace("accumulated_delta/" + name, delta);
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool AdadeltaOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error) {
    if (!ValidateAdaptiveStateHeader(
            state,
            "Adadelta",
            {{"epsilon", epsilon_}, {"rho", rho_}},
            error)) {
        return false;
    }
    std::map<std::string, Tensor> imported_average;
    std::map<std::string, Tensor> imported_delta;
    if (!SplitAdaptiveStateTensors(
            state,
            "Adadelta",
            "square_average/",
            "accumulated_delta/",
            true,
            imported_average,
            imported_delta,
            error)) {
        return false;
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    acc_grad_ = std::move(imported_average);
    acc_delta_ = std::move(imported_delta);
    error.clear();
    return true;
}

} // namespace cyxwiz
