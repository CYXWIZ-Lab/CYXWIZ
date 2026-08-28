#include "cyxwiz/optimizers/adam.h"
#include "cyxwiz/tensor.h"
#include "../arrayfire_backend_utils.h"
#include "optimizer_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

constexpr float kNAdamMomentumDecay = 0.004f;

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

void ValidateBeta(
    double value,
    const char* optimizer_name,
    const char* field_name) {
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        throw std::invalid_argument(
            std::string(optimizer_name) + " requires " + field_name +
            " in [0, 1)");
    }
}

void ValidateAdamFamilyHyperparameters(
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    const char* optimizer_name) {
    ValidateNonNegativeFinite(
        learning_rate, optimizer_name, "learning rate");
    ValidateBeta(beta1, optimizer_name, "beta1");
    ValidateBeta(beta2, optimizer_name, "beta2");
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument(
            std::string(optimizer_name) + " requires finite positive epsilon");
    }
}

void ValidateRuntimeMoment(
    const Tensor& moment,
    const Tensor& parameter,
    const char* optimizer_name,
    const std::string& parameter_name) {
    if (moment.GetDataType() != DataType::Float32 ||
        moment.Shape() != parameter.Shape()) {
        throw std::invalid_argument(
            std::string(optimizer_name) + " state does not match parameter '" +
            parameter_name + "'.");
    }
}

Tensor MakeStepTensor(int step) {
    const int32_t value = static_cast<int32_t>(step);
    return Tensor({1}, &value, DataType::Int32);
}

Tensor MakeScalarTensor(float value) {
    return Tensor({1}, &value, DataType::Float32);
}

bool ReadStepTensor(
    const Tensor& tensor,
    const std::string& key,
    int global_step,
    int& step,
    std::string& error) {
    if (tensor.GetDataType() != DataType::Int32 ||
        tensor.NumElements() != 1) {
        error = "Optimizer state tensor '" + key +
                "' must be one Int32 scalar.";
        return false;
    }
    const int32_t value = tensor.ReadData<int32_t>()[0];
    if (value < 0 || value > global_step) {
        error = "Optimizer state tensor '" + key +
                "' contains an invalid parameter step.";
        return false;
    }
    step = static_cast<int>(value);
    return true;
}

bool ReadMuProductTensor(
    const Tensor& tensor,
    const std::string& key,
    float& mu_product,
    std::string& error) {
    if (tensor.GetDataType() != DataType::Float32 ||
        tensor.NumElements() != 1) {
        error = "Optimizer state tensor '" + key +
                "' must be one Float32 scalar.";
        return false;
    }
    const float value = tensor.ReadData<float>()[0];
    if (!std::isfinite(value) || value <= 0.0f || value > 1.0f) {
        error = "Optimizer state tensor '" + key +
                "' contains an invalid momentum product.";
        return false;
    }
    mu_product = value;
    return true;
}

bool ValidateStateHeader(
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

} // namespace

// ============================================================================
// Adam Optimizer
// ============================================================================

AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    ValidateAdamFamilyHyperparameters(
        learning_rate, beta1_, beta2_, epsilon_, "Adam");
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void AdamOptimizer::Step(std::map<std::string, Tensor>& parameters,
                         const std::map<std::string, Tensor>& gradients) {
    StepImpl(parameters, gradients, "AdamOptimizer::Step", 0.0);
}

void AdamOptimizer::StepImpl(
    std::map<std::string, Tensor>& parameters,
    const std::map<std::string, Tensor>& gradients,
    const char* operation_name,
    double weight_decay) {
    for (const auto& [name, parameter] : parameters) {
        const auto gradient = gradients.find(name);
        if (gradient == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            operation_name, name, parameter, gradient->second);
        const auto first = m_.find(name);
        const auto second = v_.find(name);
        const auto parameter_step = parameter_steps_.find(name);
        const bool has_first = first != m_.end();
        if (has_first != (second != v_.end()) ||
            has_first != (parameter_step != parameter_steps_.end())) {
            throw std::invalid_argument(
                std::string(operation_name) +
                " has incomplete state for parameter '" + name + "'.");
        }
        if (has_first) {
            ValidateRuntimeMoment(first->second, parameter, "Adam", name);
            ValidateRuntimeMoment(second->second, parameter, "Adam", name);
            if (parameter_step->second < 0 ||
                parameter_step->second == (std::numeric_limits<int>::max)()) {
                throw std::invalid_argument(
                    std::string(operation_name) +
                    " has an invalid parameter step for '" + name + "'.");
            }
        }
    }
    if (step_count_ == (std::numeric_limits<int>::max)()) {
        throw std::overflow_error(
            std::string(operation_name) + " update count overflow");
    }

    const float b1 = static_cast<float>(beta1_);
    const float b2 = static_cast<float>(beta2_);
    const float lr = static_cast<float>(learning_rate_);
    const float eps = static_cast<float>(epsilon_);
    const float wd = static_cast<float>(weight_decay);
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
                operation_name, name, param, arrayfire_available);

        // Initialize state only after fallback policy authorizes work.
        if (m_.find(name) == m_.end()) {
            m_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            parameter_steps_[name] = 0;
        }

        const int parameter_step = parameter_steps_.at(name) + 1;
        const float bias_correction1 =
            1.0f - static_cast<float>(std::pow(b1, parameter_step));
        const float bias_correction2 =
            1.0f - static_cast<float>(std::pow(b2, parameter_step));

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetSemanticArray();
                af::array grad_gpu = grad.GetSemanticArray();
                af::array m_gpu = m_[name].GetSemanticArray();
                af::array v_gpu = v_[name].GetSemanticArray();

                if (wd > 0.0f) {
                    param_gpu = param_gpu * (1.0f - lr * wd);
                    param_gpu.eval();
                }

                // Update biased first moment estimate: m = b1 * m + (1 - b1) * grad
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                m_gpu.eval();

                // Update biased second moment estimate: v = b2 * v + (1 - b2) * grad^2
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;
                v_gpu.eval();

                // Compute bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                m_hat.eval();
                af::array v_hat = v_gpu / bias_correction2;
                v_hat.eval();

                // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
                param_gpu = param_gpu - lr * m_hat / (af::sqrt(v_hat) + eps);
                param_gpu.eval();

                param.SetFromSemanticArray(param_gpu, param.Shape());
                m_[name].SetFromSemanticArray(m_gpu, m_[name].Shape());
                v_[name].SetFromSemanticArray(v_gpu, v_[name].Shape());
                parameter_steps_[name] = parameter_step;
                continue;
            } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    operation_name, name, param, e.what());
            }
        }
#endif

        // CPU fallback
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::OptimizerCpuPath,
            operation_name);
        float* param_data = param.MutableData<float>();
        const float* grad_data = grad.ReadData<float>();
        float* m_data = m_[name].MutableData<float>();
        float* v_data = v_[name].MutableData<float>();

        for (size_t i = 0; i < num_elements; ++i) {
            if (wd > 0.0f) {
                param_data[i] *= (1.0f - lr * wd);
            }
            // Update biased first moment estimate
            m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];

            // Update biased second raw moment estimate
            v_data[i] = b2 * v_data[i] +
                        (1.0f - b2) * grad_data[i] * grad_data[i];

            // Compute bias-corrected estimates
            float m_hat = m_data[i] / bias_correction1;
            float v_hat = v_data[i] / bias_correction2;

            // Update parameters
            param_data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
        parameter_steps_[name] = parameter_step;
    }
    ++step_count_;
}

void AdamOptimizer::ZeroGrad() {
    // Gradients are caller-owned; moment and parameter-step state persists.
}

std::map<std::string, double> AdamOptimizer::AdamHyperparameters() const {
    return {
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"epsilon", epsilon_},
    };
}

bool AdamOptimizer::ExportState(OptimizerState& state, std::string& error) const {
    error.clear();
    state = OptimizerState{};
    state.optimizer_type = "Adam";
    state.learning_rate = learning_rate_;
    state.step_count = step_count_;
    state.hyperparameters = AdamHyperparameters();

    for (const auto& [name, tensor] : m_) {
        state.tensors.emplace("first_moment/" + name, tensor);
    }
    for (const auto& [name, tensor] : v_) {
        state.tensors.emplace("second_moment/" + name, tensor);
    }
    for (const auto& [name, step] : parameter_steps_) {
        state.tensors.emplace("parameter_step/" + name, MakeStepTensor(step));
    }
    return true;
}

bool AdamOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error)
{
    constexpr const char* kFirstPrefix = "first_moment/";
    constexpr const char* kSecondPrefix = "second_moment/";
    constexpr const char* kStepPrefix = "parameter_step/";

    if (!ValidateStateHeader(
            state,
            "Adam",
            AdamHyperparameters(),
            error)) {
        return false;
    }

    std::map<std::string, Tensor> imported_m;
    std::map<std::string, Tensor> imported_v;
    std::map<std::string, int> imported_steps;
    const ScopedArrayFireHostSyncAttribution checkpoint_attribution(
        ArrayFireHostSyncCategory::CheckpointOutput,
        "AdamOptimizer::ImportState");
    for (const auto& [key, tensor] : state.tensors) {
        std::string parameter_name;
        std::map<std::string, Tensor>* destination = nullptr;
        if (key.rfind(kFirstPrefix, 0) == 0) {
            parameter_name = key.substr(std::char_traits<char>::length(kFirstPrefix));
            destination = &imported_m;
        } else if (key.rfind(kSecondPrefix, 0) == 0) {
            parameter_name = key.substr(std::char_traits<char>::length(kSecondPrefix));
            destination = &imported_v;
        } else if (key.rfind(kStepPrefix, 0) == 0) {
            parameter_name = key.substr(std::char_traits<char>::length(kStepPrefix));
            if (parameter_name.empty()) {
                error = "Adam optimizer state contains an empty parameter name.";
                return false;
            }
            int parameter_step = 0;
            if (!ReadStepTensor(
                    tensor, key, state.step_count, parameter_step, error)) {
                return false;
            }
            if (!imported_steps.emplace(parameter_name, parameter_step).second) {
                error = "Adam optimizer state contains duplicate parameter step '" +
                        parameter_name + "'.";
                return false;
            }
            continue;
        } else {
            error = "Adam optimizer state contains unknown tensor '" + key + "'.";
            return false;
        }
        if (parameter_name.empty()) {
            error = "Adam optimizer state contains an empty parameter name.";
            return false;
        }
        if (tensor.GetDataType() != DataType::Float32) {
            error = "Adam optimizer state tensor '" + key +
                    "' must use Float32.";
            return false;
        }
        if (!destination->emplace(parameter_name, tensor).second) {
            error = "Adam optimizer state contains duplicate parameter '" +
                    parameter_name + "'.";
            return false;
        }
    }

    if (imported_m.size() != imported_v.size()) {
        error = "Adam optimizer state has incomplete moment tensor pairs.";
        return false;
    }
    for (const auto& [name, first_moment] : imported_m) {
        const auto second = imported_v.find(name);
        if (second == imported_v.end()) {
            error = "Adam optimizer state is missing the second moment for '" +
                    name + "'.";
            return false;
        }
        if (first_moment.Shape() != second->second.Shape()) {
            error = "Adam optimizer moment shape mismatch for '" + name + "'.";
            return false;
        }
    }
    if (imported_steps.empty()) {
        // Schema-v1 checkpoints written before per-parameter counters used the
        // global update count for every moment pair.
        for (const auto& [name, tensor] : imported_m) {
            (void)tensor;
            imported_steps.emplace(name, state.step_count);
        }
    } else if (imported_steps.size() != imported_m.size()) {
        error = "Adam optimizer state has incomplete parameter-step state.";
        return false;
    }
    for (const auto& [name, step] : imported_steps) {
        (void)step;
        if (imported_m.find(name) == imported_m.end()) {
            error = "Adam optimizer state has a parameter step without moments for '" +
                    name + "'.";
            return false;
        }
    }

    // Commit only after every field and tensor pair has been validated.
    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    m_ = std::move(imported_m);
    v_ = std::move(imported_v);
    parameter_steps_ = std::move(imported_steps);
    error.clear();
    return true;
}

// ============================================================================
// AdamW Optimizer
// ============================================================================

AdamWOptimizer::AdamWOptimizer(double learning_rate, double beta1, double beta2,
                               double epsilon, double weight_decay)
    : AdamOptimizer(learning_rate, beta1, beta2, epsilon), weight_decay_(weight_decay) {
    ValidateNonNegativeFinite(weight_decay_, "AdamW", "weight decay");
}

void AdamWOptimizer::Step(std::map<std::string, Tensor>& parameters,
                          const std::map<std::string, Tensor>& gradients) {
    StepImpl(parameters, gradients, "AdamWOptimizer::Step", weight_decay_);
}

bool AdamWOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const
{
    if (!AdamOptimizer::ExportState(state, error)) {
        return false;
    }
    state.optimizer_type = "AdamW";
    state.hyperparameters.emplace("weight_decay", weight_decay_);
    return true;
}

bool AdamWOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error)
{
    auto expected_hyperparameters = AdamHyperparameters();
    expected_hyperparameters.emplace("weight_decay", weight_decay_);
    if (!ValidateStateHeader(
            state, "AdamW", expected_hyperparameters, error)) {
        return false;
    }

    OptimizerState normalized = state;
    normalized.optimizer_type = "Adam";
    normalized.hyperparameters.erase("weight_decay");
    return AdamOptimizer::ImportState(normalized, error);
}

// ============================================================================
// NAdam Optimizer (Nesterov-accelerated Adam)
// ============================================================================

NAdamOptimizer::NAdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    ValidateAdamFamilyHyperparameters(
        learning_rate, beta1_, beta2_, epsilon_, "NAdam");
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void NAdamOptimizer::Step(std::map<std::string, Tensor>& parameters,
                          const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "NAdamOptimizer::Step";
    for (const auto& [name, parameter] : parameters) {
        const auto gradient = gradients.find(name);
        if (gradient == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, parameter, gradient->second);
        const auto first = m_.find(name);
        const auto second = v_.find(name);
        const auto parameter_step = parameter_steps_.find(name);
        const auto mu_product = mu_products_.find(name);
        const bool has_first = first != m_.end();
        if (has_first != (second != v_.end()) ||
            has_first != (parameter_step != parameter_steps_.end()) ||
            has_first != (mu_product != mu_products_.end())) {
            throw std::invalid_argument(
                "NAdam has incomplete state for parameter '" + name + "'.");
        }
        if (has_first) {
            ValidateRuntimeMoment(first->second, parameter, "NAdam", name);
            ValidateRuntimeMoment(second->second, parameter, "NAdam", name);
            if (parameter_step->second < 0 ||
                parameter_step->second == (std::numeric_limits<int>::max)() ||
                !std::isfinite(mu_product->second) ||
                mu_product->second <= 0.0f || mu_product->second > 1.0f) {
                throw std::invalid_argument(
                    "NAdam has invalid continuation state for parameter '" +
                    name + "'.");
            }
        }
    }
    if (step_count_ == (std::numeric_limits<int>::max)()) {
        throw std::overflow_error("NAdamOptimizer::Step update count overflow");
    }

    const float b1 = static_cast<float>(beta1_);
    const float b2 = static_cast<float>(beta2_);
    const float lr = static_cast<float>(learning_rate_);
    const float eps = static_cast<float>(epsilon_);
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

        // Initialize state only after fallback policy authorizes work.
        if (m_.find(name) == m_.end()) {
            m_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            parameter_steps_[name] = 0;
            mu_products_[name] = 1.0f;
        }

        const int parameter_step = parameter_steps_.at(name) + 1;
        const float bias_correction2 =
            1.0f - static_cast<float>(std::pow(b2, parameter_step));
        const float mu = b1 *
            (1.0f - 0.5f * static_cast<float>(std::pow(
                0.96f, parameter_step * kNAdamMomentumDecay)));
        const float mu_next = b1 *
            (1.0f - 0.5f * static_cast<float>(std::pow(
                0.96f, (parameter_step + 1) * kNAdamMomentumDecay)));
        const float mu_product = mu_products_.at(name) * mu;
        const float mu_product_next = mu_product * mu_next;

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetSemanticArray();
                af::array grad_gpu = grad.GetSemanticArray();
                af::array m_gpu = m_[name].GetSemanticArray();
                af::array v_gpu = v_[name].GetSemanticArray();

                // Update moments
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                m_gpu.eval();
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;
                v_gpu.eval();

                af::array denominator =
                    af::sqrt(v_gpu / bias_correction2) + eps;
                denominator.eval();
                af::array m_nesterov =
                    ((1.0f - mu) / (1.0f - mu_product)) * grad_gpu +
                    (mu_next / (1.0f - mu_product_next)) * m_gpu;
                m_nesterov.eval();

                // Update parameters
                param_gpu = param_gpu - lr * m_nesterov / denominator;
                param_gpu.eval();

                param.SetFromSemanticArray(param_gpu, param.Shape());
                m_[name].SetFromSemanticArray(m_gpu, m_[name].Shape());
                v_[name].SetFromSemanticArray(v_gpu, v_[name].Shape());
                parameter_steps_[name] = parameter_step;
                mu_products_[name] = mu_product;
                continue;
            } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    "NAdamOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::OptimizerCpuPath,
            kOperation);
        float* param_data = param.MutableData<float>();
        const float* grad_data = grad.ReadData<float>();
        float* m_data = m_[name].MutableData<float>();
        float* v_data = v_[name].MutableData<float>();

        for (size_t i = 0; i < num_elements; ++i) {
            m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];
            v_data[i] = b2 * v_data[i] +
                        (1.0f - b2) * grad_data[i] * grad_data[i];
            const float denominator =
                std::sqrt(v_data[i] / bias_correction2) + eps;
            const float m_nesterov =
                ((1.0f - mu) / (1.0f - mu_product)) * grad_data[i] +
                (mu_next / (1.0f - mu_product_next)) * m_data[i];
            param_data[i] -= lr * m_nesterov / denominator;
        }
        parameter_steps_[name] = parameter_step;
        mu_products_[name] = mu_product;
    }
    ++step_count_;
}

void NAdamOptimizer::ZeroGrad() {
    // Gradients are caller-owned; moments and scheduled-momentum state persist.
}

bool NAdamOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const {
    OptimizerState exported;
    exported.optimizer_type = "NAdam";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"epsilon", epsilon_},
        {"momentum_decay", kNAdamMomentumDecay},
    };
    for (const auto& [name, moment] : m_) {
        exported.tensors.emplace("first_moment/" + name, moment);
    }
    for (const auto& [name, moment] : v_) {
        exported.tensors.emplace("second_moment/" + name, moment);
    }
    for (const auto& [name, step] : parameter_steps_) {
        exported.tensors.emplace("parameter_step/" + name, MakeStepTensor(step));
    }
    for (const auto& [name, mu_product] : mu_products_) {
        exported.tensors.emplace(
            "mu_product/" + name, MakeScalarTensor(mu_product));
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool NAdamOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error) {
    if (!ValidateStateHeader(
            state,
            "NAdam",
            {{"beta1", beta1_},
             {"beta2", beta2_},
             {"epsilon", epsilon_},
             {"momentum_decay", kNAdamMomentumDecay}},
            error)) {
        return false;
    }

    constexpr const char* kFirstPrefix = "first_moment/";
    constexpr const char* kSecondPrefix = "second_moment/";
    constexpr const char* kStepPrefix = "parameter_step/";
    constexpr const char* kMuPrefix = "mu_product/";
    std::map<std::string, Tensor> imported_m;
    std::map<std::string, Tensor> imported_v;
    std::map<std::string, int> imported_steps;
    std::map<std::string, float> imported_mu_products;
    const ScopedArrayFireHostSyncAttribution checkpoint_attribution(
        ArrayFireHostSyncCategory::CheckpointOutput,
        "NAdamOptimizer::ImportState");

    for (const auto& [key, tensor] : state.tensors) {
        std::string name;
        std::map<std::string, Tensor>* destination = nullptr;
        if (key.rfind(kFirstPrefix, 0) == 0) {
            name = key.substr(std::char_traits<char>::length(kFirstPrefix));
            destination = &imported_m;
        } else if (key.rfind(kSecondPrefix, 0) == 0) {
            name = key.substr(std::char_traits<char>::length(kSecondPrefix));
            destination = &imported_v;
        } else if (key.rfind(kStepPrefix, 0) == 0) {
            name = key.substr(std::char_traits<char>::length(kStepPrefix));
            if (name.empty()) {
                error = "NAdam optimizer state contains an empty parameter name.";
                return false;
            }
            int parameter_step = 0;
            if (!ReadStepTensor(
                    tensor, key, state.step_count, parameter_step, error)) {
                return false;
            }
            if (!imported_steps.emplace(name, parameter_step).second) {
                error = "NAdam optimizer state contains duplicate parameter step '" +
                        name + "'.";
                return false;
            }
            continue;
        } else if (key.rfind(kMuPrefix, 0) == 0) {
            name = key.substr(std::char_traits<char>::length(kMuPrefix));
            if (name.empty()) {
                error = "NAdam optimizer state contains an empty parameter name.";
                return false;
            }
            float mu_product = 0.0f;
            if (!ReadMuProductTensor(tensor, key, mu_product, error)) {
                return false;
            }
            if (!imported_mu_products.emplace(name, mu_product).second) {
                error = "NAdam optimizer state contains duplicate momentum product '" +
                        name + "'.";
                return false;
            }
            continue;
        } else {
            error = "NAdam optimizer state contains unknown tensor '" + key + "'.";
            return false;
        }

        if (name.empty()) {
            error = "NAdam optimizer state contains an empty parameter name.";
            return false;
        }
        if (tensor.GetDataType() != DataType::Float32) {
            error = "NAdam optimizer state tensor '" + key +
                    "' must use Float32.";
            return false;
        }
        if (!destination->emplace(name, tensor).second) {
            error = "NAdam optimizer state contains duplicate parameter '" +
                    name + "'.";
            return false;
        }
    }

    const size_t state_size = imported_m.size();
    if (imported_v.size() != state_size ||
        imported_steps.size() != state_size ||
        imported_mu_products.size() != state_size) {
        error = "NAdam optimizer state has incomplete continuation state.";
        return false;
    }
    for (const auto& [name, first_moment] : imported_m) {
        const auto second = imported_v.find(name);
        if (second == imported_v.end() ||
            imported_steps.find(name) == imported_steps.end() ||
            imported_mu_products.find(name) == imported_mu_products.end()) {
            error = "NAdam optimizer state is incomplete for '" + name + "'.";
            return false;
        }
        if (first_moment.Shape() != second->second.Shape()) {
            error = "NAdam optimizer moment shape mismatch for '" + name + "'.";
            return false;
        }
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    m_ = std::move(imported_m);
    v_ = std::move(imported_v);
    parameter_steps_ = std::move(imported_steps);
    mu_products_ = std::move(imported_mu_products);
    error.clear();
    return true;
}

} // namespace cyxwiz



