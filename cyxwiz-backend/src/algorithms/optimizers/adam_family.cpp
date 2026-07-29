#include "cyxwiz/optimizers/adam.h"
#include "cyxwiz/tensor.h"
#include "optimizer_utils.h"

#include <cmath>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// Adam Optimizer
// ============================================================================

AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void AdamOptimizer::Step(std::map<std::string, Tensor>& parameters,
                         const std::map<std::string, Tensor>& gradients) {
    step_count_++;

    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(static_cast<float>(beta1_), step_count_);
    float bias_correction2 = 1.0f - std::pow(static_cast<float>(beta2_), step_count_);

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        // Initialize moment vectors if needed
        if (m_.find(name) == m_.end()) {
            m_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu = param.GetArray();
                af::array grad_gpu = grad.GetArray();
                af::array m_gpu = m_[name].GetArray();
                af::array v_gpu = v_[name].GetArray();

                float b1 = static_cast<float>(beta1_);
                float b2 = static_cast<float>(beta2_);
                float lr = static_cast<float>(learning_rate_);
                float eps = static_cast<float>(epsilon_);

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

                param.SetFromArray(param_gpu);
                m_[name].SetFromArray(m_gpu);
                v_[name].SetFromArray(v_gpu);
                continue;
                } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    "AdamOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* m_data = m_[name].Data<float>();
            float* v_data = v_[name].Data<float>();

            float lr = static_cast<float>(learning_rate_);
            float b1 = static_cast<float>(beta1_);
            float b2 = static_cast<float>(beta2_);
            float eps = static_cast<float>(epsilon_);

            for (size_t i = 0; i < num_elements; ++i) {
                // Update biased first moment estimate
                m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];

                // Update biased second raw moment estimate
                v_data[i] = b2 * v_data[i] + (1.0f - b2) * grad_data[i] * grad_data[i];

                // Compute bias-corrected estimates
                float m_hat = m_data[i] / bias_correction1;
                float v_hat = v_data[i] / bias_correction2;

                // Update parameters
                param_data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
}

void AdamOptimizer::ZeroGrad() {
    m_.clear();
    v_.clear();
}

bool AdamOptimizer::ExportState(OptimizerState& state, std::string& error) const {
    error.clear();
    state = OptimizerState{};
    state.optimizer_type = "Adam";
    state.learning_rate = learning_rate_;
    state.step_count = step_count_;
    state.hyperparameters = {
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"epsilon", epsilon_},
    };

    for (const auto& [name, tensor] : m_) {
        state.tensors.emplace("first_moment/" + name, tensor);
    }
    for (const auto& [name, tensor] : v_) {
        state.tensors.emplace("second_moment/" + name, tensor);
    }
    return true;
}

bool AdamOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error)
{
    constexpr const char* kFirstPrefix = "first_moment/";
    constexpr const char* kSecondPrefix = "second_moment/";

    if (state.schema_version != 1) {
        error = "Adam optimizer state schema version is unsupported.";
        return false;
    }
    if (state.optimizer_type != "Adam") {
        error = "Optimizer state type '" + state.optimizer_type +
                "' is incompatible with Adam.";
        return false;
    }
    if (!std::isfinite(state.learning_rate) || state.learning_rate <= 0.0) {
        error = "Adam optimizer state has an invalid learning rate.";
        return false;
    }
    if (state.step_count < 0) {
        error = "Adam optimizer state has a negative step count.";
        return false;
    }
    if (state.hyperparameters.size() != 3 ||
        state.hyperparameters.find("beta1") == state.hyperparameters.end() ||
        state.hyperparameters.find("beta2") == state.hyperparameters.end() ||
        state.hyperparameters.find("epsilon") == state.hyperparameters.end()) {
        error = "Adam optimizer state is missing required hyperparameters.";
        return false;
    }
    if (state.hyperparameters.at("beta1") != beta1_ ||
        state.hyperparameters.at("beta2") != beta2_ ||
        state.hyperparameters.at("epsilon") != epsilon_) {
        error = "Adam optimizer state hyperparameters do not match the active optimizer.";
        return false;
    }

    std::map<std::string, Tensor> imported_m;
    std::map<std::string, Tensor> imported_v;
    for (const auto& [key, tensor] : state.tensors) {
        std::string parameter_name;
        std::map<std::string, Tensor>* destination = nullptr;
        if (key.rfind(kFirstPrefix, 0) == 0) {
            parameter_name = key.substr(std::char_traits<char>::length(kFirstPrefix));
            destination = &imported_m;
        } else if (key.rfind(kSecondPrefix, 0) == 0) {
            parameter_name = key.substr(std::char_traits<char>::length(kSecondPrefix));
            destination = &imported_v;
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

    // Commit only after every field and tensor pair has been validated.
    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    m_ = std::move(imported_m);
    v_ = std::move(imported_v);
    error.clear();
    return true;
}

// ============================================================================
// AdamW Optimizer
// ============================================================================

AdamWOptimizer::AdamWOptimizer(double learning_rate, double beta1, double beta2,
                               double epsilon, double weight_decay)
    : AdamOptimizer(learning_rate, beta1, beta2, epsilon), weight_decay_(weight_decay) {
}

void AdamWOptimizer::Step(std::map<std::string, Tensor>& parameters,
                          const std::map<std::string, Tensor>& gradients) {
    // AdamW: Apply decoupled weight decay before Adam update
    if (weight_decay_ > 0.0) {
        float wd = static_cast<float>(weight_decay_ * learning_rate_);

        for (auto& param_pair : parameters) {
            Tensor& param = param_pair.second;
            size_t num_elements = param.NumElements();

#ifdef CYXWIZ_HAS_ARRAYFIRE
            if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
                try {
                    af::array param_gpu = param.GetArray();
                    param_gpu = param_gpu * (1.0f - wd);
                    param_gpu.eval();
                    param.SetFromArray(param_gpu);
                    continue;
    } catch (const af::exception& e) {
                    optimizer_detail::LogOptimizerFallbackOnce(
                        "AdamWOptimizer::WeightDecay", param_pair.first, param, e.what());
                    // Fall through to CPU
                }
            }
#endif

            // CPU fallback
            if (param.GetDataType() == DataType::Float32) {
                float* param_data = param.Data<float>();
                for (size_t i = 0; i < num_elements; ++i) {
                    param_data[i] *= (1.0f - wd);
                }
            }
        }
    }

    // Then apply Adam update
    AdamOptimizer::Step(parameters, gradients);
}

bool AdamWOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const
{
    state = OptimizerState{};
    error = "AdamW exact state export is not implemented yet.";
    return false;
}

bool AdamWOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error)
{
    (void)state;
    error = "AdamW exact state import is not implemented yet.";
    return false;
}

// ============================================================================
// NAdam Optimizer (Nesterov-accelerated Adam)
// ============================================================================

NAdamOptimizer::NAdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void NAdamOptimizer::Step(std::map<std::string, Tensor>& parameters,
                          const std::map<std::string, Tensor>& gradients) {
    step_count_++;

    float b1 = static_cast<float>(beta1_);
    float b2 = static_cast<float>(beta2_);
    float lr = static_cast<float>(learning_rate_);
    float eps = static_cast<float>(epsilon_);

    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(b1, step_count_);
    float bias_correction2 = 1.0f - std::pow(b2, step_count_);

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        // Initialize moment vectors if needed
        if (m_.find(name) == m_.end()) {
            m_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu = param.GetArray();
                af::array grad_gpu = grad.GetArray();
                af::array m_gpu = m_[name].GetArray();
                af::array v_gpu = v_[name].GetArray();

                // Update moments
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                m_gpu.eval();
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;
                v_gpu.eval();

                // Bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                m_hat.eval();
                af::array v_hat = v_gpu / bias_correction2;
                v_hat.eval();

                // NAdam: Nesterov momentum term
                af::array m_nesterov = b1 * m_hat + (1.0f - b1) * grad_gpu / bias_correction1;
                m_nesterov.eval();

                // Update parameters
                param_gpu = param_gpu - lr * m_nesterov / (af::sqrt(v_hat) + eps);
                param_gpu.eval();

                param.SetFromArray(param_gpu);
                m_[name].SetFromArray(m_gpu);
                v_[name].SetFromArray(v_gpu);
                continue;
            } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    "NAdamOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* m_data = m_[name].Data<float>();
            float* v_data = v_[name].Data<float>();

            for (size_t i = 0; i < num_elements; ++i) {
                m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];
                v_data[i] = b2 * v_data[i] + (1.0f - b2) * grad_data[i] * grad_data[i];

                float m_hat = m_data[i] / bias_correction1;
                float v_hat = v_data[i] / bias_correction2;
                float m_nesterov = b1 * m_hat + (1.0f - b1) * grad_data[i] / bias_correction1;

                param_data[i] -= lr * m_nesterov / (std::sqrt(v_hat) + eps);
            }
        }
    }
}

void NAdamOptimizer::ZeroGrad() {
    m_.clear();
    v_.clear();
}

} // namespace cyxwiz



