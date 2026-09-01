#include "cyxwiz/optimizers/lamb.h"
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

namespace {

void ValidateNonNegativeFinite(
    double value,
    const char* field_name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string("LAMB requires finite non-negative ") + field_name);
    }
}

void ValidateBeta(double value, const char* field_name) {
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        throw std::invalid_argument(
            std::string("LAMB requires ") + field_name + " in [0, 1)");
    }
}

void ValidateRuntimeStateTensor(
    const Tensor& state,
    const Tensor& parameter,
    const std::string& parameter_name) {
    if (state.GetDataType() != DataType::Float32 ||
        state.Shape() != parameter.Shape()) {
        throw std::invalid_argument(
            "LAMB state does not match parameter '" + parameter_name + "'.");
    }
}

bool ValidateStateHeader(const OptimizerState& state, std::string& error) {
    if (state.schema_version != 1) {
        error = "LAMB optimizer state schema version is unsupported.";
        return false;
    }
    if (state.optimizer_type != "LAMB") {
        error = "Optimizer state type '" + state.optimizer_type +
                "' is incompatible with LAMB.";
        return false;
    }
    if (!std::isfinite(state.learning_rate) || state.learning_rate < 0.0) {
        error = "LAMB optimizer state has an invalid learning rate.";
        return false;
    }
    if (state.step_count < 0) {
        error = "LAMB optimizer state has a negative step count.";
        return false;
    }
    return true;
}

} // namespace

// ============================================================================
// LAMB Optimizer (Layer-wise Adaptive Moments for Batch training)
// ============================================================================

LAMBOptimizer::LAMBOptimizer(double learning_rate, double beta1, double beta2,
                             double epsilon, double weight_decay)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay) {
    ValidateNonNegativeFinite(learning_rate, "learning rate");
    ValidateBeta(beta1_, "beta1");
    ValidateBeta(beta2_, "beta2");
    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0) {
        throw std::invalid_argument("LAMB requires finite positive epsilon");
    }
    ValidateNonNegativeFinite(weight_decay_, "weight decay");
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerArrayFireAvailable();
}

void LAMBOptimizer::Step(std::map<std::string, Tensor>& parameters,
                         const std::map<std::string, Tensor>& gradients) {
    constexpr const char* kOperation = "LAMBOptimizer::Step";
    for (const auto& [name, parameter] : parameters) {
        const auto gradient = gradients.find(name);
        if (gradient == gradients.end()) {
            continue;
        }
        optimizer_detail::ValidateOptimizerStepTensors(
            kOperation, name, parameter, gradient->second);
        const auto first_moment = m_.find(name);
        const auto second_moment = v_.find(name);
        if ((first_moment == m_.end()) != (second_moment == v_.end())) {
            throw std::invalid_argument(
                "LAMB state is incomplete for parameter '" + name + "'.");
        }
        if (first_moment != m_.end()) {
            ValidateRuntimeStateTensor(first_moment->second, parameter, name);
            ValidateRuntimeStateTensor(second_moment->second, parameter, name);
        }
    }

    const int next_step = step_count_ + 1;

    float b1 = static_cast<float>(beta1_);
    float b2 = static_cast<float>(beta2_);
    float lr = static_cast<float>(learning_rate_);
    float eps = static_cast<float>(epsilon_);
    float wd = static_cast<float>(weight_decay_);

    // Bias correction factors
    float bias_correction1 =
        1.0f - static_cast<float>(std::pow(b1, next_step));
    float bias_correction2 =
        1.0f - static_cast<float>(std::pow(b2, next_step));
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

        // Initialize moments only after fallback policy authorizes work.
        if (m_.find(name) == m_.end()) {
            m_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (!use_native_cpu) {
            try {
                af::array param_gpu = param.GetSemanticArray();
                af::array grad_gpu = grad.GetSemanticArray();
                af::array m_gpu = m_[name].GetSemanticArray();
                af::array v_gpu = v_[name].GetSemanticArray();

                // Update moments (same as Adam)
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                m_gpu.eval();
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;
                v_gpu.eval();

                // Bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                m_hat.eval();
                af::array v_hat = v_gpu / bias_correction2;
                v_hat.eval();

                // Adam update direction: m_hat / (sqrt(v_hat) + eps)
                af::array adam_update = m_hat / (af::sqrt(v_hat) + eps);
                adam_update.eval();

                // Add weight decay to update (LAMB uses decoupled weight decay)
                if (wd > 0) {
                    adam_update = adam_update + wd * param_gpu;
                    adam_update.eval();
                }

                // Compute trust ratio (layer-wise scaling)
                float weight_norm = static_cast<float>(af::norm(param_gpu));
                float update_norm = static_cast<float>(af::norm(adam_update));

                float trust_ratio = 1.0f;
                if (weight_norm > 0 && update_norm > 0) {
                    trust_ratio = weight_norm / update_norm;
                }

                // Apply scaled update
                param_gpu = param_gpu - lr * trust_ratio * adam_update;
                param_gpu.eval();

                param.SetFromSemanticArray(param_gpu, param.Shape());
                m_[name].SetFromSemanticArray(m_gpu, m_[name].Shape());
                v_[name].SetFromSemanticArray(v_gpu, v_[name].Shape());
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
        float* m_data = m_[name].MutableData<float>();
        float* v_data = v_[name].MutableData<float>();

        // First compute moments and Adam update, then compute layer norms.
        std::vector<float> adam_update(num_elements);
        float weight_norm_sq = 0.0f;
        float update_norm_sq = 0.0f;

        for (size_t i = 0; i < num_elements; ++i) {
            m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];
            v_data[i] = b2 * v_data[i] + (1.0f - b2) * grad_data[i] * grad_data[i];

            float m_hat = m_data[i] / bias_correction1;
            float v_hat = v_data[i] / bias_correction2;
            adam_update[i] = m_hat / (std::sqrt(v_hat) + eps);
            if (wd > 0) {
                adam_update[i] += wd * param_data[i];
            }
            weight_norm_sq += param_data[i] * param_data[i];
            update_norm_sq += adam_update[i] * adam_update[i];
        }

        float weight_norm = std::sqrt(weight_norm_sq);
        float update_norm = std::sqrt(update_norm_sq);
        float trust_ratio = 1.0f;
        if (weight_norm > 0 && update_norm > 0) {
            trust_ratio = weight_norm / update_norm;
        }
        for (size_t i = 0; i < num_elements; ++i) {
            param_data[i] -= lr * trust_ratio * adam_update[i];
        }
    }
    step_count_ = next_step;
}

void LAMBOptimizer::ZeroGrad() {
    // Gradients are caller-owned; moments persist across optimizer steps.
}

bool LAMBOptimizer::ExportState(
    OptimizerState& state,
    std::string& error) const {
    OptimizerState exported;
    exported.optimizer_type = "LAMB";
    exported.learning_rate = learning_rate_;
    exported.step_count = step_count_;
    exported.hyperparameters = {
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"epsilon", epsilon_},
        {"weight_decay", weight_decay_},
    };
    for (const auto& [name, moment] : m_) {
        exported.tensors.emplace("first_moment/" + name, moment);
    }
    for (const auto& [name, moment] : v_) {
        exported.tensors.emplace("second_moment/" + name, moment);
    }
    state = std::move(exported);
    error.clear();
    return true;
}

bool LAMBOptimizer::ImportState(
    const OptimizerState& state,
    std::string& error) {
    if (!ValidateStateHeader(state, error)) {
        return false;
    }
    const std::map<std::string, double> expected_hyperparameters = {
        {"beta1", beta1_},
        {"beta2", beta2_},
        {"epsilon", epsilon_},
        {"weight_decay", weight_decay_},
    };
    if (state.hyperparameters != expected_hyperparameters) {
        error = "LAMB optimizer state configuration does not match the active optimizer.";
        return false;
    }

    constexpr const char* kFirstPrefix = "first_moment/";
    constexpr const char* kSecondPrefix = "second_moment/";
    std::map<std::string, Tensor> imported_m;
    std::map<std::string, Tensor> imported_v;
    for (const auto& [key, tensor] : state.tensors) {
        std::string name;
        std::map<std::string, Tensor>* destination = nullptr;
        if (key.rfind(kFirstPrefix, 0) == 0 &&
            key.size() > std::char_traits<char>::length(kFirstPrefix)) {
            name = key.substr(std::char_traits<char>::length(kFirstPrefix));
            destination = &imported_m;
        } else if (key.rfind(kSecondPrefix, 0) == 0 &&
                   key.size() > std::char_traits<char>::length(kSecondPrefix)) {
            name = key.substr(std::char_traits<char>::length(kSecondPrefix));
            destination = &imported_v;
        } else {
            error = "LAMB optimizer state contains unknown tensor '" + key + "'.";
            return false;
        }
        if (tensor.GetDataType() != DataType::Float32) {
            error = "LAMB optimizer state tensor '" + key +
                    "' must use Float32.";
            return false;
        }
        destination->emplace(std::move(name), tensor);
    }

    if (imported_m.size() != imported_v.size()) {
        error = "LAMB optimizer state contains incomplete moment pairs.";
        return false;
    }
    for (const auto& [name, first_moment] : imported_m) {
        const auto second_moment = imported_v.find(name);
        if (second_moment == imported_v.end() ||
            second_moment->second.Shape() != first_moment.Shape()) {
            error = "LAMB optimizer state contains incomplete or shape-mismatched moment pairs.";
            return false;
        }
    }

    learning_rate_ = state.learning_rate;
    step_count_ = state.step_count;
    m_ = std::move(imported_m);
    v_ = std::move(imported_v);
    error.clear();
    return true;
}

} // namespace cyxwiz
