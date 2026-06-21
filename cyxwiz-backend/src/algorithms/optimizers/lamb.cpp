#include "cyxwiz/optimizers/lamb.h"
#include "cyxwiz/tensor.h"
#include "optimizer_utils.h"

#include <cmath>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// LAMB Optimizer (Layer-wise Adaptive Moments for Batch training)
// ============================================================================

LAMBOptimizer::LAMBOptimizer(double learning_rate, double beta1, double beta2,
                             double epsilon, double weight_decay)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void LAMBOptimizer::Step(std::map<std::string, Tensor>& parameters,
                         const std::map<std::string, Tensor>& gradients) {
    step_count_++;

    float b1 = static_cast<float>(beta1_);
    float b2 = static_cast<float>(beta2_);
    float lr = static_cast<float>(learning_rate_);
    float eps = static_cast<float>(epsilon_);
    float wd = static_cast<float>(weight_decay_);

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

                // Update moments (same as Adam)
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;

                // Bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                af::array v_hat = v_gpu / bias_correction2;

                // Adam update direction: m_hat / (sqrt(v_hat) + eps)
                af::array adam_update = m_hat / (af::sqrt(v_hat) + eps);

                // Add weight decay to update (LAMB uses decoupled weight decay)
                if (wd > 0) {
                    adam_update = adam_update + wd * param_gpu;
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

                param.SetFromArray(param_gpu);
                m_[name].SetFromArray(m_gpu);
                v_[name].SetFromArray(v_gpu);
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("LAMB GPU step failed: {}, falling back to CPU", e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* m_data = m_[name].Data<float>();
            float* v_data = v_[name].Data<float>();

            // First compute moments and adam update, then compute norms
            std::vector<float> adam_update(num_elements);
            float weight_norm_sq = 0.0f;
            float update_norm_sq = 0.0f;

            for (size_t i = 0; i < num_elements; ++i) {
                m_data[i] = b1 * m_data[i] + (1.0f - b1) * grad_data[i];
                v_data[i] = b2 * v_data[i] + (1.0f - b2) * grad_data[i] * grad_data[i];

                float m_hat = m_data[i] / bias_correction1;
                float v_hat = v_data[i] / bias_correction2;

                adam_update[i] = m_hat / (std::sqrt(v_hat) + eps);

                // Add weight decay
                if (wd > 0) {
                    adam_update[i] += wd * param_data[i];
                }

                weight_norm_sq += param_data[i] * param_data[i];
                update_norm_sq += adam_update[i] * adam_update[i];
            }

            float weight_norm = std::sqrt(weight_norm_sq);
            float update_norm = std::sqrt(update_norm_sq);

            // Compute trust ratio
            float trust_ratio = 1.0f;
            if (weight_norm > 0 && update_norm > 0) {
                trust_ratio = weight_norm / update_norm;
            }

            // Apply scaled update
            for (size_t i = 0; i < num_elements; ++i) {
                param_data[i] -= lr * trust_ratio * adam_update[i];
            }
        }
    }
}

void LAMBOptimizer::ZeroGrad() {
    m_.clear();
    v_.clear();
}

} // namespace cyxwiz
