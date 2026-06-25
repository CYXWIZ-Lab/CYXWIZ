#include "cyxwiz/optimizers/adaptive.h"
#include "cyxwiz/tensor.h"
#include "optimizer_utils.h"

#include <cmath>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// RMSprop Optimizer
// ============================================================================

RMSpropOptimizer::RMSpropOptimizer(double learning_rate, double alpha, double epsilon, double momentum)
    : alpha_(alpha), epsilon_(epsilon), momentum_(momentum) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void RMSpropOptimizer::Step(std::map<std::string, Tensor>& parameters,
                            const std::map<std::string, Tensor>& gradients) {
    float lr = static_cast<float>(learning_rate_);
    float alpha = static_cast<float>(alpha_);
    float eps = static_cast<float>(epsilon_);
    float mom = static_cast<float>(momentum_);

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        // Initialize running average if needed
        if (v_.find(name) == v_.end()) {
            v_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            if (momentum_ > 0) {
                buffer_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            }
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
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
                    "RMSpropOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* v_data = v_[name].Data<float>();

            if (momentum_ > 0) {
                float* buf_data = buffer_[name].Data<float>();
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
    }
    step_count_++;
}

void RMSpropOptimizer::ZeroGrad() {
    v_.clear();
    buffer_.clear();
}

// ============================================================================
// AdaGrad Optimizer
// ============================================================================

AdaGradOptimizer::AdaGradOptimizer(double learning_rate, double epsilon)
    : epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void AdaGradOptimizer::Step(std::map<std::string, Tensor>& parameters,
                            const std::map<std::string, Tensor>& gradients) {
    float lr = static_cast<float>(learning_rate_);
    float eps = static_cast<float>(epsilon_);

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        // Initialize cache if needed
        if (cache_.find(name) == cache_.end()) {
            cache_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
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
                    "AdaGradOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* cache_data = cache_[name].Data<float>();

            for (size_t i = 0; i < num_elements; ++i) {
                cache_data[i] += grad_data[i] * grad_data[i];
                param_data[i] -= lr * grad_data[i] / (std::sqrt(cache_data[i]) + eps);
            }
        }
    }
    step_count_++;
}

void AdaGradOptimizer::ZeroGrad() {
    cache_.clear();
}

// ============================================================================
// Adadelta Optimizer
// ============================================================================

AdadeltaOptimizer::AdadeltaOptimizer(double rho, double epsilon)
    : rho_(rho), epsilon_(epsilon) {
    // Adadelta doesn't use a global learning rate
    learning_rate_ = 1.0;  // Effective LR is computed from accumulated deltas
    step_count_ = 0;
    optimizer_detail::OptimizerGpuAvailable();
}

void AdadeltaOptimizer::Step(std::map<std::string, Tensor>& parameters,
                              const std::map<std::string, Tensor>& gradients) {
    float rho = static_cast<float>(rho_);
    float eps = static_cast<float>(epsilon_);

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

        // Initialize accumulators if needed
        if (acc_grad_.find(name) == acc_grad_.end()) {
            acc_grad_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
            acc_delta_[name] = Tensor::Zeros(param.Shape(), DataType::Float32);
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (optimizer_detail::OptimizerGpuAvailable() && param.GetDataType() == DataType::Float32) {
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
                param_gpu = param_gpu + delta;
                param_gpu.eval();

                param.SetFromArray(param_gpu);
                acc_grad_[name].SetFromArray(acc_grad_gpu);
                acc_delta_[name].SetFromArray(acc_delta_gpu);
                continue;
            } catch (const af::exception& e) {
                optimizer_detail::LogOptimizerFallbackOnce(
                    "AdadeltaOptimizer::Step", name, param, e.what());
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();
            float* acc_grad_data = acc_grad_[name].Data<float>();
            float* acc_delta_data = acc_delta_[name].Data<float>();

            for (size_t i = 0; i < num_elements; ++i) {
                // Accumulate squared gradient
                acc_grad_data[i] = rho * acc_grad_data[i] + (1.0f - rho) * grad_data[i] * grad_data[i];

                // Compute update
                float delta = -std::sqrt(acc_delta_data[i] + eps) / std::sqrt(acc_grad_data[i] + eps) * grad_data[i];

                // Accumulate squared update
                acc_delta_data[i] = rho * acc_delta_data[i] + (1.0f - rho) * delta * delta;

                // Apply update
                param_data[i] += delta;
            }
        }
    }
    step_count_++;
}

void AdadeltaOptimizer::ZeroGrad() {
    acc_grad_.clear();
    acc_delta_.clear();
}

} // namespace cyxwiz
