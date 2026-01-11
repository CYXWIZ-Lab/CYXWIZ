#include "cyxwiz/optimizer.h"
#include "cyxwiz/tensor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// Flag to track if GPU is available
static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool CheckGPUAvailable() {
    if (s_gpu_checked) return s_use_gpu;
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
    } catch (const af::exception& e) {
        s_use_gpu = false;
    }
#endif

    return s_use_gpu;
}

// ============================================================================
// SGD Optimizer
// ============================================================================

SGDOptimizer::SGDOptimizer(double learning_rate, double momentum)
    : momentum_(momentum) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    CheckGPUAvailable();
}

void SGDOptimizer::Step(std::map<std::string, Tensor>& parameters,
                        const std::map<std::string, Tensor>& gradients) {
    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) continue;

        const Tensor& grad = grad_it->second;
        size_t num_elements = param.NumElements();

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));

                if (momentum_ > 0.0) {
                    // Initialize velocity if needed
                    if (velocity_.find(name) == velocity_.end()) {
                        velocity_[name] = Tensor(param.Shape(), DataType::Float32);
                        // Use memset for faster zero-initialization (optimized from CPU loop)
                        std::memset(velocity_[name].Data<float>(), 0, num_elements * sizeof(float));
                    }

                    af::array v_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(velocity_[name].Data()));

                    // v = momentum * v + grad
                    v_gpu = static_cast<float>(momentum_) * v_gpu + grad_gpu;
                    // param = param - lr * v
                    param_gpu = param_gpu - static_cast<float>(learning_rate_) * v_gpu;

                    v_gpu.host(velocity_[name].Data<float>());
                } else {
                    // Simple SGD: param = param - lr * grad
                    param_gpu = param_gpu - static_cast<float>(learning_rate_) * grad_gpu;
                }

                param_gpu.host(param.Data<float>());
                continue;
            } catch (const af::exception& e) {
                // Fall through to CPU
            }
        }
#endif

        // CPU fallback
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = param.Data<float>();
            const float* grad_data = grad.Data<float>();

            for (size_t i = 0; i < num_elements; i++) {
                param_data[i] -= static_cast<float>(learning_rate_) * grad_data[i];
            }
        }
    }

    step_count_++;
}

void SGDOptimizer::ZeroGrad() {
    velocity_.clear();
}

// ============================================================================
// Adam Optimizer
// ============================================================================

AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    CheckGPUAvailable();
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
            m_[name] = Tensor(param.Shape(), DataType::Float32);
            v_[name] = Tensor(param.Shape(), DataType::Float32);
            // Use memset for faster zero-initialization (optimized from CPU loop)
            std::memset(m_[name].Data<float>(), 0, num_elements * sizeof(float));
            std::memset(v_[name].Data<float>(), 0, num_elements * sizeof(float));
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array m_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(m_[name].Data()));
                af::array v_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(v_[name].Data()));

                float b1 = static_cast<float>(beta1_);
                float b2 = static_cast<float>(beta2_);
                float lr = static_cast<float>(learning_rate_);
                float eps = static_cast<float>(epsilon_);

                // Update biased first moment estimate: m = b1 * m + (1 - b1) * grad
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;

                // Update biased second moment estimate: v = b2 * v + (1 - b2) * grad^2
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;

                // Compute bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                af::array v_hat = v_gpu / bias_correction2;

                // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
                param_gpu = param_gpu - lr * m_hat / (af::sqrt(v_hat) + eps);

                // Copy back
                param_gpu.host(param.Data<float>());
                m_gpu.host(m_[name].Data<float>());
                v_gpu.host(v_[name].Data<float>());
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("Adam GPU step failed: {}, falling back to CPU", e.what());
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
            if (s_use_gpu && param.GetDataType() == DataType::Float32) {
                try {
                    af::array param_gpu(static_cast<dim_t>(num_elements),
                                        static_cast<const float*>(param.Data()));
                    param_gpu = param_gpu * (1.0f - wd);
                    param_gpu.host(param.Data<float>());
                    continue;
                } catch (const af::exception& e) {
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


// ============================================================================
// RMSprop Optimizer
// ============================================================================

RMSpropOptimizer::RMSpropOptimizer(double learning_rate, double alpha, double epsilon, double momentum)
    : alpha_(alpha), epsilon_(epsilon), momentum_(momentum) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    CheckGPUAvailable();
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
            v_[name] = Tensor(param.Shape(), DataType::Float32);
            std::memset(v_[name].Data<float>(), 0, num_elements * sizeof(float));
            if (momentum_ > 0) {
                buffer_[name] = Tensor(param.Shape(), DataType::Float32);
                std::memset(buffer_[name].Data<float>(), 0, num_elements * sizeof(float));
            }
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array v_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(v_[name].Data()));

                // v = alpha * v + (1 - alpha) * grad^2
                v_gpu = alpha * v_gpu + (1.0f - alpha) * grad_gpu * grad_gpu;

                if (momentum_ > 0) {
                    af::array buf_gpu(static_cast<dim_t>(num_elements),
                                      static_cast<const float*>(buffer_[name].Data()));
                    // buf = mom * buf + grad / sqrt(v + eps)
                    buf_gpu = mom * buf_gpu + grad_gpu / (af::sqrt(v_gpu) + eps);
                    param_gpu = param_gpu - lr * buf_gpu;
                    buf_gpu.host(buffer_[name].Data<float>());
                } else {
                    param_gpu = param_gpu - lr * grad_gpu / (af::sqrt(v_gpu) + eps);
                }

                param_gpu.host(param.Data<float>());
                v_gpu.host(v_[name].Data<float>());
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("RMSprop GPU step failed: {}, falling back to CPU", e.what());
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
    CheckGPUAvailable();
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
            cache_[name] = Tensor(param.Shape(), DataType::Float32);
            std::memset(cache_[name].Data<float>(), 0, num_elements * sizeof(float));
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array cache_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(cache_[name].Data()));

                // cache += grad^2
                cache_gpu = cache_gpu + grad_gpu * grad_gpu;
                // param -= lr * grad / sqrt(cache + eps)
                param_gpu = param_gpu - lr * grad_gpu / (af::sqrt(cache_gpu) + eps);

                param_gpu.host(param.Data<float>());
                cache_gpu.host(cache_[name].Data<float>());
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("AdaGrad GPU step failed: {}, falling back to CPU", e.what());
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
// NAdam Optimizer (Nesterov-accelerated Adam)
// ============================================================================

NAdamOptimizer::NAdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    CheckGPUAvailable();
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
            m_[name] = Tensor(param.Shape(), DataType::Float32);
            v_[name] = Tensor(param.Shape(), DataType::Float32);
            std::memset(m_[name].Data<float>(), 0, num_elements * sizeof(float));
            std::memset(v_[name].Data<float>(), 0, num_elements * sizeof(float));
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array m_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(m_[name].Data()));
                af::array v_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(v_[name].Data()));

                // Update moments
                m_gpu = b1 * m_gpu + (1.0f - b1) * grad_gpu;
                v_gpu = b2 * v_gpu + (1.0f - b2) * grad_gpu * grad_gpu;

                // Bias-corrected estimates
                af::array m_hat = m_gpu / bias_correction1;
                af::array v_hat = v_gpu / bias_correction2;

                // NAdam: Nesterov momentum term
                af::array m_nesterov = b1 * m_hat + (1.0f - b1) * grad_gpu / bias_correction1;

                // Update parameters
                param_gpu = param_gpu - lr * m_nesterov / (af::sqrt(v_hat) + eps);

                param_gpu.host(param.Data<float>());
                m_gpu.host(m_[name].Data<float>());
                v_gpu.host(v_[name].Data<float>());
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("NAdam GPU step failed: {}, falling back to CPU", e.what());
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

// ============================================================================
// Adadelta Optimizer
// ============================================================================

AdadeltaOptimizer::AdadeltaOptimizer(double rho, double epsilon)
    : rho_(rho), epsilon_(epsilon) {
    // Adadelta doesn't use a global learning rate
    learning_rate_ = 1.0;  // Effective LR is computed from accumulated deltas
    step_count_ = 0;
    CheckGPUAvailable();
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
            acc_grad_[name] = Tensor(param.Shape(), DataType::Float32);
            acc_delta_[name] = Tensor(param.Shape(), DataType::Float32);
            std::memset(acc_grad_[name].Data<float>(), 0, num_elements * sizeof(float));
            std::memset(acc_delta_[name].Data<float>(), 0, num_elements * sizeof(float));
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array acc_grad_gpu(static_cast<dim_t>(num_elements),
                                       static_cast<const float*>(acc_grad_[name].Data()));
                af::array acc_delta_gpu(static_cast<dim_t>(num_elements),
                                        static_cast<const float*>(acc_delta_[name].Data()));

                // Accumulate squared gradient: E[g²]_t = ρ * E[g²]_{t-1} + (1-ρ) * g²
                acc_grad_gpu = rho * acc_grad_gpu + (1.0f - rho) * grad_gpu * grad_gpu;

                // Compute update: Δx = -sqrt(E[Δx²]_{t-1} + ε) / sqrt(E[g²]_t + ε) * g
                af::array delta = -af::sqrt(acc_delta_gpu + eps) / af::sqrt(acc_grad_gpu + eps) * grad_gpu;

                // Accumulate squared update: E[Δx²]_t = ρ * E[Δx²]_{t-1} + (1-ρ) * Δx²
                acc_delta_gpu = rho * acc_delta_gpu + (1.0f - rho) * delta * delta;

                // Apply update: x = x + Δx
                param_gpu = param_gpu + delta;

                param_gpu.host(param.Data<float>());
                acc_grad_gpu.host(acc_grad_[name].Data<float>());
                acc_delta_gpu.host(acc_delta_[name].Data<float>());
                continue;
            } catch (const af::exception& e) {
                spdlog::warn("Adadelta GPU step failed: {}, falling back to CPU", e.what());
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

// ============================================================================
// LAMB Optimizer (Layer-wise Adaptive Moments for Batch training)
// ============================================================================

LAMBOptimizer::LAMBOptimizer(double learning_rate, double beta1, double beta2,
                             double epsilon, double weight_decay)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
    CheckGPUAvailable();
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
            m_[name] = Tensor(param.Shape(), DataType::Float32);
            v_[name] = Tensor(param.Shape(), DataType::Float32);
            std::memset(m_[name].Data<float>(), 0, num_elements * sizeof(float));
            std::memset(v_[name].Data<float>(), 0, num_elements * sizeof(float));
        }

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (s_use_gpu && param.GetDataType() == DataType::Float32) {
            try {
                af::array param_gpu(static_cast<dim_t>(num_elements),
                                    static_cast<const float*>(param.Data()));
                af::array grad_gpu(static_cast<dim_t>(num_elements),
                                   static_cast<const float*>(grad.Data()));
                af::array m_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(m_[name].Data()));
                af::array v_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(v_[name].Data()));

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

                param_gpu.host(param.Data<float>());
                m_gpu.host(m_[name].Data<float>());
                v_gpu.host(v_[name].Data<float>());
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

// ============================================================================
// Learning Rate Warmup
// ============================================================================

LRWarmup::LRWarmup(std::unique_ptr<Optimizer> optimizer, int warmup_steps,
                   WarmupType warmup_type, double base_lr)
    : optimizer_(std::move(optimizer)), warmup_steps_(warmup_steps),
      warmup_type_(warmup_type), current_step_(0) {
    // If base_lr not specified, use optimizer's initial learning rate
    if (base_lr < 0) {
        base_lr_ = optimizer_->GetLearningRate();
    } else {
        base_lr_ = base_lr;
    }
}

void LRWarmup::Step(std::map<std::string, Tensor>& parameters,
                    const std::map<std::string, Tensor>& gradients) {
    current_step_++;

    // Compute warmup multiplier
    double warmup_lr = base_lr_;

    if (current_step_ <= warmup_steps_ && warmup_type_ != WarmupType::None) {
        double progress = static_cast<double>(current_step_) / warmup_steps_;

        switch (warmup_type_) {
            case WarmupType::Linear:
                // Linear warmup: lr increases linearly from 0 to base_lr
                warmup_lr = base_lr_ * progress;
                break;

            case WarmupType::Cosine:
                // Cosine warmup: smoother ramp-up using cosine curve
                // lr = base_lr * 0.5 * (1 - cos(pi * progress))
                warmup_lr = base_lr_ * 0.5 * (1.0 - std::cos(M_PI * progress));
                break;

            default:
                break;
        }
    }

    // Set adjusted learning rate
    optimizer_->SetLearningRate(warmup_lr);

    // Perform optimization step
    optimizer_->Step(parameters, gradients);
}

void LRWarmup::ZeroGrad() {
    optimizer_->ZeroGrad();
}

double LRWarmup::GetCurrentLR() const {
    return optimizer_->GetLearningRate();
}

double LRWarmup::GetWarmupProgress() const {
    if (warmup_steps_ <= 0) return 1.0;
    double progress = static_cast<double>(current_step_) / warmup_steps_;
    return progress > 1.0 ? 1.0 : progress;
}

bool LRWarmup::IsWarmupComplete() const {
    return current_step_ >= warmup_steps_;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Optimizer> CreateOptimizer(OptimizerType type, double learning_rate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGDOptimizer>(learning_rate);
        case OptimizerType::Adam:
            return std::make_unique<AdamOptimizer>(learning_rate);
        case OptimizerType::AdamW:
            return std::make_unique<AdamWOptimizer>(learning_rate);
        case OptimizerType::RMSprop:
            return std::make_unique<RMSpropOptimizer>(learning_rate);
        case OptimizerType::AdaGrad:
            return std::make_unique<AdaGradOptimizer>(learning_rate);
        case OptimizerType::NAdam:
            return std::make_unique<NAdamOptimizer>(learning_rate);
        case OptimizerType::Adadelta:
            // Adadelta doesn't use learning_rate, but we accept it for consistency
            return std::make_unique<AdadeltaOptimizer>();
        case OptimizerType::LAMB:
            return std::make_unique<LAMBOptimizer>(learning_rate);
        default:
            return nullptr;
    }
}

} // namespace cyxwiz
