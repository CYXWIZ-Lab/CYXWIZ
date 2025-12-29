#include "cyxwiz/optimizer.h"
#include "cyxwiz/tensor.h"
#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>

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
        default:
            return nullptr;
    }
}

} // namespace cyxwiz
