#include "cyxwiz/layers/linear.h"
#include "../arrayfire_backend_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include <cyxwiz/error_codes.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
void LogLinearInitializationFallbackOnce(
    const char* error_message,
    size_t in_features,
    size_t out_features) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        "in_features=" + std::to_string(in_features) +
        "; out_features=" + std::to_string(out_features));
    if (!ShouldLogArrayFireBackendFallbackOnce(
            "LinearLayer::InitializeWeights", reason, context)) {
        return;
    }

    std::string message =
        "ArrayFire LinearLayer::InitializeWeights failed (reason=" +
        std::string(BackendFallbackReasonName(reason)) +
        "); initializing weights on CPU.";
    message += " Context: ";
    message += context;
    message += ".";
    if (reason != BackendFallbackReason::CudaJitParamOverflow &&
        error_message != nullptr && error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    spdlog::warn("{}", message);
}
#endif

} // namespace

// Flag to track if GPU is available and should be used
static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool CheckGPUAvailable() {
    if (s_gpu_checked) return s_use_gpu;
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        if (s_use_gpu) {
#ifdef __APPLE__
            // On macOS, af::deviceInfo() crashes in background threads with OpenCL backend
            // This is a thread-safety issue specific to OpenCL context on macOS
            spdlog::info("LinearLayer: GPU acceleration enabled");
#else
            // On other platforms (Windows/Linux), query device name normally
            char name[256];
            af::deviceInfo(name, nullptr, nullptr, nullptr);
            spdlog::info("LinearLayer: GPU acceleration enabled using {}", name);
#endif
        }
    } catch (const af::exception& e) {
        spdlog::warn("LinearLayer: GPU check failed: {}, using CPU", e.what());
        s_use_gpu = false;
    } catch (...) {
        spdlog::warn("LinearLayer: GPU check failed with unknown error, using CPU");
        s_use_gpu = false;
    }
#endif

    return s_use_gpu;
}

LinearLayer::LinearLayer(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(use_bias)
    , weight_({out_features, in_features}, DataType::Float32)
    , weight_grad_({out_features, in_features}, DataType::Float32)
{
    if (use_bias_) {
        bias_ = Tensor({out_features}, DataType::Float32);
        bias_grad_ = Tensor({out_features}, DataType::Float32);
    }

    // Check GPU availability
    CheckGPUAvailable();

    // Initialize weights
    InitializeWeights();
}

void LinearLayer::InitializeWeights() {
    // Xavier/Glorot initialization: weights ~ U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
    double limit = std::sqrt(6.0 / (in_features_ + out_features_));

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (s_use_gpu) {
        try {
            af::array w_gpu = af::randu(static_cast<dim_t>(out_features_),
                                         static_cast<dim_t>(in_features_), f32);
            // Scale to [-limit, limit]
            w_gpu = (w_gpu * 2.0f - 1.0f) * static_cast<float>(limit);
            w_gpu.eval();

            weight_ = Tensor::FromArrayRowMajor2D(w_gpu);

            if (use_bias_) {
                bias_ = Tensor::Zeros({out_features_}, DataType::Float32);
            }

            spdlog::info("LinearLayer({}, {}) initialized with Xavier (GPU)", in_features_, out_features_);
            return;
        } catch (const af::exception& e) {
            LogLinearInitializationFallbackOnce(
                e.what(),
                in_features_,
                out_features_);
        }
    }
#endif

    // CPU fallback
    weight_ = Tensor::Random({out_features_, in_features_}, DataType::Float32);

    // Scale to [-limit, limit]
    float* weight_data = static_cast<float*>(weight_.Data());
    size_t num_weights = out_features_ * in_features_;
    for (size_t i = 0; i < num_weights; i++) {
        weight_data[i] = (weight_data[i] * 2.0f - 1.0f) * static_cast<float>(limit);
    }

    if (use_bias_) {
        bias_ = Tensor::Zeros({out_features_}, DataType::Float32);
    }

    spdlog::debug("LinearLayer({}, {}) initialized with Xavier (CPU)", in_features_, out_features_);
}

Tensor LinearLayer::Forward(const Tensor& input) {
    // Cache input for backward pass
    input_cache_ = input.Clone();

    const auto& input_shape = input.Shape();
    bool is_batched = input_shape.size() == 2;

    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("LinearLayer: Input must be 1D or 2D tensor");
    }

    size_t batch_size = is_batched ? input_shape[0] : 1;
    size_t in_features = is_batched ? input_shape[1] : input_shape[0];

    if (in_features != in_features_) {
        throw std::runtime_error("LinearLayer: Input features mismatch. Expected " +
                               std::to_string(in_features_) + ", got " +
                               std::to_string(in_features));
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (s_use_gpu) {
        try {
            af::array input_gpu;
            if (is_batched) {
                input_gpu = input.GetArrayRowMajor2D().as(af::dtype::f32);
            } else {
                input_gpu = af::moddims(input.GetArray(), 1, static_cast<dim_t>(in_features)).as(af::dtype::f32);
            }

            af::array weight_gpu = weight_.GetArrayRowMajor2D().as(af::dtype::f32);
            af::array output_gpu = af::matmul(input_gpu, weight_gpu, AF_MAT_NONE, AF_MAT_TRANS);
            output_gpu.eval();

            // Add bias if present
            if (use_bias_) {
                af::array bias_gpu = af::moddims(bias_.GetArray(), 1, static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                output_gpu = output_gpu + af::tile(bias_gpu, static_cast<unsigned int>(batch_size), 1);
                output_gpu.eval();
            }

            if (is_batched) {
                return Tensor::FromArrayRowMajor2D(output_gpu);
            }

            return Tensor(af::flat(output_gpu));
        } catch (const af::exception& e) {
            const BackendFallbackReason reason =
                ClassifyArrayFireBackendFallbackReason(e.what());
            const std::string context = BuildArrayFireBackendFallbackContext(
                "in=" + std::to_string(in_features_) +
                "; out=" + std::to_string(out_features_) +
                "; batch=" + std::to_string(batch_size) +
                "; bias=" + std::string(use_bias_ ? "true" : "false"));
            const bool log_fallback =
                ShouldLogArrayFireBackendFallbackOnce(
                    "LinearLayer::Forward", reason, context);
            if (log_fallback) {
                spdlog::warn("{}",
                             errors::FormatWarning(
                                 errors::Gpu::KernelExecutionFailed,
                                 BuildArrayFireBackendFallbackMessage(
                                     "LinearLayer::Forward", reason,
                                     reason != BackendFallbackReason::CudaJitParamOverflow,
                                     e.what(), context)));
            }
        }
    }
#endif

    // CPU fallback implementation
    if (is_batched) {
        Tensor output({batch_size, out_features_}, DataType::Float32);
        const float* input_data = static_cast<const float*>(input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());
        const float* bias_data = use_bias_ ? static_cast<const float*>(bias_.Data()) : nullptr;
        float* output_data = static_cast<float*>(output.Data());

        // Matrix multiplication: C = A @ B^T
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_features_; o++) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_features_; i++) {
                    sum += input_data[b * in_features_ + i] * weight_data[o * in_features_ + i];
                }
                if (use_bias_) {
                    sum += bias_data[o];
                }
                output_data[b * out_features_ + o] = sum;
            }
        }

        return output;
    } else {
        // Single sample (no batch dimension)
        Tensor output({out_features_}, DataType::Float32);
        const float* input_data = static_cast<const float*>(input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());
        const float* bias_data = use_bias_ ? static_cast<const float*>(bias_.Data()) : nullptr;
        float* output_data = static_cast<float*>(output.Data());

        for (size_t o = 0; o < out_features_; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features_; i++) {
                sum += input_data[i] * weight_data[o * in_features_ + i];
            }
            if (use_bias_) {
                sum += bias_data[o];
            }
            output_data[o] = sum;
        }

        return output;
    }
}

Tensor LinearLayer::Backward(const Tensor& grad_output) {
    const auto& grad_shape = grad_output.Shape();
    const auto& input_shape = input_cache_.Shape();
    (void)input_shape;  // Suppress unused variable warning
    bool is_batched = grad_shape.size() == 2;

    size_t batch_size = is_batched ? grad_shape[0] : 1;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (s_use_gpu) {
        try {
            af::array grad_gpu;
            af::array input_gpu;

            if (is_batched) {
                grad_gpu = grad_output.GetArrayRowMajor2D().as(af::dtype::f32);
                input_gpu = input_cache_.GetArrayRowMajor2D().as(af::dtype::f32);
            } else {
                grad_gpu = af::moddims(grad_output.GetArray(), 1, static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                input_gpu = af::moddims(input_cache_.GetArray(), 1, static_cast<dim_t>(in_features_)).as(af::dtype::f32);
            }

            af::array weight_gpu = weight_.GetArrayRowMajor2D().as(af::dtype::f32);

            af::array weight_grad_gpu = af::matmul(grad_gpu, input_gpu, AF_MAT_TRANS, AF_MAT_NONE);
            weight_grad_gpu.eval();
            weight_grad_gpu = weight_grad_gpu / static_cast<float>(batch_size);
            weight_grad_gpu.eval();
            weight_grad_ = Tensor::FromArrayRowMajor2D(weight_grad_gpu);

            if (use_bias_) {
                af::array bias_grad_gpu = af::flat(af::sum(grad_gpu, 0) / static_cast<float>(batch_size));
                bias_grad_gpu.eval();
                bias_grad_ = Tensor(bias_grad_gpu);
            }

            af::array grad_input_gpu = af::matmul(grad_gpu, weight_gpu);
            grad_input_gpu.eval();

            if (is_batched) {
                return Tensor::FromArrayRowMajor2D(grad_input_gpu);
            }

            return Tensor(af::flat(grad_input_gpu));
        } catch (const af::exception& e) {
            const BackendFallbackReason reason =
                ClassifyArrayFireBackendFallbackReason(e.what());
            const std::string context = BuildArrayFireBackendFallbackContext(
                "in=" + std::to_string(in_features_) +
                "; out=" + std::to_string(out_features_) +
                "; batch=" + std::to_string(batch_size) +
                "; bias=" + std::string(use_bias_ ? "true" : "false"));
            const bool log_fallback =
                ShouldLogArrayFireBackendFallbackOnce(
                    "LinearLayer::Backward", reason, context);
            if (log_fallback) {
                spdlog::warn("{}",
                             BuildArrayFireBackendFallbackMessage(
                                 "LinearLayer::Backward", reason,
                                 reason != BackendFallbackReason::CudaJitParamOverflow,
                                 e.what(), context));
            }
        }
    }
#endif

    // CPU fallback implementation
    if (is_batched) {
        const float* grad_output_data = static_cast<const float*>(grad_output.Data());
        const float* input_data = static_cast<const float*>(input_cache_.Data());
        float* weight_grad_data = static_cast<float*>(weight_grad_.Data());

        // Initialize gradients to zero
        std::memset(weight_grad_data, 0, sizeof(float) * out_features_ * in_features_);

        for (size_t o = 0; o < out_features_; o++) {
            for (size_t i = 0; i < in_features_; i++) {
                float grad_sum = 0.0f;
                for (size_t b = 0; b < batch_size; b++) {
                    grad_sum += grad_output_data[b * out_features_ + o] *
                              input_data[b * in_features_ + i];
                }
                weight_grad_data[o * in_features_ + i] = grad_sum / static_cast<float>(batch_size);
            }
        }

        if (use_bias_) {
            float* bias_grad_data = static_cast<float*>(bias_grad_.Data());
            std::memset(bias_grad_data, 0, sizeof(float) * out_features_);

            for (size_t b = 0; b < batch_size; b++) {
                for (size_t o = 0; o < out_features_; o++) {
                    bias_grad_data[o] += grad_output_data[b * out_features_ + o];
                }
            }

            for (size_t o = 0; o < out_features_; o++) {
                bias_grad_data[o] /= static_cast<float>(batch_size);
            }
        }

        // Gradient w.r.t. input
        Tensor grad_input({batch_size, in_features_}, DataType::Float32);
        float* grad_input_data = static_cast<float*>(grad_input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());

        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < in_features_; i++) {
                float sum = 0.0f;
                for (size_t o = 0; o < out_features_; o++) {
                    sum += grad_output_data[b * out_features_ + o] *
                          weight_data[o * in_features_ + i];
                }
                grad_input_data[b * in_features_ + i] = sum;
            }
        }

        return grad_input;
    } else {
        // Single sample (1D tensors)
        const float* grad_output_data = static_cast<const float*>(grad_output.Data());
        const float* input_data = static_cast<const float*>(input_cache_.Data());
        float* weight_grad_data = static_cast<float*>(weight_grad_.Data());

        for (size_t o = 0; o < out_features_; o++) {
            for (size_t i = 0; i < in_features_; i++) {
                weight_grad_data[o * in_features_ + i] = grad_output_data[o] * input_data[i];
            }
        }

        if (use_bias_) {
            std::memcpy(bias_grad_.Data(), grad_output.Data(), sizeof(float) * out_features_);
        }

        Tensor grad_input({in_features_}, DataType::Float32);
        float* grad_input_data = static_cast<float*>(grad_input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());

        for (size_t i = 0; i < in_features_; i++) {
            float sum = 0.0f;
            for (size_t o = 0; o < out_features_; o++) {
                sum += weight_data[o * in_features_ + i] * grad_output_data[o];
            }
            grad_input_data[i] = sum;
        }

        return grad_input;
    }
}

std::map<std::string, Tensor> LinearLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weight"] = weight_;
    if (use_bias_) {
        params["bias"] = bias_;
    }
    return params;
}

void LinearLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    auto weight_it = params.find("weight");
    if (weight_it != params.end()) {
        weight_ = weight_it->second.Clone();
    }

    if (use_bias_) {
        auto bias_it = params.find("bias");
        if (bias_it != params.end()) {
            bias_ = bias_it->second.Clone();
        }
    }
}

std::map<std::string, Tensor> LinearLayer::GetGradients() {
    std::map<std::string, Tensor> grads;
    grads["weight"] = weight_grad_;
    if (use_bias_) {
        grads["bias"] = bias_grad_;
    }
    return grads;
}

} // namespace cyxwiz
