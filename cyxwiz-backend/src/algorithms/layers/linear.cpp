#include "cyxwiz/layers/linear.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace cyxwiz {

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

    // Initialize weights
    InitializeWeights();
}

void LinearLayer::InitializeWeights() {
    // Xavier/Glorot initialization: weights ~ U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
    double limit = std::sqrt(6.0 / (in_features_ + out_features_));

    // For now, use simple random initialization
    // TODO: Replace with proper Xavier initialization when Random supports range
    weight_ = Tensor::Random({out_features_, in_features_}, DataType::Float32);

    // Scale to [-limit, limit]
    float* weight_data = static_cast<float*>(weight_.Data());
    size_t num_weights = out_features_ * in_features_;
    for (size_t i = 0; i < num_weights; i++) {
        weight_data[i] = (weight_data[i] * 2.0f - 1.0f) * static_cast<float>(limit);
    }

    if (use_bias_) {
        // Initialize bias to zeros
        bias_ = Tensor::Zeros({out_features_}, DataType::Float32);
    }

    spdlog::debug("LinearLayer({}, {}) initialized with Xavier", in_features_, out_features_);
}

Tensor LinearLayer::Forward(const Tensor& input) {
    // Cache input for backward pass
    input_cache_ = input.Clone();

    const auto& input_shape = input.Shape();
    bool is_batched = input_shape.size() == 2;

    if (is_batched) {
        size_t batch_size = input_shape[0];
        size_t in_features = input_shape[1];

        if (in_features != in_features_) {
            throw std::runtime_error("LinearLayer: Input features mismatch. Expected " +
                                   std::to_string(in_features_) + ", got " +
                                   std::to_string(in_features));
        }

        // output = input @ weight^T + bias
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features]

        Tensor output({batch_size, out_features_}, DataType::Float32);
        const float* input_data = static_cast<const float*>(input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());
        const float* bias_data = use_bias_ ? static_cast<const float*>(bias_.Data()) : nullptr;
        float* output_data = static_cast<float*>(output.Data());

        // Matrix multiplication: C = A @ B^T
        // A: [batch, in]
        // B: [out, in]
        // C: [batch, out]
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

    } else if (input_shape.size() == 1) {
        // Single sample (no batch dimension)
        size_t in_features = input_shape[0];

        if (in_features != in_features_) {
            throw std::runtime_error("LinearLayer: Input features mismatch. Expected " +
                                   std::to_string(in_features_) + ", got " +
                                   std::to_string(in_features));
        }

        Tensor output({out_features_}, DataType::Float32);
        const float* input_data = static_cast<const float*>(input.Data());
        const float* weight_data = static_cast<const float*>(weight_.Data());
        const float* bias_data = use_bias_ ? static_cast<const float*>(bias_.Data()) : nullptr;
        float* output_data = static_cast<float*>(output.Data());

        // Matrix-vector multiplication
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

    } else {
        throw std::runtime_error("LinearLayer: Input must be 1D or 2D tensor");
    }
}

Tensor LinearLayer::Backward(const Tensor& grad_output) {
    const auto& grad_shape = grad_output.Shape();
    const auto& input_shape = input_cache_.Shape();
    bool is_batched = grad_shape.size() == 2;

    if (is_batched) {
        size_t batch_size = grad_shape[0];

        // Gradient w.r.t. weight: grad_weight = grad_output^T @ input
        // grad_output: [batch, out_features]
        // input: [batch, in_features]
        // grad_weight: [out_features, in_features]

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

        // Gradient w.r.t. bias: grad_bias = sum(grad_output, axis=0)
        if (use_bias_) {
            float* bias_grad_data = static_cast<float*>(bias_grad_.Data());
            std::memset(bias_grad_data, 0, sizeof(float) * out_features_);

            for (size_t b = 0; b < batch_size; b++) {
                for (size_t o = 0; o < out_features_; o++) {
                    bias_grad_data[o] += grad_output_data[b * out_features_ + o];
                }
            }

            // Average over batch
            for (size_t o = 0; o < out_features_; o++) {
                bias_grad_data[o] /= static_cast<float>(batch_size);
            }
        }

        // Gradient w.r.t. input: grad_input = grad_output @ weight
        // grad_output: [batch, out_features]
        // weight: [out_features, in_features]
        // grad_input: [batch, in_features]

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

        // grad_weight = outer_product(grad_output, input)
        for (size_t o = 0; o < out_features_; o++) {
            for (size_t i = 0; i < in_features_; i++) {
                weight_grad_data[o * in_features_ + i] = grad_output_data[o] * input_data[i];
            }
        }

        // grad_bias = grad_output
        if (use_bias_) {
            std::memcpy(bias_grad_.Data(), grad_output.Data(), sizeof(float) * out_features_);
        }

        // grad_input = weight^T @ grad_output
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
