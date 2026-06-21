#include "cyxwiz/layers/convolution.h"
#include "layer_arrayfire_utils.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

Conv1DLayer::Conv1DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding, int dilation, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), use_bias_(use_bias) {
    if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 ||
        stride_ <= 0 || padding_ < 0 || dilation_ <= 0) {
        throw std::invalid_argument("Conv1D requires positive channels/kernel/stride/dilation and non-negative padding");
    }

    // Xavier initialization for weights
    float stddev = std::sqrt(2.0f / (in_channels * kernel_size + out_channels));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    weights_ = Tensor({static_cast<size_t>(out_channels),
                       static_cast<size_t>(in_channels),
                       static_cast<size_t>(kernel_size)}, DataType::Float32);

    float* w_data = weights_.Data<float>();
    for (size_t i = 0; i < weights_.NumElements(); ++i) {
        w_data[i] = dist(gen);
    }

    if (use_bias) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }

    grad_weights_ = Tensor::Zeros(weights_.Shape());
    grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
}

Tensor Conv1DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    if (input.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32 ||
        (use_bias_ && bias_.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Conv1D forward CPU fallback requires Float32 tensors");
    }
    if (input.Shape().size() != 3) {
        throw std::runtime_error("Conv1D forward expects [L, C, N] input");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(out_channels_),
                                                static_cast<size_t>(in_channels_),
                                                static_cast<size_t>(kernel_size_)}) {
        throw std::runtime_error("Conv1D forward weight shape mismatch");
    }
    if (use_bias_ && bias_.Shape() != std::vector<size_t>{static_cast<size_t>(out_channels_)}) {
        throw std::runtime_error("Conv1D forward bias shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    const size_t input_length = shape[0];
    const size_t in_channels = shape[1];
    const size_t batch_size = shape[2];
    if (input_length == 0 || in_channels == 0 || batch_size == 0) {
        throw std::runtime_error("Conv1D forward does not support empty dimensions");
    }
    if (in_channels != static_cast<size_t>(in_channels_)) {
        throw std::runtime_error("Conv1D forward input channel mismatch");
    }

    const size_t effective_kernel = static_cast<size_t>(dilation_) * static_cast<size_t>(kernel_size_ - 1) + 1;
    const size_t padded_length = input_length + static_cast<size_t>(2 * padding_);
    if (padded_length < effective_kernel) {
        throw std::runtime_error("Conv1D kernel is larger than padded input");
    }
    const size_t output_length = (padded_length - effective_kernel) / static_cast<size_t>(stride_) + 1;

    Tensor output({output_length, static_cast<size_t>(out_channels_), batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* bias_data = use_bias_ ? bias_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
            for (size_t out_pos = 0; out_pos < output_length; ++out_pos) {
                float sum = use_bias_ ? bias_data[oc] : 0.0f;
                for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
                    for (int k = 0; k < kernel_size_; ++k) {
                        const int input_pos = static_cast<int>(out_pos * static_cast<size_t>(stride_)) +
                                              k * dilation_ - padding_;
                        if (input_pos < 0 || input_pos >= static_cast<int>(input_length)) {
                            continue;
                        }
                        const size_t input_index =
                            (static_cast<size_t>(input_pos) * in_channels + ic) * batch_size + b;
                        const size_t weight_index =
                            (oc * static_cast<size_t>(in_channels_) + ic) * static_cast<size_t>(kernel_size_) +
                            static_cast<size_t>(k);
                        sum += input_data[input_index] * weight_data[weight_index];
                    }
                }
                output_data[(out_pos * static_cast<size_t>(out_channels_) + oc) * batch_size + b] = sum;
            }
        }
    }

    return output;
}

Tensor Conv1DLayer::Backward(const Tensor& grad_output) {
    if (cached_input_.GetDataType() != DataType::Float32 ||
        grad_output.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Conv1D backward CPU fallback requires Float32 tensors");
    }
    if (cached_input_.Shape().size() != 3) {
        throw std::runtime_error("Conv1D backward expects cached [L, C, N] input");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(out_channels_),
                                                static_cast<size_t>(in_channels_),
                                                static_cast<size_t>(kernel_size_)}) {
        throw std::runtime_error("Conv1D backward weight shape mismatch");
    }

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t input_length = input_shape[0];
    const size_t in_channels = input_shape[1];
    const size_t batch_size = input_shape[2];
    const size_t effective_kernel = static_cast<size_t>(dilation_) * static_cast<size_t>(kernel_size_ - 1) + 1;
    const size_t padded_length = input_length + static_cast<size_t>(2 * padding_);
    if (padded_length < effective_kernel) {
        throw std::runtime_error("Conv1D kernel is larger than padded input");
    }
    const size_t output_length = (padded_length - effective_kernel) / static_cast<size_t>(stride_) + 1;
    if (grad_output.Shape() != std::vector<size_t>{output_length, static_cast<size_t>(out_channels_), batch_size}) {
        throw std::runtime_error("Conv1D backward gradient shape mismatch");
    }

    Tensor grad_input(input_shape, DataType::Float32);
    grad_weights_ = Tensor({static_cast<size_t>(out_channels_),
                            static_cast<size_t>(in_channels_),
                            static_cast<size_t>(kernel_size_)},
                           DataType::Float32);
    if (use_bias_) {
        grad_bias_ = Tensor({static_cast<size_t>(out_channels_)}, DataType::Float32);
    }

    const float* input_data = cached_input_.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* grad_output_data = grad_output.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* grad_weight_data = grad_weights_.Data<float>();
    float* grad_bias_data = use_bias_ ? grad_bias_.Data<float>() : nullptr;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
            for (size_t out_pos = 0; out_pos < output_length; ++out_pos) {
                const float grad_value =
                    grad_output_data[(out_pos * static_cast<size_t>(out_channels_) + oc) * batch_size + b];
                if (use_bias_) {
                    grad_bias_data[oc] += grad_value;
                }
                for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
                    for (int k = 0; k < kernel_size_; ++k) {
                        const int input_pos = static_cast<int>(out_pos * static_cast<size_t>(stride_)) +
                                              k * dilation_ - padding_;
                        if (input_pos < 0 || input_pos >= static_cast<int>(input_length)) {
                            continue;
                        }
                        const size_t input_index =
                            (static_cast<size_t>(input_pos) * in_channels + ic) * batch_size + b;
                        const size_t weight_index =
                            (oc * static_cast<size_t>(in_channels_) + ic) * static_cast<size_t>(kernel_size_) +
                            static_cast<size_t>(k);
                        grad_input_data[input_index] += grad_value * weight_data[weight_index];
                        grad_weight_data[weight_index] += grad_value * input_data[input_index];
                    }
                }
            }
        }
    }

    return grad_input;
}

std::map<std::string, Tensor> Conv1DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv1DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) weights_ = params.at("weights");
    if (params.count("bias")) bias_ = params.at("bias");
}

} // namespace cyxwiz
