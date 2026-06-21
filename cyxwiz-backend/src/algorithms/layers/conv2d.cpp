#include "cyxwiz/layer.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <algorithm>
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

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      use_bias_(use_bias) {
    if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Conv2D requires positive channels/kernel/stride and non-negative padding");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize weights using Kaiming initialization
    // Shape: [kernel_size, kernel_size, in_channels, out_channels] for ArrayFire
    // (ArrayFire uses column-major order)
    int fan_in = in_channels * kernel_size * kernel_size;
    af::dim4 weight_dims(kernel_size, kernel_size, in_channels, out_channels);
    af::array w = KaimingUniform(fan_in, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        af::array b = af::constant(0.0f, af::dim4(out_channels));
        bias_ = AfToTensor(b);
    }

    // Initialize gradient accumulators
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(in_channels),
                                    static_cast<size_t>(out_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#else
    weights_ = Tensor::Random({static_cast<size_t>(kernel_size),
                                static_cast<size_t>(kernel_size),
                                static_cast<size_t>(in_channels),
                                static_cast<size_t>(out_channels)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(in_channels),
                                    static_cast<size_t>(out_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#endif
}

Tensor Conv2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "Conv2D");
    if (weights_.GetDataType() != DataType::Float32 || (use_bias_ && bias_.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Conv2D forward CPU fallback requires Float32 parameters");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(in_channels_),
                                                static_cast<size_t>(out_channels_)}) {
        throw std::runtime_error("Conv2D forward weight shape mismatch");
    }
    if (use_bias_ && bias_.Shape() != std::vector<size_t>{static_cast<size_t>(out_channels_)}) {
        throw std::runtime_error("Conv2D forward bias shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t in_channels = shape[2];
    const size_t batch_size = shape[3];
    if (in_channels != static_cast<size_t>(in_channels_)) {
        throw std::runtime_error("Conv2D forward input channel mismatch");
    }

    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(kernel_size_) || padded_w < static_cast<size_t>(kernel_size_)) {
        throw std::runtime_error("Conv2D kernel is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(kernel_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(kernel_size_)) / static_cast<size_t>(stride_) + 1;

    Tensor output({out_h, out_w, static_cast<size_t>(out_channels_), batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* bias_data = use_bias_ ? bias_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = use_bias_ ? bias_data[oc] : 0.0f;
                    for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + kh - padding_;
                                const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + kw - padding_;
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }
                                sum += input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                             static_cast<size_t>(iw),
                                                             ic, b, in_w, in_channels, batch_size)] *
                                       weight_data[Pool4DIndex(static_cast<size_t>(kh),
                                                              static_cast<size_t>(kw),
                                                              ic, oc,
                                                              static_cast<size_t>(kernel_size_),
                                                              static_cast<size_t>(in_channels_),
                                                              static_cast<size_t>(out_channels_))];
                            }
                        }
                    }
                    output_data[Pool4DIndex(oh, ow, oc, b, out_w, static_cast<size_t>(out_channels_), batch_size)] = sum;
                }
            }
        }
    }

    return output;
}

Tensor Conv2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "Conv2D");
    if (grad_output.GetDataType() != DataType::Float32 || weights_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Conv2D backward CPU fallback requires Float32 tensors");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(in_channels_),
                                                static_cast<size_t>(out_channels_)}) {
        throw std::runtime_error("Conv2D backward weight shape mismatch");
    }

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t in_channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (in_channels != static_cast<size_t>(in_channels_)) {
        throw std::runtime_error("Conv2D backward cached input channel mismatch");
    }

    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(kernel_size_) || padded_w < static_cast<size_t>(kernel_size_)) {
        throw std::runtime_error("Conv2D kernel is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(kernel_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(kernel_size_)) / static_cast<size_t>(stride_) + 1;
    if (grad_output.Shape() != std::vector<size_t>{out_h, out_w, static_cast<size_t>(out_channels_), batch_size}) {
        throw std::runtime_error("Conv2D backward gradient shape mismatch");
    }

    Tensor grad_input(input_shape, DataType::Float32);
    grad_weights_ = Tensor({static_cast<size_t>(kernel_size_),
                            static_cast<size_t>(kernel_size_),
                            static_cast<size_t>(in_channels_),
                            static_cast<size_t>(out_channels_)},
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
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const float grad_value =
                        grad_output_data[Pool4DIndex(oh, ow, oc, b, out_w, static_cast<size_t>(out_channels_), batch_size)];
                    if (use_bias_) {
                        grad_bias_data[oc] += grad_value;
                    }
                    for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + kh - padding_;
                                const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + kw - padding_;
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }

                                const size_t input_index = Pool4DIndex(static_cast<size_t>(ih),
                                                                       static_cast<size_t>(iw),
                                                                       ic, b, in_w, in_channels, batch_size);
                                const size_t weight_index = Pool4DIndex(static_cast<size_t>(kh),
                                                                        static_cast<size_t>(kw),
                                                                        ic, oc,
                                                                        static_cast<size_t>(kernel_size_),
                                                                        static_cast<size_t>(in_channels_),
                                                                        static_cast<size_t>(out_channels_));
                                grad_input_data[input_index] += grad_value * weight_data[weight_index];
                                grad_weight_data[weight_index] += grad_value * input_data[input_index];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

std::map<std::string, Tensor> Conv2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) {
        weights_ = params.at("weights");
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
    }
}

} // namespace cyxwiz