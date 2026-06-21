#include "cyxwiz/layers/convolution.h"
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

ConvTranspose2DLayer::ConvTranspose2DLayer(int in_channels, int out_channels,
                                           int kernel_size, int stride, int padding,
                                           int output_padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      output_padding_(output_padding), use_bias_(use_bias) {
    if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 ||
        stride_ <= 0 || padding_ < 0 || output_padding_ < 0 || output_padding_ >= stride_) {
        throw std::invalid_argument("ConvTranspose2D requires positive channels/kernel/stride, non-negative padding, and output_padding < stride");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Weights: [kernel_size, kernel_size, out_channels, in_channels]
    // Note: transposed conv weights are "flipped" relative to conv2d
    int fan_in = in_channels * kernel_size * kernel_size;
    af::dim4 weight_dims(kernel_size, kernel_size, out_channels, in_channels);
    af::array w = KaimingUniform(fan_in, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        af::array b = af::constant(0.0f, af::dim4(out_channels));
        bias_ = AfToTensor(b);
    }

    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(out_channels),
                                    static_cast<size_t>(in_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#else
    weights_ = Tensor::Random({static_cast<size_t>(kernel_size),
                                static_cast<size_t>(kernel_size),
                                static_cast<size_t>(out_channels),
                                static_cast<size_t>(in_channels)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(out_channels),
                                    static_cast<size_t>(in_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#endif
}

Tensor ConvTranspose2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "ConvTranspose2D");
    if (weights_.GetDataType() != DataType::Float32 || (use_bias_ && bias_.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("ConvTranspose2D forward CPU fallback requires Float32 parameters");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(out_channels_),
                                                static_cast<size_t>(in_channels_)}) {
        throw std::runtime_error("ConvTranspose2D forward weight shape mismatch");
    }
    if (use_bias_ && bias_.Shape() != std::vector<size_t>{static_cast<size_t>(out_channels_)}) {
        throw std::runtime_error("ConvTranspose2D forward bias shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t in_channels = shape[2];
    const size_t batch_size = shape[3];
    if (in_channels != static_cast<size_t>(in_channels_)) {
        throw std::runtime_error("ConvTranspose2D forward input channel mismatch");
    }

    const int out_h_signed = (static_cast<int>(in_h) - 1) * stride_ - 2 * padding_ +
                             kernel_size_ + output_padding_;
    const int out_w_signed = (static_cast<int>(in_w) - 1) * stride_ - 2 * padding_ +
                             kernel_size_ + output_padding_;
    if (out_h_signed <= 0 || out_w_signed <= 0) {
        throw std::runtime_error("ConvTranspose2D output shape is not positive");
    }
    const size_t out_h = static_cast<size_t>(out_h_signed);
    const size_t out_w = static_cast<size_t>(out_w_signed);

    Tensor output({out_h, out_w, static_cast<size_t>(out_channels_), batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* bias_data = use_bias_ ? bias_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
            for (size_t ih = 0; ih < in_h; ++ih) {
                for (size_t iw = 0; iw < in_w; ++iw) {
                    const float value = input_data[Pool4DIndex(ih, iw, ic, b, in_w, in_channels, batch_size)];
                    for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int oh = static_cast<int>(ih * static_cast<size_t>(stride_)) - padding_ + kh;
                                const int ow = static_cast<int>(iw * static_cast<size_t>(stride_)) - padding_ + kw;
                                if (oh < 0 || ow < 0 || oh >= out_h_signed || ow >= out_w_signed) {
                                    continue;
                                }
                                const size_t weight_index = Pool4DIndex(static_cast<size_t>(kh),
                                                                        static_cast<size_t>(kw),
                                                                        oc, ic,
                                                                        static_cast<size_t>(kernel_size_),
                                                                        static_cast<size_t>(out_channels_),
                                                                        static_cast<size_t>(in_channels_));
                                output_data[Pool4DIndex(static_cast<size_t>(oh),
                                                       static_cast<size_t>(ow),
                                                       oc, b,
                                                       out_w,
                                                       static_cast<size_t>(out_channels_),
                                                       batch_size)] += value * weight_data[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }

    if (use_bias_) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        output_data[Pool4DIndex(oh, ow, oc, b, out_w, static_cast<size_t>(out_channels_), batch_size)] +=
                            bias_data[oc];
                    }
                }
            }
        }
    }

    return output;
}

Tensor ConvTranspose2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "ConvTranspose2D");
    if (grad_output.GetDataType() != DataType::Float32 || weights_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("ConvTranspose2D backward CPU fallback requires Float32 tensors");
    }
    if (weights_.Shape() != std::vector<size_t>{static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(kernel_size_),
                                                static_cast<size_t>(out_channels_),
                                                static_cast<size_t>(in_channels_)}) {
        throw std::runtime_error("ConvTranspose2D backward weight shape mismatch");
    }

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t in_channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (in_channels != static_cast<size_t>(in_channels_)) {
        throw std::runtime_error("ConvTranspose2D backward cached input channel mismatch");
    }

    const int out_h_signed = (static_cast<int>(in_h) - 1) * stride_ - 2 * padding_ +
                             kernel_size_ + output_padding_;
    const int out_w_signed = (static_cast<int>(in_w) - 1) * stride_ - 2 * padding_ +
                             kernel_size_ + output_padding_;
    if (out_h_signed <= 0 || out_w_signed <= 0) {
        throw std::runtime_error("ConvTranspose2D output shape is not positive");
    }
    const size_t out_h = static_cast<size_t>(out_h_signed);
    const size_t out_w = static_cast<size_t>(out_w_signed);
    if (grad_output.Shape() != std::vector<size_t>{out_h, out_w, static_cast<size_t>(out_channels_), batch_size}) {
        throw std::runtime_error("ConvTranspose2D backward gradient shape mismatch");
    }

    Tensor grad_input(input_shape, DataType::Float32);
    grad_weights_ = Tensor({static_cast<size_t>(kernel_size_),
                            static_cast<size_t>(kernel_size_),
                            static_cast<size_t>(out_channels_),
                            static_cast<size_t>(in_channels_)},
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
        for (size_t ic = 0; ic < static_cast<size_t>(in_channels_); ++ic) {
            for (size_t ih = 0; ih < in_h; ++ih) {
                for (size_t iw = 0; iw < in_w; ++iw) {
                    const float input_value = input_data[Pool4DIndex(ih, iw, ic, b, in_w, in_channels, batch_size)];
                    for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int oh = static_cast<int>(ih * static_cast<size_t>(stride_)) - padding_ + kh;
                                const int ow = static_cast<int>(iw * static_cast<size_t>(stride_)) - padding_ + kw;
                                if (oh < 0 || ow < 0 || oh >= out_h_signed || ow >= out_w_signed) {
                                    continue;
                                }

                                const size_t grad_index = Pool4DIndex(static_cast<size_t>(oh),
                                                                      static_cast<size_t>(ow),
                                                                      oc, b,
                                                                      out_w,
                                                                      static_cast<size_t>(out_channels_),
                                                                      batch_size);
                                const size_t weight_index = Pool4DIndex(static_cast<size_t>(kh),
                                                                        static_cast<size_t>(kw),
                                                                        oc, ic,
                                                                        static_cast<size_t>(kernel_size_),
                                                                        static_cast<size_t>(out_channels_),
                                                                        static_cast<size_t>(in_channels_));
                                const size_t input_index = Pool4DIndex(ih, iw, ic, b, in_w, in_channels, batch_size);
                                const float grad_value = grad_output_data[grad_index];
                                grad_input_data[input_index] += grad_value * weight_data[weight_index];
                                grad_weight_data[weight_index] += input_value * grad_value;
                            }
                        }
                    }
                }
            }
        }
    }

    if (use_bias_) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < static_cast<size_t>(out_channels_); ++oc) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        grad_bias_data[oc] +=
                            grad_output_data[Pool4DIndex(oh, ow, oc, b, out_w, static_cast<size_t>(out_channels_), batch_size)];
                    }
                }
            }
        }
    }

    return grad_input;
}

std::map<std::string, Tensor> ConvTranspose2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void ConvTranspose2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) weights_ = params.at("weights");
    if (params.count("bias") && use_bias_) bias_ = params.at("bias");
}

} // namespace cyxwiz
