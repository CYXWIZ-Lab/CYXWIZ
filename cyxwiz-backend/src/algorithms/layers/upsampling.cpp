#include "cyxwiz/layers/upsampling.h"
#include "layer_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

Upsample2DLayer::Upsample2DLayer(int scale_factor, UpsampleMode mode)
    : scale_factor_(scale_factor), mode_(mode) {
    if (scale_factor_ <= 0) {
        throw std::invalid_argument("Upsample2D scale_factor must be positive");
    }
}

Tensor Upsample2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;
    ValidateSpatial4DInput(input, "Upsample2D");

    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t scale = static_cast<size_t>(scale_factor_);
    const size_t out_h = in_h * scale;
    const size_t out_w = in_w * scale;

    Tensor output({out_h, out_w, channels, batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const size_t out_index = Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
                    if (mode_ == UpsampleMode::Nearest) {
                        const size_t ih = oh / scale;
                        const size_t iw = ow / scale;
                        output_data[out_index] =
                            input_data[Pool4DIndex(ih, iw, c, b, in_w, channels, batch_size)];
                        continue;
                    }

                    const ResizeLinearSample h_sample = ComputeResizeLinearSample(oh, in_h, scale_factor_);
                    const ResizeLinearSample w_sample = ComputeResizeLinearSample(ow, in_w, scale_factor_);
                    const float h0_weight = 1.0f - h_sample.upper_weight;
                    const float w0_weight = 1.0f - w_sample.upper_weight;
                    const float v00 = input_data[Pool4DIndex(h_sample.lower, w_sample.lower, c, b, in_w, channels, batch_size)];
                    const float v01 = input_data[Pool4DIndex(h_sample.lower, w_sample.upper, c, b, in_w, channels, batch_size)];
                    const float v10 = input_data[Pool4DIndex(h_sample.upper, w_sample.lower, c, b, in_w, channels, batch_size)];
                    const float v11 = input_data[Pool4DIndex(h_sample.upper, w_sample.upper, c, b, in_w, channels, batch_size)];
                    output_data[out_index] =
                        h0_weight * (w0_weight * v00 + w_sample.upper_weight * v01) +
                        h_sample.upper_weight * (w0_weight * v10 + w_sample.upper_weight * v11);
                }
            }
        }
    }

    return output;
}

Tensor Upsample2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "Upsample2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Upsample2D backward CPU fallback requires Float32 grad_output");
    }

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t scale = static_cast<size_t>(scale_factor_);
    const size_t out_h = in_h * scale;
    const size_t out_w = in_w * scale;
    if (grad_output.Shape() != std::vector<size_t>{out_h, out_w, channels, batch_size}) {
        throw std::runtime_error("Upsample2D backward gradient shape mismatch");
    }

    Tensor grad_input = Tensor::Zeros(input_shape, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    float* grad_input_data = grad_input.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            if (mode_ == UpsampleMode::Nearest) {
                for (size_t ih = 0; ih < in_h; ++ih) {
                    for (size_t iw = 0; iw < in_w; ++iw) {
                        float sum = 0.0f;
                        for (int sh = 0; sh < scale_factor_; ++sh) {
                            for (int sw = 0; sw < scale_factor_; ++sw) {
                                const size_t oh = ih * scale + static_cast<size_t>(sh);
                                const size_t ow = iw * scale + static_cast<size_t>(sw);
                                sum += grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)];
                            }
                        }
                        grad_input_data[Pool4DIndex(ih, iw, c, b, in_w, channels, batch_size)] = sum;
                    }
                }
                continue;
            }

            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const ResizeLinearSample h_sample = ComputeResizeLinearSample(oh, in_h, scale_factor_);
                    const ResizeLinearSample w_sample = ComputeResizeLinearSample(ow, in_w, scale_factor_);
                    const float h0_weight = 1.0f - h_sample.upper_weight;
                    const float w0_weight = 1.0f - w_sample.upper_weight;
                    const float grad_value = grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)];
                    grad_input_data[Pool4DIndex(h_sample.lower, w_sample.lower, c, b, in_w, channels, batch_size)] +=
                        grad_value * h0_weight * w0_weight;
                    grad_input_data[Pool4DIndex(h_sample.lower, w_sample.upper, c, b, in_w, channels, batch_size)] +=
                        grad_value * h0_weight * w_sample.upper_weight;
                    grad_input_data[Pool4DIndex(h_sample.upper, w_sample.lower, c, b, in_w, channels, batch_size)] +=
                        grad_value * h_sample.upper_weight * w0_weight;
                    grad_input_data[Pool4DIndex(h_sample.upper, w_sample.upper, c, b, in_w, channels, batch_size)] +=
                        grad_value * h_sample.upper_weight * w_sample.upper_weight;
                }
            }
        }
    }

    return grad_input;
}

// ============================================================================
// PixelShuffle Layer Implementation
// ============================================================================

PixelShuffleLayer::PixelShuffleLayer(int upscale_factor)
    : upscale_factor_(upscale_factor) {
    if (upscale_factor_ <= 0) {
        throw std::invalid_argument("PixelShuffle upscale_factor must be positive");
    }
}

Tensor PixelShuffleLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "PixelShuffle");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t in_c = shape[2];
    const size_t batch_size = shape[3];
    const size_t r = static_cast<size_t>(upscale_factor_);
    cached_in_channels_ = static_cast<int>(in_c);
    if (in_c % (r * r) != 0) {
        throw std::runtime_error("PixelShuffle input channels must be divisible by upscale_factor^2");
    }

    const size_t out_c = in_c / (r * r);
    const size_t out_h = in_h * r;
    const size_t out_w = in_w * r;
    Tensor output({out_h, out_w, out_c, batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t rh = 0; rh < r; ++rh) {
                for (size_t rw = 0; rw < r; ++rw) {
                    const size_t ic = oc * r * r + rh * r + rw;
                    for (size_t ih = 0; ih < in_h; ++ih) {
                        for (size_t iw = 0; iw < in_w; ++iw) {
                            output_data[Pool4DIndex(ih * r + rh, iw * r + rw, oc, b,
                                                     out_w, out_c, batch_size)] =
                                input_data[Pool4DIndex(ih, iw, ic, b, in_w, in_c, batch_size)];
                        }
                    }
                }
            }
        }
    }

    return output;
}

Tensor PixelShuffleLayer::Backward(const Tensor& grad_output) {
    if (grad_output.GetDataType() != DataType::Float32 || grad_output.Shape().size() != 4) {
        throw std::runtime_error("PixelShuffle backward CPU fallback expects Float32 [H, W, C, N] grad_output");
    }
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    const size_t out_h = grad_shape[0];
    const size_t out_w = grad_shape[1];
    const size_t out_c = grad_shape[2];
    const size_t batch_size = grad_shape[3];
    const size_t r = static_cast<size_t>(upscale_factor_);
    if (out_h % r != 0 || out_w % r != 0) {
        throw std::runtime_error("PixelShuffle backward output spatial shape must be divisible by upscale_factor");
    }

    const size_t in_h = out_h / r;
    const size_t in_w = out_w / r;
    const size_t in_c = out_c * r * r;
    Tensor grad_input({in_h, in_w, in_c, batch_size}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    float* grad_input_data = grad_input.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t rh = 0; rh < r; ++rh) {
                for (size_t rw = 0; rw < r; ++rw) {
                    const size_t ic = oc * r * r + rh * r + rw;
                    for (size_t ih = 0; ih < in_h; ++ih) {
                        for (size_t iw = 0; iw < in_w; ++iw) {
                            grad_input_data[Pool4DIndex(ih, iw, ic, b, in_w, in_c, batch_size)] =
                                grad_data[Pool4DIndex(ih * r + rh, iw * r + rw, oc, b,
                                                       out_w, out_c, batch_size)];
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

} // namespace cyxwiz
