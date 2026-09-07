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

    // This primitive executes its backward formula on native CPU. Construct
    // the host tensor with the semantic shape so trailing singleton channel
    // and batch dimensions are preserved; Tensor::Zeros may round-trip
    // through ArrayFire and collapse those dimensions.
    Tensor grad_input(input_shape, DataType::Float32);
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


} // namespace cyxwiz
