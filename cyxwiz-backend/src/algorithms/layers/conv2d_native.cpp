#include "conv2d_native.h"

#include "../arrayfire_backend_utils.h"
#include "layer_utils.h"

#include <cstdint>
#include <vector>

namespace cyxwiz {

Tensor Conv2DForwardNative(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias,
                           const Conv2DNativeGeometry& geometry,
                           const Conv2DNativeConfig& config) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "Conv2DLayer::Forward");
    Tensor output(
        {
            geometry.out_h,
            geometry.out_w,
            config.out_channels,
            geometry.batch_size,
        },
        DataType::Float32);
    const float* input_data = input.ReadData<float>();
    const float* weight_data = weights.ReadData<float>();
    const float* bias_data =
        config.use_bias ? bias.ReadData<float>() : nullptr;
    float* output_data = output.MutableData<float>();

    for (size_t b = 0; b < geometry.batch_size; ++b) {
        for (size_t oc = 0; oc < config.out_channels; ++oc) {
            for (size_t oh = 0; oh < geometry.out_h; ++oh) {
                for (size_t ow = 0; ow < geometry.out_w; ++ow) {
                    float sum = config.use_bias ? bias_data[oc] : 0.0f;
                    for (size_t ic = 0; ic < geometry.in_channels; ++ic) {
                        for (int kh = 0; kh < config.kernel_size; ++kh) {
                            for (int kw = 0; kw < config.kernel_size; ++kw) {
                                const int64_t ih =
                                    static_cast<int64_t>(oh) * config.stride +
                                    kh - config.padding;
                                const int64_t iw =
                                    static_cast<int64_t>(ow) * config.stride +
                                    kw - config.padding;
                                if (ih < 0 || iw < 0 ||
                                    ih >= static_cast<int64_t>(geometry.in_h) ||
                                    iw >= static_cast<int64_t>(geometry.in_w)) {
                                    continue;
                                }
                                sum += input_data[Pool4DIndex(
                                           static_cast<size_t>(ih),
                                           static_cast<size_t>(iw),
                                           ic,
                                           b,
                                           geometry.in_w,
                                           geometry.in_channels,
                                           geometry.batch_size)] *
                                       weight_data[Pool4DIndex(
                                           static_cast<size_t>(kh),
                                           static_cast<size_t>(kw),
                                           ic,
                                           oc,
                                           static_cast<size_t>(
                                               config.kernel_size),
                                           geometry.in_channels,
                                           config.out_channels)];
                            }
                        }
                    }
                    output_data[Pool4DIndex(
                        oh,
                        ow,
                        oc,
                        b,
                        geometry.out_w,
                        config.out_channels,
                        geometry.batch_size)] = sum;
                }
            }
        }
    }
    return output;
}

Tensor Conv2DBackwardNative(const Tensor& cached_input,
                            const Tensor& grad_output,
                            const Tensor& weights,
                            Tensor& grad_weights,
                            Tensor& grad_bias,
                            const Conv2DNativeGeometry& geometry,
                            const Conv2DNativeConfig& config) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "Conv2DLayer::Backward");
    Tensor grad_input(cached_input.Shape(), DataType::Float32);
    grad_weights = Tensor(
        {
            static_cast<size_t>(config.kernel_size),
            static_cast<size_t>(config.kernel_size),
            geometry.in_channels,
            config.out_channels,
        },
        DataType::Float32);
    if (config.use_bias) {
        grad_bias = Tensor({config.out_channels}, DataType::Float32);
    }

    const float* input_data = cached_input.ReadData<float>();
    const float* weight_data = weights.ReadData<float>();
    const float* grad_output_data = grad_output.ReadData<float>();
    float* grad_input_data = grad_input.MutableData<float>();
    float* grad_weight_data = grad_weights.MutableData<float>();
    float* grad_bias_data =
        config.use_bias ? grad_bias.MutableData<float>() : nullptr;

    for (size_t b = 0; b < geometry.batch_size; ++b) {
        for (size_t oc = 0; oc < config.out_channels; ++oc) {
            for (size_t oh = 0; oh < geometry.out_h; ++oh) {
                for (size_t ow = 0; ow < geometry.out_w; ++ow) {
                    const float grad_value = grad_output_data[Pool4DIndex(
                        oh,
                        ow,
                        oc,
                        b,
                        geometry.out_w,
                        config.out_channels,
                        geometry.batch_size)];
                    if (config.use_bias) {
                        grad_bias_data[oc] += grad_value;
                    }
                    for (size_t ic = 0; ic < geometry.in_channels; ++ic) {
                        for (int kh = 0; kh < config.kernel_size; ++kh) {
                            for (int kw = 0; kw < config.kernel_size; ++kw) {
                                const int64_t ih =
                                    static_cast<int64_t>(oh) * config.stride +
                                    kh - config.padding;
                                const int64_t iw =
                                    static_cast<int64_t>(ow) * config.stride +
                                    kw - config.padding;
                                if (ih < 0 || iw < 0 ||
                                    ih >= static_cast<int64_t>(geometry.in_h) ||
                                    iw >= static_cast<int64_t>(geometry.in_w)) {
                                    continue;
                                }

                                const size_t input_index = Pool4DIndex(
                                    static_cast<size_t>(ih),
                                    static_cast<size_t>(iw),
                                    ic,
                                    b,
                                    geometry.in_w,
                                    geometry.in_channels,
                                    geometry.batch_size);
                                const size_t weight_index = Pool4DIndex(
                                    static_cast<size_t>(kh),
                                    static_cast<size_t>(kw),
                                    ic,
                                    oc,
                                    static_cast<size_t>(config.kernel_size),
                                    geometry.in_channels,
                                    config.out_channels);
                                grad_input_data[input_index] +=
                                    grad_value * weight_data[weight_index];
                                grad_weight_data[weight_index] +=
                                    grad_value * input_data[input_index];
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}

} // namespace cyxwiz
