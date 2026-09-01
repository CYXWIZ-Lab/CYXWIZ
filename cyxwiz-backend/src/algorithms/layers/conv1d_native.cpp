#include "conv1d_native.h"

#include "../arrayfire_backend_utils.h"

#include <cstdint>

namespace cyxwiz {

namespace {

size_t Conv1DInputIndex(size_t position,
                        size_t channel,
                        size_t batch,
                        size_t channels,
                        size_t batch_size) {
    return (position * channels + channel) * batch_size + batch;
}

size_t Conv1DWeightIndex(size_t output_channel,
                         size_t input_channel,
                         size_t kernel,
                         size_t input_channels,
                         size_t kernel_size) {
    return (output_channel * input_channels + input_channel) * kernel_size +
           kernel;
}

size_t Conv1DOutputIndex(size_t position,
                         size_t channel,
                         size_t batch,
                         size_t channels,
                         size_t batch_size) {
    return (position * channels + channel) * batch_size + batch;
}

} // namespace

Tensor Conv1DForwardNative(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias,
                           const Conv1DNativeGeometry& geometry,
                           const Conv1DNativeConfig& config) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "Conv1DLayer::Forward");
    Tensor output(
        {
            geometry.output_length,
            config.out_channels,
            geometry.batch_size,
        },
        DataType::Float32);
    const float* input_data = input.ReadData<float>();
    const float* weight_data = weights.ReadData<float>();
    const float* bias_data =
        config.use_bias ? bias.ReadData<float>() : nullptr;
    float* output_data = output.MutableData<float>();

    for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        for (size_t output_channel = 0;
             output_channel < config.out_channels;
             ++output_channel) {
            for (size_t output_position = 0;
                 output_position < geometry.output_length;
                 ++output_position) {
                float sum = config.use_bias
                    ? bias_data[output_channel]
                    : 0.0f;
                for (size_t input_channel = 0;
                     input_channel < geometry.in_channels;
                     ++input_channel) {
                    for (int kernel = 0;
                         kernel < config.kernel_size;
                         ++kernel) {
                        const int64_t input_position =
                            static_cast<int64_t>(output_position) *
                                config.stride +
                            static_cast<int64_t>(kernel) * config.dilation -
                            config.padding;
                        if (input_position < 0 ||
                            input_position >=
                                static_cast<int64_t>(geometry.input_length)) {
                            continue;
                        }
                        sum += input_data[Conv1DInputIndex(
                                   static_cast<size_t>(input_position),
                                   input_channel,
                                   batch,
                                   geometry.in_channels,
                                   geometry.batch_size)] *
                               weight_data[Conv1DWeightIndex(
                                   output_channel,
                                   input_channel,
                                   static_cast<size_t>(kernel),
                                   geometry.in_channels,
                                   static_cast<size_t>(config.kernel_size))];
                    }
                }
                output_data[Conv1DOutputIndex(
                    output_position,
                    output_channel,
                    batch,
                    config.out_channels,
                    geometry.batch_size)] = sum;
            }
        }
    }
    return output;
}

Tensor Conv1DBackwardNative(const Tensor& cached_input,
                            const Tensor& grad_output,
                            const Tensor& weights,
                            Tensor& grad_weights,
                            Tensor& grad_bias,
                            const Conv1DNativeGeometry& geometry,
                            const Conv1DNativeConfig& config) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "Conv1DLayer::Backward");
    Tensor grad_input(cached_input.Shape(), DataType::Float32);
    grad_weights = Tensor(
        {
            config.out_channels,
            geometry.in_channels,
            static_cast<size_t>(config.kernel_size),
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

    for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        for (size_t output_channel = 0;
             output_channel < config.out_channels;
             ++output_channel) {
            for (size_t output_position = 0;
                 output_position < geometry.output_length;
                 ++output_position) {
                const float grad_value = grad_output_data[Conv1DOutputIndex(
                    output_position,
                    output_channel,
                    batch,
                    config.out_channels,
                    geometry.batch_size)];
                if (config.use_bias) {
                    grad_bias_data[output_channel] += grad_value;
                }
                for (size_t input_channel = 0;
                     input_channel < geometry.in_channels;
                     ++input_channel) {
                    for (int kernel = 0;
                         kernel < config.kernel_size;
                         ++kernel) {
                        const int64_t input_position =
                            static_cast<int64_t>(output_position) *
                                config.stride +
                            static_cast<int64_t>(kernel) * config.dilation -
                            config.padding;
                        if (input_position < 0 ||
                            input_position >=
                                static_cast<int64_t>(geometry.input_length)) {
                            continue;
                        }

                        const size_t input_index = Conv1DInputIndex(
                            static_cast<size_t>(input_position),
                            input_channel,
                            batch,
                            geometry.in_channels,
                            geometry.batch_size);
                        const size_t weight_index = Conv1DWeightIndex(
                            output_channel,
                            input_channel,
                            static_cast<size_t>(kernel),
                            geometry.in_channels,
                            static_cast<size_t>(config.kernel_size));
                        grad_input_data[input_index] +=
                            grad_value * weight_data[weight_index];
                        grad_weight_data[weight_index] +=
                            grad_value * input_data[input_index];
                    }
                }
            }
        }
    }
    return grad_input;
}

} // namespace cyxwiz
