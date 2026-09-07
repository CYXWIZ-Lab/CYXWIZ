#include "conv_transpose2d_native.h"

#include "../arrayfire_backend_utils.h"
#include "layer_utils.h"

#include <vector>

namespace cyxwiz {

Tensor
ConvTranspose2DForwardNative(const Tensor &input, const Tensor &weights,
                             const Tensor &bias,
                             const ConvTranspose2DNativeGeometry &geometry,
                             const ConvTranspose2DNativeConfig &config) {
  const ScopedArrayFireHostSyncAttribution attribution(
      ArrayFireHostSyncCategory::LayerCpuPath, "ConvTranspose2DLayer::Forward");
  Tensor output(
      {
          geometry.out_h,
          geometry.out_w,
          config.out_channels,
          geometry.batch_size,
      },
      DataType::Float32);
  const float *input_data = input.ReadData<float>();
  const float *weight_data = weights.ReadData<float>();
  const float *bias_data = config.use_bias ? bias.ReadData<float>() : nullptr;
  float *output_data = output.MutableData<float>();

  for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
    for (size_t input_channel = 0; input_channel < geometry.in_channels;
         ++input_channel) {
      for (size_t input_h = 0; input_h < geometry.in_h; ++input_h) {
        for (size_t input_w = 0; input_w < geometry.in_w; ++input_w) {
          const float value = input_data[Pool4DIndex(
              input_h, input_w, input_channel, batch, geometry.in_w,
              geometry.in_channels, geometry.batch_size)];
          for (size_t output_channel = 0; output_channel < config.out_channels;
               ++output_channel) {
            for (int kernel_h = 0; kernel_h < config.kernel_size; ++kernel_h) {
              for (int kernel_w = 0; kernel_w < config.kernel_size;
                   ++kernel_w) {
                const size_t padded_h =
                    input_h * static_cast<size_t>(config.stride) +
                    static_cast<size_t>(kernel_h);
                const size_t padded_w =
                    input_w * static_cast<size_t>(config.stride) +
                    static_cast<size_t>(kernel_w);
                const size_t pad = static_cast<size_t>(config.padding);
                if (padded_h < pad || padded_w < pad)
                  continue;
                const size_t output_h = padded_h - pad;
                const size_t output_w = padded_w - pad;
                if (output_h >= geometry.out_h || output_w >= geometry.out_w) {
                  continue;
                }
                const size_t weight_index = Pool4DIndex(
                    static_cast<size_t>(kernel_h),
                    static_cast<size_t>(kernel_w), output_channel,
                    input_channel, static_cast<size_t>(config.kernel_size),
                    config.out_channels, geometry.in_channels);
                const size_t output_index = Pool4DIndex(
                    static_cast<size_t>(output_h),
                    static_cast<size_t>(output_w), output_channel, batch,
                    geometry.out_w, config.out_channels, geometry.batch_size);
                output_data[output_index] += value * weight_data[weight_index];
              }
            }
          }
        }
      }
    }
  }

  if (config.use_bias) {
    for (size_t output_h = 0; output_h < geometry.out_h; ++output_h) {
      for (size_t output_w = 0; output_w < geometry.out_w; ++output_w) {
        for (size_t output_channel = 0; output_channel < config.out_channels;
             ++output_channel) {
          for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
            output_data[Pool4DIndex(output_h, output_w, output_channel, batch,
                                    geometry.out_w, config.out_channels,
                                    geometry.batch_size)] +=
                bias_data[output_channel];
          }
        }
      }
    }
  }

  return output;
}

Tensor
ConvTranspose2DBackwardNative(const Tensor &cached_input,
                              const Tensor &grad_output, const Tensor &weights,
                              Tensor &grad_weights, Tensor &grad_bias,
                              const ConvTranspose2DNativeGeometry &geometry,
                              const ConvTranspose2DNativeConfig &config) {
  const ScopedArrayFireHostSyncAttribution attribution(
      ArrayFireHostSyncCategory::LayerCpuPath,
      "ConvTranspose2DLayer::Backward");
  const std::vector<size_t> input_shape{
      geometry.in_h,
      geometry.in_w,
      geometry.in_channels,
      geometry.batch_size,
  };
  Tensor grad_input(input_shape, DataType::Float32);
  grad_weights = Tensor(
      {
          static_cast<size_t>(config.kernel_size),
          static_cast<size_t>(config.kernel_size),
          config.out_channels,
          geometry.in_channels,
      },
      DataType::Float32);
  if (config.use_bias) {
    grad_bias = Tensor({config.out_channels}, DataType::Float32);
  }

  const float *input_data = cached_input.ReadData<float>();
  const float *weight_data = weights.ReadData<float>();
  const float *grad_output_data = grad_output.ReadData<float>();
  float *grad_input_data = grad_input.MutableData<float>();
  float *grad_weight_data = grad_weights.MutableData<float>();
  float *grad_bias_data =
      config.use_bias ? grad_bias.MutableData<float>() : nullptr;

  for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
    for (size_t input_channel = 0; input_channel < geometry.in_channels;
         ++input_channel) {
      for (size_t input_h = 0; input_h < geometry.in_h; ++input_h) {
        for (size_t input_w = 0; input_w < geometry.in_w; ++input_w) {
          const size_t input_index =
              Pool4DIndex(input_h, input_w, input_channel, batch, geometry.in_w,
                          geometry.in_channels, geometry.batch_size);
          const float input_value = input_data[input_index];
          for (size_t output_channel = 0; output_channel < config.out_channels;
               ++output_channel) {
            for (int kernel_h = 0; kernel_h < config.kernel_size; ++kernel_h) {
              for (int kernel_w = 0; kernel_w < config.kernel_size;
                   ++kernel_w) {
                const size_t padded_h =
                    input_h * static_cast<size_t>(config.stride) +
                    static_cast<size_t>(kernel_h);
                const size_t padded_w =
                    input_w * static_cast<size_t>(config.stride) +
                    static_cast<size_t>(kernel_w);
                const size_t pad = static_cast<size_t>(config.padding);
                if (padded_h < pad || padded_w < pad)
                  continue;
                const size_t output_h = padded_h - pad;
                const size_t output_w = padded_w - pad;
                if (output_h >= geometry.out_h || output_w >= geometry.out_w) {
                  continue;
                }

                const size_t grad_index = Pool4DIndex(
                    static_cast<size_t>(output_h),
                    static_cast<size_t>(output_w), output_channel, batch,
                    geometry.out_w, config.out_channels, geometry.batch_size);
                const size_t weight_index = Pool4DIndex(
                    static_cast<size_t>(kernel_h),
                    static_cast<size_t>(kernel_w), output_channel,
                    input_channel, static_cast<size_t>(config.kernel_size),
                    config.out_channels, geometry.in_channels);
                const float grad_value = grad_output_data[grad_index];
                grad_input_data[input_index] +=
                    grad_value * weight_data[weight_index];
                grad_weight_data[weight_index] += input_value * grad_value;
              }
            }
          }
        }
      }
    }
  }

  if (config.use_bias) {
    for (size_t output_h = 0; output_h < geometry.out_h; ++output_h) {
      for (size_t output_w = 0; output_w < geometry.out_w; ++output_w) {
        for (size_t output_channel = 0; output_channel < config.out_channels;
             ++output_channel) {
          for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
            grad_bias_data[output_channel] += grad_output_data[Pool4DIndex(
                output_h, output_w, output_channel, batch, geometry.out_w,
                config.out_channels, geometry.batch_size)];
          }
        }
      }
    }
  }

  return grad_input;
}

} // namespace cyxwiz
