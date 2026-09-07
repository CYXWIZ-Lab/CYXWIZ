#pragma once

#include "cyxwiz/tensor.h"

#include <cstddef>

namespace cyxwiz {

struct ConvTranspose2DNativeGeometry {
  size_t in_h;
  size_t in_w;
  size_t in_channels;
  size_t batch_size;
  size_t out_h;
  size_t out_w;
};

struct ConvTranspose2DNativeConfig {
  size_t out_channels;
  int kernel_size;
  int stride;
  int padding;
  bool use_bias;
};

// Native compatibility route. The caller validates tensor dtype, shapes, and
// geometry and records fallback policy before entering these functions.
Tensor
ConvTranspose2DForwardNative(const Tensor &input, const Tensor &weights,
                             const Tensor &bias,
                             const ConvTranspose2DNativeGeometry &geometry,
                             const ConvTranspose2DNativeConfig &config);

Tensor
ConvTranspose2DBackwardNative(const Tensor &cached_input,
                              const Tensor &grad_output, const Tensor &weights,
                              Tensor &grad_weights, Tensor &grad_bias,
                              const ConvTranspose2DNativeGeometry &geometry,
                              const ConvTranspose2DNativeConfig &config);

} // namespace cyxwiz
