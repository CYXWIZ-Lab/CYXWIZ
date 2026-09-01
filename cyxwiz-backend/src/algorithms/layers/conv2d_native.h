#pragma once

#include "cyxwiz/tensor.h"

#include <cstddef>

namespace cyxwiz {

struct Conv2DNativeGeometry {
    size_t in_h;
    size_t in_w;
    size_t in_channels;
    size_t batch_size;
    size_t out_h;
    size_t out_w;
};

struct Conv2DNativeConfig {
    size_t out_channels;
    int kernel_size;
    int stride;
    int padding;
    bool use_bias;
};

// Native compatibility route. The caller validates tensor dtype, shapes, and
// geometry and records fallback policy before entering these functions.
Tensor Conv2DForwardNative(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias,
                           const Conv2DNativeGeometry& geometry,
                           const Conv2DNativeConfig& config);

Tensor Conv2DBackwardNative(const Tensor& cached_input,
                            const Tensor& grad_output,
                            const Tensor& weights,
                            Tensor& grad_weights,
                            Tensor& grad_bias,
                            const Conv2DNativeGeometry& geometry,
                            const Conv2DNativeConfig& config);

} // namespace cyxwiz
