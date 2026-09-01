#pragma once

#include "cyxwiz/tensor.h"

#include <cstddef>

namespace cyxwiz {

struct Conv1DNativeGeometry {
    size_t input_length;
    size_t in_channels;
    size_t batch_size;
    size_t output_length;
};

struct Conv1DNativeConfig {
    size_t out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool use_bias;
};

// Native compatibility route. The caller validates tensor dtype, shapes, and
// geometry and records fallback policy before entering these functions.
Tensor Conv1DForwardNative(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias,
                           const Conv1DNativeGeometry& geometry,
                           const Conv1DNativeConfig& config);

Tensor Conv1DBackwardNative(const Tensor& cached_input,
                            const Tensor& grad_output,
                            const Tensor& weights,
                            Tensor& grad_weights,
                            Tensor& grad_bias,
                            const Conv1DNativeGeometry& geometry,
                            const Conv1DNativeConfig& config);

} // namespace cyxwiz
