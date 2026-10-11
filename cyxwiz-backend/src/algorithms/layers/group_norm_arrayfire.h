#pragma once

// GroupNorm on ArrayFire, shared by GroupNormLayer and InstanceNorm2DLayer
// (InstanceNorm is GroupNorm with one channel per group). Inputs are row-major
// [H, W, C, B] Float32; statistics are per (group, batch) over H x W x C/groups.
#include "cyxwiz/tensor.h"

#include <cstddef>

namespace cyxwiz {

struct GroupNormForward {
    Tensor output;
    Tensor normalized;  // [H, W, C, B]
    Tensor std_inv;     // [groups, B]
};

struct GroupNormBackward {
    Tensor grad_input;
    Tensor grad_gamma;  // [C], empty without affine
    Tensor grad_beta;   // [C], empty without affine
};

// gamma / beta are null without affine.
GroupNormForward GroupNormForwardArrayFire(const Tensor& input, size_t groups, float eps,
                                           const Tensor* gamma, const Tensor* beta);
GroupNormBackward GroupNormBackwardArrayFire(const Tensor& grad_output, const Tensor& normalized,
                                             const Tensor& std_inv, size_t groups, const Tensor* gamma);

}  // namespace cyxwiz
