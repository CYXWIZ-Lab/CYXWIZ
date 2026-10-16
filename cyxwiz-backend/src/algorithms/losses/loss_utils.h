#pragma once

#include "cyxwiz/loss.h"
#include "cyxwiz/tensor.h"
#include "../arrayfire_backend_utils.h"

#include <vector>

#include <arrayfire.h>

namespace cyxwiz {
namespace loss_detail {

void ValidateFloat32Pair(const Tensor& predictions, const Tensor& targets, const char* name);

// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops.
[[noreturn]] void ThrowLossDeviceError(const char* operation_name, const af::exception& error);

af::array TensorToAf(const Tensor& t);
Tensor AfToTensor(const af::array& arr);
Tensor AfToTensor(const af::array& arr,
                  const std::vector<size_t>& semantic_shape);
af::array ApplyReduction(const af::array& loss, Reduction reduction);
af::array StableSoftmax(const af::array& x, int axis = 0);
af::array SignLike(const af::array& x);

} // namespace loss_detail
} // namespace cyxwiz
