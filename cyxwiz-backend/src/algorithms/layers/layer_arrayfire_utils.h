#pragma once

#include "cyxwiz/tensor.h"
#include <cstddef>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array TensorToAf(const Tensor& t);
af::array TensorToAf3DRowMajor(const Tensor& t);
Tensor AfToTensor(const af::array& arr);
Tensor AfToTensor3DRowMajor(const af::array& arr);
int CheckedIntDim(size_t value, const char* name);
af::array XavierUniform(int fan_in, int fan_out, af::dim4 dims);
af::array KaimingUniform(int fan_in, af::dim4 dims);
#endif

} // namespace cyxwiz
