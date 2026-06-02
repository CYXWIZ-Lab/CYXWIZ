#pragma once

#include "cyxwiz/tensor.h"

#include <cstddef>
#include <vector>

namespace cyxwiz::tensor_utils {

bool SafeMultiply(size_t a, size_t b, size_t& result);
size_t ElementSize(DataType dtype);
int NormalizeDim(int dim, int rank, bool allow_end = false);
size_t CheckedProduct(const std::vector<size_t>& values,
                      size_t begin,
                      size_t end,
                      const char* overflow_message);
std::vector<size_t> RowMajorStrides(const std::vector<size_t>& shape,
                                    const char* overflow_message,
                                    bool broadcast_zero_stride = false);
void CopyElement(const Tensor& src_tensor, Tensor& dst_tensor, size_t src_linear, size_t dst_linear);

} // namespace cyxwiz::tensor_utils
