#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cyxwiz {
namespace spatial_shape_detail {
inline size_t CheckedFloatElements(size_t left, size_t right) {
  if (right == 0 ||
      right > static_cast<size_t>((std::numeric_limits<int>::max)()))
    throw std::invalid_argument(
        "Spatial dimensions must be positive and fit the backend int range");
  if (left > (std::numeric_limits<size_t>::max)() / sizeof(float) / right)
    throw std::overflow_error(
        "Spatial Float32 tensor byte count overflows size_t");
  return left * right;
}
} // namespace spatial_shape_detail

// Shared metadata-only contract for independently linked compiler and runtime
// consumers. No Tensor allocation, device selection, or backend dependency.
inline size_t SpatialSampleElements(const std::vector<size_t> &sample_shape) {
  if (sample_shape.size() != 3)
    throw std::invalid_argument(
        "Spatial sample shape must be [H,W,C], without a batch axis");
  size_t elements = 1;
  for (size_t extent : sample_shape)
    elements = spatial_shape_detail::CheckedFloatElements(elements, extent);
  return elements;
}

inline std::vector<size_t>
SpatialRuntimeShape(const std::vector<size_t> &sample_shape,
                    size_t batch_size) {
  spatial_shape_detail::CheckedFloatElements(
      SpatialSampleElements(sample_shape), batch_size);
  return {sample_shape[0], sample_shape[1], sample_shape[2], batch_size};
}
} // namespace cyxwiz
