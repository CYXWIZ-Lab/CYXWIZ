#pragma once

// Row-major index arithmetic for host references in Tensor contract tests.
#include <cstddef>
#include <vector>

namespace tensor_test {

// Row-major coordinates of a flat index.
inline std::vector<size_t> Unravel(size_t index, const std::vector<size_t>& shape) {
    std::vector<size_t> coordinates(shape.size());
    for (size_t axis = shape.size(); axis-- > 0;) {
        coordinates[axis] = index % shape[axis];
        index /= shape[axis];
    }
    return coordinates;
}

// Flat row-major index of coordinates.
inline size_t Ravel(const std::vector<size_t>& coordinates, const std::vector<size_t>& shape) {
    size_t index = 0;
    for (size_t axis = 0; axis < shape.size(); ++axis) index = index * shape[axis] + coordinates[axis];
    return index;
}

}  // namespace tensor_test
