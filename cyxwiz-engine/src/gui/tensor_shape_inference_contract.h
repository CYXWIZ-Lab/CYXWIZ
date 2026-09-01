#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace gui::tensor_shape_inference {

std::optional<std::vector<size_t>> InferPool2DOutputShape(
    const std::vector<size_t>& input_shape,
    int pool_size,
    int stride);

std::optional<std::vector<size_t>> ResolveReshapeOutputShape(
    const std::vector<size_t>& input_shape,
    std::string_view requested_shape);

}  // namespace gui::tensor_shape_inference
