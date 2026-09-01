#include "layer_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cyxwiz {

size_t Pool4DIndex(size_t h, size_t w, size_t c, size_t b,
                   size_t width, size_t channels, size_t batch_size) {
    return ((h * width + w) * channels + c) * batch_size + b;
}

void ValidateSpatial4DInput(const Tensor& input, const char* name) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " requires Float32 input");
    }
    if (input.Shape().size() != 4) {
        throw std::runtime_error(std::string(name) + " expects [H, W, C, N] input");
    }
    if (input.Shape()[0] == 0 || input.Shape()[1] == 0 ||
        input.Shape()[2] == 0 || input.Shape()[3] == 0) {
        throw std::runtime_error(std::string(name) + " does not support empty dimensions");
    }
}

void ValidatePoolInput(const Tensor& input, const char* name) {
    ValidateSpatial4DInput(input, name);
}

ResizeLinearSample ComputeResizeLinearSample(size_t out_index, size_t in_size, int scale_factor) {
    float source = (static_cast<float>(out_index) + 0.5f) / static_cast<float>(scale_factor) - 0.5f;
    source = std::clamp(source, 0.0f, static_cast<float>(in_size - 1));
    const size_t lower = static_cast<size_t>(std::floor(source));
    const size_t upper = std::min(lower + 1, in_size - 1);
    return {lower, upper, source - static_cast<float>(lower)};
}

} // namespace cyxwiz
