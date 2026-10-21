#include "layer_utils.h"


#include <limits>
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

size_t CheckedSpatialPaddedExtent(size_t input_extent,
                                  int padding,
                                  const char* layer_name) {
    if (padding < 0) {
        throw std::invalid_argument(
            std::string(layer_name) + " requires non-negative padding");
    }
    const size_t pad = static_cast<size_t>(padding);
    if (pad > ((std::numeric_limits<size_t>::max)() - input_extent) / 2) {
        throw std::overflow_error(
            std::string(layer_name) + " padded input extent overflow");
    }
    return input_extent + pad * 2;
}

size_t CheckedLayerProduct(size_t left,
                           size_t right,
                           const char* layer_name,
                           const char* quantity) {
    if (left != 0 && right > (std::numeric_limits<size_t>::max)() / left) {
        throw std::overflow_error(
            std::string(layer_name) + " " + quantity + " overflow");
    }
    return left * right;
}

} // namespace cyxwiz
