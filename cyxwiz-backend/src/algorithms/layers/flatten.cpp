#include "cyxwiz/layers/flatten.h"

#include <stdexcept>

namespace cyxwiz {

Tensor FlattenLayer::Forward(const Tensor& input) {
    Tensor output = input.Flatten(1);
    input_shape_ = input.Shape();
    output_shape_ = output.Shape();
    output_dtype_ = output.GetDataType();
    return output;
}

Tensor FlattenLayer::Backward(const Tensor& grad_output) {
    if (input_shape_.empty()) {
        throw std::logic_error(
            "FlattenLayer::Backward requires a successful Forward call");
    }
    if (grad_output.Shape() != output_shape_) {
        throw std::runtime_error(
            "FlattenLayer::Backward gradient shape does not match Forward output");
    }
    if (grad_output.GetDataType() != output_dtype_) {
        throw std::runtime_error(
            "FlattenLayer::Backward gradient dtype does not match Forward output");
    }
    return grad_output.Reshape(input_shape_);
}

} // namespace cyxwiz
