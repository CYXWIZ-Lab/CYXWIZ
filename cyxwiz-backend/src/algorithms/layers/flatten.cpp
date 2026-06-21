#include "cyxwiz/layer.h"

namespace cyxwiz {

Tensor FlattenLayer::Forward(const Tensor& input) {
    input_shape_ = input.Shape();
    const auto& shape = input.Shape();

    // Flatten is a pure reshape — no GPU computation needed. We just
    // change the shape from [batch, d1, d2, ...] to [batch, d1*d2*...]
    // and keep the data buffer in place (row-major layout stays correct
    // for LinearLayer which expects row-major [batch, features]).
    //
    // Going through ArrayFire for this was wrong: the generic
    // TensorToAf/AfToTensor round-trip scrambles the data layout
    // (column-major vs row-major mismatch with LinearLayer's manual
    // transpose in its Forward). Pure CPU reshape avoids the issue.
    if (shape.size() <= 2) {
        return input;  // already 1D or 2D, nothing to flatten
    }

    size_t batch = shape[0];
    size_t flat = 1;
    for (size_t i = 1; i < shape.size(); i++) {
        flat *= shape[i];
    }

    return Tensor({batch, flat}, input.Data(), input.GetDataType());
}

Tensor FlattenLayer::Backward(const Tensor& grad_output) {
    // Pure CPU reshape back to the original shape saved in Forward.
    // Same reasoning as Forward: no GPU needed, just change the shape.
    return Tensor(input_shape_, grad_output.Data(), grad_output.GetDataType());
}

} // namespace cyxwiz
