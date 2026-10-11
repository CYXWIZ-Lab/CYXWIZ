#include "cyxwiz/tensor.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"
#endif

namespace cyxwiz {

namespace {

std::vector<size_t> LeftPadShape(const std::vector<size_t>& shape, size_t rank) {
    std::vector<size_t> padded(rank, 1);
    const size_t offset = rank - shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        padded[offset + i] = shape[i];
    }
    return padded;
}

// Any rank on the device: ranks above 4 collapse neighbouring axes that
// broadcast the same way (tensor_math_utils::BroadcastArray).
Tensor MaterializeBroadcast(const Tensor& input,
                            const std::vector<size_t>& target_shape,
                            const char* operation_name) {
    if (target_shape.size() < input.Shape().size()) {
        throw std::runtime_error(
            std::string(operation_name) +
            ": target rank must be >= source rank");
    }

    const std::vector<size_t> padded_shape =
        LeftPadShape(input.Shape(), target_shape.size());
    for (size_t axis = 0; axis < target_shape.size(); ++axis) {
        if (padded_shape[axis] != 1 &&
            padded_shape[axis] != target_shape[axis]) {
            throw std::runtime_error(
                std::string(operation_name) +
                ": incompatible target shape");
        }
    }

    const size_t total = tensor_utils::CheckedProduct(
        target_shape,
        0,
        target_shape.size(),
        "Tensor broadcasting: output shape overflow");
    if (total == 0) {
        return Tensor(target_shape, input.GetDataType());
    }

    if (input.Shape() == target_shape) return input.Clone();
    if (padded_shape == target_shape) {
        return input.Reshape(target_shape);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array output = tensor_math_utils::BroadcastArray(input, target_shape, input.GetDataType());
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, target_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(operation_name, e));
    }
#else
    throw std::runtime_error(std::string(operation_name) + " runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace

bool Tensor::IsBroadcastable(const std::vector<size_t>& shape1,
                             const std::vector<size_t>& shape2) {
    const size_t rank = (std::max)(shape1.size(), shape2.size());
    for (size_t i = 0; i < rank; i++) {
        const size_t d1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        const size_t d2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> Tensor::BroadcastShape(const std::vector<size_t>& shape1,
                                           const std::vector<size_t>& shape2) {
    if (!IsBroadcastable(shape1, shape2)) {
        throw std::runtime_error("Tensor::BroadcastShape: shapes are not broadcastable");
    }

    const size_t rank = (std::max)(shape1.size(), shape2.size());
    std::vector<size_t> result(rank, 1);
    for (size_t i = 0; i < rank; i++) {
        const size_t d1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        const size_t d2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
        if (d1 == d2) {
            result[rank - 1 - i] = d1;
        } else if (d1 == 1) {
            result[rank - 1 - i] = d2;
        } else {
            result[rank - 1 - i] = d1;
        }
    }
    return result;
}

Tensor Tensor::BroadcastTo(const std::vector<size_t>& target_shape) const {
    if (BroadcastShape(shape_, target_shape) != target_shape) {
        throw std::runtime_error("Tensor::BroadcastTo: target shape is not the broadcast result");
    }
    return MaterializeBroadcast(
        *this, target_shape, "Tensor::BroadcastTo");
}

Tensor Tensor::Expand(const std::vector<size_t>& target_shape) const {
    return MaterializeBroadcast(
        *this, target_shape, "Tensor::Expand");
}

} // namespace cyxwiz
