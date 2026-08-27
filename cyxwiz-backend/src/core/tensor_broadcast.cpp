#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordBroadcastArrayFireFallback(
    const char* operation_name,
    const Tensor& input,
    const std::vector<size_t>& output_shape,
    const std::string& attributes,
    const char* error_message) {
    const std::string message =
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(input.GetDataType()),
            tensor_backend_observation::BuildTensorOpSignature(
                {input.Shape()},
                output_shape,
                input.GetDataType(),
                attributes),
            error_message);
    spdlog::warn("{}", message);
}

Tensor ApplyBroadcastArrayFire(const Tensor& input,
                               const std::vector<size_t>& target_shape) {
    tensor_math_utils::ArrayFireDims(target_shape);
    af::array output = tensor_math_utils::BroadcastArray(
        input, target_shape, input.GetDataType());
    output.eval();
    return Tensor::FromSemanticArray(output, target_shape);
}
#endif

Tensor ApplyBroadcastNative(const Tensor& input,
                            const std::vector<size_t>& target_shape,
                            const std::vector<size_t>& padded_shape) {
    Tensor result(target_shape, input.GetDataType());
    const size_t total = result.NumElements();
    if (total == 0) return result;

    const size_t element_size =
        tensor_utils::ElementSize(input.GetDataType());
    const auto src_strides = tensor_utils::RowMajorStrides(
        padded_shape,
        "Tensor broadcasting: source stride overflow",
        true);
    const auto dst_strides = tensor_utils::RowMajorStrides(
        target_shape,
        "Tensor broadcasting: destination stride overflow");
    const auto* src =
        static_cast<const unsigned char*>(input.ReadData());
    auto* dst = static_cast<unsigned char*>(result.MutableData());

    for (size_t dst_linear = 0; dst_linear < total; ++dst_linear) {
        size_t remaining = dst_linear;
        size_t src_linear = 0;
        for (size_t axis = 0; axis < target_shape.size(); ++axis) {
            const size_t coordinate = remaining / dst_strides[axis];
            remaining %= dst_strides[axis];
            src_linear += coordinate * src_strides[axis];
        }
        std::memcpy(
            dst + dst_linear * element_size,
            src + src_linear * element_size,
            element_size);
    }
    return result;
}

Tensor MaterializeBroadcast(const Tensor& input,
                            const std::vector<size_t>& target_shape,
                            const char* operation_name,
                            [[maybe_unused]] const char* operation_attribute) {
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const std::string attributes =
        std::string(operation_attribute) +
        ";rank=" + std::to_string(target_shape.size());
    if (target_shape.size() > 4) {
        RecordBroadcastArrayFireFallback(
            operation_name,
            input,
            target_shape,
            attributes,
            "ArrayFire Tensor broadcasting supports ranks up to 4");
        return ApplyBroadcastNative(input, target_shape, padded_shape);
    }
#endif

    if (input.Shape() == target_shape) return input.Clone();
    if (padded_shape == target_shape) {
        return input.Reshape(target_shape);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return ApplyBroadcastArrayFire(input, target_shape);
    } catch (const af::exception& error) {
        RecordBroadcastArrayFireFallback(
            operation_name,
            input,
            target_shape,
            attributes,
            error.what());
    }
#endif

    return ApplyBroadcastNative(input, target_shape, padded_shape);
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
        *this, target_shape, "Tensor::BroadcastTo", "op=broadcast_to");
}

Tensor Tensor::Expand(const std::vector<size_t>& target_shape) const {
    return MaterializeBroadcast(
        *this, target_shape, "Tensor::Expand", "op=expand");
}

} // namespace cyxwiz
