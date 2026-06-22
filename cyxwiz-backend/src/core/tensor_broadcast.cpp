#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

std::vector<size_t> LeftPadShape(const std::vector<size_t>& shape, size_t rank) {
    if (shape.size() > rank) {
        throw std::runtime_error("Tensor::Expand: target shape rank is too small");
    }

    std::vector<size_t> padded(rank, 1);
    const size_t offset = rank - shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        padded[offset + i] = shape[i];
    }
    return padded;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool IsArrayFire2DExpandSupported(const Tensor& input,
                                  const std::vector<size_t>& target_shape,
                                  const std::vector<size_t>& padded_shape) {
    if (input.GetDataType() != DataType::Float32 &&
        input.GetDataType() != DataType::Float64) {
        return false;
    }
    if (input.Shape().size() != 2 || target_shape.size() != 2) {
        return false;
    }
    if (input.NumElements() == 0 || target_shape[0] == 0 || target_shape[1] == 0) {
        return false;
    }
    for (size_t i = 0; i < target_shape.size(); i++) {
        if (padded_shape[i] != target_shape[i] &&
            padded_shape[i] != 1) {
            return false;
        }
    }
    return true;
}
#endif

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
        result[rank - 1 - i] = (std::max)(d1, d2);
    }
    return result;
}

Tensor Tensor::BroadcastTo(const std::vector<size_t>& target_shape) const {
    if (BroadcastShape(shape_, target_shape) != target_shape) {
        throw std::runtime_error("Tensor::BroadcastTo: target shape is not the broadcast result");
    }
    return Expand(target_shape);
}

Tensor Tensor::Expand(const std::vector<size_t>& target_shape) const {
    if (target_shape.size() < shape_.size()) {
        throw std::runtime_error("Tensor::Expand: target rank must be >= source rank");
    }

    const std::vector<size_t> padded_shape = LeftPadShape(shape_, target_shape.size());
    for (size_t i = 0; i < target_shape.size(); i++) {
        if (padded_shape[i] != 1 && padded_shape[i] != target_shape[i]) {
            throw std::runtime_error("Tensor::Expand: incompatible target shape");
        }
    }

    if (padded_shape == target_shape) {
        return padded_shape == shape_ ? Clone() : Reshape(target_shape);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsArrayFire2DExpandSupported(*this, target_shape, padded_shape)) {
        try {
            const unsigned tile_rows = padded_shape[0] == target_shape[0]
                ? 1u
                : static_cast<unsigned>(target_shape[0]);
            const unsigned tile_cols = padded_shape[1] == target_shape[1]
                ? 1u
                : static_cast<unsigned>(target_shape[1]);
            return Tensor::FromArrayRowMajor2D(
                af::tile(GetArrayRowMajor2D(), tile_rows, tile_cols));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Expand: ArrayFire expand failed, falling back to CPU: {}", e.what());
        }
    }
#endif

    Tensor result(target_shape, dtype_);
    const size_t element_size = tensor_utils::ElementSize(dtype_);
    const auto src_strides = tensor_utils::RowMajorStrides(
        padded_shape,
        "Tensor broadcasting: stride overflow",
        true);
    const auto dst_strides = tensor_utils::RowMajorStrides(
        target_shape,
        "Tensor broadcasting: stride overflow");
    const auto* src = static_cast<const unsigned char*>(Data());
    auto* dst = static_cast<unsigned char*>(result.Data());

    const size_t total = result.NumElements();
    for (size_t dst_linear = 0; dst_linear < total; dst_linear++) {
        size_t remaining = dst_linear;
        size_t src_linear = 0;
        for (size_t axis = 0; axis < target_shape.size(); axis++) {
            const size_t coord = remaining / dst_strides[axis];
            remaining %= dst_strides[axis];
            src_linear += coord * src_strides[axis];
        }

        std::memcpy(dst + dst_linear * element_size, src + src_linear * element_size, element_size);
    }

    return result;
}

} // namespace cyxwiz
