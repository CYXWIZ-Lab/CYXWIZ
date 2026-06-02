#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cyxwiz {

Tensor Tensor::View(const std::vector<size_t>& new_shape) const {
    return Reshape(new_shape);
}

Tensor Tensor::Squeeze(int dim) const {
    std::vector<size_t> new_shape;

    if (dim == -1) {
        for (size_t size : shape_) {
            if (size != 1) {
                new_shape.push_back(size);
            }
        }
    } else {
        const int rank = static_cast<int>(shape_.size());
        const int normalized = tensor_utils::NormalizeDim(dim, rank);
        if (shape_[static_cast<size_t>(normalized)] != 1) {
            throw std::runtime_error("Tensor::Squeeze: selected dimension must have size 1");
        }

        new_shape.reserve(shape_.size() - 1);
        for (size_t i = 0; i < shape_.size(); i++) {
            if (i != static_cast<size_t>(normalized)) {
                new_shape.push_back(shape_[i]);
            }
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }
    return Reshape(new_shape);
}

Tensor Tensor::Unsqueeze(int dim) const {
    const int rank = static_cast<int>(shape_.size());
    const int normalized = tensor_utils::NormalizeDim(dim, rank, true);

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() + 1);
    for (int i = 0; i < rank; i++) {
        if (i == normalized) {
            new_shape.push_back(1);
        }
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }
    if (normalized == rank) {
        new_shape.push_back(1);
    }

    return Reshape(new_shape);
}

Tensor Tensor::Flatten() const {
    return Reshape({NumElements()});
}

Tensor Tensor::Flatten(int start_dim, int end_dim) const {
    const int rank = static_cast<int>(shape_.size());
    if (rank == 0) {
        return Reshape({1});
    }

    const int start = tensor_utils::NormalizeDim(start_dim, rank);
    const int end = tensor_utils::NormalizeDim(end_dim, rank);
    if (start > end) {
        throw std::runtime_error("Tensor::Flatten: start_dim must be <= end_dim");
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() - static_cast<size_t>(end - start));

    for (int i = 0; i < start; i++) {
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }
    new_shape.push_back(tensor_utils::CheckedProduct(
        shape_,
        static_cast<size_t>(start),
        static_cast<size_t>(end + 1),
        "Tensor shape operation: integer overflow in dimension product"));
    for (int i = end + 1; i < rank; i++) {
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }

    return Reshape(new_shape);
}

Tensor Tensor::Transpose(int dim0, int dim1) const {
    const int rank = static_cast<int>(shape_.size());
    const int first = tensor_utils::NormalizeDim(dim0, rank);
    const int second = tensor_utils::NormalizeDim(dim1, rank);

    if (first == second) {
        return Reshape(shape_);
    }

    std::vector<size_t> new_shape = shape_;
    std::swap(new_shape[static_cast<size_t>(first)], new_shape[static_cast<size_t>(second)]);

    Tensor result(new_shape, dtype_);
    const size_t element_size = tensor_utils::ElementSize(dtype_);
    const auto src_strides = tensor_utils::RowMajorStrides(shape_, "Tensor stride calculation overflow");
    const auto dst_strides = tensor_utils::RowMajorStrides(new_shape, "Tensor stride calculation overflow");

    const auto* src = static_cast<const unsigned char*>(Data());
    auto* dst = static_cast<unsigned char*>(result.Data());

    const size_t total = NumElements();
    std::vector<size_t> src_index(shape_.size(), 0);
    std::vector<size_t> dst_index(shape_.size(), 0);

    for (size_t linear = 0; linear < total; linear++) {
        size_t rem = linear;
        for (size_t axis = 0; axis < shape_.size(); axis++) {
            src_index[axis] = rem / src_strides[axis];
            rem %= src_strides[axis];
        }

        dst_index = src_index;
        std::swap(dst_index[static_cast<size_t>(first)], dst_index[static_cast<size_t>(second)]);

        size_t dst_linear = 0;
        for (size_t axis = 0; axis < dst_index.size(); axis++) {
            dst_linear += dst_index[axis] * dst_strides[axis];
        }

        std::memcpy(dst + dst_linear * element_size, src + linear * element_size, element_size);
    }

    return result;
}

Tensor Tensor::Permute(const std::vector<int>& dims) const {
    const int rank = static_cast<int>(shape_.size());
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Tensor::Permute: dims must match tensor rank");
    }

    std::vector<int> normalized_dims;
    normalized_dims.reserve(dims.size());
    std::vector<bool> seen(dims.size(), false);
    std::vector<size_t> new_shape;
    new_shape.reserve(dims.size());

    for (int dim : dims) {
        const int normalized = tensor_utils::NormalizeDim(dim, rank);
        const size_t axis = static_cast<size_t>(normalized);
        if (seen[axis]) {
            throw std::runtime_error("Tensor::Permute: dims must not contain duplicates");
        }
        seen[axis] = true;
        normalized_dims.push_back(normalized);
        new_shape.push_back(shape_[axis]);
    }

    Tensor result(new_shape, dtype_);
    const size_t element_size = tensor_utils::ElementSize(dtype_);
    const auto src_strides = tensor_utils::RowMajorStrides(shape_, "Tensor stride calculation overflow");
    const auto dst_strides = tensor_utils::RowMajorStrides(new_shape, "Tensor stride calculation overflow");

    const auto* src = static_cast<const unsigned char*>(Data());
    auto* dst = static_cast<unsigned char*>(result.Data());

    const size_t total = NumElements();
    std::vector<size_t> src_index(shape_.size(), 0);
    std::vector<size_t> dst_index(shape_.size(), 0);

    for (size_t linear = 0; linear < total; linear++) {
        size_t rem = linear;
        for (size_t axis = 0; axis < shape_.size(); axis++) {
            src_index[axis] = rem / src_strides[axis];
            rem %= src_strides[axis];
        }

        for (size_t axis = 0; axis < normalized_dims.size(); axis++) {
            dst_index[axis] = src_index[static_cast<size_t>(normalized_dims[axis])];
        }

        size_t dst_linear = 0;
        for (size_t axis = 0; axis < dst_index.size(); axis++) {
            dst_linear += dst_index[axis] * dst_strides[axis];
        }

        std::memcpy(dst + dst_linear * element_size, src + linear * element_size, element_size);
    }

    return result;
}

} // namespace cyxwiz
