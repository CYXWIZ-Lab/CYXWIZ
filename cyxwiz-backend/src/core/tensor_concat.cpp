#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cyxwiz {

namespace {

bool SafeAdd(size_t a, size_t b, size_t& result) {
    if (b > (std::numeric_limits<size_t>::max)() - a) {
        return false;
    }
    result = a + b;
    return true;
}

void ValidateSplitSize(int value, const char* name) {
    if (value <= 0) {
        throw std::runtime_error(name);
    }
}

} // namespace

Tensor Tensor::Cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Tensor::Cat: tensor list must not be empty");
    }

    const auto& ref_shape = tensors.front().Shape();
    const int rank = static_cast<int>(ref_shape.size());
    const int axis = tensor_utils::NormalizeDim(dim, rank);
    const DataType dtype = tensors.front().GetDataType();

    std::vector<size_t> out_shape = ref_shape;
    out_shape[static_cast<size_t>(axis)] = 0;

    for (const Tensor& tensor : tensors) {
        if (tensor.GetDataType() != dtype) {
            throw std::runtime_error("Tensor::Cat: all tensors must have the same data type");
        }
        if (tensor.Shape().size() != ref_shape.size()) {
            throw std::runtime_error("Tensor::Cat: all tensors must have the same rank");
        }

        for (size_t i = 0; i < ref_shape.size(); i++) {
            if (i != static_cast<size_t>(axis) && tensor.Shape()[i] != ref_shape[i]) {
                throw std::runtime_error("Tensor::Cat: shapes must match except along concat dimension");
            }
        }

        size_t total = 0;
        if (!SafeAdd(out_shape[static_cast<size_t>(axis)],
                     tensor.Shape()[static_cast<size_t>(axis)],
                     total)) {
            throw std::overflow_error("Tensor::Cat: concat dimension overflow");
        }
        out_shape[static_cast<size_t>(axis)] = total;
    }

    Tensor result(out_shape, dtype);
    const auto dst_strides = tensor_utils::RowMajorStrides(out_shape, "Tensor concat: stride overflow");

    size_t axis_offset = 0;
    for (const Tensor& tensor : tensors) {
        const auto src_strides = tensor_utils::RowMajorStrides(tensor.Shape(), "Tensor concat: stride overflow");
        std::vector<size_t> index(tensor.Shape().size(), 0);

        for (size_t src_linear = 0; src_linear < tensor.NumElements(); src_linear++) {
            size_t remaining = src_linear;
            for (size_t i = 0; i < tensor.Shape().size(); i++) {
                index[i] = remaining / src_strides[i];
                remaining %= src_strides[i];
            }

            size_t dst_linear = 0;
            for (size_t i = 0; i < index.size(); i++) {
                const size_t coord = i == static_cast<size_t>(axis) ? index[i] + axis_offset : index[i];
                dst_linear += coord * dst_strides[i];
            }

            tensor_utils::CopyElement(tensor, result, src_linear, dst_linear);
        }

        axis_offset += tensor.Shape()[static_cast<size_t>(axis)];
    }

    return result;
}

Tensor Tensor::Stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Tensor::Stack: tensor list must not be empty");
    }

    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(tensors.front().Shape().size()), true);
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        expanded.push_back(tensor.Unsqueeze(axis));
    }
    return Cat(expanded, axis);
}

std::vector<Tensor> Tensor::Split(int split_size, int dim) const {
    ValidateSplitSize(split_size, "Tensor::Split: split_size must be positive");

    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];
    std::vector<Tensor> result;

    for (size_t start = 0; start < dim_size; start += static_cast<size_t>(split_size)) {
        const size_t end = (std::min)(start + static_cast<size_t>(split_size), dim_size);
        result.push_back(Slice(axis, static_cast<int>(start), static_cast<int>(end)));
    }
    return result;
}

std::vector<Tensor> Tensor::Split(const std::vector<int>& sizes, int dim) const {
    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];

    std::vector<Tensor> result;
    result.reserve(sizes.size());

    size_t start = 0;
    for (int size : sizes) {
        ValidateSplitSize(size, "Tensor::Split: sizes must be positive");
        size_t end = 0;
        if (!SafeAdd(start, static_cast<size_t>(size), end) || end > dim_size) {
            throw std::runtime_error("Tensor::Split: split sizes exceed dimension size");
        }
        result.push_back(Slice(axis, static_cast<int>(start), static_cast<int>(end)));
        start = end;
    }

    if (start != dim_size) {
        throw std::runtime_error("Tensor::Split: split sizes must cover the full dimension");
    }
    return result;
}

std::vector<Tensor> Tensor::Chunk(int chunks, int dim) const {
    ValidateSplitSize(chunks, "Tensor::Chunk: chunks must be positive");

    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];
    if (dim_size == 0) {
        return {};
    }

    const size_t chunk_size = (dim_size + static_cast<size_t>(chunks) - 1) / static_cast<size_t>(chunks);
    return Split(static_cast<int>(chunk_size), axis);
}

} // namespace cyxwiz
