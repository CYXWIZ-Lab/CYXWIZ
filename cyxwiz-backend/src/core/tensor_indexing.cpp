#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <cstdint>
#include <stdexcept>

namespace cyxwiz {

namespace {

size_t CheckedLinearIndex(const std::vector<size_t>& shape, std::initializer_list<size_t> indices) {
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Tensor index rank does not match tensor rank");
    }

    const auto strides = tensor_utils::RowMajorStrides(shape, "Tensor indexing: stride overflow");
    size_t linear = 0;
    size_t axis = 0;
    for (size_t index : indices) {
        if (index >= shape[axis]) {
            throw std::out_of_range("Tensor index out of range");
        }
        linear += index * strides[axis];
        axis++;
    }
    return linear;
}

float ReadScalarAsFloat(const Tensor& tensor, size_t index) {
    switch (tensor.GetDataType()) {
        case DataType::Float32: return tensor.Data<float>()[index];
        case DataType::Float64: return static_cast<float>(tensor.Data<double>()[index]);
        case DataType::Int32: return static_cast<float>(tensor.Data<int32_t>()[index]);
        case DataType::Int64: return static_cast<float>(tensor.Data<int64_t>()[index]);
        case DataType::UInt8: return static_cast<float>(tensor.Data<uint8_t>()[index]);
    }
    throw std::runtime_error("Tensor::At: unsupported data type");
}

void WriteScalarFromFloat(Tensor& tensor, size_t index, float value) {
    switch (tensor.GetDataType()) {
        case DataType::Float32: tensor.Data<float>()[index] = value; return;
        case DataType::Float64: tensor.Data<double>()[index] = static_cast<double>(value); return;
        case DataType::Int32: tensor.Data<int32_t>()[index] = static_cast<int32_t>(value); return;
        case DataType::Int64: tensor.Data<int64_t>()[index] = static_cast<int64_t>(value); return;
        case DataType::UInt8: tensor.Data<uint8_t>()[index] = static_cast<uint8_t>(value); return;
    }
    throw std::runtime_error("Tensor::Set: unsupported data type");
}

int NormalizeSliceIndex(int index, int dim_size, bool default_end) {
    if (default_end && index == -1) {
        return dim_size;
    }
    if (index < 0) {
        index += dim_size;
    }
    if (index < 0) {
        return 0;
    }
    if (index > dim_size) {
        return dim_size;
    }
    return index;
}

} // namespace

float Tensor::At(size_t idx) const {
    if (idx >= NumElements()) {
        throw std::out_of_range("Tensor::At: index out of range");
    }
    return ReadScalarAsFloat(*this, idx);
}

float Tensor::At(size_t i, size_t j) const {
    return At(CheckedLinearIndex(shape_, {i, j}));
}

float Tensor::At(size_t i, size_t j, size_t k) const {
    return At(CheckedLinearIndex(shape_, {i, j, k}));
}

float Tensor::At(size_t i, size_t j, size_t k, size_t l) const {
    return At(CheckedLinearIndex(shape_, {i, j, k, l}));
}

void Tensor::Set(size_t idx, float value) {
    if (idx >= NumElements()) {
        throw std::out_of_range("Tensor::Set: index out of range");
    }
    WriteScalarFromFloat(*this, idx, value);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    MarkHostModified();
#endif
}

void Tensor::Set(size_t i, size_t j, float value) {
    Set(CheckedLinearIndex(shape_, {i, j}), value);
}

void Tensor::Set(size_t i, size_t j, size_t k, float value) {
    Set(CheckedLinearIndex(shape_, {i, j, k}), value);
}

void Tensor::Set(size_t i, size_t j, size_t k, size_t l, float value) {
    Set(CheckedLinearIndex(shape_, {i, j, k, l}), value);
}

Tensor Tensor::Slice(int dim, int start, int end, int step) const {
    if (step <= 0) {
        throw std::runtime_error("Tensor::Slice: step must be positive");
    }

    const int rank = static_cast<int>(shape_.size());
    const int axis = tensor_utils::NormalizeDim(dim, rank);
    const int dim_size = static_cast<int>(shape_[static_cast<size_t>(axis)]);
    const int begin = NormalizeSliceIndex(start, dim_size, false);
    const int finish = NormalizeSliceIndex(end, dim_size, true);

    if (begin >= finish) {
        throw std::runtime_error("Tensor::Slice: start must be before end");
    }

    const size_t slice_size = static_cast<size_t>((finish - begin + step - 1) / step);
    std::vector<size_t> out_shape = shape_;
    out_shape[static_cast<size_t>(axis)] = slice_size;
    Tensor result(out_shape, dtype_);

    const auto src_strides = tensor_utils::RowMajorStrides(shape_, "Tensor indexing: stride overflow");
    const auto dst_strides = tensor_utils::RowMajorStrides(out_shape, "Tensor indexing: stride overflow");
    const size_t total = result.NumElements();

    std::vector<size_t> index(out_shape.size(), 0);
    for (size_t dst_linear = 0; dst_linear < total; dst_linear++) {
        size_t remaining = dst_linear;
        for (size_t i = 0; i < out_shape.size(); i++) {
            index[i] = remaining / dst_strides[i];
            remaining %= dst_strides[i];
        }

        index[static_cast<size_t>(axis)] =
            static_cast<size_t>(begin) + index[static_cast<size_t>(axis)] * static_cast<size_t>(step);

        size_t src_linear = 0;
        for (size_t i = 0; i < index.size(); i++) {
            src_linear += index[i] * src_strides[i];
        }
        tensor_utils::CopyElement(*this, result, src_linear, dst_linear);
    }

    return result;
}

Tensor Tensor::IndexSelect(int dim, const std::vector<int>& indices) const {
    const int rank = static_cast<int>(shape_.size());
    const int axis = tensor_utils::NormalizeDim(dim, rank);
    const int dim_size = static_cast<int>(shape_[static_cast<size_t>(axis)]);

    std::vector<size_t> out_shape = shape_;
    out_shape[static_cast<size_t>(axis)] = indices.size();
    Tensor result(out_shape, dtype_);

    const auto src_strides = tensor_utils::RowMajorStrides(shape_, "Tensor indexing: stride overflow");
    const auto dst_strides = tensor_utils::RowMajorStrides(out_shape, "Tensor indexing: stride overflow");
    const size_t total = result.NumElements();

    std::vector<size_t> index(out_shape.size(), 0);
    for (size_t dst_linear = 0; dst_linear < total; dst_linear++) {
        size_t remaining = dst_linear;
        for (size_t i = 0; i < out_shape.size(); i++) {
            index[i] = remaining / dst_strides[i];
            remaining %= dst_strides[i];
        }

        int selected = indices[index[static_cast<size_t>(axis)]];
        if (selected < 0) {
            selected += dim_size;
        }
        if (selected < 0 || selected >= dim_size) {
            throw std::out_of_range("Tensor::IndexSelect: selected index out of range");
        }
        index[static_cast<size_t>(axis)] = static_cast<size_t>(selected);

        size_t src_linear = 0;
        for (size_t i = 0; i < index.size(); i++) {
            src_linear += index[i] * src_strides[i];
        }
        tensor_utils::CopyElement(*this, result, src_linear, dst_linear);
    }

    return result;
}

} // namespace cyxwiz
