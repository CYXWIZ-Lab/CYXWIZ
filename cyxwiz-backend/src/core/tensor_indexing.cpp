#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"

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
        case DataType::Float32: return tensor.ReadData<float>()[index];
        case DataType::Float64: return static_cast<float>(tensor.ReadData<double>()[index]);
        case DataType::Int32: return static_cast<float>(tensor.ReadData<int32_t>()[index]);
        case DataType::Int64: return static_cast<float>(tensor.ReadData<int64_t>()[index]);
        case DataType::UInt8: return static_cast<float>(tensor.ReadData<uint8_t>()[index]);
    }
    throw std::runtime_error("Tensor::At: unsupported data type");
}

void WriteScalarFromFloat(Tensor& tensor, size_t index, float value) {
    switch (tensor.GetDataType()) {
        case DataType::Float32: tensor.MutableData<float>()[index] = value; return;
        case DataType::Float64: tensor.MutableData<double>()[index] = static_cast<double>(value); return;
        case DataType::Int32: tensor.MutableData<int32_t>()[index] = static_cast<int32_t>(value); return;
        case DataType::Int64: tensor.MutableData<int64_t>()[index] = static_cast<int64_t>(value); return;
        case DataType::UInt8: tensor.MutableData<uint8_t>()[index] = static_cast<uint8_t>(value); return;
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

int CheckedIndexDimensionSize(size_t dimension, const char* operation_name) {
    if (dimension > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::overflow_error(
            std::string(operation_name) +
            ": dimension exceeds the supported indexing range");
    }
    return static_cast<int>(dimension);
}

std::vector<unsigned> NormalizeIndexSelectIndices(const std::vector<int>& indices,
                                                  int dim_size) {
    std::vector<unsigned> normalized;
    normalized.reserve(indices.size());
    for (int selected : indices) {
        if (selected < 0) {
            selected += dim_size;
        }
        if (selected < 0 || selected >= dim_size) {
            throw std::out_of_range("Tensor::IndexSelect: selected index out of range");
        }
        normalized.push_back(static_cast<unsigned>(selected));
    }
    return normalized;
}

// Gather positions along axis on the device. Up to rank 4 on the semantic
// view; above, the row-major flat storage viewed as [after, extent, before]
// and gathered on ArrayFire dim 1 (already the output's row-major order).
Tensor LookupArrayFire(const Tensor& input,
                       const std::vector<unsigned>& indices,
                       int axis,
                       const std::vector<size_t>& output_shape) {
    const af::array index_array(
        static_cast<dim_t>(indices.size()), indices.data());
    const std::vector<size_t>& shape = input.Shape();
    if (shape.size() <= 4) {
        af::array output = af::lookup(input.GetSemanticArray(), index_array, static_cast<unsigned>(axis));
        output.eval();
        return Tensor::FromSemanticArray(output, output_shape);
    }
    dim_t before = 1;
    dim_t after = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i < static_cast<size_t>(axis)) before *= static_cast<dim_t>(shape[i]);
        if (i > static_cast<size_t>(axis)) after *= static_cast<dim_t>(shape[i]);
    }
    const af::array view = af::moddims(input.GetRowMajorFlatArray(), after,
                                       static_cast<dim_t>(shape[static_cast<size_t>(axis)]), before);
    af::array output = af::flat(af::lookup(view, index_array, 1));
    output.eval();
    return Tensor::FromRowMajorFlatArray(output, output_shape);
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
    const int dim_size = CheckedIndexDimensionSize(
        shape_[static_cast<size_t>(axis)], "Tensor::Slice");
    const int begin = NormalizeSliceIndex(start, dim_size, false);
    const int finish = NormalizeSliceIndex(end, dim_size, true);

    const size_t slice_size = begin >= finish
        ? 0
        : static_cast<size_t>((finish - begin + step - 1) / step);
    std::vector<size_t> out_shape = shape_;
    out_shape[static_cast<size_t>(axis)] = slice_size;

    if (tensor_utils::CheckedProduct(
            out_shape, 0, out_shape.size(),
            "Tensor::Slice: output shape overflow") == 0) {
        return Tensor(out_shape, dtype_);
    }

    std::vector<unsigned> selected_indices;
    selected_indices.reserve(slice_size);
    for (size_t index = 0; index < slice_size; ++index) {
        selected_indices.push_back(static_cast<unsigned>(
            begin + static_cast<int>(index) * step));
    }

    try {
        return LookupArrayFire(*this, selected_indices, axis, out_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Slice", e));
    }
}

Tensor Tensor::IndexSelect(int dim, const std::vector<int>& indices) const {
    const int rank = static_cast<int>(shape_.size());
    const int axis = tensor_utils::NormalizeDim(dim, rank);
    const int dim_size = CheckedIndexDimensionSize(
        shape_[static_cast<size_t>(axis)], "Tensor::IndexSelect");
    const std::vector<unsigned> normalized_indices =
        NormalizeIndexSelectIndices(indices, dim_size);

    std::vector<size_t> out_shape = shape_;
    out_shape[static_cast<size_t>(axis)] = indices.size();

    if (tensor_utils::CheckedProduct(
            out_shape, 0, out_shape.size(),
            "Tensor::IndexSelect: output shape overflow") == 0) {
        return Tensor(out_shape, dtype_);
    }

    try {
        return LookupArrayFire(*this, normalized_indices, axis, out_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::IndexSelect", e));
    }
}

} // namespace cyxwiz
