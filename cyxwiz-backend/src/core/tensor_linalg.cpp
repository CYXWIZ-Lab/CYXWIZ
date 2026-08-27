#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

template <typename T>
Tensor DotTyped(const Tensor& left, const Tensor& right) {
    const size_t count = left.NumElements();
    const T* a = left.ReadData<T>();
    const T* b = right.ReadData<T>();
    T total{};
    for (size_t i = 0; i < count; i++) {
        total = static_cast<T>(total + a[i] * b[i]);
    }
    return Tensor({1}, &total, left.GetDataType());
}

template <typename T>
Tensor RowWiseDotTyped(const Tensor& left, const Tensor& right) {
    const auto& shape = left.Shape();
    const size_t batch = shape[0];
    const size_t features = shape[1];
    const T* a = left.ReadData<T>();
    const T* b = right.ReadData<T>();
    Tensor result({batch, 1}, left.GetDataType());
    T* out = result.MutableData<T>();

    for (size_t row = 0; row < batch; row++) {
        T total{};
        for (size_t col = 0; col < features; col++) {
            const size_t idx = row * features + col;
            total = static_cast<T>(total + a[idx] * b[idx]);
        }
        out[row] = total;
    }

    return result;
}

Tensor DotPreservingType(const Tensor& left, const Tensor& right) {
    switch (left.GetDataType()) {
        case DataType::Float32: return DotTyped<float>(left, right);
        case DataType::Float64: return DotTyped<double>(left, right);
        case DataType::Int32: return DotTyped<int32_t>(left, right);
        case DataType::Int64: return DotTyped<int64_t>(left, right);
        case DataType::UInt8: return DotTyped<uint8_t>(left, right);
    }
    throw std::runtime_error("Tensor::Dot: unsupported data type");
}

Tensor RowWiseDotPreservingType(const Tensor& left, const Tensor& right) {
    switch (left.GetDataType()) {
        case DataType::Float32: return RowWiseDotTyped<float>(left, right);
        case DataType::Float64: return RowWiseDotTyped<double>(left, right);
        case DataType::Int32: return RowWiseDotTyped<int32_t>(left, right);
        case DataType::Int64: return RowWiseDotTyped<int64_t>(left, right);
        case DataType::UInt8: return RowWiseDotTyped<uint8_t>(left, right);
    }
    throw std::runtime_error("Tensor::Dot: unsupported data type");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordLinalgArrayFireFallback(
    const char* operation_name,
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    const std::string& attributes,
    const char* error_message) {
    const std::string message =
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(left.GetDataType()),
            tensor_backend_observation::BuildTensorOpSignature(
                {left.Shape(), right.Shape()},
                output_shape,
                left.GetDataType(),
                attributes),
            error_message);
    spdlog::warn("{}", message);
}

Tensor DotArrayFire(const Tensor& left, const Tensor& right) {
    af::array output = af::sum(
        left.GetSemanticArray() * right.GetSemanticArray());
    output = output.as(
        tensor_math_utils::ArrayFireType(left.GetDataType()));
    output.eval();
    return Tensor::FromSemanticArray(output, {1});
}

Tensor RowWiseDotArrayFire(const Tensor& left, const Tensor& right) {
    af::array products = left.GetArrayRowMajor2D() * right.GetArrayRowMajor2D();
    af::array output = af::sum(products, 1).as(
        tensor_math_utils::ArrayFireType(left.GetDataType()));
    output.eval();
    return Tensor::FromSemanticArray(output, {left.Shape()[0], 1});
}

Tensor BatchMatMulArrayFire(const Tensor& left,
                            const Tensor& right,
                            const std::vector<size_t>& output_shape) {
    af::array left_matrices =
        af::reorder(left.GetArrayRowMajor3D(), 1, 2, 0);
    af::array right_matrices =
        af::reorder(right.GetArrayRowMajor3D(), 1, 2, 0);
    af::array output_matrices = af::matmul(left_matrices, right_matrices);
    af::array output = af::reorder(output_matrices, 2, 0, 1);
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}
#endif

template <typename T>
Tensor BatchMatMulTyped(const Tensor& left, const Tensor& right) {
    const auto& a_shape = left.Shape();
    const auto& b_shape = right.Shape();
    const size_t batch = a_shape[0];
    const size_t rows = a_shape[1];
    const size_t shared = a_shape[2];
    const size_t cols = b_shape[2];

    Tensor result({batch, rows, cols}, left.GetDataType());
    const T* a = left.ReadData<T>();
    const T* b = right.ReadData<T>();
    T* out = result.MutableData<T>();

    for (size_t batch_idx = 0; batch_idx < batch; batch_idx++) {
        const size_t a_batch_offset = batch_idx * rows * shared;
        const size_t b_batch_offset = batch_idx * shared * cols;
        const size_t out_batch_offset = batch_idx * rows * cols;
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                T total{};
                for (size_t k = 0; k < shared; k++) {
                    const T left_value = a[a_batch_offset + row * shared + k];
                    const T right_value = b[b_batch_offset + k * cols + col];
                    total = static_cast<T>(total + left_value * right_value);
                }
                out[out_batch_offset + row * cols + col] = total;
            }
        }
    }

    return result;
}

Tensor BatchMatMulPreservingType(const Tensor& left, const Tensor& right) {
    switch (left.GetDataType()) {
        case DataType::Float32: return BatchMatMulTyped<float>(left, right);
        case DataType::Float64: return BatchMatMulTyped<double>(left, right);
        case DataType::Int32: return BatchMatMulTyped<int32_t>(left, right);
        case DataType::Int64: return BatchMatMulTyped<int64_t>(left, right);
        case DataType::UInt8: return BatchMatMulTyped<uint8_t>(left, right);
    }
    throw std::runtime_error("Tensor::BatchMatMul: unsupported data type");
}

void RequireSameDType(const Tensor& left, const Tensor& right, const char* message) {
    if (left.GetDataType() != right.GetDataType()) {
        throw std::runtime_error(message);
    }
}

size_t ValidateLinalgOutputShape(const std::vector<size_t>& shape,
                                 DataType dtype) {
    const size_t count = tensor_utils::CheckedProduct(
        shape,
        0,
        shape.size(),
        "Tensor linalg: output shape element count overflow");
    size_t bytes = 0;
    if (!tensor_utils::SafeMultiply(
            count, tensor_utils::ElementSize(dtype), bytes)) {
        throw std::overflow_error(
            "Tensor linalg: output shape byte count overflow");
    }
    return count;
}

} // namespace

Tensor Tensor::Dot(const Tensor& other) const {
    RequireSameDType(*this, other, "Tensor::Dot: data types must match");
    if (shape_.size() == 1 && other.Shape().size() == 1) {
        if (shape_[0] != other.Shape()[0]) {
            throw std::runtime_error("Tensor::Dot: vector sizes must match");
        }
        const std::vector<size_t> output_shape{1};
        ValidateLinalgOutputShape(output_shape, dtype_);
        if (NumElements() == 0) return Tensor::Zeros(output_shape, dtype_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
        try {
            return DotArrayFire(*this, other);
        } catch (const af::exception& error) {
            RecordLinalgArrayFireFallback(
                "Tensor::Dot",
                *this,
                other,
                output_shape,
                "mode=vector",
                error.what());
        }
#endif
        return DotPreservingType(*this, other);
    }

    if (shape_.size() == 2 && other.Shape().size() == 2) {
        if (shape_ != other.Shape()) {
            throw std::runtime_error("Tensor::Dot: 2D row-wise input shapes must match");
        }
        const std::vector<size_t> output_shape{shape_[0], 1};
        const size_t output_count =
            ValidateLinalgOutputShape(output_shape, dtype_);
        if (output_count == 0) return Tensor(output_shape, dtype_);
        if (shape_[1] == 0) return Tensor::Zeros(output_shape, dtype_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
        try {
            return RowWiseDotArrayFire(*this, other);
        } catch (const af::exception& error) {
            RecordLinalgArrayFireFallback(
                "Tensor::Dot",
                *this,
                other,
                output_shape,
                "mode=rowwise",
                error.what());
        }
#endif
        return RowWiseDotPreservingType(*this, other);
    }

    throw std::runtime_error("Tensor::Dot: both tensors must be 1D or both 2D");
}

Tensor Tensor::BatchMatMul(const Tensor& other) const {
    RequireSameDType(*this, other, "Tensor::BatchMatMul: data types must match");
    if (shape_.size() != 3 || other.Shape().size() != 3) {
        throw std::runtime_error("Tensor::BatchMatMul: both tensors must be 3D");
    }
    if (shape_[0] != other.Shape()[0]) {
        throw std::runtime_error("Tensor::BatchMatMul: batch dimensions must match");
    }
    if (shape_[2] != other.Shape()[1]) {
        throw std::runtime_error("Tensor::BatchMatMul: inner dimensions must match");
    }

    const std::vector<size_t> output_shape{
        shape_[0], shape_[1], other.Shape()[2]};
    const size_t output_count =
        ValidateLinalgOutputShape(output_shape, dtype_);
    if (output_count == 0) {
        return Tensor(output_shape, dtype_);
    }
    if (shape_[2] == 0) {
        return Tensor::Zeros(output_shape, dtype_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (dtype_ != DataType::Float32 && dtype_ != DataType::Float64) {
        RecordLinalgArrayFireFallback(
            "Tensor::BatchMatMul",
            *this,
            other,
            output_shape,
            "mode=torch_bmm;batch_broadcast=false",
            "unsupported dtype: ArrayFire BatchMatMul supports "
            "floating-point data types only");
        return BatchMatMulPreservingType(*this, other);
    }

    try {
        return BatchMatMulArrayFire(*this, other, output_shape);
    } catch (const af::exception& error) {
        RecordLinalgArrayFireFallback(
            "Tensor::BatchMatMul",
            *this,
            other,
            output_shape,
            "mode=torch_bmm;batch_broadcast=false",
            error.what());
    }
#endif

    return BatchMatMulPreservingType(*this, other);
}

} // namespace cyxwiz
