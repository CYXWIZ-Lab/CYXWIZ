#include "cyxwiz/tensor.h"

#include <cstdint>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

template <typename T>
Tensor DotTyped(const Tensor& left, const Tensor& right) {
    const size_t count = left.NumElements();
    const T* a = left.Data<T>();
    const T* b = right.Data<T>();
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
    const T* a = left.Data<T>();
    const T* b = right.Data<T>();
    Tensor result({batch, 1}, left.GetDataType());
    T* out = result.Data<T>();

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
bool SupportsArrayFireDot(DataType dtype) {
    return dtype == DataType::Float32 || dtype == DataType::Float64;
}

Tensor DotArrayFire(const Tensor& left, const Tensor& right) {
    af::array products = left.GetArray() * right.GetArray();
    return Tensor(af::sum(products));
}

Tensor RowWiseDotArrayFire(const Tensor& left, const Tensor& right) {
    af::array products = left.GetArrayRowMajor2D() * right.GetArrayRowMajor2D();
    return Tensor::FromArrayRowMajor2D(af::sum(products, 1));
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
    const T* a = left.Data<T>();
    const T* b = right.Data<T>();
    T* out = result.Data<T>();

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

} // namespace

Tensor Tensor::Dot(const Tensor& other) const {
    RequireSameDType(*this, other, "Tensor::Dot: data types must match");
    if (shape_.size() == 1 && other.Shape().size() == 1) {
        if (shape_[0] != other.Shape()[0]) {
            throw std::runtime_error("Tensor::Dot: vector sizes must match");
        }
#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (NumElements() > 0 && SupportsArrayFireDot(dtype_)) {
            try {
                return DotArrayFire(*this, other);
            } catch (const af::exception& e) {
                spdlog::warn("ArrayFire Tensor::Dot failed, using CPU fallback: {}", e.what());
            }
        }
#endif
        return DotPreservingType(*this, other);
    }
    if (shape_.size() == 2 && other.Shape().size() == 2) {
        if (shape_ != other.Shape()) {
            throw std::runtime_error("Tensor::Dot: 2D row-wise input shapes must match");
        }
#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (NumElements() > 0 && SupportsArrayFireDot(dtype_)) {
            try {
                return RowWiseDotArrayFire(*this, other);
            } catch (const af::exception& e) {
                spdlog::warn("ArrayFire row-wise Tensor::Dot failed, using CPU fallback: {}", e.what());
            }
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
    return BatchMatMulPreservingType(*this, other);
}

} // namespace cyxwiz
