#include "cyxwiz/tensor.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>
#include "../algorithms/arrayfire_backend_utils.h"
#include "../algorithms/arrayfire_host_values.h"

namespace cyxwiz {

namespace {

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

// Integer matrices multiply in float64 (ArrayFire has no integer matmul).
// Exact while every partial sum stays below 2^53, which max(|A| |B|) bounds;
// the result then wraps as integer bmm does (mod 2^8 / 2^32; Int64 needs no
// wrap below 2^53). Beyond the bound, or without float64, it refuses.
af::array IntegerMatMul(const af::array& left, const af::array& right, DataType dtype) {
    if (!DeviceHasDouble()) {
        throw std::runtime_error(
            "Tensor::BatchMatMul: integer matrices multiply in float64, which this device does not support");
    }
    const af::array a = left.as(f64);
    const af::array b = right.as(f64);
    constexpr double kExactLimit = 9007199254740992.0;  // 2^53
    const af::array bound = (af::max)(af::flat(af::matmul(af::abs(a), af::abs(b))));
    const Tensor bound_tensor = Tensor::FromSemanticArray(bound, {});
    {
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::MetricScalarReadback, "Tensor::BatchMatMul integer bound");
        if (!(bound_tensor.ReadData<double>()[0] < kExactLimit)) {
            throw std::runtime_error(
                "Tensor::BatchMatMul: integer products reach 2^53 and cannot be computed exactly in float64");
        }
    }
    af::array product = af::matmul(a, b);
    if (dtype == DataType::UInt8) {
        product = af::mod(product, 256.0);
        product = af::select(product < 0.0, product + 256.0, product);
    } else if (dtype == DataType::Int32) {
        constexpr double kTwo32 = 4294967296.0;
        constexpr double kTwo31 = 2147483648.0;
        product = af::mod(product + kTwo31, kTwo32);
        product = af::select(product < 0.0, product + kTwo32, product) - kTwo31;
    }
    return product.as(tensor_math_utils::ArrayFireType(dtype));
}

Tensor BatchMatMulArrayFire(const Tensor& left,
                            const Tensor& right,
                            const std::vector<size_t>& output_shape) {
    const af::array left_matrices = af::reorder(left.GetArrayRowMajor3D(), 1, 2, 0);
    const af::array right_matrices = af::reorder(right.GetArrayRowMajor3D(), 1, 2, 0);
    const DataType dtype = left.GetDataType();
    const af::array output_matrices = (dtype == DataType::Float32 || dtype == DataType::Float64)
        ? af::matmul(left_matrices, right_matrices)
        : IntegerMatMul(left_matrices, right_matrices, dtype);
    af::array output = af::reorder(output_matrices, 2, 0, 1);
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
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
        try {
            return DotArrayFire(*this, other);
        } catch (const af::exception& e) {
            throw std::runtime_error(DeviceError("Tensor::Dot", e));
        }
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
        try {
            return RowWiseDotArrayFire(*this, other);
        } catch (const af::exception& e) {
            throw std::runtime_error(DeviceError("Tensor::Dot", e));
        }
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

    try {
        return BatchMatMulArrayFire(*this, other, output_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::BatchMatMul", e));
    }
}

} // namespace cyxwiz
