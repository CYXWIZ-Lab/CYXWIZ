#include "cyxwiz/tensor.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"

namespace cyxwiz {

namespace {

enum class CompareOp {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual
};

const char* TensorComparisonOperationName(CompareOp op) {
    switch (op) {
        case CompareOp::Greater: return "Tensor::operator>";
        case CompareOp::GreaterEqual: return "Tensor::operator>=";
        case CompareOp::Less: return "Tensor::operator<";
        case CompareOp::LessEqual: return "Tensor::operator<=";
        case CompareOp::Equal: return "Tensor::operator==";
        case CompareOp::NotEqual: return "Tensor::operator!=";
    }
    return "Tensor::comparison";
}

const char* ScalarComparisonOperationName(CompareOp op) {
    switch (op) {
        case CompareOp::Greater: return "Tensor::operator>(scalar)";
        case CompareOp::GreaterEqual: return "Tensor::operator>=(scalar)";
        case CompareOp::Less: return "Tensor::operator<(scalar)";
        case CompareOp::LessEqual: return "Tensor::operator<=(scalar)";
        case CompareOp::Equal: return "Tensor::operator==(scalar)";
        case CompareOp::NotEqual: return "Tensor::operator!=(scalar)";
    }
    return "Tensor::scalar_comparison";
}

af::array CompareArrays(const af::array& left,
                        const af::array& right,
                        CompareOp op) {
    switch (op) {
        case CompareOp::Greater: return left > right;
        case CompareOp::GreaterEqual: return left >= right;
        case CompareOp::Less: return left < right;
        case CompareOp::LessEqual: return left <= right;
        case CompareOp::Equal: return left == right;
        case CompareOp::NotEqual: return left != right;
    }
    throw std::runtime_error("Tensor comparison: invalid ArrayFire operation");
}

// Scalars compare in the input's real type; the result is UInt8 0 / 1.
Tensor ApplyScalarComparison(const Tensor& input, float scalar, CompareOp op) {
    const DataType common_dtype =
        tensor_math_utils::RealType(input.GetDataType());
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), DataType::UInt8);
    }
    try {
        const af::dtype type = tensor_math_utils::ArrayFireType(common_dtype);
        const af::array values = tensor_math_utils::ElementwiseArray(input).as(type);
        af::array output = CompareArrays(values, af::constant(scalar, values.dims(), type), op).as(u8);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(ScalarComparisonOperationName(op), e));
    }
}

Tensor CompareTensors(const Tensor& left, const Tensor& right, CompareOp op) {
    const std::vector<size_t> output_shape =
        Tensor::BroadcastShape(left.Shape(), right.Shape());
    const DataType common_dtype = tensor_math_utils::PromoteTypes(
        left.GetDataType(), right.GetDataType());
    if (tensor_utils::CheckedProduct(
            output_shape, 0, output_shape.size(),
            "Tensor comparison: output shape overflow") == 0) {
        return Tensor(output_shape, DataType::UInt8);
    }
    try {
        af::array output = CompareArrays(tensor_math_utils::BroadcastArray(left, output_shape, common_dtype),
                                         tensor_math_utils::BroadcastArray(right, output_shape, common_dtype),
                                         op).as(u8);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, output_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(TensorComparisonOperationName(op), e));
    }
}

} // namespace

Tensor Tensor::operator>(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::Greater);
}

Tensor Tensor::operator>(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::Greater);
}

Tensor Tensor::operator>=(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::GreaterEqual);
}

Tensor Tensor::operator>=(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::GreaterEqual);
}

Tensor Tensor::operator<(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::Less);
}

Tensor Tensor::operator<(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::Less);
}

Tensor Tensor::operator<=(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::LessEqual);
}

Tensor Tensor::operator<=(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::LessEqual);
}

Tensor Tensor::operator==(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::Equal);
}

Tensor Tensor::operator==(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::Equal);
}

Tensor Tensor::operator!=(const Tensor& other) const {
    return CompareTensors(*this, other, CompareOp::NotEqual);
}

Tensor Tensor::operator!=(float scalar) const {
    return ApplyScalarComparison(*this, scalar, CompareOp::NotEqual);
}

} // namespace cyxwiz
