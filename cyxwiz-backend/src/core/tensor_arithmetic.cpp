#include "cyxwiz/tensor.h"

#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"
#endif

namespace cyxwiz {

namespace {

enum class ArithmeticOp {
    Add,
    Subtract,
    Multiply,
    Divide
};

const char* TensorOperationName(ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return "Tensor::operator+";
        case ArithmeticOp::Subtract: return "Tensor::operator-";
        case ArithmeticOp::Multiply: return "Tensor::operator*";
        case ArithmeticOp::Divide: return "Tensor::operator/";
    }
    return "Tensor::arithmetic";
}

const char* ScalarOperationName(ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return "Tensor::operator+(scalar)";
        case ArithmeticOp::Subtract: return "Tensor::operator-(scalar)";
        case ArithmeticOp::Multiply: return "Tensor::operator*(scalar)";
        case ArithmeticOp::Divide: return "Tensor::operator/(scalar)";
    }
    return "Tensor::scalar_arithmetic";
}

DataType ArithmeticResultType(DataType left,
                              DataType right,
                              ArithmeticOp operation) {
    const DataType promoted = tensor_math_utils::PromoteTypes(left, right);
    if (operation == ArithmeticOp::Divide &&
        tensor_math_utils::IsIntegralType(promoted)) {
        return DataType::Float32;
    }
    return promoted;
}

DataType ScalarResultType(DataType input) {
    return input == DataType::Float64 ? DataType::Float64 : DataType::Float32;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array Arithmetic(const af::array& left, const af::array& right, ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return left + right;
        case ArithmeticOp::Subtract: return left - right;
        case ArithmeticOp::Multiply: return left * right;
        case ArithmeticOp::Divide: break;
    }
    return left / right;
}

af::array ScalarArithmetic(const af::array& values, float scalar, ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return values + scalar;
        case ArithmeticOp::Subtract: return values - scalar;
        case ArithmeticOp::Multiply: return values * scalar;
        case ArithmeticOp::Divide: break;
    }
    return values / scalar;
}
#endif

Tensor ApplyTensorArithmetic(const Tensor& left,
                             const Tensor& right,
                             ArithmeticOp operation) {
    const std::vector<size_t> output_shape =
        Tensor::BroadcastShape(left.Shape(), right.Shape());
    const DataType output_dtype = ArithmeticResultType(
        left.GetDataType(), right.GetDataType(), operation);
    if (tensor_utils::CheckedProduct(
            output_shape, 0, output_shape.size(),
            "Tensor arithmetic: output shape overflow") == 0) {
        return Tensor(output_shape, output_dtype);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::dtype type = tensor_math_utils::ArrayFireType(output_dtype);
        af::array output = Arithmetic(tensor_math_utils::BroadcastArray(left, output_shape, output_dtype),
                                      tensor_math_utils::BroadcastArray(right, output_shape, output_dtype),
                                      operation).as(type);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, output_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(TensorOperationName(operation), e));
    }
#else
    throw std::runtime_error(std::string(TensorOperationName(operation)) +
                             " runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor ApplyScalarArithmetic(const Tensor& input,
                             float scalar,
                             ArithmeticOp operation) {
    const DataType output_dtype = ScalarResultType(input.GetDataType());
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), output_dtype);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::dtype type = tensor_math_utils::ArrayFireType(output_dtype);
        af::array output = ScalarArithmetic(tensor_math_utils::ElementwiseArray(input).as(type),
                                            scalar, operation).as(type);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(ScalarOperationName(operation), e));
    }
#else
    (void)scalar;
    throw std::runtime_error(std::string(ScalarOperationName(operation)) +
                             " runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace

Tensor Tensor::operator+(const Tensor& other) const {
    return ApplyTensorArithmetic(*this, other, ArithmeticOp::Add);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return ApplyTensorArithmetic(*this, other, ArithmeticOp::Subtract);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return ApplyTensorArithmetic(*this, other, ArithmeticOp::Multiply);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return ApplyTensorArithmetic(*this, other, ArithmeticOp::Divide);
}

Tensor Tensor::operator+(float scalar) const {
    return ApplyScalarArithmetic(*this, scalar, ArithmeticOp::Add);
}

Tensor Tensor::operator-(float scalar) const {
    return ApplyScalarArithmetic(*this, scalar, ArithmeticOp::Subtract);
}

Tensor Tensor::operator*(float scalar) const {
    return ApplyScalarArithmetic(*this, scalar, ArithmeticOp::Multiply);
}

Tensor Tensor::operator/(float scalar) const {
    return ApplyScalarArithmetic(*this, scalar, ArithmeticOp::Divide);
}

} // namespace cyxwiz
