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

enum class UnaryOp {
    Pow,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sign,
    Clip,
    Negate
};

const char* UnaryOperationName(UnaryOp operation) {
    switch (operation) {
        case UnaryOp::Pow: return "Tensor::Pow(scalar)";
        case UnaryOp::Sqrt: return "Tensor::Sqrt";
        case UnaryOp::Exp: return "Tensor::Exp";
        case UnaryOp::Log: return "Tensor::Log";
        case UnaryOp::Sin: return "Tensor::Sin";
        case UnaryOp::Cos: return "Tensor::Cos";
        case UnaryOp::Abs: return "Tensor::Abs";
        case UnaryOp::Sign: return "Tensor::Sign";
        case UnaryOp::Clip: return "Tensor::Clip";
        case UnaryOp::Negate: return "Tensor::operator-()";
    }
    return "Tensor::unary";
}

af::array RealUnary(const af::array& values, UnaryOp operation, float first, float second) {
    switch (operation) {
        case UnaryOp::Pow: return af::pow(values, first);
        case UnaryOp::Sqrt: return af::sqrt(values);
        case UnaryOp::Exp: return af::exp(values);
        case UnaryOp::Log: return af::log(values);
        case UnaryOp::Sin: return af::sin(values);
        case UnaryOp::Cos: return af::cos(values);
        case UnaryOp::Clip: return (af::min)((af::max)(values, first), second);
        default: break;
    }
    throw std::runtime_error("Tensor real unary: invalid ArrayFire operation");
}

af::array ArrayFireSign(const af::array& values) {
    const af::array one = af::constant(1, values.dims(), values.type());
    const af::array zero = af::constant(0, values.dims(), values.type());
    const af::array negative_one =
        af::constant(-1, values.dims(), values.type());
    return af::select(
        values > 0, one, af::select(values < 0, negative_one, zero));
}

af::array PreservingUnary(const af::array& values, UnaryOp operation) {
    switch (operation) {
        case UnaryOp::Abs: return af::abs(values);
        case UnaryOp::Sign: return ArrayFireSign(values);
        // Integers wrap like two's complement (0 - x), floats negate.
        case UnaryOp::Negate: return af::constant(0, values.dims(), values.type()) - values;
        default: break;
    }
    throw std::runtime_error("Tensor preserving unary: invalid ArrayFire operation");
}

// Integer powers by square-and-multiply in the output type (wrapping);
// negative exponents give 0. Floats use af::pow.
af::array TensorPow(const af::array& left, const af::array& right, DataType output_dtype) {
    if (!tensor_math_utils::IsIntegralType(output_dtype)) {
        return af::pow(left, right);
    }
    const af::dtype type = tensor_math_utils::ArrayFireType(output_dtype);
    const unsigned bit_count = output_dtype == DataType::UInt8 ? 8u
        : output_dtype == DataType::Int32 ? 32u : 64u;
    const af::array nonnegative = right >= 0;
    af::array remaining = af::select(
        nonnegative, right, af::constant(0, right.dims(), right.type())).as(u64);
    af::array factor = left;
    af::array output = af::constant(1, left.dims(), type);
    for (unsigned bit = 0; bit < bit_count; ++bit) {
        const af::array odd = (remaining & 1u) != 0u;
        output = af::select(odd, (output * factor).as(type), output);
        remaining = remaining >> 1u;
        if (bit + 1u < bit_count) {
            factor = (factor * factor).as(type);
        }
    }
    return af::select(nonnegative, output, af::constant(0, output.dims(), output.type()));
}

Tensor ApplyRealUnary(const Tensor& input,
                      UnaryOp operation,
                      float first = 0.0f,
                      float second = 0.0f) {
    const DataType output_dtype =
        tensor_math_utils::RealType(input.GetDataType());
    if (input.NumElements() == 0) return Tensor(input.Shape(), output_dtype);
    try {
        const af::dtype type = tensor_math_utils::ArrayFireType(output_dtype);
        af::array output = RealUnary(tensor_math_utils::ElementwiseArray(input).as(type),
                                     operation, first, second).as(type);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(UnaryOperationName(operation), e));
    }
}

Tensor ApplyPreservingUnary(const Tensor& input, UnaryOp operation) {
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), input.GetDataType());
    }
    try {
        af::array output = PreservingUnary(tensor_math_utils::ElementwiseArray(input), operation)
                               .as(tensor_math_utils::ArrayFireType(input.GetDataType()));
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(UnaryOperationName(operation), e));
    }
}

} // namespace

Tensor Tensor::Pow(float exponent) const {
    return ApplyRealUnary(*this, UnaryOp::Pow, exponent);
}

Tensor Tensor::Pow(const Tensor& exponent) const {
    const std::vector<size_t> output_shape =
        Tensor::BroadcastShape(shape_, exponent.Shape());
    const DataType output_dtype =
        tensor_math_utils::PromoteTypes(dtype_, exponent.GetDataType());
    if (tensor_utils::CheckedProduct(
            output_shape, 0, output_shape.size(),
            "Tensor::Pow: output shape overflow") == 0) {
        return Tensor(output_shape, output_dtype);
    }
    try {
        af::array output = TensorPow(tensor_math_utils::BroadcastArray(*this, output_shape, output_dtype),
                                     tensor_math_utils::BroadcastArray(exponent, output_shape, output_dtype),
                                     output_dtype)
                               .as(tensor_math_utils::ArrayFireType(output_dtype));
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, output_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Pow(tensor)", e));
    }
}

Tensor Tensor::Sqrt() const {
    return ApplyRealUnary(*this, UnaryOp::Sqrt);
}

Tensor Tensor::Exp() const {
    return ApplyRealUnary(*this, UnaryOp::Exp);
}

Tensor Tensor::Log() const {
    return ApplyRealUnary(*this, UnaryOp::Log);
}

Tensor Tensor::Sin() const {
    return ApplyRealUnary(*this, UnaryOp::Sin);
}

Tensor Tensor::Cos() const {
    return ApplyRealUnary(*this, UnaryOp::Cos);
}

Tensor Tensor::Abs() const {
    return ApplyPreservingUnary(*this, UnaryOp::Abs);
}

Tensor Tensor::Sign() const {
    return ApplyPreservingUnary(*this, UnaryOp::Sign);
}

Tensor Tensor::Clip(float min_val, float max_val) const {
    return ApplyRealUnary(*this, UnaryOp::Clip, min_val, max_val);
}

Tensor Tensor::operator-() const {
    return ApplyPreservingUnary(*this, UnaryOp::Negate);
}

} // namespace cyxwiz
