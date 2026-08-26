#include "cyxwiz/tensor.h"

#include "tensor_backend_observation_utils.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

enum class UnaryOp {
    Pow,
    Sqrt,
    Exp,
    Log,
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
        case UnaryOp::Abs: return "Tensor::Abs";
        case UnaryOp::Sign: return "Tensor::Sign";
        case UnaryOp::Clip: return "Tensor::Clip";
        case UnaryOp::Negate: return "Tensor::operator-()";
    }
    return "Tensor::unary";
}

const char* UnaryName(UnaryOp operation) {
    switch (operation) {
        case UnaryOp::Pow: return "pow";
        case UnaryOp::Sqrt: return "sqrt";
        case UnaryOp::Exp: return "exp";
        case UnaryOp::Log: return "log";
        case UnaryOp::Abs: return "abs";
        case UnaryOp::Sign: return "sign";
        case UnaryOp::Clip: return "clip";
        case UnaryOp::Negate: return "negate";
    }
    return "unknown";
}

template <typename T>
T NegateWrapped(T value) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = Unsigned{} - static_cast<Unsigned>(value);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return -value;
    }
}

template <typename T>
T MultiplyWrapped(T left, T right) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result =
            static_cast<Unsigned>(left) * static_cast<Unsigned>(right);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return left * right;
    }
}

template <typename T>
T IntegerPower(T base, T exponent) {
    if constexpr (std::is_signed_v<T>) {
        if (exponent < 0) return T{};
    }
    using UnsignedExponent = std::make_unsigned_t<T>;
    UnsignedExponent remaining = static_cast<UnsignedExponent>(exponent);
    T result = static_cast<T>(1);
    while (remaining != 0) {
        if ((remaining & static_cast<UnsignedExponent>(1)) != 0) {
            result = MultiplyWrapped(result, base);
        }
        remaining >>= 1;
        if (remaining != 0) {
            base = MultiplyWrapped(base, base);
        }
    }
    return result;
}

template <typename In, typename Out>
Tensor NativeRealUnaryTyped(const Tensor& input,
                            UnaryOp operation,
                            float first,
                            float second,
                            DataType output_dtype) {
    Tensor result(input.Shape(), output_dtype);
    const In* source = input.ReadData<In>();
    Out* output = result.MutableData<Out>();
    for (size_t index = 0; index < input.NumElements(); ++index) {
        const Out value = static_cast<Out>(source[index]);
        switch (operation) {
            case UnaryOp::Pow:
                output[index] = std::pow(value, static_cast<Out>(first));
                break;
            case UnaryOp::Sqrt: output[index] = std::sqrt(value); break;
            case UnaryOp::Exp: output[index] = std::exp(value); break;
            case UnaryOp::Log: output[index] = std::log(value); break;
            case UnaryOp::Clip:
                output[index] = (std::min)(
                    (std::max)(value, static_cast<Out>(first)),
                    static_cast<Out>(second));
                break;
            default:
                throw std::runtime_error("Tensor real unary: invalid operation");
        }
    }
    return result;
}

template <typename In>
Tensor NativeRealUnaryOutput(const Tensor& input,
                             UnaryOp operation,
                             float first,
                             float second,
                             DataType output_dtype) {
    if (output_dtype == DataType::Float64) {
        return NativeRealUnaryTyped<In, double>(
            input, operation, first, second, output_dtype);
    }
    return NativeRealUnaryTyped<In, float>(
        input, operation, first, second, output_dtype);
}

Tensor NativeRealUnary(const Tensor& input,
                       UnaryOp operation,
                       float first,
                       float second,
                       DataType output_dtype) {
    switch (input.GetDataType()) {
        case DataType::Float32: return NativeRealUnaryOutput<float>(input, operation, first, second, output_dtype);
        case DataType::Float64: return NativeRealUnaryOutput<double>(input, operation, first, second, output_dtype);
        case DataType::Int32: return NativeRealUnaryOutput<int32_t>(input, operation, first, second, output_dtype);
        case DataType::Int64: return NativeRealUnaryOutput<int64_t>(input, operation, first, second, output_dtype);
        case DataType::UInt8: return NativeRealUnaryOutput<uint8_t>(input, operation, first, second, output_dtype);
    }
    throw std::runtime_error("Tensor real unary: unsupported data type");
}

template <typename T>
Tensor NativePreservingUnaryTyped(const Tensor& input, UnaryOp operation) {
    Tensor result(input.Shape(), input.GetDataType());
    const T* source = input.ReadData<T>();
    T* output = result.MutableData<T>();
    for (size_t index = 0; index < input.NumElements(); ++index) {
        const T value = source[index];
        switch (operation) {
            case UnaryOp::Abs:
                output[index] = value < T{} ? NegateWrapped(value) : value;
                break;
            case UnaryOp::Sign:
                output[index] = value > T{} ? static_cast<T>(1)
                    : value < T{} ? static_cast<T>(-1) : T{};
                break;
            case UnaryOp::Negate:
                output[index] = NegateWrapped(value);
                break;
            default:
                throw std::runtime_error("Tensor preserving unary: invalid operation");
        }
    }
    return result;
}

Tensor NativePreservingUnary(const Tensor& input, UnaryOp operation) {
    switch (input.GetDataType()) {
        case DataType::Float32: return NativePreservingUnaryTyped<float>(input, operation);
        case DataType::Float64: return NativePreservingUnaryTyped<double>(input, operation);
        case DataType::Int32: return NativePreservingUnaryTyped<int32_t>(input, operation);
        case DataType::Int64: return NativePreservingUnaryTyped<int64_t>(input, operation);
        case DataType::UInt8: return NativePreservingUnaryTyped<uint8_t>(input, operation);
    }
    throw std::runtime_error("Tensor preserving unary: unsupported data type");
}

template <typename Base, typename Exponent, typename Out>
Tensor NativeTensorPowTyped(const Tensor& base,
                            const Tensor& exponent,
                            const std::vector<size_t>& output_shape,
                            DataType output_dtype) {
    Tensor result(output_shape, output_dtype);
    const size_t count = result.NumElements();
    if (count == 0) return result;
    const auto output_strides = tensor_utils::RowMajorStrides(
        output_shape, "Tensor::Pow: output stride overflow");
    const auto base_strides = tensor_utils::RowMajorStrides(
        base.Shape(), "Tensor::Pow: base stride overflow", true);
    const auto exponent_strides = tensor_utils::RowMajorStrides(
        exponent.Shape(), "Tensor::Pow: exponent stride overflow", true);
    const Base* base_data = base.ReadData<Base>();
    const Exponent* exponent_data = exponent.ReadData<Exponent>();
    Out* output = result.MutableData<Out>();
    for (size_t index = 0; index < count; ++index) {
        const size_t base_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, base.Shape(), base_strides);
        const size_t exponent_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides,
            exponent.Shape(), exponent_strides);
        const Out left = static_cast<Out>(base_data[base_index]);
        const Out right = static_cast<Out>(exponent_data[exponent_index]);
        if constexpr (std::is_integral_v<Out>) {
            output[index] = IntegerPower(left, right);
        } else {
            output[index] = std::pow(left, right);
        }
    }
    return result;
}

template <typename Base, typename Exponent>
Tensor DispatchPowOutput(const Tensor& base,
                         const Tensor& exponent,
                         const std::vector<size_t>& output_shape,
                         DataType output_dtype) {
    switch (output_dtype) {
        case DataType::Float32: return NativeTensorPowTyped<Base, Exponent, float>(base, exponent, output_shape, output_dtype);
        case DataType::Float64: return NativeTensorPowTyped<Base, Exponent, double>(base, exponent, output_shape, output_dtype);
        case DataType::Int32: return NativeTensorPowTyped<Base, Exponent, int32_t>(base, exponent, output_shape, output_dtype);
        case DataType::Int64: return NativeTensorPowTyped<Base, Exponent, int64_t>(base, exponent, output_shape, output_dtype);
        case DataType::UInt8: return NativeTensorPowTyped<Base, Exponent, uint8_t>(base, exponent, output_shape, output_dtype);
    }
    throw std::runtime_error("Tensor::Pow: unsupported output data type");
}

template <typename Base>
Tensor DispatchPowExponent(const Tensor& base,
                           const Tensor& exponent,
                           const std::vector<size_t>& output_shape,
                           DataType output_dtype) {
    switch (exponent.GetDataType()) {
        case DataType::Float32: return DispatchPowOutput<Base, float>(base, exponent, output_shape, output_dtype);
        case DataType::Float64: return DispatchPowOutput<Base, double>(base, exponent, output_shape, output_dtype);
        case DataType::Int32: return DispatchPowOutput<Base, int32_t>(base, exponent, output_shape, output_dtype);
        case DataType::Int64: return DispatchPowOutput<Base, int64_t>(base, exponent, output_shape, output_dtype);
        case DataType::UInt8: return DispatchPowOutput<Base, uint8_t>(base, exponent, output_shape, output_dtype);
    }
    throw std::runtime_error("Tensor::Pow: unsupported exponent data type");
}

Tensor NativeTensorPow(const Tensor& base,
                       const Tensor& exponent,
                       const std::vector<size_t>& output_shape,
                       DataType output_dtype) {
    switch (base.GetDataType()) {
        case DataType::Float32: return DispatchPowExponent<float>(base, exponent, output_shape, output_dtype);
        case DataType::Float64: return DispatchPowExponent<double>(base, exponent, output_shape, output_dtype);
        case DataType::Int32: return DispatchPowExponent<int32_t>(base, exponent, output_shape, output_dtype);
        case DataType::Int64: return DispatchPowExponent<int64_t>(base, exponent, output_shape, output_dtype);
        case DataType::UInt8: return DispatchPowExponent<uint8_t>(base, exponent, output_shape, output_dtype);
    }
    throw std::runtime_error("Tensor::Pow: unsupported base data type");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordUnaryArrayFireFallback(const char* operation_name,
                         const Tensor& input,
                         const std::vector<size_t>& output_shape,
                         DataType output_dtype,
                         const std::string& attributes,
                         const char* error_message) {
    const std::string message = tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(output_dtype),
        tensor_backend_observation::BuildTensorOpSignature(
            {input.Shape()}, output_shape, output_dtype, attributes),
        error_message);
    spdlog::warn("{}", message);
}

void RecordPowArrayFireFallback(const Tensor& base,
                       const Tensor& exponent,
                       const std::vector<size_t>& output_shape,
                       DataType output_dtype,
                       const char* error_message) {
    const std::string message = tensor_backend_observation::RecordArrayFireFallback(
        "Tensor::Pow(tensor)",
        tensor_backend_observation::DataTypeName(output_dtype),
        tensor_backend_observation::BuildTensorOpSignature(
            {base.Shape(), exponent.Shape()}, output_shape, output_dtype,
            "op=pow;exponent=tensor;broadcast=true"),
        error_message);
    spdlog::warn("{}", message);
}

Tensor ArrayFireRealUnary(const Tensor& input,
                          UnaryOp operation,
                          float first,
                          float second,
                          DataType output_dtype) {
    const af::array values =
        input.GetSemanticArray().as(
            tensor_math_utils::ArrayFireType(output_dtype));
    af::array output;
    switch (operation) {
        case UnaryOp::Pow: output = af::pow(values, first); break;
        case UnaryOp::Sqrt: output = af::sqrt(values); break;
        case UnaryOp::Exp: output = af::exp(values); break;
        case UnaryOp::Log: output = af::log(values); break;
        case UnaryOp::Clip:
            output = (af::min)((af::max)(values, first), second);
            break;
        default:
            throw std::runtime_error("Tensor real unary: invalid ArrayFire operation");
    }
    output = output.as(tensor_math_utils::ArrayFireType(output_dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, input.Shape());
}

af::array ArrayFireSign(const af::array& values) {
    const af::array one = af::constant(1, values.dims(), values.type());
    const af::array zero = af::constant(0, values.dims(), values.type());
    const af::array negative_one =
        af::constant(-1, values.dims(), values.type());
    return af::select(
        values > 0, one, af::select(values < 0, negative_one, zero));
}

Tensor ArrayFirePreservingUnary(const Tensor& input, UnaryOp operation) {
    const af::array values = input.GetSemanticArray();
    af::array output;
    switch (operation) {
        case UnaryOp::Abs: output = af::abs(values); break;
        case UnaryOp::Sign: output = ArrayFireSign(values); break;
        case UnaryOp::Negate:
            output = af::constant(0, values.dims(), values.type()) - values;
            break;
        default:
            throw std::runtime_error(
                "Tensor preserving unary: invalid ArrayFire operation");
    }
    output = output.as(
        tensor_math_utils::ArrayFireType(input.GetDataType()));
    output.eval();
    return Tensor::FromSemanticArray(output, input.Shape());
}

Tensor ArrayFireTensorPow(const Tensor& base,
                          const Tensor& exponent,
                          const std::vector<size_t>& output_shape,
                          DataType output_dtype) {
    af::array left = tensor_math_utils::BroadcastArray(
        base, output_shape, output_dtype);
    af::array right =
        tensor_math_utils::BroadcastArray(
            exponent, output_shape, output_dtype);
    af::array output;
    if (tensor_math_utils::IsIntegralType(output_dtype)) {
        const unsigned bit_count = output_dtype == DataType::UInt8 ? 8u
            : output_dtype == DataType::Int32 ? 32u : 64u;
        const af::array nonnegative = right >= 0;
        af::array remaining = af::select(
            nonnegative, right, af::constant(0, right.dims(), right.type())).as(u64);
        af::array factor = left;
        output = af::constant(
            1, left.dims(), tensor_math_utils::ArrayFireType(output_dtype));
        for (unsigned bit = 0; bit < bit_count; ++bit) {
            const af::array odd = (remaining & 1u) != 0u;
            output = af::select(
                odd,
                (output * factor).as(
                    tensor_math_utils::ArrayFireType(output_dtype)),
                output);
            remaining = remaining >> 1u;
            if (bit + 1u < bit_count) {
                factor = (factor * factor).as(
                    tensor_math_utils::ArrayFireType(output_dtype));
            }
        }
        output = af::select(
            nonnegative,
            output,
            af::constant(0, output.dims(), output.type()));
    } else {
        output = af::pow(left, right);
    }
    output = output.as(tensor_math_utils::ArrayFireType(output_dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}
#endif

Tensor ApplyRealUnary(const Tensor& input,
                      UnaryOp operation,
                      float first = 0.0f,
                      float second = 0.0f) {
    const DataType output_dtype =
        tensor_math_utils::RealType(input.GetDataType());
    if (input.NumElements() == 0) return Tensor(input.Shape(), output_dtype);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = UnaryOperationName(operation);
    const std::string attributes =
        "op=" + std::string(UnaryName(operation));
    if (input.Shape().size() <= 4) {
        try {
            return ArrayFireRealUnary(
                input, operation, first, second, output_dtype);
        } catch (const af::exception& error) {
            RecordUnaryArrayFireFallback(
                operation_name, input, input.Shape(), output_dtype,
                attributes, error.what());
        }
    } else {
        RecordUnaryArrayFireFallback(
            operation_name, input, input.Shape(), output_dtype, attributes,
            "ArrayFire Tensor elementwise operations support ranks up to 4");
    }
#endif
    return NativeRealUnary(
        input, operation, first, second, output_dtype);
}

Tensor ApplyPreservingUnary(const Tensor& input, UnaryOp operation) {
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), input.GetDataType());
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = UnaryOperationName(operation);
    const std::string attributes =
        "op=" + std::string(UnaryName(operation));
    if (input.Shape().size() <= 4) {
        try {
            return ArrayFirePreservingUnary(input, operation);
        } catch (const af::exception& error) {
            RecordUnaryArrayFireFallback(
                operation_name, input, input.Shape(), input.GetDataType(),
                attributes, error.what());
        }
    } else {
        RecordUnaryArrayFireFallback(
            operation_name, input, input.Shape(), input.GetDataType(),
            attributes,
            "ArrayFire Tensor elementwise operations support ranks up to 4");
    }
#endif
    return NativePreservingUnary(input, operation);
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (output_shape.size() <= 4) {
        try {
            return ArrayFireTensorPow(
                *this, exponent, output_shape, output_dtype);
        } catch (const af::exception& error) {
            RecordPowArrayFireFallback(
                *this, exponent, output_shape, output_dtype, error.what());
        }
    } else {
        RecordPowArrayFireFallback(
            *this, exponent, output_shape, output_dtype,
            "ArrayFire Tensor elementwise operations support ranks up to 4");
    }
#endif
    return NativeTensorPow(
        *this, exponent, output_shape, output_dtype);
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
