#include "cyxwiz/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cyxwiz {

namespace {

enum class UnaryRealOp {
    Pow,
    Sqrt,
    Exp,
    Log
};

template <typename In, typename Out, typename Fn>
void ApplyUnary(const In* src, Out* dst, size_t count, Fn fn) {
    for (size_t i = 0; i < count; i++) {
        dst[i] = static_cast<Out>(fn(src[i]));
    }
}

template <typename T, typename Fn>
Tensor ApplyPreservingType(const Tensor& input, Fn fn) {
    Tensor result(input.Shape(), input.GetDataType());
    ApplyUnary(input.Data<T>(), result.Data<T>(), input.NumElements(), fn);
    return result;
}

template <typename Fn>
Tensor ApplyPreservingType(const Tensor& input, Fn fn) {
    switch (input.GetDataType()) {
        case DataType::Float32: return ApplyPreservingType<float>(input, fn);
        case DataType::Float64: return ApplyPreservingType<double>(input, fn);
        case DataType::Int32: return ApplyPreservingType<int32_t>(input, fn);
        case DataType::Int64: return ApplyPreservingType<int64_t>(input, fn);
        case DataType::UInt8: return ApplyPreservingType<uint8_t>(input, fn);
    }
    throw std::runtime_error("Tensor elementwise operation: unsupported data type");
}

template <typename T>
Tensor ApplyRealMath(const Tensor& input, UnaryRealOp op, float value = 0.0f) {
    const size_t count = input.NumElements();
    const T* src = input.Data<T>();

    if (input.GetDataType() == DataType::Float64) {
        Tensor result(input.Shape(), DataType::Float64);
        double* dst = result.Data<double>();
        for (size_t i = 0; i < count; i++) {
            const double x = static_cast<double>(src[i]);
            switch (op) {
                case UnaryRealOp::Pow: dst[i] = std::pow(x, static_cast<double>(value)); break;
                case UnaryRealOp::Sqrt: dst[i] = std::sqrt(x); break;
                case UnaryRealOp::Exp: dst[i] = std::exp(x); break;
                case UnaryRealOp::Log: dst[i] = std::log(x); break;
            }
        }
        return result;
    }

    Tensor result(input.Shape(), DataType::Float32);
    float* dst = result.Data<float>();
    for (size_t i = 0; i < count; i++) {
        const float x = static_cast<float>(src[i]);
        switch (op) {
            case UnaryRealOp::Pow: dst[i] = std::pow(x, value); break;
            case UnaryRealOp::Sqrt: dst[i] = std::sqrt(x); break;
            case UnaryRealOp::Exp: dst[i] = std::exp(x); break;
            case UnaryRealOp::Log: dst[i] = std::log(x); break;
        }
    }
    return result;
}

Tensor ApplyRealMath(const Tensor& input, UnaryRealOp op, float value = 0.0f) {
    switch (input.GetDataType()) {
        case DataType::Float32: return ApplyRealMath<float>(input, op, value);
        case DataType::Float64: return ApplyRealMath<double>(input, op, value);
        case DataType::Int32: return ApplyRealMath<int32_t>(input, op, value);
        case DataType::Int64: return ApplyRealMath<int64_t>(input, op, value);
        case DataType::UInt8: return ApplyRealMath<uint8_t>(input, op, value);
    }
    throw std::runtime_error("Tensor real-valued operation: unsupported data type");
}

template <typename Base, typename Exp>
Tensor ApplyTensorPow(const Tensor& base, const Tensor& exponent, DataType result_dtype) {
    const size_t count = base.NumElements();
    const Base* base_data = base.Data<Base>();
    const Exp* exponent_data = exponent.Data<Exp>();

    if (result_dtype == DataType::Float64) {
        Tensor result(base.Shape(), DataType::Float64);
        double* dst = result.Data<double>();
        for (size_t i = 0; i < count; i++) {
            dst[i] = std::pow(static_cast<double>(base_data[i]), static_cast<double>(exponent_data[i]));
        }
        return result;
    }

    Tensor result(base.Shape(), DataType::Float32);
    float* dst = result.Data<float>();
    for (size_t i = 0; i < count; i++) {
        dst[i] = std::pow(static_cast<float>(base_data[i]), static_cast<float>(exponent_data[i]));
    }
    return result;
}

template <typename Base>
Tensor DispatchTensorPowExponent(const Tensor& base, const Tensor& exponent, DataType result_dtype) {
    switch (exponent.GetDataType()) {
        case DataType::Float32: return ApplyTensorPow<Base, float>(base, exponent, result_dtype);
        case DataType::Float64: return ApplyTensorPow<Base, double>(base, exponent, result_dtype);
        case DataType::Int32: return ApplyTensorPow<Base, int32_t>(base, exponent, result_dtype);
        case DataType::Int64: return ApplyTensorPow<Base, int64_t>(base, exponent, result_dtype);
        case DataType::UInt8: return ApplyTensorPow<Base, uint8_t>(base, exponent, result_dtype);
    }
    throw std::runtime_error("Tensor::Pow: unsupported exponent data type");
}

} // namespace

Tensor Tensor::operator+(float scalar) const {
    return ApplyPreservingType(*this, [scalar](auto value) {
        return value + scalar;
    });
}

Tensor Tensor::operator-(float scalar) const {
    return ApplyPreservingType(*this, [scalar](auto value) {
        return value - scalar;
    });
}

Tensor Tensor::operator*(float scalar) const {
    return ApplyPreservingType(*this, [scalar](auto value) {
        return value * scalar;
    });
}

Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Tensor::operator/: division by zero");
    }
    return ApplyPreservingType(*this, [scalar](auto value) {
        return value / scalar;
    });
}

Tensor Tensor::Pow(float exponent) const {
    return ApplyRealMath(*this, UnaryRealOp::Pow, exponent);
}

Tensor Tensor::Pow(const Tensor& exponent) const {
    if (shape_ != exponent.Shape()) {
        throw std::runtime_error("Tensor::Pow: shapes must match");
    }

    const DataType result_dtype =
        (dtype_ == DataType::Float64 || exponent.GetDataType() == DataType::Float64)
            ? DataType::Float64
            : DataType::Float32;

    switch (dtype_) {
        case DataType::Float32: return DispatchTensorPowExponent<float>(*this, exponent, result_dtype);
        case DataType::Float64: return DispatchTensorPowExponent<double>(*this, exponent, result_dtype);
        case DataType::Int32: return DispatchTensorPowExponent<int32_t>(*this, exponent, result_dtype);
        case DataType::Int64: return DispatchTensorPowExponent<int64_t>(*this, exponent, result_dtype);
        case DataType::UInt8: return DispatchTensorPowExponent<uint8_t>(*this, exponent, result_dtype);
    }
    throw std::runtime_error("Tensor::Pow: unsupported base data type");
}

Tensor Tensor::Sqrt() const {
    return ApplyRealMath(*this, UnaryRealOp::Sqrt);
}

Tensor Tensor::Exp() const {
    return ApplyRealMath(*this, UnaryRealOp::Exp);
}

Tensor Tensor::Log() const {
    return ApplyRealMath(*this, UnaryRealOp::Log);
}

Tensor Tensor::Abs() const {
    return ApplyPreservingType(*this, [](auto value) {
        return value < 0 ? -value : value;
    });
}

Tensor Tensor::Sign() const {
    return ApplyPreservingType(*this, [](auto value) {
        using T = decltype(value);
        if (value > static_cast<T>(0)) {
            return static_cast<T>(1);
        }
        if (value < static_cast<T>(0)) {
            return static_cast<T>(-1);
        }
        return static_cast<T>(0);
    });
}

Tensor Tensor::Clip(float min_val, float max_val) const {
    if (min_val > max_val) {
        throw std::runtime_error("Tensor::Clip: min_val must be <= max_val");
    }
    return ApplyPreservingType(*this, [min_val, max_val](auto value) {
        using T = decltype(value);
        const T lo = static_cast<T>(min_val);
        const T hi = static_cast<T>(max_val);
        return (std::min)((std::max)(value, lo), hi);
    });
}

Tensor Tensor::operator-() const {
    return ApplyPreservingType(*this, [](auto value) {
        return -value;
    });
}

} // namespace cyxwiz
