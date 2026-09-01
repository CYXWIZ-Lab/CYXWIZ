#include "cyxwiz/tensor.h"

#include "tensor_backend_observation_utils.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <bit>
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

enum class ArithmeticOp {
    Add,
    Subtract,
    Multiply,
    Divide
};

const char* ArithmeticName(ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return "add";
        case ArithmeticOp::Subtract: return "subtract";
        case ArithmeticOp::Multiply: return "multiply";
        case ArithmeticOp::Divide: return "divide";
    }
    return "unknown";
}

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

template <typename T>
T AddWrapped(T left, T right) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result =
            static_cast<Unsigned>(left) + static_cast<Unsigned>(right);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return left + right;
    }
}

template <typename T>
T SubtractWrapped(T left, T right) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result =
            static_cast<Unsigned>(left) - static_cast<Unsigned>(right);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return left - right;
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
T ApplyArithmeticValue(T left, T right, ArithmeticOp operation) {
    switch (operation) {
        case ArithmeticOp::Add: return AddWrapped(left, right);
        case ArithmeticOp::Subtract: return SubtractWrapped(left, right);
        case ArithmeticOp::Multiply: return MultiplyWrapped(left, right);
        case ArithmeticOp::Divide: return left / right;
    }
    return T{};
}

template <typename Left, typename Right, typename Out>
Tensor NativeTensorArithmetic(const Tensor& left,
                              const Tensor& right,
                              const std::vector<size_t>& output_shape,
                              DataType output_dtype,
                              ArithmeticOp operation) {
    Tensor result(output_shape, output_dtype);
    const size_t count = result.NumElements();
    if (count == 0) {
        return result;
    }
    const auto output_strides = tensor_utils::RowMajorStrides(
        output_shape, "Tensor arithmetic: output stride overflow");
    const auto left_strides = tensor_utils::RowMajorStrides(
        left.Shape(), "Tensor arithmetic: left stride overflow", true);
    const auto right_strides = tensor_utils::RowMajorStrides(
        right.Shape(), "Tensor arithmetic: right stride overflow", true);
    const Left* left_data = left.ReadData<Left>();
    const Right* right_data = right.ReadData<Right>();
    Out* output = result.MutableData<Out>();
    for (size_t index = 0; index < count; ++index) {
        const size_t left_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, left.Shape(), left_strides);
        const size_t right_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, right.Shape(), right_strides);
        output[index] = ApplyArithmeticValue(
            static_cast<Out>(left_data[left_index]),
            static_cast<Out>(right_data[right_index]), operation);
    }
    return result;
}

template <typename Left, typename Right>
Tensor DispatchArithmeticOutput(const Tensor& left,
                                const Tensor& right,
                                const std::vector<size_t>& output_shape,
                                DataType output_dtype,
                                ArithmeticOp operation) {
    switch (output_dtype) {
        case DataType::Float32:
            return NativeTensorArithmetic<Left, Right, float>(
                left, right, output_shape, output_dtype, operation);
        case DataType::Float64:
            return NativeTensorArithmetic<Left, Right, double>(
                left, right, output_shape, output_dtype, operation);
        case DataType::Int32:
            return NativeTensorArithmetic<Left, Right, int32_t>(
                left, right, output_shape, output_dtype, operation);
        case DataType::Int64:
            return NativeTensorArithmetic<Left, Right, int64_t>(
                left, right, output_shape, output_dtype, operation);
        case DataType::UInt8:
            return NativeTensorArithmetic<Left, Right, uint8_t>(
                left, right, output_shape, output_dtype, operation);
    }
    throw std::runtime_error("Tensor arithmetic: unsupported output data type");
}

template <typename Left>
Tensor DispatchArithmeticRight(const Tensor& left,
                               const Tensor& right,
                               const std::vector<size_t>& output_shape,
                               DataType output_dtype,
                               ArithmeticOp operation) {
    switch (right.GetDataType()) {
        case DataType::Float32: return DispatchArithmeticOutput<Left, float>(left, right, output_shape, output_dtype, operation);
        case DataType::Float64: return DispatchArithmeticOutput<Left, double>(left, right, output_shape, output_dtype, operation);
        case DataType::Int32: return DispatchArithmeticOutput<Left, int32_t>(left, right, output_shape, output_dtype, operation);
        case DataType::Int64: return DispatchArithmeticOutput<Left, int64_t>(left, right, output_shape, output_dtype, operation);
        case DataType::UInt8: return DispatchArithmeticOutput<Left, uint8_t>(left, right, output_shape, output_dtype, operation);
    }
    throw std::runtime_error("Tensor arithmetic: unsupported right data type");
}

Tensor NativeTensorArithmetic(const Tensor& left,
                              const Tensor& right,
                              const std::vector<size_t>& output_shape,
                              DataType output_dtype,
                              ArithmeticOp operation) {
    switch (left.GetDataType()) {
        case DataType::Float32: return DispatchArithmeticRight<float>(left, right, output_shape, output_dtype, operation);
        case DataType::Float64: return DispatchArithmeticRight<double>(left, right, output_shape, output_dtype, operation);
        case DataType::Int32: return DispatchArithmeticRight<int32_t>(left, right, output_shape, output_dtype, operation);
        case DataType::Int64: return DispatchArithmeticRight<int64_t>(left, right, output_shape, output_dtype, operation);
        case DataType::UInt8: return DispatchArithmeticRight<uint8_t>(left, right, output_shape, output_dtype, operation);
    }
    throw std::runtime_error("Tensor arithmetic: unsupported left data type");
}

template <typename In, typename Out>
Tensor NativeScalarArithmeticTyped(const Tensor& input,
                                   float scalar,
                                   ArithmeticOp operation,
                                   DataType output_dtype) {
    Tensor result(input.Shape(), output_dtype);
    const In* source = input.ReadData<In>();
    Out* output = result.MutableData<Out>();
    const Out converted_scalar = static_cast<Out>(scalar);
    for (size_t index = 0; index < input.NumElements(); ++index) {
        output[index] = ApplyArithmeticValue(
            static_cast<Out>(source[index]), converted_scalar, operation);
    }
    return result;
}

template <typename In>
Tensor NativeScalarArithmeticOutput(const Tensor& input,
                                    float scalar,
                                    ArithmeticOp operation,
                                    DataType output_dtype) {
    if (output_dtype == DataType::Float64) {
        return NativeScalarArithmeticTyped<In, double>(
            input, scalar, operation, output_dtype);
    }
    return NativeScalarArithmeticTyped<In, float>(
        input, scalar, operation, output_dtype);
}

Tensor NativeScalarArithmetic(const Tensor& input,
                              float scalar,
                              ArithmeticOp operation,
                              DataType output_dtype) {
    switch (input.GetDataType()) {
        case DataType::Float32: return NativeScalarArithmeticOutput<float>(input, scalar, operation, output_dtype);
        case DataType::Float64: return NativeScalarArithmeticOutput<double>(input, scalar, operation, output_dtype);
        case DataType::Int32: return NativeScalarArithmeticOutput<int32_t>(input, scalar, operation, output_dtype);
        case DataType::Int64: return NativeScalarArithmeticOutput<int64_t>(input, scalar, operation, output_dtype);
        case DataType::UInt8: return NativeScalarArithmeticOutput<uint8_t>(input, scalar, operation, output_dtype);
    }
    throw std::runtime_error("Tensor scalar arithmetic: unsupported data type");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordArithmeticArrayFireFallback(const char* operation_name,
                              const Tensor& left,
                              const Tensor* right,
                              const std::vector<size_t>& output_shape,
                              DataType output_dtype,
                              const std::string& attributes,
                              const char* error_message) {
    std::vector<std::vector<size_t>> input_shapes{left.Shape()};
    if (right != nullptr) input_shapes.push_back(right->Shape());
    const std::string signature =
        tensor_backend_observation::BuildTensorOpSignature(
            input_shapes, output_shape, output_dtype, attributes);
    const std::string message = tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(output_dtype),
        signature,
        error_message);
    spdlog::warn("{}", message);
}

Tensor ArrayFireTensorArithmetic(const Tensor& left,
                                 const Tensor& right,
                                 const std::vector<size_t>& output_shape,
                                 DataType output_dtype,
                                 ArithmeticOp operation) {
    af::array left_values = tensor_math_utils::BroadcastArray(
        left, output_shape, output_dtype);
    af::array right_values = tensor_math_utils::BroadcastArray(
        right, output_shape, output_dtype);
    af::array output;
    switch (operation) {
        case ArithmeticOp::Add: output = left_values + right_values; break;
        case ArithmeticOp::Subtract: output = left_values - right_values; break;
        case ArithmeticOp::Multiply: output = left_values * right_values; break;
        case ArithmeticOp::Divide: output = left_values / right_values; break;
    }
    output = output.as(tensor_math_utils::ArrayFireType(output_dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}

Tensor ArrayFireScalarArithmetic(const Tensor& input,
                                 float scalar,
                                 DataType output_dtype,
                                 ArithmeticOp operation) {
    const af::array values = input.GetSemanticArray().as(
        tensor_math_utils::ArrayFireType(output_dtype));
    af::array output;
    switch (operation) {
        case ArithmeticOp::Add: output = values + scalar; break;
        case ArithmeticOp::Subtract: output = values - scalar; break;
        case ArithmeticOp::Multiply: output = values * scalar; break;
        case ArithmeticOp::Divide: output = values / scalar; break;
    }
    output = output.as(tensor_math_utils::ArrayFireType(output_dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, input.Shape());
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
    const char* operation_name = TensorOperationName(operation);
    const std::string attributes =
        "op=" + std::string(ArithmeticName(operation)) + ";broadcast=true";
    if (output_shape.size() <= 4) {
        try {
            return ArrayFireTensorArithmetic(
                left, right, output_shape, output_dtype, operation);
        } catch (const af::exception& error) {
            RecordArithmeticArrayFireFallback(
                operation_name, left, &right, output_shape, output_dtype,
                attributes, error.what());
        }
    } else {
        RecordArithmeticArrayFireFallback(
            operation_name, left, &right, output_shape, output_dtype,
            attributes,
            "ArrayFire Tensor arithmetic supports ranks up to 4");
    }
#endif
    return NativeTensorArithmetic(
        left, right, output_shape, output_dtype, operation);
}

Tensor ApplyScalarArithmetic(const Tensor& input,
                             float scalar,
                             ArithmeticOp operation) {
    const DataType output_dtype = ScalarResultType(input.GetDataType());
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), output_dtype);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = ScalarOperationName(operation);
    const std::string attributes =
        "op=" + std::string(ArithmeticName(operation)) + ";rhs=scalar";
    if (input.Shape().size() <= 4) {
        try {
            return ArrayFireScalarArithmetic(
                input, scalar, output_dtype, operation);
        } catch (const af::exception& error) {
            RecordArithmeticArrayFireFallback(
                operation_name, input, nullptr, input.Shape(), output_dtype,
                attributes, error.what());
        }
    } else {
        RecordArithmeticArrayFireFallback(
            operation_name, input, nullptr, input.Shape(), output_dtype,
            attributes,
            "ArrayFire Tensor scalar arithmetic supports ranks up to 4");
    }
#endif
    return NativeScalarArithmetic(input, scalar, operation, output_dtype);
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
