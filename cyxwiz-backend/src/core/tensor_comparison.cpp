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

enum class CompareOp {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual
};

template <typename L, typename R>
bool CompareValues(L left, R right, CompareOp op) {
    switch (op) {
        case CompareOp::Greater: return left > right;
        case CompareOp::GreaterEqual: return left >= right;
        case CompareOp::Less: return left < right;
        case CompareOp::LessEqual: return left <= right;
        case CompareOp::Equal: return left == right;
        case CompareOp::NotEqual: return left != right;
    }
    return false;
}

const char* CompareOpName(CompareOp op) {
    switch (op) {
        case CompareOp::Greater: return "greater";
        case CompareOp::GreaterEqual: return "greater_equal";
        case CompareOp::Less: return "less";
        case CompareOp::LessEqual: return "less_equal";
        case CompareOp::Equal: return "equal";
        case CompareOp::NotEqual: return "not_equal";
    }
    return "unknown";
}

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

template <typename L, typename R, typename Common>
Tensor ApplyTensorComparisonTyped(
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    CompareOp op) {
    Tensor result(output_shape, DataType::UInt8);
    const size_t count = result.NumElements();
    if (count == 0) return result;
    const auto output_strides = tensor_utils::RowMajorStrides(
        output_shape, "Tensor comparison: output stride overflow");
    const auto left_strides = tensor_utils::RowMajorStrides(
        left.Shape(), "Tensor comparison: left stride overflow", true);
    const auto right_strides = tensor_utils::RowMajorStrides(
        right.Shape(), "Tensor comparison: right stride overflow", true);
    const L* left_data = left.ReadData<L>();
    const R* right_data = right.ReadData<R>();
    uint8_t* output = result.MutableData<uint8_t>();

    for (size_t index = 0; index < count; ++index) {
        const size_t left_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, left.Shape(), left_strides);
        const size_t right_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, right.Shape(), right_strides);
        output[index] = CompareValues(
            static_cast<Common>(left_data[left_index]),
            static_cast<Common>(right_data[right_index]), op) ? 1 : 0;
    }
    return result;
}

template <typename L, typename R>
Tensor DispatchComparisonCommon(
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    DataType common_dtype,
    CompareOp op) {
    switch (common_dtype) {
        case DataType::Float32:
            return ApplyTensorComparisonTyped<L, R, float>(
                left, right, output_shape, op);
        case DataType::Float64:
            return ApplyTensorComparisonTyped<L, R, double>(
                left, right, output_shape, op);
        case DataType::Int32:
            return ApplyTensorComparisonTyped<L, R, int32_t>(
                left, right, output_shape, op);
        case DataType::Int64:
            return ApplyTensorComparisonTyped<L, R, int64_t>(
                left, right, output_shape, op);
        case DataType::UInt8:
            return ApplyTensorComparisonTyped<L, R, uint8_t>(
                left, right, output_shape, op);
    }
    throw std::runtime_error("Tensor comparison: unsupported common data type");
}

template <typename L>
Tensor DispatchRightComparison(
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    DataType common_dtype,
    CompareOp op) {
    switch (right.GetDataType()) {
        case DataType::Float32:
            return DispatchComparisonCommon<L, float>(
                left, right, output_shape, common_dtype, op);
        case DataType::Float64:
            return DispatchComparisonCommon<L, double>(
                left, right, output_shape, common_dtype, op);
        case DataType::Int32:
            return DispatchComparisonCommon<L, int32_t>(
                left, right, output_shape, common_dtype, op);
        case DataType::Int64:
            return DispatchComparisonCommon<L, int64_t>(
                left, right, output_shape, common_dtype, op);
        case DataType::UInt8:
            return DispatchComparisonCommon<L, uint8_t>(
                left, right, output_shape, common_dtype, op);
    }
    throw std::runtime_error("Tensor comparison: unsupported right data type");
}

Tensor ApplyTensorComparison(
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    DataType common_dtype,
    CompareOp op) {
    switch (left.GetDataType()) {
        case DataType::Float32:
            return DispatchRightComparison<float>(
                left, right, output_shape, common_dtype, op);
        case DataType::Float64:
            return DispatchRightComparison<double>(
                left, right, output_shape, common_dtype, op);
        case DataType::Int32:
            return DispatchRightComparison<int32_t>(
                left, right, output_shape, common_dtype, op);
        case DataType::Int64:
            return DispatchRightComparison<int64_t>(
                left, right, output_shape, common_dtype, op);
        case DataType::UInt8:
            return DispatchRightComparison<uint8_t>(
                left, right, output_shape, common_dtype, op);
    }
    throw std::runtime_error("Tensor comparison: unsupported left data type");
}

template <typename Input, typename Common>
Tensor ApplyScalarComparisonTyped(
    const Tensor& input,
    float scalar,
    CompareOp op) {
    const Input* data = input.ReadData<Input>();
    Tensor result(input.Shape(), DataType::UInt8);
    uint8_t* out = result.MutableData<uint8_t>();

    const Common value = static_cast<Common>(scalar);
    for (size_t i = 0; i < input.NumElements(); i++) {
        out[i] = CompareValues(
            static_cast<Common>(data[i]), value, op) ? 1 : 0;
    }
    return result;
}

template <typename Input>
Tensor DispatchScalarCommon(const Tensor& input,
                            float scalar,
                            DataType common_dtype,
                            CompareOp op) {
    if (common_dtype == DataType::Float64) {
        return ApplyScalarComparisonTyped<Input, double>(input, scalar, op);
    }
    return ApplyScalarComparisonTyped<Input, float>(input, scalar, op);
}

Tensor ApplyScalarComparisonNative(const Tensor& input,
                                   float scalar,
                                   DataType common_dtype,
                                   CompareOp op) {
    switch (input.GetDataType()) {
        case DataType::Float32:
            return DispatchScalarCommon<float>(input, scalar, common_dtype, op);
        case DataType::Float64:
            return DispatchScalarCommon<double>(input, scalar, common_dtype, op);
        case DataType::Int32:
            return DispatchScalarCommon<int32_t>(input, scalar, common_dtype, op);
        case DataType::Int64:
            return DispatchScalarCommon<int64_t>(input, scalar, common_dtype, op);
        case DataType::UInt8:
            return DispatchScalarCommon<uint8_t>(input, scalar, common_dtype, op);
    }
    throw std::runtime_error("Tensor scalar comparison: unsupported data type");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
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

void RecordComparisonArrayFireFallback(
    const char* operation_name,
    const Tensor& left,
    const Tensor* right,
    const std::vector<size_t>& output_shape,
    DataType common_dtype,
    const std::string& attributes,
    const char* error_message) {
    std::vector<std::vector<size_t>> input_shapes{left.Shape()};
    if (right != nullptr) input_shapes.push_back(right->Shape());
    const std::string message =
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(DataType::UInt8),
            tensor_backend_observation::BuildTensorOpSignature(
                input_shapes, output_shape, DataType::UInt8,
                attributes + ";common_dtype=" +
                    tensor_backend_observation::DataTypeName(common_dtype)),
            error_message);
    spdlog::warn("{}", message);
}

Tensor ApplyTensorComparisonArrayFire(
    const Tensor& left,
    const Tensor& right,
    const std::vector<size_t>& output_shape,
    DataType common_dtype,
    CompareOp op) {
    const af::array left_values = tensor_math_utils::BroadcastArray(
        left, output_shape, common_dtype);
    const af::array right_values = tensor_math_utils::BroadcastArray(
        right, output_shape, common_dtype);
    af::array output = CompareArrays(left_values, right_values, op).as(u8);
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}

Tensor ApplyScalarComparisonArrayFire(const Tensor& input,
                                      float scalar,
                                      DataType common_dtype,
                                      CompareOp op) {
    const af::array values = input.GetSemanticArray().as(
        tensor_math_utils::ArrayFireType(common_dtype));
    const af::array right = af::constant(
        scalar, values.dims(), tensor_math_utils::ArrayFireType(common_dtype));
    af::array output = CompareArrays(values, right, op).as(u8);
    output.eval();
    return Tensor::FromSemanticArray(output, input.Shape());
}
#endif

Tensor ApplyScalarComparison(const Tensor& input, float scalar, CompareOp op) {
    const DataType common_dtype =
        tensor_math_utils::RealType(input.GetDataType());
    if (input.NumElements() == 0) {
        return Tensor(input.Shape(), DataType::UInt8);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = ScalarComparisonOperationName(op);
    const std::string attributes =
        std::string("op=") + CompareOpName(op) + ";rhs=scalar";
    if (input.Shape().size() <= 4) {
        try {
            return ApplyScalarComparisonArrayFire(
                input, scalar, common_dtype, op);
        } catch (const af::exception& error) {
            RecordComparisonArrayFireFallback(
                operation_name, input, nullptr, input.Shape(), common_dtype,
                attributes, error.what());
        }
    } else {
        RecordComparisonArrayFireFallback(
            operation_name, input, nullptr, input.Shape(), common_dtype,
            attributes,
            "ArrayFire Tensor scalar comparisons support ranks up to 4");
    }
#endif
    return ApplyScalarComparisonNative(input, scalar, common_dtype, op);
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = TensorComparisonOperationName(op);
    const std::string attributes =
        std::string("op=") + CompareOpName(op) + ";broadcast=true";
    if (output_shape.size() <= 4) {
        try {
            return ApplyTensorComparisonArrayFire(
                left, right, output_shape, common_dtype, op);
        } catch (const af::exception& error) {
            RecordComparisonArrayFireFallback(
                operation_name, left, &right, output_shape, common_dtype,
                attributes, error.what());
        }
    } else {
        RecordComparisonArrayFireFallback(
            operation_name, left, &right, output_shape, common_dtype,
            attributes,
            "ArrayFire Tensor comparisons support ranks up to 4");
    }
#endif
    return ApplyTensorComparison(
        left, right, output_shape, common_dtype, op);
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
