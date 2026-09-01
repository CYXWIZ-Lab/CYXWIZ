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

enum class LogicalOp {
    And,
    Or
};

const char* LogicalOpName(LogicalOp op) {
    switch (op) {
        case LogicalOp::And: return "and";
        case LogicalOp::Or: return "or";
    }
    return "unknown";
}

const char* LogicalOperationName(LogicalOp op) {
    switch (op) {
        case LogicalOp::And: return "Tensor::operator&&";
        case LogicalOp::Or: return "Tensor::operator||";
    }
    return "Tensor::logical";
}

template <typename T>
bool IsTruthy(T value) {
    return value != static_cast<T>(0);
}

template <typename L, typename R>
Tensor ApplyTensorLogical(const Tensor& left,
                          const Tensor& right,
                          const std::vector<size_t>& output_shape,
                          LogicalOp op) {
    Tensor result(output_shape, DataType::UInt8);
    const size_t count = result.NumElements();
    if (count == 0) return result;
    const auto output_strides = tensor_utils::RowMajorStrides(
        output_shape, "Tensor logical: output stride overflow");
    const auto left_strides = tensor_utils::RowMajorStrides(
        left.Shape(), "Tensor logical: left stride overflow", true);
    const auto right_strides = tensor_utils::RowMajorStrides(
        right.Shape(), "Tensor logical: right stride overflow", true);
    const L* left_data = left.ReadData<L>();
    const R* right_data = right.ReadData<R>();
    uint8_t* output = result.MutableData<uint8_t>();

    for (size_t index = 0; index < count; ++index) {
        const size_t left_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, left.Shape(), left_strides);
        const size_t right_index = tensor_math_utils::BroadcastIndex(
            index, output_shape, output_strides, right.Shape(), right_strides);
        const bool lhs = IsTruthy(left_data[left_index]);
        const bool rhs = IsTruthy(right_data[right_index]);
        output[index] =
            (op == LogicalOp::And ? lhs && rhs : lhs || rhs) ? 1 : 0;
    }
    return result;
}

template <typename L>
Tensor DispatchRightLogical(const Tensor& left,
                            const Tensor& right,
                            const std::vector<size_t>& output_shape,
                            LogicalOp op) {
    switch (right.GetDataType()) {
        case DataType::Float32:
            return ApplyTensorLogical<L, float>(
                left, right, output_shape, op);
        case DataType::Float64:
            return ApplyTensorLogical<L, double>(
                left, right, output_shape, op);
        case DataType::Int32:
            return ApplyTensorLogical<L, int32_t>(
                left, right, output_shape, op);
        case DataType::Int64:
            return ApplyTensorLogical<L, int64_t>(
                left, right, output_shape, op);
        case DataType::UInt8:
            return ApplyTensorLogical<L, uint8_t>(
                left, right, output_shape, op);
    }
    throw std::runtime_error("Tensor logical operation: unsupported right data type");
}

Tensor ApplyTensorLogical(const Tensor& left,
                          const Tensor& right,
                          const std::vector<size_t>& output_shape,
                          LogicalOp op) {
    switch (left.GetDataType()) {
        case DataType::Float32:
            return DispatchRightLogical<float>(
                left, right, output_shape, op);
        case DataType::Float64:
            return DispatchRightLogical<double>(
                left, right, output_shape, op);
        case DataType::Int32:
            return DispatchRightLogical<int32_t>(
                left, right, output_shape, op);
        case DataType::Int64:
            return DispatchRightLogical<int64_t>(
                left, right, output_shape, op);
        case DataType::UInt8:
            return DispatchRightLogical<uint8_t>(
                left, right, output_shape, op);
    }
    throw std::runtime_error("Tensor logical operation: unsupported left data type");
}

template <typename T>
Tensor ApplyLogicalNot(const Tensor& input) {
    const size_t count = input.NumElements();
    const T* data = input.ReadData<T>();
    Tensor result(input.Shape(), DataType::UInt8);
    uint8_t* out = result.MutableData<uint8_t>();

    for (size_t i = 0; i < count; i++) {
        out[i] = IsTruthy(data[i]) ? 0 : 1;
    }
    return result;
}

Tensor ApplyLogicalNot(const Tensor& input) {
    switch (input.GetDataType()) {
        case DataType::Float32: return ApplyLogicalNot<float>(input);
        case DataType::Float64: return ApplyLogicalNot<double>(input);
        case DataType::Int32: return ApplyLogicalNot<int32_t>(input);
        case DataType::Int64: return ApplyLogicalNot<int64_t>(input);
        case DataType::UInt8: return ApplyLogicalNot<uint8_t>(input);
    }
    throw std::runtime_error("Tensor logical not: unsupported data type");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
Tensor ApplyTensorLogicalArrayFire(const Tensor& left,
                                   const Tensor& right,
                                   const std::vector<size_t>& output_shape,
                                   LogicalOp op) {
    const af::array lhs = tensor_math_utils::BroadcastArray(
        left, output_shape, left.GetDataType()) != 0;
    const af::array rhs = tensor_math_utils::BroadcastArray(
        right, output_shape, right.GetDataType()) != 0;
    af::array output =
        (op == LogicalOp::And ? lhs && rhs : lhs || rhs).as(u8);
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}

Tensor ApplyLogicalNotArrayFire(const Tensor& input) {
    af::array output = (input.GetSemanticArray() == 0).as(u8);
    output.eval();
    return Tensor::FromSemanticArray(output, input.Shape());
}

void RecordLogicalArrayFireFallback(
    const char* operation_name,
    const Tensor& left,
    const Tensor* right,
    const std::vector<size_t>& output_shape,
    const std::string& attributes,
    const char* error_message) {
    std::vector<std::vector<size_t>> input_shapes{left.Shape()};
    if (right != nullptr) input_shapes.push_back(right->Shape());
    const std::string message =
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(DataType::UInt8),
            tensor_backend_observation::BuildTensorOpSignature(
                input_shapes, output_shape, DataType::UInt8, attributes),
            error_message);
    spdlog::warn("{}", message);
}
#endif

Tensor LogicalTensors(const Tensor& left, const Tensor& right, LogicalOp op) {
    const std::vector<size_t> output_shape =
        Tensor::BroadcastShape(left.Shape(), right.Shape());
    if (tensor_utils::CheckedProduct(
            output_shape, 0, output_shape.size(),
            "Tensor logical: output shape overflow") == 0) {
        return Tensor(output_shape, DataType::UInt8);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const char* operation_name = LogicalOperationName(op);
    const std::string attributes =
        std::string("op=") + LogicalOpName(op) + ";broadcast=true";
    if (output_shape.size() <= 4) {
        try {
            return ApplyTensorLogicalArrayFire(
                left, right, output_shape, op);
        } catch (const af::exception& error) {
            RecordLogicalArrayFireFallback(
                operation_name, left, &right, output_shape,
                attributes, error.what());
        }
    } else {
        RecordLogicalArrayFireFallback(
            operation_name, left, &right, output_shape, attributes,
            "ArrayFire Tensor logical operations support ranks up to 4");
    }
#endif
    return ApplyTensorLogical(left, right, output_shape, op);
}

} // namespace

Tensor Tensor::operator&&(const Tensor& other) const {
    return LogicalTensors(*this, other, LogicalOp::And);
}

Tensor Tensor::operator||(const Tensor& other) const {
    return LogicalTensors(*this, other, LogicalOp::Or);
}

Tensor Tensor::operator!() const {
    if (NumElements() == 0) {
        return Tensor(shape_, DataType::UInt8);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return ApplyLogicalNotArrayFire(*this);
        } catch (const af::exception& error) {
            RecordLogicalArrayFireFallback(
                "Tensor::operator!", *this, nullptr, shape_,
                "op=not", error.what());
        }
    } else {
        RecordLogicalArrayFireFallback(
            "Tensor::operator!", *this, nullptr, shape_, "op=not",
            "ArrayFire Tensor logical operations support ranks up to 4");
    }
#endif
    return ApplyLogicalNot(*this);
}

} // namespace cyxwiz
