#include "cyxwiz/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace cyxwiz {

namespace {

enum class LogicalOp {
    And,
    Or
};

template <typename T>
bool IsTruthy(T value) {
    return value != static_cast<T>(0);
}

template <typename L, typename R>
Tensor ApplyTensorLogical(const Tensor& left, const Tensor& right, LogicalOp op) {
    const size_t count = left.NumElements();
    const L* left_data = left.Data<L>();
    const R* right_data = right.Data<R>();
    Tensor result(left.Shape(), DataType::UInt8);
    uint8_t* out = result.Data<uint8_t>();

    for (size_t i = 0; i < count; i++) {
        const bool lhs = IsTruthy(left_data[i]);
        const bool rhs = IsTruthy(right_data[i]);
        out[i] = (op == LogicalOp::And ? lhs && rhs : lhs || rhs) ? 1 : 0;
    }
    return result;
}

template <typename L>
Tensor DispatchRightLogical(const Tensor& left, const Tensor& right, LogicalOp op) {
    switch (right.GetDataType()) {
        case DataType::Float32: return ApplyTensorLogical<L, float>(left, right, op);
        case DataType::Float64: return ApplyTensorLogical<L, double>(left, right, op);
        case DataType::Int32: return ApplyTensorLogical<L, int32_t>(left, right, op);
        case DataType::Int64: return ApplyTensorLogical<L, int64_t>(left, right, op);
        case DataType::UInt8: return ApplyTensorLogical<L, uint8_t>(left, right, op);
    }
    throw std::runtime_error("Tensor logical operation: unsupported right data type");
}

Tensor ApplyTensorLogical(const Tensor& left, const Tensor& right, LogicalOp op) {
    switch (left.GetDataType()) {
        case DataType::Float32: return DispatchRightLogical<float>(left, right, op);
        case DataType::Float64: return DispatchRightLogical<double>(left, right, op);
        case DataType::Int32: return DispatchRightLogical<int32_t>(left, right, op);
        case DataType::Int64: return DispatchRightLogical<int64_t>(left, right, op);
        case DataType::UInt8: return DispatchRightLogical<uint8_t>(left, right, op);
    }
    throw std::runtime_error("Tensor logical operation: unsupported left data type");
}

template <typename T>
Tensor ApplyLogicalNot(const Tensor& input) {
    const size_t count = input.NumElements();
    const T* data = input.Data<T>();
    Tensor result(input.Shape(), DataType::UInt8);
    uint8_t* out = result.Data<uint8_t>();

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

Tensor LogicalTensors(const Tensor& left, const Tensor& right, LogicalOp op) {
    const std::vector<size_t> out_shape = Tensor::BroadcastShape(left.Shape(), right.Shape());
    const Tensor left_expanded = left.Shape() == out_shape ? left.Clone() : left.Expand(out_shape);
    const Tensor right_expanded = right.Shape() == out_shape ? right.Clone() : right.Expand(out_shape);
    return ApplyTensorLogical(left_expanded, right_expanded, op);
}

} // namespace

Tensor Tensor::operator&&(const Tensor& other) const {
    return LogicalTensors(*this, other, LogicalOp::And);
}

Tensor Tensor::operator||(const Tensor& other) const {
    return LogicalTensors(*this, other, LogicalOp::Or);
}

Tensor Tensor::operator!() const {
    return ApplyLogicalNot(*this);
}

} // namespace cyxwiz
