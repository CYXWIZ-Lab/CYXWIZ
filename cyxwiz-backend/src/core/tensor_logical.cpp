#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

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

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool IsArrayFireLogicalSupported(DataType dtype) {
    return dtype == DataType::Float32 || dtype == DataType::Float64;
}

Tensor ApplyTensorLogicalArrayFire(const Tensor& left,
                                   const Tensor& right,
                                   LogicalOp op) {
    if (left.Shape().size() == 2 && right.Shape().size() == 2) {
        const af::array lhs = left.GetArrayRowMajor2D() != 0;
        const af::array rhs = right.GetArrayRowMajor2D() != 0;
        const af::array mask = op == LogicalOp::And ? lhs && rhs : lhs || rhs;
        return Tensor::FromArrayRowMajor2D(mask.as(af::dtype::u8));
    }
    const af::array lhs = left.GetArray() != 0;
    const af::array rhs = right.GetArray() != 0;
    const af::array mask = op == LogicalOp::And ? lhs && rhs : lhs || rhs;
    return Tensor(mask.as(af::dtype::u8));
}

Tensor ApplyLogicalNotArrayFire(const Tensor& input) {
    if (input.Shape().size() == 2) {
        return Tensor::FromArrayRowMajor2D((input.GetArrayRowMajor2D() == 0).as(af::dtype::u8));
    }
    return Tensor((input.GetArray() == 0).as(af::dtype::u8));
}
#endif

Tensor LogicalTensors(const Tensor& left, const Tensor& right, LogicalOp op) {
    const std::vector<size_t> out_shape = Tensor::BroadcastShape(left.Shape(), right.Shape());
    const Tensor left_expanded = left.Shape() == out_shape ? left.Clone() : left.Expand(out_shape);
    const Tensor right_expanded = right.Shape() == out_shape ? right.Clone() : right.Expand(out_shape);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (left_expanded.GetDataType() == right_expanded.GetDataType() &&
        IsArrayFireLogicalSupported(left_expanded.GetDataType())) {
        try {
            return ApplyTensorLogicalArrayFire(left_expanded, right_expanded, op);
        } catch (const af::exception& e) {
            tensor_backend_observation::RecordArrayFireFallback(
                "Tensor::Logical",
                tensor_backend_observation::DataTypeName(left_expanded.GetDataType()),
                tensor_backend_observation::BuildTensorOpSignature(
                    {left_expanded.Shape(), right_expanded.Shape()},
                    out_shape,
                    left_expanded.GetDataType(),
                    std::string("op=") + LogicalOpName(op)),
                e.what());
            spdlog::warn("Tensor logical operation: ArrayFire path failed, falling back to CPU: {}", e.what());
        }
    }
#endif
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsArrayFireLogicalSupported(dtype_)) {
        try {
            return ApplyLogicalNotArrayFire(*this);
        } catch (const af::exception& e) {
            tensor_backend_observation::RecordArrayFireFallback(
                "Tensor::LogicalNot",
                tensor_backend_observation::DataTypeName(dtype_),
                tensor_backend_observation::BuildTensorOpSignature(
                    {shape_},
                    shape_,
                    dtype_,
                    "op=not"),
                e.what());
            spdlog::warn("Tensor logical not: ArrayFire path failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return ApplyLogicalNot(*this);
}

} // namespace cyxwiz
