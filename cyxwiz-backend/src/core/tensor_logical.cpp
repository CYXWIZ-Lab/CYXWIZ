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

enum class LogicalOp {
    And,
    Or
};

const char* LogicalOperationName(LogicalOp op) {
    switch (op) {
        case LogicalOp::And: return "Tensor::operator&&";
        case LogicalOp::Or: return "Tensor::operator||";
    }
    return "Tensor::logical";
}

// Nonzero is true; the result is UInt8 0 / 1.
Tensor LogicalTensors(const Tensor& left, const Tensor& right, LogicalOp op) {
    const std::vector<size_t> output_shape =
        Tensor::BroadcastShape(left.Shape(), right.Shape());
    if (tensor_utils::CheckedProduct(
            output_shape, 0, output_shape.size(),
            "Tensor logical: output shape overflow") == 0) {
        return Tensor(output_shape, DataType::UInt8);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array lhs = tensor_math_utils::BroadcastArray(left, output_shape, left.GetDataType()) != 0;
        const af::array rhs = tensor_math_utils::BroadcastArray(right, output_shape, right.GetDataType()) != 0;
        af::array output = (op == LogicalOp::And ? lhs && rhs : lhs || rhs).as(u8);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, output_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(LogicalOperationName(op), e));
    }
#else
    throw std::runtime_error(std::string(LogicalOperationName(op)) +
                             " runs on ArrayFire, and this build has no ArrayFire");
#endif
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
    try {
        af::array output = (tensor_math_utils::ElementwiseArray(*this) == 0).as(u8);
        output.eval();
        return tensor_math_utils::FromElementwiseArray(output, shape_);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::operator!", e));
    }
#else
    throw std::runtime_error("Tensor::operator! runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace cyxwiz
