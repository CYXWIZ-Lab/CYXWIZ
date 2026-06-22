#include "cyxwiz/tensor.h"

#include <cstdint>
#include <stdexcept>

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

template <typename L, typename R>
Tensor ApplyTensorComparison(const Tensor& left, const Tensor& right, CompareOp op) {
    const size_t count = left.NumElements();
    const L* left_data = left.Data<L>();
    const R* right_data = right.Data<R>();
    Tensor result(left.Shape(), DataType::UInt8);
    uint8_t* out = result.Data<uint8_t>();

    for (size_t i = 0; i < count; i++) {
        out[i] = CompareValues(left_data[i], right_data[i], op) ? 1 : 0;
    }
    return result;
}

template <typename L>
Tensor DispatchRightComparison(const Tensor& left, const Tensor& right, CompareOp op) {
    switch (right.GetDataType()) {
        case DataType::Float32: return ApplyTensorComparison<L, float>(left, right, op);
        case DataType::Float64: return ApplyTensorComparison<L, double>(left, right, op);
        case DataType::Int32: return ApplyTensorComparison<L, int32_t>(left, right, op);
        case DataType::Int64: return ApplyTensorComparison<L, int64_t>(left, right, op);
        case DataType::UInt8: return ApplyTensorComparison<L, uint8_t>(left, right, op);
    }
    throw std::runtime_error("Tensor comparison: unsupported right data type");
}

Tensor ApplyTensorComparison(const Tensor& left, const Tensor& right, CompareOp op) {
    switch (left.GetDataType()) {
        case DataType::Float32: return DispatchRightComparison<float>(left, right, op);
        case DataType::Float64: return DispatchRightComparison<double>(left, right, op);
        case DataType::Int32: return DispatchRightComparison<int32_t>(left, right, op);
        case DataType::Int64: return DispatchRightComparison<int64_t>(left, right, op);
        case DataType::UInt8: return DispatchRightComparison<uint8_t>(left, right, op);
    }
    throw std::runtime_error("Tensor comparison: unsupported left data type");
}

template <typename T>
Tensor ApplyScalarComparison(const Tensor& input, float scalar, CompareOp op) {
    const size_t count = input.NumElements();
    const T* data = input.Data<T>();
    Tensor result(input.Shape(), DataType::UInt8);
    uint8_t* out = result.Data<uint8_t>();

    const T value = static_cast<T>(scalar);
    for (size_t i = 0; i < count; i++) {
        out[i] = CompareValues(data[i], value, op) ? 1 : 0;
    }
    return result;
}

Tensor ApplyScalarComparison(const Tensor& input, float scalar, CompareOp op) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (input.GetDataType() == DataType::Float32 ||
        input.GetDataType() == DataType::Float64) {
        try {
            const af::array values = input.Shape().size() == 2
                ? input.GetArrayRowMajor2D()
                : input.GetArray();
            af::array mask;
            switch (op) {
                case CompareOp::Greater:
                    mask = values > scalar;
                    break;
                case CompareOp::GreaterEqual:
                    mask = values >= scalar;
                    break;
                case CompareOp::Less:
                    mask = values < scalar;
                    break;
                case CompareOp::LessEqual:
                    mask = values <= scalar;
                    break;
                case CompareOp::Equal:
                    mask = values == scalar;
                    break;
                case CompareOp::NotEqual:
                    mask = values != scalar;
                    break;
            }
            if (input.Shape().size() == 2) {
                return Tensor::FromArrayRowMajor2D(mask.as(u8));
            }
            return Tensor(mask.as(u8));
        } catch (const af::exception& e) {
            spdlog::warn(
                "ArrayFire tensor scalar comparison failed, using CPU fallback: {}",
                e.what());
        }
    }
#endif
    switch (input.GetDataType()) {
        case DataType::Float32: return ApplyScalarComparison<float>(input, scalar, op);
        case DataType::Float64: return ApplyScalarComparison<double>(input, scalar, op);
        case DataType::Int32: return ApplyScalarComparison<int32_t>(input, scalar, op);
        case DataType::Int64: return ApplyScalarComparison<int64_t>(input, scalar, op);
        case DataType::UInt8: return ApplyScalarComparison<uint8_t>(input, scalar, op);
    }
    throw std::runtime_error("Tensor scalar comparison: unsupported data type");
}

Tensor CompareTensors(const Tensor& left, const Tensor& right, CompareOp op) {
    const std::vector<size_t> out_shape = Tensor::BroadcastShape(left.Shape(), right.Shape());
    const Tensor left_expanded = left.Shape() == out_shape ? left.Clone() : left.Expand(out_shape);
    const Tensor right_expanded = right.Shape() == out_shape ? right.Clone() : right.Expand(out_shape);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if ((left_expanded.GetDataType() == DataType::Float32 ||
         left_expanded.GetDataType() == DataType::Float64) &&
        left_expanded.GetDataType() == right_expanded.GetDataType()) {
        try {
            const bool row_major_2d = out_shape.size() == 2;
            const af::array lhs = row_major_2d
                ? left_expanded.GetArrayRowMajor2D()
                : left_expanded.GetArray();
            const af::array rhs = row_major_2d
                ? right_expanded.GetArrayRowMajor2D()
                : right_expanded.GetArray();
            af::array mask;
            switch (op) {
                case CompareOp::Greater:
                    mask = lhs > rhs;
                    break;
                case CompareOp::GreaterEqual:
                    mask = lhs >= rhs;
                    break;
                case CompareOp::Less:
                    mask = lhs < rhs;
                    break;
                case CompareOp::LessEqual:
                    mask = lhs <= rhs;
                    break;
                case CompareOp::Equal:
                    mask = lhs == rhs;
                    break;
                case CompareOp::NotEqual:
                    mask = lhs != rhs;
                    break;
            }
            if (row_major_2d) {
                return Tensor::FromArrayRowMajor2D(mask.as(u8));
            }
            return Tensor(mask.as(u8));
        } catch (const af::exception& e) {
            spdlog::warn(
                "ArrayFire tensor comparison failed, using CPU fallback: {}",
                e.what());
        }
    }
#endif
    return ApplyTensorComparison(left_expanded, right_expanded, op);
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
