#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

template <typename T>
T SumValues(const T* data, size_t count) {
    T total{};
    for (size_t i = 0; i < count; i++) {
        total += data[i];
    }
    return total;
}

template <typename T>
T ProdValues(const T* data, size_t count) {
    T product = static_cast<T>(1);
    for (size_t i = 0; i < count; i++) {
        product *= data[i];
    }
    return product;
}

template <typename T>
T MaxValues(const T* data, size_t count) {
    if (count == 0) {
        throw std::runtime_error("Tensor::Max: cannot reduce an empty tensor");
    }
    T value = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] > value) {
            value = data[i];
        }
    }
    return value;
}

template <typename T>
T MinValues(const T* data, size_t count) {
    if (count == 0) {
        throw std::runtime_error("Tensor::Min: cannot reduce an empty tensor");
    }
    T value = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] < value) {
            value = data[i];
        }
    }
    return value;
}

template <typename T>
Tensor ScalarTensor(DataType dtype, T value) {
    return Tensor({1}, &value, dtype);
}

std::vector<size_t> ReducedShape(const std::vector<size_t>& shape, int axis, bool keepdim) {
    std::vector<size_t> out_shape;
    out_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        if (i == static_cast<size_t>(axis)) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }
    return out_shape;
}

size_t Product(const std::vector<size_t>& values, size_t begin, size_t end, const char* message) {
    return tensor_utils::CheckedProduct(values, begin, end, message);
}

template <typename T, typename InitFn, typename StepFn, typename FinishFn>
Tensor ReduceDimPreservingType(const Tensor& input,
                               int dim,
                               bool keepdim,
                               InitFn init,
                               StepFn step,
                               FinishFn finish,
                               const char* empty_message) {
    const int axis = tensor_utils::NormalizeDim(dim, input.NumDimensions());
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0 && empty_message != nullptr) {
        throw std::runtime_error(empty_message);
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), input.GetDataType());
    const T* src = input.Data<T>();
    T* dst = result.Data<T>();

    for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            T accum = static_cast<T>(init());
            for (size_t k = 0; k < axis_size; k++) {
                const size_t src_idx = outer_idx * axis_size * inner + k * inner + inner_idx;
                accum = static_cast<T>(step(accum, src[src_idx]));
            }
            dst[outer_idx * inner + inner_idx] = finish(accum, axis_size);
        }
    }
    return result;
}

template <typename InitFn, typename StepFn, typename FinishFn>
Tensor ReduceDimPreservingType(const Tensor& input,
                               int dim,
                               bool keepdim,
                               InitFn init,
                               StepFn step,
                               FinishFn finish,
                               const char* empty_message) {
    switch (input.GetDataType()) {
        case DataType::Float32: return ReduceDimPreservingType<float>(input, dim, keepdim, init, step, finish, empty_message);
        case DataType::Float64: return ReduceDimPreservingType<double>(input, dim, keepdim, init, step, finish, empty_message);
        case DataType::Int32: return ReduceDimPreservingType<int32_t>(input, dim, keepdim, init, step, finish, empty_message);
        case DataType::Int64: return ReduceDimPreservingType<int64_t>(input, dim, keepdim, init, step, finish, empty_message);
        case DataType::UInt8: return ReduceDimPreservingType<uint8_t>(input, dim, keepdim, init, step, finish, empty_message);
    }
    throw std::runtime_error("Tensor reduction: unsupported data type");
}

template <typename T, typename StepFn>
Tensor ReduceDimFromFirstValue(const Tensor& input,
                               int dim,
                               bool keepdim,
                               StepFn step,
                               const char* empty_message) {
    const int axis = tensor_utils::NormalizeDim(dim, input.NumDimensions());
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0) {
        throw std::runtime_error(empty_message);
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), input.GetDataType());
    const T* src = input.Data<T>();
    T* dst = result.Data<T>();

    for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            T accum = src[outer_idx * axis_size * inner + inner_idx];
            for (size_t k = 1; k < axis_size; k++) {
                const size_t src_idx = outer_idx * axis_size * inner + k * inner + inner_idx;
                accum = step(accum, src[src_idx]);
            }
            dst[outer_idx * inner + inner_idx] = accum;
        }
    }
    return result;
}

template <typename StepFn>
Tensor ReduceDimFromFirstValue(const Tensor& input,
                               int dim,
                               bool keepdim,
                               StepFn step,
                               const char* empty_message) {
    switch (input.GetDataType()) {
        case DataType::Float32: return ReduceDimFromFirstValue<float>(input, dim, keepdim, step, empty_message);
        case DataType::Float64: return ReduceDimFromFirstValue<double>(input, dim, keepdim, step, empty_message);
        case DataType::Int32: return ReduceDimFromFirstValue<int32_t>(input, dim, keepdim, step, empty_message);
        case DataType::Int64: return ReduceDimFromFirstValue<int64_t>(input, dim, keepdim, step, empty_message);
        case DataType::UInt8: return ReduceDimFromFirstValue<uint8_t>(input, dim, keepdim, step, empty_message);
    }
    throw std::runtime_error("Tensor reduction: unsupported data type");
}

template <typename T>
Tensor MeanDimAsReal(const Tensor& input, int dim, bool keepdim, DataType out_dtype) {
    const int axis = tensor_utils::NormalizeDim(dim, input.NumDimensions());
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0) {
        throw std::runtime_error("Tensor::Mean: cannot reduce an empty dimension");
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), out_dtype);
    const T* src = input.Data<T>();

    if (out_dtype == DataType::Float64) {
        double* dst = result.Data<double>();
        for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
            for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
                double total = 0.0;
                for (size_t k = 0; k < axis_size; k++) {
                    total += static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]);
                }
                dst[outer_idx * inner + inner_idx] = total / static_cast<double>(axis_size);
            }
        }
        return result;
    }

    float* dst = result.Data<float>();
    for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            double total = 0.0;
            for (size_t k = 0; k < axis_size; k++) {
                total += static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]);
            }
            dst[outer_idx * inner + inner_idx] = static_cast<float>(total / static_cast<double>(axis_size));
        }
    }
    return result;
}

Tensor MeanDimAsReal(const Tensor& input, int dim, bool keepdim) {
    const DataType out_dtype = input.GetDataType() == DataType::Float64 ? DataType::Float64 : DataType::Float32;
    switch (input.GetDataType()) {
        case DataType::Float32: return MeanDimAsReal<float>(input, dim, keepdim, out_dtype);
        case DataType::Float64: return MeanDimAsReal<double>(input, dim, keepdim, out_dtype);
        case DataType::Int32: return MeanDimAsReal<int32_t>(input, dim, keepdim, out_dtype);
        case DataType::Int64: return MeanDimAsReal<int64_t>(input, dim, keepdim, out_dtype);
        case DataType::UInt8: return MeanDimAsReal<uint8_t>(input, dim, keepdim, out_dtype);
    }
    throw std::runtime_error("Tensor::Mean: unsupported data type");
}

template <typename T>
Tensor VarDimAsReal(const Tensor& input, int dim, bool keepdim, DataType out_dtype) {
    const int axis = tensor_utils::NormalizeDim(dim, input.NumDimensions());
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0) {
        throw std::runtime_error("Tensor::Var: cannot reduce an empty dimension");
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), out_dtype);
    const T* src = input.Data<T>();

    if (out_dtype == DataType::Float64) {
        double* dst = result.Data<double>();
        for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
            for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
                double total = 0.0;
                for (size_t k = 0; k < axis_size; k++) {
                    total += static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]);
                }
                const double mean = total / static_cast<double>(axis_size);
                double variance = 0.0;
                for (size_t k = 0; k < axis_size; k++) {
                    const double diff = static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]) - mean;
                    variance += diff * diff;
                }
                dst[outer_idx * inner + inner_idx] = variance / static_cast<double>(axis_size);
            }
        }
        return result;
    }

    float* dst = result.Data<float>();
    for (size_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (size_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            double total = 0.0;
            for (size_t k = 0; k < axis_size; k++) {
                total += static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]);
            }
            const double mean = total / static_cast<double>(axis_size);
            double variance = 0.0;
            for (size_t k = 0; k < axis_size; k++) {
                const double diff = static_cast<double>(src[outer_idx * axis_size * inner + k * inner + inner_idx]) - mean;
                variance += diff * diff;
            }
            dst[outer_idx * inner + inner_idx] = static_cast<float>(variance / static_cast<double>(axis_size));
        }
    }
    return result;
}

Tensor VarDimAsReal(const Tensor& input, int dim, bool keepdim) {
    const DataType out_dtype = input.GetDataType() == DataType::Float64 ? DataType::Float64 : DataType::Float32;
    switch (input.GetDataType()) {
        case DataType::Float32: return VarDimAsReal<float>(input, dim, keepdim, out_dtype);
        case DataType::Float64: return VarDimAsReal<double>(input, dim, keepdim, out_dtype);
        case DataType::Int32: return VarDimAsReal<int32_t>(input, dim, keepdim, out_dtype);
        case DataType::Int64: return VarDimAsReal<int64_t>(input, dim, keepdim, out_dtype);
        case DataType::UInt8: return VarDimAsReal<uint8_t>(input, dim, keepdim, out_dtype);
    }
    throw std::runtime_error("Tensor::Var: unsupported data type");
}

template <typename T>
double MeanAsDouble(const T* data, size_t count) {
    double total = 0.0;
    for (size_t i = 0; i < count; i++) {
        total += static_cast<double>(data[i]);
    }
    return total / static_cast<double>(count);
}

template <typename T>
double VarianceAsDouble(const T* data, size_t count) {
    if (count == 0) {
        throw std::runtime_error("Tensor::Var: cannot reduce an empty tensor");
    }

    const double mean = MeanAsDouble(data, count);
    double variance = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double diff = static_cast<double>(data[i]) - mean;
        variance += diff * diff;
    }
    return variance / static_cast<double>(count);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool IsArrayFireRealReductionSupported(DataType dtype) {
    return dtype == DataType::Float32 || dtype == DataType::Float64;
}

Tensor FinishArrayFire2DReduction(const af::array& reduced, bool keepdim) {
    if (keepdim) {
        return Tensor::FromArrayRowMajor2D(reduced);
    }
    return Tensor(af::flat(reduced));
}

template <typename ReduceFn>
Tensor ReduceDimArrayFire2D(const Tensor& input,
                            int dim,
                            bool keepdim,
                            ReduceFn reduce) {
    const int axis = tensor_utils::NormalizeDim(dim, input.NumDimensions());
    return FinishArrayFire2DReduction(
        reduce(input.GetArrayRowMajor2D(), axis),
        keepdim);
}
#endif

} // namespace

Tensor Tensor::Sum() const {
    const size_t count = NumElements();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (count > 0 && IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return Tensor(af::sum(af::flat(GetArray())));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Sum: ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(DataType::Float32, SumValues(Data<float>(), count));
        case DataType::Float64: return ScalarTensor(DataType::Float64, SumValues(Data<double>(), count));
        case DataType::Int32: return ScalarTensor(DataType::Int32, SumValues(Data<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(DataType::Int64, SumValues(Data<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(DataType::UInt8, SumValues(Data<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Sum: unsupported data type");
}

Tensor Tensor::Sum(int dim, bool keepdim) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const int axis = tensor_utils::NormalizeDim(dim, NumDimensions());
    if (shape_.size() == 2 &&
        shape_[static_cast<size_t>(axis)] > 0 &&
        IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return ReduceDimArrayFire2D(*this, dim, keepdim, [](const af::array& arr, int axis) {
                return af::sum(arr, axis);
            });
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Sum(dim): ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return ReduceDimPreservingType(
        *this,
        dim,
        keepdim,
        []() { return 0; },
        [](auto accum, auto value) { return accum + value; },
        [](auto accum, size_t) { return accum; },
        nullptr);
}

Tensor Tensor::Mean() const {
    const size_t count = NumElements();
    if (count == 0) {
        throw std::runtime_error("Tensor::Mean: cannot reduce an empty tensor");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return Tensor(af::mean(af::flat(GetArray())));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Mean: ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif

    if (dtype_ == DataType::Float64) {
        const double value = SumValues(Data<double>(), count) / static_cast<double>(count);
        return ScalarTensor(DataType::Float64, value);
    }

    double total = 0.0;
    switch (dtype_) {
        case DataType::Float32: {
            const float* data = Data<float>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Int32: {
            const int32_t* data = Data<int32_t>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Int64: {
            const int64_t* data = Data<int64_t>();
            for (size_t i = 0; i < count; i++) total += static_cast<double>(data[i]);
            break;
        }
        case DataType::UInt8: {
            const uint8_t* data = Data<uint8_t>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Float64:
            break;
    }

    const float value = static_cast<float>(total / static_cast<double>(count));
    return ScalarTensor(DataType::Float32, value);
}

Tensor Tensor::Mean(int dim, bool keepdim) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const int axis = tensor_utils::NormalizeDim(dim, NumDimensions());
    if (shape_.size() == 2 &&
        shape_[static_cast<size_t>(axis)] > 0 &&
        IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return ReduceDimArrayFire2D(*this, dim, keepdim, [](const af::array& arr, int axis) {
                return af::mean(arr, axis);
            });
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Mean(dim): ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return MeanDimAsReal(*this, dim, keepdim);
}

Tensor Tensor::Max() const {
    const size_t count = NumElements();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (count > 0 && IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return Tensor((af::max)(af::flat(GetArray())));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Max: ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(DataType::Float32, MaxValues(Data<float>(), count));
        case DataType::Float64: return ScalarTensor(DataType::Float64, MaxValues(Data<double>(), count));
        case DataType::Int32: return ScalarTensor(DataType::Int32, MaxValues(Data<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(DataType::Int64, MaxValues(Data<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(DataType::UInt8, MaxValues(Data<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Max: unsupported data type");
}

Tensor Tensor::Max(int dim, bool keepdim) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const int axis = tensor_utils::NormalizeDim(dim, NumDimensions());
    if (shape_.size() == 2 &&
        shape_[static_cast<size_t>(axis)] > 0 &&
        IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return ReduceDimArrayFire2D(*this, dim, keepdim, [](const af::array& arr, int axis) {
                return (af::max)(arr, axis);
            });
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Max(dim): ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return ReduceDimFromFirstValue(
        *this,
        dim,
        keepdim,
        [](auto accum, auto value) { return value > accum ? value : accum; },
        "Tensor::Max: cannot reduce an empty dimension");
}

Tensor Tensor::Min() const {
    const size_t count = NumElements();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (count > 0 && IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return Tensor((af::min)(af::flat(GetArray())));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Min: ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(DataType::Float32, MinValues(Data<float>(), count));
        case DataType::Float64: return ScalarTensor(DataType::Float64, MinValues(Data<double>(), count));
        case DataType::Int32: return ScalarTensor(DataType::Int32, MinValues(Data<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(DataType::Int64, MinValues(Data<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(DataType::UInt8, MinValues(Data<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Min: unsupported data type");
}

Tensor Tensor::Min(int dim, bool keepdim) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const int axis = tensor_utils::NormalizeDim(dim, NumDimensions());
    if (shape_.size() == 2 &&
        shape_[static_cast<size_t>(axis)] > 0 &&
        IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return ReduceDimArrayFire2D(*this, dim, keepdim, [](const af::array& arr, int axis) {
                return (af::min)(arr, axis);
            });
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Min(dim): ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return ReduceDimFromFirstValue(
        *this,
        dim,
        keepdim,
        [](auto accum, auto value) { return value < accum ? value : accum; },
        "Tensor::Min: cannot reduce an empty dimension");
}

Tensor Tensor::Prod() const {
    const size_t count = NumElements();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (count > 0 && IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return Tensor(af::product(af::flat(GetArray())));
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Prod: ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(DataType::Float32, ProdValues(Data<float>(), count));
        case DataType::Float64: return ScalarTensor(DataType::Float64, ProdValues(Data<double>(), count));
        case DataType::Int32: return ScalarTensor(DataType::Int32, ProdValues(Data<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(DataType::Int64, ProdValues(Data<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(DataType::UInt8, ProdValues(Data<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Prod: unsupported data type");
}

Tensor Tensor::Prod(int dim, bool keepdim) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const int axis = tensor_utils::NormalizeDim(dim, NumDimensions());
    if (shape_.size() == 2 &&
        shape_[static_cast<size_t>(axis)] > 0 &&
        IsArrayFireRealReductionSupported(dtype_)) {
        try {
            return ReduceDimArrayFire2D(*this, dim, keepdim, [](const af::array& arr, int axis) {
                return af::product(arr, axis);
            });
        } catch (const af::exception& e) {
            spdlog::warn("Tensor::Prod(dim): ArrayFire reduction failed, falling back to CPU: {}", e.what());
        }
    }
#endif
    return ReduceDimPreservingType(
        *this,
        dim,
        keepdim,
        []() { return 1; },
        [](auto accum, auto value) { return accum * value; },
        [](auto accum, size_t) { return accum; },
        nullptr);
}

Tensor Tensor::Var() const {
    const size_t count = NumElements();
    if (dtype_ == DataType::Float64) {
        const double value = VarianceAsDouble(Data<double>(), count);
        return ScalarTensor(DataType::Float64, value);
    }

    double value = 0.0;
    switch (dtype_) {
        case DataType::Float32: value = VarianceAsDouble(Data<float>(), count); break;
        case DataType::Int32: value = VarianceAsDouble(Data<int32_t>(), count); break;
        case DataType::Int64: value = VarianceAsDouble(Data<int64_t>(), count); break;
        case DataType::UInt8: value = VarianceAsDouble(Data<uint8_t>(), count); break;
        case DataType::Float64: break;
    }

    const float out = static_cast<float>(value);
    return ScalarTensor(DataType::Float32, out);
}

Tensor Tensor::Var(int dim, bool keepdim) const {
    return VarDimAsReal(*this, dim, keepdim);
}

Tensor Tensor::Std() const {
    Tensor variance = Var();
    if (variance.GetDataType() == DataType::Float64) {
        const double value = std::sqrt(variance.Data<double>()[0]);
        return ScalarTensor(DataType::Float64, value);
    }

    const float value = std::sqrt(variance.Data<float>()[0]);
    return ScalarTensor(DataType::Float32, value);
}

Tensor Tensor::Std(int dim, bool keepdim) const {
    return Var(dim, keepdim).Sqrt();
}

} // namespace cyxwiz
