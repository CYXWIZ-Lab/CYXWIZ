#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"
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

template <typename T>
T AddPreservingType(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result =
            static_cast<Unsigned>(lhs) + static_cast<Unsigned>(rhs);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return lhs + rhs;
    }
}

template <typename T>
T MultiplyPreservingType(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result =
            static_cast<Unsigned>(lhs) * static_cast<Unsigned>(rhs);
        if constexpr (std::is_signed_v<T>) {
            return std::bit_cast<T>(result);
        } else {
            return static_cast<T>(result);
        }
    } else {
        return lhs * rhs;
    }
}

template <typename T>
T SumValues(const T* data, size_t count) {
    T total{};
    for (size_t i = 0; i < count; i++) {
        total = AddPreservingType(total, data[i]);
    }
    return total;
}

template <typename T>
T ProdValues(const T* data, size_t count) {
    T product = static_cast<T>(1);
    for (size_t i = 0; i < count; i++) {
        product = MultiplyPreservingType(product, data[i]);
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
    return Tensor({}, &value, dtype);
}

int NormalizeReductionDim(int dim, int rank) {
    if (rank == 0) {
        if (dim == 0 || dim == -1) {
            return 0;
        }
        throw std::runtime_error("Tensor dimension out of range");
    }
    return tensor_utils::NormalizeDim(dim, rank);
}

std::vector<size_t> ReducedShape(const std::vector<size_t>& shape, int axis, bool keepdim) {
    if (shape.empty()) {
        return {};
    }
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
    return out_shape;
}

DataType RealReductionType(DataType dtype) {
    return dtype == DataType::Float64 ? DataType::Float64 : DataType::Float32;
}

double VarianceDenominator(size_t count, int64_t correction) {
    const long double denominator =
        static_cast<long double>(count) - static_cast<long double>(correction);
    return denominator > 0.0L ? static_cast<double>(denominator) : 0.0;
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
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    if (input.Shape().empty()) {
        return input.Clone();
    }
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0 && empty_message != nullptr) {
        throw std::runtime_error(empty_message);
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), input.GetDataType());
    const T* src = input.ReadData<T>();
    T* dst = result.MutableData<T>();

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
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    if (input.Shape().empty()) {
        return input.Clone();
    }
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];
    if (axis_size == 0) {
        throw std::runtime_error(empty_message);
    }

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), input.GetDataType());
    const T* src = input.ReadData<T>();
    T* dst = result.MutableData<T>();

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
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    if (input.Shape().empty()) {
        return input.Mean();
    }
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), out_dtype);

    if (out_dtype == DataType::Float64) {
        double* dst = result.MutableData<double>();
        if (axis_size == 0) {
            for (size_t index = 0; index < result.NumElements(); ++index) {
                dst[index] = std::numeric_limits<double>::quiet_NaN();
            }
            return result;
        }
        const T* src = input.ReadData<T>();
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

    float* dst = result.MutableData<float>();
    if (axis_size == 0) {
        for (size_t index = 0; index < result.NumElements(); ++index) {
            dst[index] = std::numeric_limits<float>::quiet_NaN();
        }
        return result;
    }
    const T* src = input.ReadData<T>();
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
    const DataType out_dtype = RealReductionType(input.GetDataType());
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
Tensor VarDimAsReal(const Tensor& input,
                    int dim,
                    bool keepdim,
                    DataType out_dtype,
                    int64_t correction) {
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    if (input.Shape().empty()) {
        return input.VarWithCorrection(correction);
    }
    const auto& shape = input.Shape();
    const size_t axis_size = shape[static_cast<size_t>(axis)];

    const size_t outer = Product(shape, 0, static_cast<size_t>(axis), "Tensor reduction: outer product overflow");
    const size_t inner = Product(shape, static_cast<size_t>(axis + 1), shape.size(), "Tensor reduction: inner product overflow");
    Tensor result(ReducedShape(shape, axis, keepdim), out_dtype);
    const double denominator = VarianceDenominator(axis_size, correction);

    if (out_dtype == DataType::Float64) {
        double* dst = result.MutableData<double>();
        if (axis_size == 0) {
            for (size_t index = 0; index < result.NumElements(); ++index) {
                dst[index] = std::numeric_limits<double>::quiet_NaN();
            }
            return result;
        }
        const T* src = input.ReadData<T>();
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
                dst[outer_idx * inner + inner_idx] = variance / denominator;
            }
        }
        return result;
    }

    float* dst = result.MutableData<float>();
    if (axis_size == 0) {
        for (size_t index = 0; index < result.NumElements(); ++index) {
            dst[index] = std::numeric_limits<float>::quiet_NaN();
        }
        return result;
    }
    const T* src = input.ReadData<T>();
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
            dst[outer_idx * inner + inner_idx] =
                static_cast<float>(variance / denominator);
        }
    }
    return result;
}

Tensor VarDimAsReal(const Tensor& input,
                    int dim,
                    bool keepdim,
                    int64_t correction) {
    const DataType out_dtype = RealReductionType(input.GetDataType());
    switch (input.GetDataType()) {
        case DataType::Float32: return VarDimAsReal<float>(input, dim, keepdim, out_dtype, correction);
        case DataType::Float64: return VarDimAsReal<double>(input, dim, keepdim, out_dtype, correction);
        case DataType::Int32: return VarDimAsReal<int32_t>(input, dim, keepdim, out_dtype, correction);
        case DataType::Int64: return VarDimAsReal<int64_t>(input, dim, keepdim, out_dtype, correction);
        case DataType::UInt8: return VarDimAsReal<uint8_t>(input, dim, keepdim, out_dtype, correction);
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
double VarianceAsDouble(const T* data, size_t count, int64_t correction) {
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double mean = MeanAsDouble(data, count);
    double variance = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double diff = static_cast<double>(data[i]) - mean;
        variance += diff * diff;
    }
    return variance / VarianceDenominator(count, correction);
}

Tensor NativeVariance(const Tensor& input, int64_t correction) {
    const size_t count = input.NumElements();
    const DataType output_dtype = RealReductionType(input.GetDataType());
    if (count == 0) {
        if (output_dtype == DataType::Float64) {
            return ScalarTensor(
                output_dtype, std::numeric_limits<double>::quiet_NaN());
        }
        return ScalarTensor(
            output_dtype, std::numeric_limits<float>::quiet_NaN());
    }
    if (output_dtype == DataType::Float64) {
        return ScalarTensor(
            output_dtype,
            VarianceAsDouble(input.ReadData<double>(), count, correction));
    }

    double value = 0.0;
    switch (input.GetDataType()) {
        case DataType::Float32:
            value = VarianceAsDouble(
                input.ReadData<float>(), count, correction);
            break;
        case DataType::Int32:
            value = VarianceAsDouble(
                input.ReadData<int32_t>(), count, correction);
            break;
        case DataType::Int64:
            value = VarianceAsDouble(
                input.ReadData<int64_t>(), count, correction);
            break;
        case DataType::UInt8:
            value = VarianceAsDouble(
                input.ReadData<uint8_t>(), count, correction);
            break;
        case DataType::Float64:
            break;
    }
    return ScalarTensor(output_dtype, static_cast<float>(value));
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::dtype ArrayFireType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return f32;
        case DataType::Float64: return f64;
        case DataType::Int32: return s32;
        case DataType::Int64: return s64;
        case DataType::UInt8: return u8;
    }
    throw std::runtime_error("Tensor reduction: unsupported ArrayFire data type");
}

af::dim4 ArrayFireDims(const std::vector<size_t>& shape) {
    af::dim4 dimensions(1, 1, 1, 1);
    for (size_t index = 0; index < shape.size(); ++index) {
        if (shape[index] >
            static_cast<size_t>((std::numeric_limits<dim_t>::max)())) {
            throw std::overflow_error(
                "Tensor reduction: dimension exceeds ArrayFire range");
        }
        dimensions[static_cast<unsigned>(index)] =
            static_cast<dim_t>(shape[index]);
    }
    return dimensions;
}

Tensor FinishArrayFireReduction(const af::array& reduced,
                                const std::vector<size_t>& output_shape) {
    af::array output = af::moddims(reduced, ArrayFireDims(output_shape));
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}

Tensor ArrayFireConstantReduction(const std::vector<size_t>& output_shape,
                                  DataType dtype,
                                  double value) {
    if (Product(output_shape, 0, output_shape.size(),
                "Tensor reduction: output shape overflow") == 0) {
        return Tensor(output_shape, dtype);
    }
    af::array output = af::constant(
        value, ArrayFireDims(output_shape), ArrayFireType(dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, output_shape);
}

af::array RealArray(const Tensor& input) {
    const af::array values = input.GetSemanticArray();
    return input.GetDataType() == DataType::Float64
        ? values.as(f64)
        : values.as(f32);
}

af::array ArrayFireVariance(const af::array& values,
                            size_t reduction_count,
                            int64_t correction) {
    const af::array flattened = af::flat(values);
    const af::array mean = af::mean(flattened);
    const af::array centered = flattened - mean;
    return af::sum(centered * centered) /
           VarianceDenominator(reduction_count, correction);
}

af::array ArrayFireVariance(const af::array& values,
                            int axis,
                            size_t reduction_count,
                            int64_t correction) {
    const af::array mean = af::mean(values, axis);
    const af::array centered = values - mean;
    return af::sum(centered * centered, axis) /
           VarianceDenominator(reduction_count, correction);
}

void RecordReductionArrayFireFallback(
    const char* operation_name,
    const Tensor& input,
    const std::vector<size_t>& output_shape,
    const std::string& attributes,
    const char* error_message) {
    const std::string shape_signature =
        tensor_backend_observation::BuildTensorOpSignature(
            {input.Shape()},
            output_shape,
            input.GetDataType(),
            attributes);
    tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(input.GetDataType()),
        shape_signature,
        error_message);
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context =
        BuildArrayFireBackendFallbackContext(shape_signature);
    if (ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context)) {
        spdlog::warn(
            "{}",
            BuildArrayFireBackendFallbackMessage(
                operation_name,
                reason,
                reason != BackendFallbackReason::CudaJitParamOverflow,
                error_message,
                context));
    }
}

void RecordUnsupportedReductionRank(
    const char* operation_name,
    const Tensor& input,
    const std::vector<size_t>& output_shape,
    const std::string& attributes) {
    RecordReductionArrayFireFallback(
        operation_name,
        input,
        output_shape,
        attributes,
        "ArrayFire Tensor reductions support ranks up to 4");
}

template <typename ReduceFn>
Tensor ReduceDimArrayFire(const Tensor& input,
                          int axis,
                          bool keepdim,
                          ReduceFn reduce) {
    const std::vector<size_t> output_shape =
        ReducedShape(input.Shape(), axis, keepdim);
    return FinishArrayFireReduction(
        reduce(input.GetSemanticArray(), axis).as(
            ArrayFireType(input.GetDataType())),
        output_shape);
}
#endif

Tensor VarianceGlobal(const Tensor& input,
                      int64_t correction,
                      const char* operation_name,
                      const char* operation,
                      const std::string& attributes_override = {}) {
    const std::string attributes = attributes_override.empty()
        ? "op=" + std::string(operation) +
              ";axis=all;correction=" + std::to_string(correction)
        : attributes_override;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const size_t count = input.NumElements();
    if (input.Shape().size() <= 4) {
        try {
            if (count == 0) {
                return ArrayFireConstantReduction(
                    {}, RealReductionType(input.GetDataType()),
                    std::numeric_limits<double>::quiet_NaN());
            }
            return FinishArrayFireReduction(
                ArrayFireVariance(
                    RealArray(input), count, correction),
                {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                operation_name, input, {}, attributes, error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            operation_name, input, {}, attributes);
    }
#else
    (void)operation_name;
#endif
    return NativeVariance(input, correction);
}

Tensor VarianceDim(const Tensor& input,
                   int dim,
                   bool keepdim,
                   int64_t correction,
                   const char* operation_name,
                   const char* operation) {
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    const std::string attributes =
        "op=" + std::string(operation) +
        ";axis=" + std::to_string(axis) +
        ";correction=" + std::to_string(correction);
    if (input.Shape().empty()) {
        return VarianceGlobal(
            input, correction, operation_name, operation, attributes);
    }
    const std::vector<size_t> output_shape =
        ReducedShape(input.Shape(), axis, keepdim);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const DataType output_dtype = RealReductionType(input.GetDataType());
    const size_t reduction_count =
        input.Shape()[static_cast<size_t>(axis)];
    if (input.Shape().size() <= 4) {
        try {
            if (reduction_count == 0) {
                return ArrayFireConstantReduction(
                    output_shape, output_dtype,
                    std::numeric_limits<double>::quiet_NaN());
            }
            return FinishArrayFireReduction(
                ArrayFireVariance(
                    RealArray(input), axis, reduction_count, correction),
                output_shape);
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                operation_name, input, output_shape,
                attributes, error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            operation_name, input, output_shape, attributes);
    }
#endif
    return VarDimAsReal(input, dim, keepdim, correction);
}

} // namespace

Tensor Tensor::Sum() const {
    const size_t count = NumElements();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            if (count == 0) {
                return ArrayFireConstantReduction({}, dtype_, 0.0);
            }
            return FinishArrayFireReduction(
                af::sum(af::flat(GetSemanticArray())).as(
                    ArrayFireType(dtype_)),
                {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Sum", *this, {}, "op=sum;axis=all",
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Sum", *this, {}, "op=sum;axis=all");
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(dtype_, SumValues(ReadData<float>(), count));
        case DataType::Float64: return ScalarTensor(dtype_, SumValues(ReadData<double>(), count));
        case DataType::Int32: return ScalarTensor(dtype_, SumValues(ReadData<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(dtype_, SumValues(ReadData<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(dtype_, SumValues(ReadData<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Sum: unsupported data type");
}

Tensor Tensor::Sum(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Sum();
    }
    const std::vector<size_t> output_shape =
        ReducedShape(shape_, axis, keepdim);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            if (shape_[static_cast<size_t>(axis)] == 0) {
                return ArrayFireConstantReduction(output_shape, dtype_, 0.0);
            }
            return ReduceDimArrayFire(
                *this, axis, keepdim,
                [](const af::array& values, int reduction_axis) {
                    return af::sum(values, reduction_axis);
                });
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Sum(dim)", *this, output_shape,
                "op=sum;axis=" + std::to_string(axis),
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Sum(dim)", *this, output_shape,
            "op=sum;axis=" + std::to_string(axis));
    }
#endif
    return ReduceDimPreservingType(
        *this,
        dim,
        keepdim,
        []() { return 0; },
        [](auto accum, auto value) {
            return AddPreservingType(accum, value);
        },
        [](auto accum, size_t) { return accum; },
        nullptr);
}

Tensor Tensor::Mean() const {
    const size_t count = NumElements();
    const DataType output_dtype = RealReductionType(dtype_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            if (count == 0) {
                return ArrayFireConstantReduction(
                    {}, output_dtype,
                    std::numeric_limits<double>::quiet_NaN());
            }
            return FinishArrayFireReduction(
                af::mean(af::flat(RealArray(*this))), {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Mean", *this, {}, "op=mean;axis=all",
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Mean", *this, {}, "op=mean;axis=all");
    }
#endif

    if (count == 0) {
        if (output_dtype == DataType::Float64) {
            return ScalarTensor(
                output_dtype, std::numeric_limits<double>::quiet_NaN());
        }
        return ScalarTensor(
            output_dtype, std::numeric_limits<float>::quiet_NaN());
    }
    if (dtype_ == DataType::Float64) {
        const double value =
            SumValues(ReadData<double>(), count) / static_cast<double>(count);
        return ScalarTensor(output_dtype, value);
    }

    double total = 0.0;
    switch (dtype_) {
        case DataType::Float32: {
            const float* data = ReadData<float>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Int32: {
            const int32_t* data = ReadData<int32_t>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Int64: {
            const int64_t* data = ReadData<int64_t>();
            for (size_t i = 0; i < count; i++) total += static_cast<double>(data[i]);
            break;
        }
        case DataType::UInt8: {
            const uint8_t* data = ReadData<uint8_t>();
            for (size_t i = 0; i < count; i++) total += data[i];
            break;
        }
        case DataType::Float64:
            break;
    }

    const float value = static_cast<float>(total / static_cast<double>(count));
    return ScalarTensor(output_dtype, value);
}

Tensor Tensor::Mean(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Mean();
    }
    const std::vector<size_t> output_shape =
        ReducedShape(shape_, axis, keepdim);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const DataType output_dtype = RealReductionType(dtype_);
    if (shape_.size() <= 4) {
        try {
            if (shape_[static_cast<size_t>(axis)] == 0) {
                return ArrayFireConstantReduction(
                    output_shape, output_dtype,
                    std::numeric_limits<double>::quiet_NaN());
            }
            return FinishArrayFireReduction(
                af::mean(RealArray(*this), axis), output_shape);
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Mean(dim)", *this, output_shape,
                "op=mean;axis=" + std::to_string(axis),
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Mean(dim)", *this, output_shape,
            "op=mean;axis=" + std::to_string(axis));
    }
#endif
    return MeanDimAsReal(*this, dim, keepdim);
}

Tensor Tensor::Max() const {
    const size_t count = NumElements();
    if (count == 0) {
        throw std::runtime_error("Tensor::Max: cannot reduce an empty tensor");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return FinishArrayFireReduction(
                (af::max)(af::flat(GetSemanticArray())).as(
                    ArrayFireType(dtype_)),
                {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Max", *this, {}, "op=max;axis=all",
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Max", *this, {}, "op=max;axis=all");
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(dtype_, MaxValues(ReadData<float>(), count));
        case DataType::Float64: return ScalarTensor(dtype_, MaxValues(ReadData<double>(), count));
        case DataType::Int32: return ScalarTensor(dtype_, MaxValues(ReadData<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(dtype_, MaxValues(ReadData<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(dtype_, MaxValues(ReadData<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Max: unsupported data type");
}

Tensor Tensor::Max(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Max();
    }
    const std::vector<size_t> output_shape =
        ReducedShape(shape_, axis, keepdim);
    if (shape_[static_cast<size_t>(axis)] == 0) {
        throw std::runtime_error("Tensor::Max: cannot reduce an empty dimension");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return ReduceDimArrayFire(
                *this, axis, keepdim,
                [](const af::array& values, int reduction_axis) {
                    return (af::max)(values, reduction_axis);
                });
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Max(dim)", *this, output_shape,
                "op=max;axis=" + std::to_string(axis),
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Max(dim)", *this, output_shape,
            "op=max;axis=" + std::to_string(axis));
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
    if (count == 0) {
        throw std::runtime_error("Tensor::Min: cannot reduce an empty tensor");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return FinishArrayFireReduction(
                (af::min)(af::flat(GetSemanticArray())).as(
                    ArrayFireType(dtype_)),
                {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Min", *this, {}, "op=min;axis=all",
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Min", *this, {}, "op=min;axis=all");
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(dtype_, MinValues(ReadData<float>(), count));
        case DataType::Float64: return ScalarTensor(dtype_, MinValues(ReadData<double>(), count));
        case DataType::Int32: return ScalarTensor(dtype_, MinValues(ReadData<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(dtype_, MinValues(ReadData<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(dtype_, MinValues(ReadData<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Min: unsupported data type");
}

Tensor Tensor::Min(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Min();
    }
    const std::vector<size_t> output_shape =
        ReducedShape(shape_, axis, keepdim);
    if (shape_[static_cast<size_t>(axis)] == 0) {
        throw std::runtime_error("Tensor::Min: cannot reduce an empty dimension");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return ReduceDimArrayFire(
                *this, axis, keepdim,
                [](const af::array& values, int reduction_axis) {
                    return (af::min)(values, reduction_axis);
                });
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Min(dim)", *this, output_shape,
                "op=min;axis=" + std::to_string(axis),
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Min(dim)", *this, output_shape,
            "op=min;axis=" + std::to_string(axis));
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
    if (shape_.size() <= 4) {
        try {
            if (count == 0) {
                return ArrayFireConstantReduction({}, dtype_, 1.0);
            }
            return FinishArrayFireReduction(
                af::product(af::flat(GetSemanticArray())).as(
                    ArrayFireType(dtype_)),
                {});
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Prod", *this, {}, "op=prod;axis=all",
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Prod", *this, {}, "op=prod;axis=all");
    }
#endif
    switch (dtype_) {
        case DataType::Float32: return ScalarTensor(dtype_, ProdValues(ReadData<float>(), count));
        case DataType::Float64: return ScalarTensor(dtype_, ProdValues(ReadData<double>(), count));
        case DataType::Int32: return ScalarTensor(dtype_, ProdValues(ReadData<int32_t>(), count));
        case DataType::Int64: return ScalarTensor(dtype_, ProdValues(ReadData<int64_t>(), count));
        case DataType::UInt8: return ScalarTensor(dtype_, ProdValues(ReadData<uint8_t>(), count));
    }
    throw std::runtime_error("Tensor::Prod: unsupported data type");
}

Tensor Tensor::Prod(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Prod();
    }
    const std::vector<size_t> output_shape =
        ReducedShape(shape_, axis, keepdim);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            if (shape_[static_cast<size_t>(axis)] == 0) {
                return ArrayFireConstantReduction(output_shape, dtype_, 1.0);
            }
            return ReduceDimArrayFire(
                *this, axis, keepdim,
                [](const af::array& values, int reduction_axis) {
                    return af::product(values, reduction_axis);
                });
        } catch (const af::exception& error) {
            RecordReductionArrayFireFallback(
                "Tensor::Prod(dim)", *this, output_shape,
                "op=prod;axis=" + std::to_string(axis),
                error.what());
        }
    } else {
        RecordUnsupportedReductionRank(
            "Tensor::Prod(dim)", *this, output_shape,
            "op=prod;axis=" + std::to_string(axis));
    }
#endif
    return ReduceDimPreservingType(
        *this,
        dim,
        keepdim,
        []() { return 1; },
        [](auto accum, auto value) {
            return MultiplyPreservingType(accum, value);
        },
        [](auto accum, size_t) { return accum; },
        nullptr);
}

Tensor Tensor::Var() const {
    return VarWithCorrection(0);
}

Tensor Tensor::VarWithCorrection(int64_t correction) const {
    return VarianceGlobal(
        *this, correction, "Tensor::Var", "var");
}

Tensor Tensor::Var(int dim, bool keepdim) const {
    return Var(dim, keepdim, 0);
}

Tensor Tensor::Var(int dim, bool keepdim, int64_t correction) const {
    return VarianceDim(
        *this, dim, keepdim, correction, "Tensor::Var(dim)", "var");
}

Tensor Tensor::Std() const {
    return StdWithCorrection(0);
}

Tensor Tensor::StdWithCorrection(int64_t correction) const {
    const Tensor variance = VarianceGlobal(
        *this, correction, "Tensor::Std", "std");
    return variance.Sqrt().Reshape(variance.Shape());
}

Tensor Tensor::Std(int dim, bool keepdim) const {
    return Std(dim, keepdim, 0);
}

Tensor Tensor::Std(int dim, bool keepdim, int64_t correction) const {
    const Tensor variance = VarianceDim(
        *this, dim, keepdim, correction,
        "Tensor::Std(dim)", "std");
    return variance.Sqrt().Reshape(variance.Shape());
}

} // namespace cyxwiz
