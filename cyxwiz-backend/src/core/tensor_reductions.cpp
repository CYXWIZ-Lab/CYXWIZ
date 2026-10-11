#include "cyxwiz/tensor.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"

namespace cyxwiz {

namespace {

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

using AxisReduceFn = std::function<af::array(const af::array&, int)>;

// The result of an empty reduction (0 for sum, 1 for prod, NaN for mean / var).
Tensor ConstantReduction(const std::vector<size_t>& output_shape, DataType dtype, double value) {
    const size_t count = tensor_utils::CheckedProduct(
        output_shape, 0, output_shape.size(), "Tensor reduction: output shape overflow");
    if (count == 0) return Tensor(output_shape, dtype);
    af::array output = af::constant(value, af::dim4(static_cast<dim_t>(count)),
                                    tensor_math_utils::ArrayFireType(dtype));
    output.eval();
    return Tensor::FromRowMajorFlatArray(output, output_shape);
}

// All elements as one column (order is irrelevant to every global reduction).
af::array AllValues(const Tensor& input) {
    return af::flat(tensor_math_utils::ElementwiseArray(input));
}

af::array AsReal(const af::array& values, DataType dtype) {
    return values.as(dtype == DataType::Float64 ? f64 : f32);
}

Tensor FinishGlobal(const af::array& reduced, DataType dtype) {
    af::array output = reduced.as(tensor_math_utils::ArrayFireType(dtype));
    output.eval();
    return Tensor::FromSemanticArray(output, {});
}

// Reduce one axis. Up to rank 4 on the semantic view; above, on the row-major
// flat storage viewed as [after, axis, before], reducing ArrayFire dim 1: the
// result is then already in the output's row-major order.
Tensor ReduceAxis(const Tensor& input, int axis, bool keepdim, DataType output_dtype,
                  const AxisReduceFn& reduce, bool real_input) {
    const std::vector<size_t>& shape = input.Shape();
    const std::vector<size_t> output_shape = ReducedShape(shape, axis, keepdim);
    const af::dtype type = tensor_math_utils::ArrayFireType(output_dtype);
    if (shape.size() <= 4) {
        af::array values = input.GetSemanticArray();
        if (real_input) values = AsReal(values, input.GetDataType());
        af::array output = af::moddims(reduce(values, axis).as(type),
                                       tensor_math_utils::ArrayFireDims(output_shape));
        output.eval();
        return Tensor::FromSemanticArray(output, output_shape);
    }
    const auto product = [&shape](size_t begin, size_t end) {
        dim_t value = 1;
        for (size_t i = begin; i < end; ++i) value *= static_cast<dim_t>(shape[i]);
        return value;
    };
    const size_t a = static_cast<size_t>(axis);
    af::array values = af::moddims(input.GetRowMajorFlatArray(), product(a + 1, shape.size()),
                                   static_cast<dim_t>(shape[a]), product(0, a));
    if (real_input) values = AsReal(values, input.GetDataType());
    af::array output = af::flat(reduce(values, 1)).as(type);
    output.eval();
    return Tensor::FromRowMajorFlatArray(output, output_shape);
}

af::array Variance(const af::array& values, int axis, size_t count, int64_t correction) {
    const af::array mean = af::mean(values, axis);
    af::dim4 tiles(1, 1, 1, 1);
    tiles[static_cast<unsigned>(axis)] = values.dims(axis);
    const af::array centered = values - af::tile(mean, tiles);
    return af::sum(centered * centered, axis) / VarianceDenominator(count, correction);
}

Tensor VarianceGlobal(const Tensor& input, int64_t correction, const char* operation_name) {
    const DataType output_dtype = RealReductionType(input.GetDataType());
    try {
        const size_t count = input.NumElements();
        if (count == 0) {
            return ConstantReduction({}, output_dtype, std::numeric_limits<double>::quiet_NaN());
        }
        return FinishGlobal(Variance(AsReal(AllValues(input), input.GetDataType()), 0, count, correction),
                            output_dtype);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(operation_name, e));
    }
}

Tensor VarianceDim(const Tensor& input, int dim, bool keepdim, int64_t correction,
                   const char* operation_name) {
    const int axis = NormalizeReductionDim(dim, input.NumDimensions());
    if (input.Shape().empty()) {
        return VarianceGlobal(input, correction, operation_name);
    }
    const DataType output_dtype = RealReductionType(input.GetDataType());
    const size_t count = input.Shape()[static_cast<size_t>(axis)];
    try {
        if (count == 0) {
            return ConstantReduction(ReducedShape(input.Shape(), axis, keepdim), output_dtype,
                                     std::numeric_limits<double>::quiet_NaN());
        }
        return ReduceAxis(input, axis, keepdim, output_dtype,
                          [count, correction](const af::array& values, int reduction_axis) {
                              return Variance(values, reduction_axis, count, correction);
                          }, true);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(operation_name, e));
    }
}

} // namespace

Tensor Tensor::Sum() const {
    try {
        if (NumElements() == 0) return ConstantReduction({}, dtype_, 0.0);
        return FinishGlobal(af::sum(AllValues(*this)), dtype_);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Sum", e));
    }
}

Tensor Tensor::Sum(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Sum();
    }
    try {
        if (shape_[static_cast<size_t>(axis)] == 0) {
            return ConstantReduction(ReducedShape(shape_, axis, keepdim), dtype_, 0.0);
        }
        return ReduceAxis(*this, axis, keepdim, dtype_,
                          [](const af::array& values, int reduction_axis) { return af::sum(values, reduction_axis); },
                          false);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Sum(dim)", e));
    }
}

Tensor Tensor::Mean() const {
    const DataType output_dtype = RealReductionType(dtype_);
    try {
        if (NumElements() == 0) {
            return ConstantReduction({}, output_dtype, std::numeric_limits<double>::quiet_NaN());
        }
        return FinishGlobal(af::mean(AsReal(AllValues(*this), dtype_)), output_dtype);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Mean", e));
    }
}

Tensor Tensor::Mean(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Mean();
    }
    const DataType output_dtype = RealReductionType(dtype_);
    try {
        if (shape_[static_cast<size_t>(axis)] == 0) {
            return ConstantReduction(ReducedShape(shape_, axis, keepdim), output_dtype,
                                     std::numeric_limits<double>::quiet_NaN());
        }
        return ReduceAxis(*this, axis, keepdim, output_dtype,
                          [](const af::array& values, int reduction_axis) { return af::mean(values, reduction_axis); },
                          true);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Mean(dim)", e));
    }
}

Tensor Tensor::Max() const {
    if (NumElements() == 0) {
        throw std::runtime_error("Tensor::Max: cannot reduce an empty tensor");
    }
    try {
        return FinishGlobal((af::max)(AllValues(*this)), dtype_);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Max", e));
    }
}

Tensor Tensor::Max(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Max();
    }
    if (shape_[static_cast<size_t>(axis)] == 0) {
        throw std::runtime_error("Tensor::Max: cannot reduce an empty dimension");
    }
    try {
        return ReduceAxis(*this, axis, keepdim, dtype_,
                          [](const af::array& values, int reduction_axis) { return (af::max)(values, reduction_axis); },
                          false);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Max(dim)", e));
    }
}

Tensor Tensor::Min() const {
    if (NumElements() == 0) {
        throw std::runtime_error("Tensor::Min: cannot reduce an empty tensor");
    }
    try {
        return FinishGlobal((af::min)(AllValues(*this)), dtype_);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Min", e));
    }
}

Tensor Tensor::Min(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Min();
    }
    if (shape_[static_cast<size_t>(axis)] == 0) {
        throw std::runtime_error("Tensor::Min: cannot reduce an empty dimension");
    }
    try {
        return ReduceAxis(*this, axis, keepdim, dtype_,
                          [](const af::array& values, int reduction_axis) { return (af::min)(values, reduction_axis); },
                          false);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Min(dim)", e));
    }
}

Tensor Tensor::Prod() const {
    try {
        if (NumElements() == 0) return ConstantReduction({}, dtype_, 1.0);
        return FinishGlobal(af::product(AllValues(*this)), dtype_);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Prod", e));
    }
}

Tensor Tensor::Prod(int dim, bool keepdim) const {
    const int axis = NormalizeReductionDim(dim, NumDimensions());
    if (shape_.empty()) {
        return Prod();
    }
    try {
        if (shape_[static_cast<size_t>(axis)] == 0) {
            return ConstantReduction(ReducedShape(shape_, axis, keepdim), dtype_, 1.0);
        }
        return ReduceAxis(*this, axis, keepdim, dtype_,
                          [](const af::array& values, int reduction_axis) { return af::product(values, reduction_axis); },
                          false);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Prod(dim)", e));
    }
}

Tensor Tensor::Var() const {
    return VarWithCorrection(0);
}

Tensor Tensor::VarWithCorrection(int64_t correction) const {
    return VarianceGlobal(*this, correction, "Tensor::Var");
}

Tensor Tensor::Var(int dim, bool keepdim) const {
    return Var(dim, keepdim, 0);
}

Tensor Tensor::Var(int dim, bool keepdim, int64_t correction) const {
    return VarianceDim(*this, dim, keepdim, correction, "Tensor::Var(dim)");
}

Tensor Tensor::Std() const {
    return StdWithCorrection(0);
}

Tensor Tensor::StdWithCorrection(int64_t correction) const {
    const Tensor variance = VarianceGlobal(*this, correction, "Tensor::Std");
    return variance.Sqrt().Reshape(variance.Shape());
}

Tensor Tensor::Std(int dim, bool keepdim) const {
    return Std(dim, keepdim, 0);
}

Tensor Tensor::Std(int dim, bool keepdim, int64_t correction) const {
    const Tensor variance = VarianceDim(*this, dim, keepdim, correction, "Tensor::Std(dim)");
    return variance.Sqrt().Reshape(variance.Shape());
}

} // namespace cyxwiz
