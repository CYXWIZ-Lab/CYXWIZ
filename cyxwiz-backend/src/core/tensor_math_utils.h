#pragma once

#include "cyxwiz/tensor.h"

#include <limits>
#include <stdexcept>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::tensor_math_utils {

inline bool IsIntegralType(DataType dtype) {
    return dtype == DataType::Int32 || dtype == DataType::Int64 ||
           dtype == DataType::UInt8;
}

inline DataType PromoteTypes(DataType left, DataType right) {
    if (left == DataType::Float64 || right == DataType::Float64) {
        return DataType::Float64;
    }
    if (left == DataType::Float32 || right == DataType::Float32) {
        return DataType::Float32;
    }
    if (left == DataType::Int64 || right == DataType::Int64) {
        return DataType::Int64;
    }
    if (left == DataType::Int32 || right == DataType::Int32) {
        return DataType::Int32;
    }
    return DataType::UInt8;
}

inline DataType RealType(DataType input) {
    return input == DataType::Float64 ? DataType::Float64 : DataType::Float32;
}

inline size_t BroadcastIndex(
    size_t output_index,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& output_strides,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& input_strides) {
    const size_t offset = output_shape.size() - input_shape.size();
    size_t remaining = output_index;
    size_t input_index = 0;
    for (size_t axis = 0; axis < output_shape.size(); ++axis) {
        const size_t coordinate = remaining / output_strides[axis];
        remaining %= output_strides[axis];
        if (axis >= offset && input_shape[axis - offset] != 1) {
            input_index += coordinate * input_strides[axis - offset];
        }
    }
    return input_index;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
inline af::dtype ArrayFireType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return f32;
        case DataType::Float64: return f64;
        case DataType::Int32: return s32;
        case DataType::Int64: return s64;
        case DataType::UInt8: return u8;
    }
    throw std::runtime_error("Tensor math: unsupported ArrayFire data type");
}

inline af::dim4 ArrayFireDims(const std::vector<size_t>& shape) {
    af::dim4 dimensions(1, 1, 1, 1);
    for (size_t index = 0; index < shape.size(); ++index) {
        if (shape[index] >
            static_cast<size_t>((std::numeric_limits<dim_t>::max)())) {
            throw std::overflow_error(
                "Tensor math: dimension exceeds ArrayFire range");
        }
        dimensions[static_cast<unsigned>(index)] =
            static_cast<dim_t>(shape[index]);
    }
    return dimensions;
}

inline af::array BroadcastArray(const Tensor& input,
                                const std::vector<size_t>& output_shape,
                                DataType output_dtype) {
    std::vector<size_t> padded(output_shape.size(), 1);
    const size_t offset = output_shape.size() - input.Shape().size();
    for (size_t index = 0; index < input.Shape().size(); ++index) {
        padded[offset + index] = input.Shape()[index];
    }
    af::array values =
        input.GetSemanticArray().as(ArrayFireType(output_dtype));
    values = af::moddims(values, ArrayFireDims(padded));
    af::dim4 factors(1, 1, 1, 1);
    for (size_t axis = 0; axis < output_shape.size(); ++axis) {
        factors[static_cast<unsigned>(axis)] = static_cast<dim_t>(
            padded[axis] == output_shape[axis] ? 1 : output_shape[axis]);
    }
    return af::tile(values, factors);
}
#endif

} // namespace cyxwiz::tensor_math_utils
