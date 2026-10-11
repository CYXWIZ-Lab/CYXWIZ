#pragma once

#include "cyxwiz/tensor.h"

#include <limits>
#include <stdexcept>
#include <vector>

#include <arrayfire.h>

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

// ArrayFire dims of a rank <= 4 shape, reversed (row-major storage order).
inline af::dim4 ReversedArrayFireDims(const std::vector<size_t>& shape) {
    std::vector<size_t> reversed(shape.rbegin(), shape.rend());
    return ArrayFireDims(reversed);
}

// Row-major flat storage <-> the semantic view of a rank <= 4 shape.
inline af::array RowMajorFlatToSemantic(const af::array& flat, const std::vector<size_t>& shape) {
    const af::array storage = af::moddims(flat, ReversedArrayFireDims(shape));
    if (shape.size() == 2) return af::transpose(storage);
    if (shape.size() == 3) return af::reorder(storage, 2, 1, 0);
    if (shape.size() == 4) return af::reorder(storage, 3, 2, 1, 0);
    return storage;
}

inline af::array SemanticToRowMajorFlat(const af::array& semantic, size_t rank) {
    if (rank == 2) return af::flat(af::transpose(semantic));
    if (rank == 3) return af::flat(af::reorder(semantic, 2, 1, 0));
    if (rank == 4) return af::flat(af::reorder(semantic, 3, 2, 1, 0));
    return af::flat(semantic);
}

// Elementwise view of a tensor: the semantic array up to rank 4, the
// row-major flat array above (TOFIX140 F4). Results go back through
// FromElementwiseArray with the same shape.
inline af::array ElementwiseArray(const Tensor& input) {
    return input.Shape().size() <= 4 ? input.GetSemanticArray() : input.GetRowMajorFlatArray();
}

inline Tensor FromElementwiseArray(const af::array& values, const std::vector<size_t>& shape) {
    return shape.size() <= 4 ? Tensor::FromSemanticArray(values, shape)
                             : Tensor::FromRowMajorFlatArray(values, shape);
}

// Merge neighbouring axes that broadcast the same way (both kept, or both
// size 1 in the input); output-size-1 axes join their neighbour. Row-major
// order makes each merge exact. Returns false when more than four groups stay.
inline bool CollapseBroadcastAxes(const std::vector<size_t>& padded_input,
                                  const std::vector<size_t>& output_shape,
                                  std::vector<size_t>& input_groups,
                                  std::vector<size_t>& output_groups) {
    input_groups.clear();
    output_groups.clear();
    int group_kind = -1;  // 0 = kept, 1 = broadcast
    for (size_t axis = 0; axis < output_shape.size(); ++axis) {
        const size_t out = output_shape[axis];
        const size_t in = padded_input[axis];
        if (out == 1 && !output_groups.empty()) continue;
        const int kind = (in == out) ? 0 : 1;
        if (kind == group_kind) {
            input_groups.back() *= in;
            output_groups.back() *= out;
        } else {
            input_groups.push_back(in);
            output_groups.push_back(out);
            group_kind = kind;
        }
    }
    if (output_groups.empty()) {
        input_groups.push_back(1);
        output_groups.push_back(1);
    }
    return output_groups.size() <= 4;
}

// General broadcast of row-major flat storage: each output position's source
// index is computed on the device (div / mod per axis) and gathered. Used
// when collapsed axes still exceed four.
inline af::array BroadcastGatherFlat(const af::array& input_flat,
                                     const std::vector<size_t>& padded_input,
                                     const std::vector<size_t>& output_shape) {
    if (static_cast<unsigned long long>(input_flat.elements()) > (std::numeric_limits<unsigned>::max)()) {
        throw std::runtime_error("Tensor broadcast: input exceeds the ArrayFire index range");
    }
    dim_t total = 1;
    for (size_t dim : output_shape) total *= static_cast<dim_t>(dim);
    af::array remaining = af::range(af::dim4(total), 0, s64);
    af::array source = af::constant(0, af::dim4(total), s64);
    long long input_stride = 1;
    for (size_t axis = output_shape.size(); axis-- > 0;) {
        // Extents and strides stay far below 2^53, so double scalars are exact.
        const double extent = static_cast<double>(output_shape[axis]);
        const af::array coordinate = af::mod(remaining, extent);
        remaining = (remaining - coordinate) / extent;
        if (padded_input[axis] != 1) source = source + coordinate * static_cast<double>(input_stride);
        input_stride *= static_cast<long long>(padded_input[axis]);
    }
    return input_flat(source.as(u32));
}

// The input broadcast to output_shape, in FromElementwiseArray's layout for
// that shape (semantic up to rank 4, row-major flat above).
inline af::array BroadcastArray(const Tensor& input,
                                const std::vector<size_t>& output_shape,
                                DataType output_dtype) {
    std::vector<size_t> padded(output_shape.size(), 1);
    const size_t offset = output_shape.size() - input.Shape().size();
    for (size_t index = 0; index < input.Shape().size(); ++index) {
        padded[offset + index] = input.Shape()[index];
    }
    if (output_shape.size() > 4) {
        std::vector<size_t> input_groups, output_groups;
        const af::array flat = input.GetRowMajorFlatArray().as(ArrayFireType(output_dtype));
        if (!CollapseBroadcastAxes(padded, output_shape, input_groups, output_groups)) {
            return BroadcastGatherFlat(flat, padded, output_shape);
        }
        const af::array values = RowMajorFlatToSemantic(flat, input_groups);
        af::dim4 factors(1, 1, 1, 1);
        for (size_t axis = 0; axis < output_groups.size(); ++axis) {
            factors[static_cast<unsigned>(axis)] = static_cast<dim_t>(
                input_groups[axis] == output_groups[axis] ? 1 : output_groups[axis]);
        }
        return SemanticToRowMajorFlat(af::tile(values, factors), output_groups.size());
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

} // namespace cyxwiz::tensor_math_utils
