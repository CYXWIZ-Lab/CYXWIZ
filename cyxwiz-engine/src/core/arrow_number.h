#pragma once

// Reading Arrow number columns as doubles, shared by the plot sources, Data
// Studio > Results and a test's per-row table read back (TOFIX157).

#include <arrow/api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace cyxwiz {

// An integer or floating-point type (not bool).
inline bool IsArrowNumber(arrow::Type::type id) {
    switch (id) {
        case arrow::Type::INT8: case arrow::Type::INT16: case arrow::Type::INT32: case arrow::Type::INT64:
        case arrow::Type::UINT8: case arrow::Type::UINT16: case arrow::Type::UINT32: case arrow::Type::UINT64:
        case arrow::Type::HALF_FLOAT: case arrow::Type::FLOAT: case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

// One value of a chunk as a double: a bool as 0 / 1, NaN when missing or not
// a number.
inline double ArrowNumberAt(const arrow::Array& a, int64_t i) {
    if (a.IsNull(i)) return std::nan("");
    switch (a.type_id()) {
        case arrow::Type::INT8: return static_cast<const arrow::Int8Array&>(a).Value(i);
        case arrow::Type::INT16: return static_cast<const arrow::Int16Array&>(a).Value(i);
        case arrow::Type::INT32: return static_cast<const arrow::Int32Array&>(a).Value(i);
        case arrow::Type::INT64: return static_cast<double>(static_cast<const arrow::Int64Array&>(a).Value(i));
        case arrow::Type::UINT8: return static_cast<const arrow::UInt8Array&>(a).Value(i);
        case arrow::Type::UINT16: return static_cast<const arrow::UInt16Array&>(a).Value(i);
        case arrow::Type::UINT32: return static_cast<const arrow::UInt32Array&>(a).Value(i);
        case arrow::Type::UINT64: return static_cast<double>(static_cast<const arrow::UInt64Array&>(a).Value(i));
        case arrow::Type::FLOAT: return static_cast<const arrow::FloatArray&>(a).Value(i);
        case arrow::Type::DOUBLE: return static_cast<const arrow::DoubleArray&>(a).Value(i);
        case arrow::Type::BOOL: return static_cast<const arrow::BooleanArray&>(a).Value(i) ? 1.0 : 0.0;
        default: {
            const auto scalar = a.GetScalar(i);
            if (!scalar.ok() || !(*scalar)->is_valid) return std::nan("");
            const auto cast = (*scalar)->CastTo(arrow::float64());
            return cast.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*cast)->value : std::nan("");
        }
    }
}

// A whole number column as doubles (NaN when missing); empty when the column
// is absent or not a number.
inline std::vector<double> ArrowNumbers(const std::shared_ptr<arrow::ChunkedArray>& column) {
    std::vector<double> out;
    if (!column || !IsArrowNumber(column->type()->id())) return out;
    out.reserve(static_cast<size_t>(column->length()));
    for (const auto& chunk : column->chunks())
        for (int64_t i = 0; i < chunk->length(); ++i) out.push_back(ArrowNumberAt(*chunk, i));
    return out;
}

}  // namespace cyxwiz
