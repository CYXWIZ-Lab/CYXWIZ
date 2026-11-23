#pragma once

// One typed reader of numeric table columns for every tabular batcher, in
// memory and from Parquet: any of the types IsNumericBatchFeatureType takes,
// with any chunking (TOFIX222 TAB-01 / TAB-18: the in-memory multi-chunk
// reader once lacked five of the types its filter accepted, and the Parquet
// reader read only single-chunk columns and had no uint64 case; both left the
// values 0).

#include "label_column_resolver.h"

#include <arrow/api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

// Reads `count` rows of one column into out[b * stride] as float, for any
// chunking: the chunk of a row is found by binary search over chunk starts.
// A row outside [0, num_rows) is left as it is.
template <typename ArrowArrayT>
void GatherTypedColumn(const arrow::ChunkedArray& column, const int64_t* rows, size_t count, int64_t num_rows,
                       float* out, size_t stride) {
    using Value = typename ArrowArrayT::TypeClass::c_type;
    const int num_chunks = column.num_chunks();
    if (num_chunks == 1) {
        // One chunk (every in-memory CSV load, most row groups): no lookup.
        const Value* values = static_cast<const ArrowArrayT&>(*column.chunk(0)).raw_values();
        for (size_t b = 0; b < count; ++b) {
            const int64_t row = rows[b];
            if (row >= 0 && row < num_rows) out[b * stride] = static_cast<float>(values[row]);
        }
        return;
    }
    std::vector<const Value*> values(static_cast<size_t>(num_chunks));
    std::vector<int64_t> starts(static_cast<size_t>(num_chunks));
    int64_t start = 0;
    for (int c = 0; c < num_chunks; ++c) {
        const auto& chunk = column.chunk(c);
        values[static_cast<size_t>(c)] = static_cast<const ArrowArrayT&>(*chunk).raw_values();
        starts[static_cast<size_t>(c)] = start;
        start += chunk->length();
    }
    for (size_t b = 0; b < count; ++b) {
        const int64_t row = rows[b];
        if (row < 0 || row >= num_rows) continue;
        // The last chunk starting at or before `row` (empty chunks share
        // their start with the next one, so they are never picked).
        const size_t c =
            static_cast<size_t>(std::upper_bound(starts.begin(), starts.end(), row) - starts.begin()) - 1;
        out[b * stride] = static_cast<float>(values[c][row - starts[c]]);
    }
}

// A bool column (bit-packed, so no raw values) as 0 / 1: label columns only;
// IsNumericBatchFeatureType does not take bool features.
inline void GatherBoolColumn(const arrow::ChunkedArray& column, const int64_t* rows, size_t count, int64_t num_rows,
                             float* out, size_t stride) {
    std::vector<int64_t> starts;
    int64_t start = 0;
    for (const auto& chunk : column.chunks()) {
        starts.push_back(start);
        start += chunk->length();
    }
    for (size_t b = 0; b < count; ++b) {
        const int64_t row = rows[b];
        if (row < 0 || row >= num_rows || starts.empty()) continue;
        const size_t c =
            static_cast<size_t>(std::upper_bound(starts.begin(), starts.end(), row) - starts.begin()) - 1;
        const auto& values = static_cast<const arrow::BooleanArray&>(*column.chunk(static_cast<int>(c)));
        out[b * stride] = values.Value(row - starts[c]) ? 1.0f : 0.0f;
    }
}

// False for a type IsNumericBatchFeatureType does not take, other than bool
// (nothing read): callers check the types first, on their own thread
// (RequireNumberColumns).
inline bool GatherNumberColumn(const arrow::ChunkedArray& column, const int64_t* rows, size_t count,
                               int64_t num_rows, float* out, size_t stride) {
    switch (column.type()->id()) {
        case arrow::Type::DOUBLE: GatherTypedColumn<arrow::DoubleArray>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::FLOAT: GatherTypedColumn<arrow::FloatArray>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::INT64: GatherTypedColumn<arrow::Int64Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::INT32: GatherTypedColumn<arrow::Int32Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::INT16: GatherTypedColumn<arrow::Int16Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::INT8: GatherTypedColumn<arrow::Int8Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::UINT64: GatherTypedColumn<arrow::UInt64Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::UINT32: GatherTypedColumn<arrow::UInt32Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::UINT16: GatherTypedColumn<arrow::UInt16Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::UINT8: GatherTypedColumn<arrow::UInt8Array>(column, rows, count, num_rows, out, stride); return true;
        case arrow::Type::BOOL: GatherBoolColumn(column, rows, count, num_rows, out, stride); return true;
        default: return false;
    }
}

// Throws, naming the column, when one of `columns` in `table` is not a number
// type the gather reads. Call it before starting worker threads: a throw
// inside a worker ends the process.
inline void RequireNumberColumns(const char* batcher, const arrow::Table& table, const std::vector<int>& columns,
                                 const char* role, bool allow_bool = false) {
    for (const int col_idx : columns) {
        if (col_idx < 0 || col_idx >= table.num_columns()) continue;
        const auto& field = table.schema()->field(col_idx);
        if (!IsNumericBatchFeatureType(field->type()) && !(allow_bool && field->type()->id() == arrow::Type::BOOL)) {
            throw std::runtime_error(std::string(batcher) + ": " + role + " column '" + field->name() +
                                     "' has type " + field->type()->ToString() +
                                     ", which training cannot read as a number");
        }
    }
}

}  // namespace cyxwiz
