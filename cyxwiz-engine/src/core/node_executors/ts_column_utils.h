#pragma once

#include <arrow/api.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

inline std::string NormalizeTimeSeriesParameterChoice(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};

    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return normalized;
}

/// Read an entire ChunkedArray column into a std::vector<float>, casting
/// supported numeric types to float32. Returns false on unsupported types
/// (caller surfaces as an Arrow error). Null values become NaN.
/// Shared by LogTransform, Differencing, and TimeSeriesWindow operators.
inline bool ReadColumnAsFloat(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<float>& out,
    std::string& error_type_name,
    const std::function<bool()>& cancellation_requested = {},
    bool* cancelled = nullptr) {

    out.clear();
    if (cancelled) *cancelled = false;
    out.reserve(static_cast<size_t>(column->length()));

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();

        auto is_null = [&](int64_t i) -> bool { return chunk->IsNull(i); };
        auto stop_requested = [&](int64_t i) {
            if ((i & 1023) != 0 || !cancellation_requested ||
                !cancellation_requested()) {
                return false;
            }
            if (cancelled) *cancelled = true;
            return true;
        };
        const float nan_val = std::numeric_limits<float>::quiet_NaN();

        switch (chunk->type_id()) {
            case arrow::Type::DOUBLE: {
                auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                const double* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(data[i]));
                }
                break;
            }
            case arrow::Type::FLOAT: {
                auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                const float* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : data[i]);
                }
                break;
            }
            case arrow::Type::INT64: {
                auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT32: {
                auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT16: {
                auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT8: {
                auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT64: {
                auto arr = std::static_pointer_cast<arrow::UInt64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT32: {
                auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT16: {
                auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT8: {
                auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                const uint8_t* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    if (stop_requested(i)) return false;
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(data[i]));
                }
                break;
            }
            default:
                error_type_name = chunk->type()->ToString();
                return false;
        }
    }

    return true;
}

/// Build a float32 Arrow column from a std::vector<float> and replace
/// a column in the table by index. Returns the new table. The replaced
/// field is retyped to float32 (preserving the column name) — callers
/// that feed this an int64 `Passengers` column end up with a float32
/// `Passengers` column in the output table, which is what downstream
/// windowing + regression expect.
inline arrow::Result<std::shared_ptr<arrow::Table>> ReplaceColumnWithFloat(
    const std::shared_ptr<arrow::Table>& table,
    int col_idx,
    const std::vector<float>& values,
    int64_t num_rows) {

    arrow::FloatBuilder builder(arrow::default_memory_pool());
    ARROW_RETURN_NOT_OK(builder.Reserve(num_rows));
    for (int64_t i = 0; i < num_rows; ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(values[i]));
    }
    std::shared_ptr<arrow::Array> new_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&new_array));

    auto new_chunked = std::make_shared<arrow::ChunkedArray>(new_array);
    // New field with same name but float32 type; otherwise SetColumn
    // throws "Field type did not match data type" when the original
    // column was int / double.
    const std::string col_name = table->schema()->field(col_idx)->name();
    auto new_field = arrow::field(col_name, arrow::float32());
    return table->SetColumn(col_idx, new_field, new_chunked);
}

} // namespace cyxwiz
