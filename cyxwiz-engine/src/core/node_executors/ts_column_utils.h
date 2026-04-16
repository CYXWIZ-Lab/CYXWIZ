#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

/// Read an entire ChunkedArray column into a std::vector<float>, casting
/// supported numeric types to float32. Returns false on unsupported types
/// (caller surfaces as an Arrow error). Null values become NaN.
/// Shared by LogTransform, Differencing, and TimeSeriesWindow operators.
inline bool ReadColumnAsFloat(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<float>& out,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();

        auto is_null = [&](int64_t i) -> bool { return chunk->IsNull(i); };
        const float nan_val = std::numeric_limits<float>::quiet_NaN();

        switch (chunk->type_id()) {
            case arrow::Type::DOUBLE: {
                auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                const double* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(data[i]));
                break;
            }
            case arrow::Type::FLOAT: {
                auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                const float* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : data[i]);
                break;
            }
            case arrow::Type::INT64: {
                auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::INT32: {
                auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::INT16: {
                auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::INT8: {
                auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::UINT64: {
                auto arr = std::static_pointer_cast<arrow::UInt64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::UINT32: {
                auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::UINT16: {
                auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(arr->Value(i)));
                break;
            }
            case arrow::Type::UINT8: {
                auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                const uint8_t* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i)
                    out.push_back(is_null(i) ? nan_val : static_cast<float>(data[i]));
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
/// a column in the table by index. Returns the new table.
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
    return table->SetColumn(col_idx, table->schema()->field(col_idx), new_chunked);
}

} // namespace cyxwiz
