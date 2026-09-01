#include "arrow_converters.h"
#include "algorithms/arrayfire_host_materialization.h"
#include <arrow/compute/api.h>
#include <arrow/type.h>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <cmath>

// ArrayFire header (from backend)
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace arrow_converters {

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Check if Arrow type is numeric
 */
bool IsNumericType(std::shared_ptr<arrow::DataType> type) {
    switch (type->id()) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::BOOL:
            return true;
        default:
            return false;
    }
}

/**
 * Convert Arrow column to float vector
 */
std::vector<float> ColumnToFloatVector(
    std::shared_ptr<arrow::ChunkedArray> column,
    const ConversionOptions& options) {

    std::vector<float> result;
    int64_t total_length = column->length();
    result.reserve(total_length);

    // Process each chunk
    for (const auto& chunk : column->chunks()) {
        auto array = chunk;

        // Handle different types
        switch (array->type_id()) {
            case arrow::Type::FLOAT: {
                auto data = std::static_pointer_cast<arrow::FloatArray>(array);
                for (int64_t i = 0; i < data->length(); ++i) {
                    if (data->IsNull(i)) {
                        result.push_back(options.null_value);
                    } else {
                        float val = data->Value(i);
                        if (options.normalize && val > 1.0f) {
                            val /= 255.0f;  // Simple normalization for images
                        }
                        result.push_back(val);
                    }
                }
                break;
            }
            case arrow::Type::DOUBLE: {
                auto data = std::static_pointer_cast<arrow::DoubleArray>(array);
                for (int64_t i = 0; i < data->length(); ++i) {
                    if (data->IsNull(i)) {
                        result.push_back(options.null_value);
                    } else {
                        result.push_back(static_cast<float>(data->Value(i)));
                    }
                }
                break;
            }
            case arrow::Type::INT8:
            case arrow::Type::INT16:
            case arrow::Type::INT32:
            case arrow::Type::INT64: {
                auto data = std::static_pointer_cast<arrow::Int64Array>(array);
                for (int64_t i = 0; i < data->length(); ++i) {
                    if (data->IsNull(i)) {
                        result.push_back(options.null_value);
                    } else {
                        result.push_back(static_cast<float>(data->Value(i)));
                    }
                }
                break;
            }
            case arrow::Type::UINT8:
            case arrow::Type::UINT16:
            case arrow::Type::UINT32:
            case arrow::Type::UINT64: {
                auto data = std::static_pointer_cast<arrow::UInt64Array>(array);
                for (int64_t i = 0; i < data->length(); ++i) {
                    if (data->IsNull(i)) {
                        result.push_back(options.null_value);
                    } else {
                        float val = static_cast<float>(data->Value(i));
                        if (options.normalize && val > 1.0f) {
                            val /= 255.0f;
                        }
                        result.push_back(val);
                    }
                }
                break;
            }
            case arrow::Type::BOOL: {
                auto data = std::static_pointer_cast<arrow::BooleanArray>(array);
                for (int64_t i = 0; i < data->length(); ++i) {
                    if (data->IsNull(i)) {
                        result.push_back(options.null_value);
                    } else {
                        result.push_back(data->Value(i) ? 1.0f : 0.0f);
                    }
                }
                break;
            }
            default:
                spdlog::warn("Unsupported Arrow type for tensor conversion: {}",
                             array->type()->ToString());
                // Fill with null value
                for (int64_t i = 0; i < array->length(); ++i) {
                    result.push_back(options.null_value);
                }
        }
    }

    return result;
}

// =============================================================================
// Main Conversion Functions
// =============================================================================

std::shared_ptr<af::array> ArrowToTensor(
    std::shared_ptr<arrow::Table> table,
    const ConversionOptions& options) {

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::error("ArrowToTensor: ArrayFire not available in this build");
    return nullptr;
#else

    if (!table) {
        spdlog::error("ArrowToTensor: null table");
        return nullptr;
    }

    // Determine columns to convert
    std::vector<std::string> columns_to_use;
    if (!options.selected_columns.empty()) {
        columns_to_use = options.selected_columns;
    } else {
        // Use all numeric columns
        columns_to_use = GetNumericColumns(table);
    }

    if (columns_to_use.empty()) {
        spdlog::error("ArrowToTensor: no numeric columns found");
        return nullptr;
    }

    int64_t num_rows = table->num_rows();
    int64_t num_cols = columns_to_use.size();

    spdlog::debug("Converting Arrow table to tensor: {} rows, {} columns", num_rows, num_cols);

    // Collect data column by column (column-major for ArrayFire)
    std::vector<float> data;
    data.reserve(num_rows * num_cols);

    for (const auto& col_name : columns_to_use) {
        auto column = table->GetColumnByName(col_name);
        if (!column) {
            spdlog::error("Column '{}' not found in table", col_name);
            return nullptr;
        }

        // Convert column to float vector
        auto col_data = ColumnToFloatVector(column, options);
        if (col_data.size() != static_cast<size_t>(num_rows)) {
            spdlog::error("Column '{}' has mismatched length: {} vs expected {}",
                          col_name, col_data.size(), num_rows);
            return nullptr;
        }

        // Append to data (column-major layout)
        data.insert(data.end(), col_data.begin(), col_data.end());
    }

    // Create ArrayFire array
    try {
        // ArrayFire uses column-major by default
        // Shape: [num_rows, num_cols]
        auto tensor = std::make_shared<::af::array>(
            ::af::dim4(num_rows, num_cols),
            data.data()
        );

        spdlog::info("Converted Arrow table to tensor: [{}, {}]", num_rows, num_cols);
        return tensor;

    } catch (const ::af::exception& e) {
        spdlog::error("ArrayFire error in ArrowToTensor: {}", e.what());
        return nullptr;
    }
#endif
}

std::shared_ptr<af::array> ArrowColumnToTensor(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name,
    const ConversionOptions& options) {

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::error("ArrowColumnToTensor: ArrayFire not available");
    return nullptr;
#else

    if (!table) {
        spdlog::error("ArrowColumnToTensor: null table");
        return nullptr;
    }

    auto column = table->GetColumnByName(column_name);
    if (!column) {
        spdlog::error("Column '{}' not found", column_name);
        return nullptr;
    }

    // Convert to float vector
    auto data = ColumnToFloatVector(column, options);

    try {
        // Create 1D tensor
        auto tensor = std::make_shared<::af::array>(
            static_cast<::af::dim4>(data.size()),
            data.data()
        );

        spdlog::debug("Converted Arrow column '{}' to tensor: [{}]",
                      column_name, data.size());
        return tensor;

    } catch (const ::af::exception& e) {
        spdlog::error("ArrayFire error in ArrowColumnToTensor: {}", e.what());
        return nullptr;
    }
#endif
}

std::shared_ptr<arrow::Table> TensorToArrow(
    const ::af::array& tensor,
    const std::vector<std::string>& column_names,
    const std::string& table_name) {

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::error("TensorToArrow: ArrayFire not available");
    return nullptr;
#else

    try {
        // Get tensor dimensions
        ::af::dim4 dims = tensor.dims();
        int64_t num_rows = dims[0];
        int64_t num_cols = (dims.ndims() > 1) ? dims[1] : 1;

        if (static_cast<size_t>(num_cols) != column_names.size()) {
            spdlog::error("Column name count mismatch: {} vs tensor cols {}",
                          column_names.size(), num_cols);
            return nullptr;
        }

        // Copy data from device to host
        std::vector<float> host_data(num_rows * num_cols);
        MaterializeArrayFireToHost(
            tensor,
            host_data.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "ArrowConverters::ArrayFireToRecordBatch",
            "arrayfire_column_major");

        // Build Arrow arrays for each column
        arrow::FieldVector fields;
        arrow::ArrayVector arrays;

        for (int64_t col = 0; col < num_cols; ++col) {
            // Extract column data (column-major layout)
            std::vector<float> col_data(num_rows);
            for (int64_t row = 0; row < num_rows; ++row) {
                col_data[row] = host_data[row + col * num_rows];
            }

            // Create Arrow array
            arrow::FloatBuilder builder;
            auto status = builder.AppendValues(col_data);
            if (!status.ok()) {
                spdlog::error("Arrow builder error: {}", status.ToString());
                return nullptr;
            }

            std::shared_ptr<arrow::Array> array;
            status = builder.Finish(&array);
            if (!status.ok()) {
                spdlog::error("Arrow finish error: {}", status.ToString());
                return nullptr;
            }

            fields.push_back(arrow::field(column_names[col], arrow::float32()));
            arrays.push_back(array);
        }

        // Create schema and table
        auto schema = arrow::schema(fields);
        auto table = arrow::Table::Make(schema, arrays);

        spdlog::info("Converted tensor to Arrow table '{}': {} rows, {} columns",
                     table_name, num_rows, num_cols);
        return table;

    } catch (const ::af::exception& e) {
        spdlog::error("ArrayFire error in TensorToArrow: {}", e.what());
        return nullptr;
    }
#endif
}

std::shared_ptr<af::array> ArrowBatchesToTensor(
    const std::vector<std::shared_ptr<arrow::Table>>& batches,
    const ConversionOptions& options) {

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::error("ArrowBatchesToTensor: ArrayFire not available");
    return nullptr;
#else

    if (batches.empty()) {
        spdlog::error("ArrowBatchesToTensor: empty batch list");
        return nullptr;
    }

    // Concatenate all tables vertically
    auto concat_result = arrow::ConcatenateTables(batches);
    if (!concat_result.ok()) {
        spdlog::error("Failed to concatenate Arrow batches: {}",
                      concat_result.status().ToString());
        return nullptr;
    }

    auto combined_table = concat_result.ValueOrDie();
    return ArrowToTensor(combined_table, options);
#endif
}

// =============================================================================
// Statistics and Utilities
// =============================================================================

std::vector<ColumnStats> ComputeArrowStats(std::shared_ptr<arrow::Table> table) {
    std::vector<ColumnStats> stats;

    if (!table) {
        return stats;
    }

    for (int i = 0; i < table->num_columns(); ++i) {
        auto column = table->column(i);
        auto field = table->schema()->field(i);

        if (!IsNumericType(field->type())) {
            continue;  // Skip non-numeric columns
        }

        ColumnStats col_stat;
        col_stat.name = field->name();
        col_stat.count = column->length();
        col_stat.null_count = column->null_count();

        // Compute min/max/sum using Arrow compute functions
        auto min_result = arrow::compute::MinMax(column);
        auto sum_result = arrow::compute::Sum(column);

        if (min_result.ok()) {
            auto minmax = min_result.ValueOrDie().scalar_as<arrow::StructScalar>();
            col_stat.min = std::static_pointer_cast<arrow::DoubleScalar>(
                minmax.value[0])->value;
            col_stat.max = std::static_pointer_cast<arrow::DoubleScalar>(
                minmax.value[1])->value;
        }

        if (sum_result.ok()) {
            auto sum_scalar = sum_result.ValueOrDie().scalar_as<arrow::DoubleScalar>();
            col_stat.mean = sum_scalar.value / (col_stat.count - col_stat.null_count);
        }

        // Compute std deviation (simplified - use two-pass algorithm)
        double variance = 0.0;
        int64_t valid_count = col_stat.count - col_stat.null_count;

        if (valid_count > 0) {
            auto data = ColumnToFloatVector(column, ConversionOptions{});
            for (float val : data) {
                double diff = val - col_stat.mean;
                variance += diff * diff;
            }
            variance /= valid_count;
            col_stat.std_dev = std::sqrt(variance);
        }

        stats.push_back(col_stat);
    }

    return stats;
}

bool IsConvertibleToTensor(std::shared_ptr<arrow::Table> table) {
    if (!table) {
        return false;
    }

    // Check if at least one numeric column exists
    for (int i = 0; i < table->num_columns(); ++i) {
        if (IsNumericType(table->schema()->field(i)->type())) {
            return true;
        }
    }

    return false;
}

std::vector<std::string> GetNumericColumns(std::shared_ptr<arrow::Table> table) {
    std::vector<std::string> numeric_cols;

    if (!table) {
        return numeric_cols;
    }

    for (int i = 0; i < table->num_columns(); ++i) {
        auto field = table->schema()->field(i);
        if (IsNumericType(field->type())) {
            numeric_cols.push_back(field->name());
        }
    }

    return numeric_cols;
}

// =============================================================================
// Categorical Encoding
// =============================================================================

std::shared_ptr<arrow::Table> EncodeCategorical(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name,
    std::vector<std::string>& out_vocabulary) {

    if (!table) {
        spdlog::error("EncodeCategorical: null table");
        return nullptr;
    }

    auto column = table->GetColumnByName(column_name);
    if (!column) {
        spdlog::error("Column '{}' not found", column_name);
        return nullptr;
    }

    // Build vocabulary from unique values
    std::unordered_map<std::string, int32_t> value_to_code;
    out_vocabulary.clear();

    for (const auto& chunk : column->chunks()) {
        auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);
        for (int64_t i = 0; i < string_array->length(); ++i) {
            if (!string_array->IsNull(i)) {
                std::string value = string_array->GetString(i);
                if (value_to_code.find(value) == value_to_code.end()) {
                    int32_t code = static_cast<int32_t>(out_vocabulary.size());
                    value_to_code[value] = code;
                    out_vocabulary.push_back(value);
                }
            }
        }
    }

    spdlog::info("Encoded column '{}': {} unique values", column_name, out_vocabulary.size());

    // Create encoded column
    arrow::Int32Builder builder;
    for (const auto& chunk : column->chunks()) {
        auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);
        for (int64_t i = 0; i < string_array->length(); ++i) {
            if (string_array->IsNull(i)) {
                const auto append_status = builder.AppendNull();
                if (!append_status.ok()) {
                    spdlog::error("Failed to append null encoded value: {}",
                                  append_status.ToString());
                    return nullptr;
                }
            } else {
                std::string value = string_array->GetString(i);
                const auto code = value_to_code.find(value);
                if (code == value_to_code.end()) {
                    spdlog::error(
                        "Categorical value '{}' is missing from the vocabulary",
                        value);
                    return nullptr;
                }
                const auto append_status = builder.Append(code->second);
                if (!append_status.ok()) {
                    spdlog::error("Failed to append encoded value: {}",
                                  append_status.ToString());
                    return nullptr;
                }
            }
        }
    }

    std::shared_ptr<arrow::Array> encoded_array;
    auto status = builder.Finish(&encoded_array);
    if (!status.ok()) {
        spdlog::error("Failed to build encoded array: {}", status.ToString());
        return nullptr;
    }

    // Add new column to table
    std::string new_col_name = column_name + "_encoded";
    auto new_field = arrow::field(new_col_name, arrow::int32());
    auto new_schema = table->schema()->AddField(
        table->num_columns(), new_field).ValueOrDie();

    auto chunked_array = std::make_shared<arrow::ChunkedArray>(encoded_array);
    auto new_table = table->AddColumn(table->num_columns(), new_field, chunked_array);

    if (!new_table.ok()) {
        spdlog::error("Failed to add encoded column: {}", new_table.status().ToString());
        return nullptr;
    }

    return new_table.ValueOrDie();
}

std::shared_ptr<arrow::Table> OneHotEncode(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name) {

    // First do categorical encoding
    std::vector<std::string> vocabulary;
    auto encoded_table = EncodeCategorical(table, column_name, vocabulary);
    if (!encoded_table) {
        return nullptr;
    }

    std::string encoded_col = column_name + "_encoded";
    auto encoded_column = encoded_table->GetColumnByName(encoded_col);

    // Create N binary columns (one per category)
    arrow::FieldVector new_fields;
    arrow::ArrayVector new_arrays;

    for (size_t cat = 0; cat < vocabulary.size(); ++cat) {
        arrow::BooleanBuilder builder;

        for (const auto& chunk : encoded_column->chunks()) {
            auto int_array = std::static_pointer_cast<arrow::Int32Array>(chunk);
            for (int64_t i = 0; i < int_array->length(); ++i) {
                if (int_array->IsNull(i)) {
                    const auto append_status = builder.AppendNull();
                    if (!append_status.ok()) {
                        spdlog::error("Failed to append null one-hot value: {}",
                                      append_status.ToString());
                        return nullptr;
                    }
                } else {
                    const auto append_status = builder.Append(
                        int_array->Value(i) == static_cast<int32_t>(cat));
                    if (!append_status.ok()) {
                        spdlog::error("Failed to append one-hot value: {}",
                                      append_status.ToString());
                        return nullptr;
                    }
                }
            }
        }

        std::shared_ptr<arrow::Array> binary_array;
        auto status = builder.Finish(&binary_array);
        if (!status.ok()) {
            spdlog::error("Failed to build one-hot column: {}", status.ToString());
            return nullptr;
        }

        std::string col_name = column_name + "_" + vocabulary[cat];
        new_fields.push_back(arrow::field(col_name, arrow::boolean()));
        new_arrays.push_back(binary_array);
    }

    // Add all one-hot columns to table
    auto result_table = encoded_table;
    for (size_t i = 0; i < new_fields.size(); ++i) {
        auto chunked = std::make_shared<arrow::ChunkedArray>(new_arrays[i]);
        auto add_result = result_table->AddColumn(
            result_table->num_columns(), new_fields[i], chunked);

        if (!add_result.ok()) {
            spdlog::error("Failed to add one-hot column: {}", add_result.status().ToString());
            return nullptr;
        }

        result_table = add_result.ValueOrDie();
    }

    spdlog::info("One-hot encoded column '{}' into {} binary columns",
                 column_name, vocabulary.size());

    return result_table;
}

} // namespace arrow_converters
} // namespace cyxwiz
