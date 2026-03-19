#pragma once

#include <arrow/api.h>
#include <memory>
#include <string>
#include <vector>

// Forward declaration for ArrayFire array (global namespace)
namespace af {
    class array;
}

namespace cyxwiz {

/**
 * Arrow <-> Tensor Conversion Utilities (Phase 0 Day 2)
 *
 * Bridges Apache Arrow columnar data with ArrayFire tensors for ML training.
 * Supports zero-copy views where possible, with automatic fallback to deep copy.
 */
namespace arrow_converters {

/**
 * Conversion options for fine-grained control
 */
struct ConversionOptions {
    // Data type conversion
    bool normalize = false;           // Normalize float data to [0,1]
    bool cast_to_float32 = true;      // Cast all numeric types to float32

    // Memory optimization
    bool zero_copy = true;             // Attempt zero-copy when possible
    bool column_major = true;          // ArrayFire uses column-major by default

    // Missing value handling
    float null_value = 0.0f;          // Replace null/NaN with this value
    bool skip_null_rows = false;       // Skip rows with any null values

    // Column selection
    std::vector<std::string> selected_columns;  // Empty = all columns
    bool include_index = false;        // Include row index as first column
};

/**
 * Arrow Table -> Tensor Conversion
 *
 * Converts Arrow Table to ArrayFire tensor for training.
 *
 * @param table Arrow table (columnar data)
 * @param options Conversion configuration
 * @return ArrayFire array [rows, cols] or nullptr on failure
 *
 * Supported column types:
 * - Numeric: int8/16/32/64, uint8/16/32/64, float, double
 * - Boolean: converted to 0.0/1.0
 * - String: skipped (use categorical encoding first)
 *
 * Example:
 *   auto table = LoadCSV("data.csv");  // 1000 rows, 10 columns
 *   auto tensor = ArrowToTensor(table);  // shape: [1000, 10]
 */
std::shared_ptr<::af::array> ArrowToTensor(
    std::shared_ptr<arrow::Table> table,
    const ConversionOptions& options = ConversionOptions{}
);

/**
 * Arrow Column -> Tensor (single column extraction)
 *
 * @param table Arrow table
 * @param column_name Column to extract
 * @param options Conversion configuration
 * @return ArrayFire array [rows, 1] or nullptr on failure
 */
std::shared_ptr<::af::array> ArrowColumnToTensor(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name,
    const ConversionOptions& options = ConversionOptions{}
);

/**
 * Tensor -> Arrow Table Conversion
 *
 * Converts ArrayFire tensor back to Arrow Table for export/storage.
 *
 * @param tensor ArrayFire array [rows, cols]
 * @param column_names Column names (must match tensor cols)
 * @param table_name Table identifier
 * @return Arrow table or nullptr on failure
 *
 * Example:
 *   ::af::array predictions = model.predict(input);  // [1000, 1]
 *   auto table = TensorToArrow(predictions, {"prediction"}, "results");
 *   ExportParquet(table, "predictions.parquet");
 */
std::shared_ptr<arrow::Table> TensorToArrow(
    const ::af::array& tensor,
    const std::vector<std::string>& column_names,
    const std::string& table_name = "tensor_data"
);

/**
 * Batch Arrow Tables -> Tensor (for mini-batch training)
 *
 * Efficiently converts multiple Arrow record batches to a single tensor.
 * Used in streaming/chunked data pipelines.
 *
 * @param batches Vector of Arrow tables
 * @param options Conversion configuration
 * @return ArrayFire array [total_rows, cols] or nullptr on failure
 */
std::shared_ptr<::af::array> ArrowBatchesToTensor(
    const std::vector<std::shared_ptr<arrow::Table>>& batches,
    const ConversionOptions& options = ConversionOptions{}
);

/**
 * Arrow Table Statistics (for data profiling)
 *
 * Computes basic statistics on numeric columns without full conversion.
 * Useful for data exploration in Data Studio.
 */
struct ColumnStats {
    std::string name;
    int64_t count;
    int64_t null_count;
    double min;
    double max;
    double mean;
    double std_dev;
};

std::vector<ColumnStats> ComputeArrowStats(std::shared_ptr<arrow::Table> table);

/**
 * Type compatibility check
 *
 * @param table Arrow table
 * @return true if all columns can be converted to tensor
 */
bool IsConvertibleToTensor(std::shared_ptr<arrow::Table> table);

/**
 * Get convertible column names
 *
 * Filters to only numeric/boolean columns that can be tensorized.
 *
 * @param table Arrow table
 * @return Vector of numeric column names
 */
std::vector<std::string> GetNumericColumns(std::shared_ptr<arrow::Table> table);

/**
 * Categorical encoding for string columns
 *
 * Converts string column to integer codes + vocabulary.
 * Required before tensorization of text data.
 *
 * @param table Arrow table
 * @param column_name String column to encode
 * @param out_vocabulary Maps code -> original string
 * @return Arrow table with new column "<column_name>_encoded" (int32)
 */
std::shared_ptr<arrow::Table> EncodeCategorical(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name,
    std::vector<std::string>& out_vocabulary
);

/**
 * One-hot encoding for categorical columns
 *
 * Expands categorical column into N binary columns.
 *
 * @param table Arrow table
 * @param column_name Column to one-hot encode
 * @return Arrow table with new binary columns
 */
std::shared_ptr<arrow::Table> OneHotEncode(
    std::shared_ptr<arrow::Table> table,
    const std::string& column_name
);

} // namespace arrow_converters

} // namespace cyxwiz
