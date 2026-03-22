#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <arrow/ipc/api.h>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Arrow-based dataset storage
 *
 * Provides zero-copy columnar data storage using Apache Arrow.
 * This is the primary format for Data Studio pipelines, with lazy
 * conversion to ArrayFire tensors when needed for ML training.
 *
 * Key benefits:
 * - Zero-copy data operations via DuckDB
 * - Larger-than-memory support via chunked format
 * - Industry standard (Parquet, Feather, CSV compatibility)
 * - Efficient columnar operations
 */
class ArrowDataset {
public:
    /**
     * Create from Arrow Table (primary constructor for Data Studio)
     * @param table Arrow table containing dataset
     * @param name Unique identifier
     */
    ArrowDataset(std::shared_ptr<arrow::Table> table, const std::string& name);

    /**
     * Load from file (CSV, Parquet, Feather, Arrow IPC)
     * @param path Path to data file
     * @param name Unique identifier
     * @return ArrowDataset instance or nullptr on failure
     */
    static std::shared_ptr<ArrowDataset> FromFile(const std::string& path,
                                                   const std::string& name);

    /**
     * Load from CSV file with options
     */
    static std::shared_ptr<ArrowDataset> FromCSV(
        const std::string& path,
        const std::string& name,
        const arrow::csv::ReadOptions& read_options = arrow::csv::ReadOptions::Defaults(),
        const arrow::csv::ParseOptions& parse_options = arrow::csv::ParseOptions::Defaults(),
        const arrow::csv::ConvertOptions& convert_options = arrow::csv::ConvertOptions::Defaults()
    );

    /**
     * Load from Parquet file
     */
    static std::shared_ptr<ArrowDataset> FromParquet(const std::string& path,
                                                     const std::string& name);

    /**
     * Load from Feather file (Arrow IPC)
     */
    static std::shared_ptr<ArrowDataset> FromFeather(const std::string& path,
                                                     const std::string& name);

    /**
     * Get underlying Arrow table
     * @return Shared pointer to Arrow table (zero-copy)
     */
    std::shared_ptr<arrow::Table> GetArrowTable() const { return table_; }

    /**
     * Get dataset name
     */
    const std::string& GetName() const { return name_; }

    /**
     * Get number of rows
     */
    int64_t GetNumRows() const { return table_ ? table_->num_rows() : 0; }

    /**
     * Get number of columns
     */
    int64_t GetNumColumns() const { return table_ ? table_->num_columns() : 0; }

    /**
     * Get column names
     */
    std::vector<std::string> GetColumnNames() const;

    /**
     * Get schema
     */
    std::shared_ptr<arrow::Schema> GetSchema() const {
        return table_ ? table_->schema() : nullptr;
    }

    /**
     * Get memory usage in bytes
     * Calculates total size of all buffers in the table
     */
    size_t GetMemoryUsage() const;

    /**
     * Select subset of columns (zero-copy view)
     * @param column_names Names of columns to keep
     * @return New ArrowDataset with only selected columns
     */
    std::shared_ptr<ArrowDataset> SelectColumns(
        const std::vector<std::string>& column_names) const;

    /**
     * Filter rows by indices (creates new table)
     * @param indices Row indices to keep
     * @return New ArrowDataset with filtered rows
     */
    std::shared_ptr<ArrowDataset> FilterRows(
        const std::vector<int64_t>& indices) const;

    /**
     * Get slice of table (zero-copy view)
     * @param offset Starting row index
     * @param length Number of rows
     * @return New ArrowDataset with sliced data
     */
    std::shared_ptr<ArrowDataset> Slice(int64_t offset, int64_t length) const;

    /**
     * Export to CSV file
     */
    bool ExportCSV(const std::string& path) const;

    /**
     * Export to Parquet file
     */
    bool ExportParquet(const std::string& path, bool compress = true) const;

    /**
     * Export to Feather file (Arrow IPC)
     */
    bool ExportFeather(const std::string& path) const;

    /**
     * Print table summary (for debugging)
     */
    void Print(int64_t max_rows = 10) const;

private:
    std::shared_ptr<arrow::Table> table_;
    std::string name_;
};

/**
 * Utility functions for Arrow operations
 */
namespace arrow_utils {

/**
 * Detect file format from extension
 */
enum class FileFormat {
    Unknown,
    CSV,
    TSV,
    Parquet,
    Feather,
    ArrowIPC
};

FileFormat DetectFormat(const std::string& path);

/**
 * Create Arrow table from raw data (for testing/prototyping)
 */
std::shared_ptr<arrow::Table> CreateTableFromVectors(
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<double>>& columns
);

/**
 * Concatenate multiple Arrow tables vertically (row-wise)
 */
arrow::Result<std::shared_ptr<arrow::Table>> ConcatenateTables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables
);

/**
 * Join two Arrow tables (requires DuckDB for complex joins)
 * This is a simple column concatenation - for SQL joins, use DuckDB
 */
arrow::Result<std::shared_ptr<arrow::Table>> ConcatenateColumns(
    const std::shared_ptr<arrow::Table>& left,
    const std::shared_ptr<arrow::Table>& right
);

} // namespace arrow_utils

} // namespace cyxwiz
