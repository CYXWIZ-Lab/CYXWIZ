#pragma once

#include "api_export.h"
#include "tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace cyxwiz {

/**
 * @brief Configuration for DataLoader
 */
struct CYXWIZ_API DataLoaderConfig {
    size_t batch_size = 1024;           // Default batch size for iterators
    size_t memory_limit_mb = 4096;      // Memory limit in MB before warning
    size_t num_threads = 4;             // Number of threads for parallel operations
    bool verbose = false;               // Print verbose logging
};

/**
 * @brief Information about a column in a dataset
 */
struct CYXWIZ_API ColumnInfo {
    std::string name;
    std::string type;                   // "FLOAT", "DOUBLE", "INTEGER", "BIGINT", "VARCHAR", etc.
    bool nullable = true;
    size_t index = 0;
};

/**
 * @brief High-performance data loader using DuckDB
 *
 * Supports:
 * - Loading Parquet, CSV, JSON files directly into Tensors
 * - SQL queries on files (SELECT, JOIN, WHERE, etc.)
 * - Batch iteration for large datasets
 * - Schema inspection
 *
 * Example usage:
 * @code
 *   DataLoader loader;
 *   Tensor data = loader.LoadCSV("data.csv");
 *   Tensor query_result = loader.Query("SELECT * FROM 'data.parquet' WHERE x > 0");
 *
 *   // Batch iteration
 *   auto iter = loader.CreateBatchIterator("SELECT * FROM 'large.parquet'", 1000);
 *   while (iter.HasNext()) {
 *       Tensor batch = iter.Next();
 *       // Process batch...
 *   }
 * @endcode
 */
class CYXWIZ_API DataLoader {
public:
    /**
     * @brief Iterator for streaming large datasets in batches
     */
    class CYXWIZ_API BatchIterator {
    public:
        BatchIterator();
        ~BatchIterator();

        // Move only (no copy)
        BatchIterator(BatchIterator&& other) noexcept;
        BatchIterator& operator=(BatchIterator&& other) noexcept;
        BatchIterator(const BatchIterator&) = delete;
        BatchIterator& operator=(const BatchIterator&) = delete;

        /**
         * @brief Check if more batches are available
         */
        bool HasNext() const;

        /**
         * @brief Get next batch as Tensor
         * @throws std::runtime_error if no more batches
         */
        Tensor Next();

        /**
         * @brief Reset iterator to beginning
         */
        void Reset();

        /**
         * @brief Get total number of rows (may be expensive to compute)
         */
        size_t TotalRows() const;

        /**
         * @brief Get current batch index (0-based)
         */
        size_t CurrentBatch() const { return current_batch_; }

        /**
         * @brief Get batch size
         */
        size_t BatchSize() const { return batch_size_; }

    private:
        friend class DataLoader;

        // Private constructor - only DataLoader can create
        BatchIterator(const std::string& sql, size_t batch_size, void* connection);

        std::string sql_;
        size_t batch_size_;
        size_t current_batch_;
        size_t total_rows_;
        bool total_rows_computed_;
        void* connection_;  // duckdb_connection (opaque)
    };

    /**
     * @brief Construct DataLoader with default configuration
     */
    DataLoader();

    /**
     * @brief Construct DataLoader with custom configuration
     */
    explicit DataLoader(const DataLoaderConfig& config);

    ~DataLoader();

    // Move only (database connection is not copyable)
    DataLoader(DataLoader&& other) noexcept;
    DataLoader& operator=(DataLoader&& other) noexcept;
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    /**
     * @brief Check if DuckDB is available
     */
    static bool IsAvailable();

    /**
     * @brief Get DuckDB version string
     */
    static std::string GetVersion();

    // ============ File Loading ============

    /**
     * @brief Load Parquet file into Tensor
     * @param path Path to Parquet file
     * @param columns Optional list of column names (empty = all columns)
     * @return Tensor with shape [rows, columns], Float32 dtype
     */
    Tensor LoadParquet(const std::string& path,
                       const std::vector<std::string>& columns = {});

    /**
     * @brief Load CSV file into Tensor
     * @param path Path to CSV file
     * @param columns Optional list of column names (empty = all columns)
     * @param delimiter Column delimiter (default ',')
     * @param has_header Whether first row is header (default true)
     * @return Tensor with shape [rows, columns], Float32 dtype
     */
    Tensor LoadCSV(const std::string& path,
                   const std::vector<std::string>& columns = {},
                   char delimiter = ',',
                   bool has_header = true);

    /**
     * @brief Load JSON file into Tensor
     * @param path Path to JSON file (newline-delimited or array)
     * @param columns Optional list of column names (empty = all columns)
     * @return Tensor with shape [rows, columns], Float32 dtype
     */
    Tensor LoadJSON(const std::string& path,
                    const std::vector<std::string>& columns = {});

    // ============ SQL Queries ============

    /**
     * @brief Execute SQL query and return result as Tensor
     * @param sql SQL query (can reference files directly like 'file.parquet')
     * @return Tensor with shape [rows, columns], Float32 dtype
     *
     * Example queries:
     *   "SELECT * FROM 'data.parquet'"
     *   "SELECT a, b FROM 'data.csv' WHERE c > 10"
     *   "SELECT * FROM 'a.parquet' JOIN 'b.parquet' ON a.id = b.id"
     */
    Tensor Query(const std::string& sql);

    /**
     * @brief Execute SQL query and return result as vector of column Tensors
     * @param sql SQL query
     * @return Vector of Tensors, one per column
     */
    std::vector<Tensor> QueryColumns(const std::string& sql);

    // ============ Batch Iteration ============

    /**
     * @brief Create batch iterator for streaming large datasets
     * @param sql SQL query to iterate over
     * @param batch_size Number of rows per batch (default from config)
     * @return BatchIterator for streaming results
     */
    BatchIterator CreateBatchIterator(const std::string& sql,
                                       size_t batch_size = 0);

    // ============ Schema Inspection ============

    /**
     * @brief Get schema information for a file
     * @param path Path to data file (Parquet, CSV, JSON)
     * @return Vector of ColumnInfo structs
     */
    std::vector<ColumnInfo> GetSchema(const std::string& path);

    /**
     * @brief Get column names for a file
     * @param path Path to data file
     * @return Vector of column names
     */
    std::vector<std::string> GetColumns(const std::string& path);

    /**
     * @brief Get row count for a file
     * @param path Path to data file
     * @return Number of rows
     */
    size_t GetRowCount(const std::string& path);

    // ============ File Conversion ============

    /**
     * @brief Convert CSV file to Parquet format
     * @param csv_path Input CSV path
     * @param parquet_path Output Parquet path
     * @param compression Compression type ("snappy", "zstd", "gzip", "none")
     */
    void ConvertCSVToParquet(const std::string& csv_path,
                             const std::string& parquet_path,
                             const std::string& compression = "snappy");

    /**
     * @brief Convert JSON file to Parquet format
     * @param json_path Input JSON path
     * @param parquet_path Output Parquet path
     * @param compression Compression type
     */
    void ConvertJSONToParquet(const std::string& json_path,
                              const std::string& parquet_path,
                              const std::string& compression = "snappy");

    // ============ Configuration ============

    /**
     * @brief Get current configuration
     */
    const DataLoaderConfig& GetConfig() const { return config_; }

    /**
     * @brief Update configuration
     */
    void SetConfig(const DataLoaderConfig& config) { config_ = config; }

private:
    DataLoaderConfig config_;

    // DuckDB handles (opaque pointers)
    void* database_;     // duckdb_database
    void* connection_;   // duckdb_connection

    // Internal helpers
    void Initialize();
    void Cleanup();
    std::string NormalizePath(const std::string& path) const;
    Tensor ResultToTensor(void* result);  // duckdb_result*
};

} // namespace cyxwiz
