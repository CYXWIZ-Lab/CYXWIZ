#include "cyxwiz/data_loader.h"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <filesystem>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {

// ============ Static Methods ============

bool DataLoader::IsAvailable() {
#ifdef CYXWIZ_HAS_DUCKDB
    return true;
#else
    return false;
#endif
}

std::string DataLoader::GetVersion() {
#ifdef CYXWIZ_HAS_DUCKDB
    return duckdb_library_version();
#else
    return "DuckDB not available";
#endif
}

// ============ DataLoader Implementation ============

DataLoader::DataLoader()
    : database_(nullptr)
    , connection_(nullptr)
{
    Initialize();
}

DataLoader::DataLoader(const DataLoaderConfig& config)
    : config_(config)
    , database_(nullptr)
    , connection_(nullptr)
{
    Initialize();
}

DataLoader::~DataLoader() {
    Cleanup();
}

DataLoader::DataLoader(DataLoader&& other) noexcept
    : config_(std::move(other.config_))
    , database_(other.database_)
    , connection_(other.connection_)
{
    other.database_ = nullptr;
    other.connection_ = nullptr;
}

DataLoader& DataLoader::operator=(DataLoader&& other) noexcept {
    if (this != &other) {
        Cleanup();
        config_ = std::move(other.config_);
        database_ = other.database_;
        connection_ = other.connection_;
        other.database_ = nullptr;
        other.connection_ = nullptr;
    }
    return *this;
}

void DataLoader::Initialize() {
#ifdef CYXWIZ_HAS_DUCKDB
    duckdb_database db;
    if (duckdb_open(nullptr, &db) != DuckDBSuccess) {
        throw std::runtime_error("Failed to open DuckDB in-memory database");
    }
    database_ = db;

    duckdb_connection con;
    if (duckdb_connect(static_cast<duckdb_database>(database_), &con) != DuckDBSuccess) {
        duckdb_close(&db);
        database_ = nullptr;
        throw std::runtime_error("Failed to create DuckDB connection");
    }
    connection_ = con;

    if (config_.verbose) {
        spdlog::info("DataLoader initialized with DuckDB {}", GetVersion());
    }
#else
    spdlog::warn("DataLoader: DuckDB not available - data loading disabled");
#endif
}

void DataLoader::Cleanup() {
#ifdef CYXWIZ_HAS_DUCKDB
    if (connection_) {
        duckdb_connection con = static_cast<duckdb_connection>(connection_);
        duckdb_disconnect(&con);
        connection_ = nullptr;
    }
    if (database_) {
        duckdb_database db = static_cast<duckdb_database>(database_);
        duckdb_close(&db);
        database_ = nullptr;
    }
#endif
}

std::string DataLoader::NormalizePath(const std::string& path) const {
    // Convert backslashes to forward slashes for DuckDB
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

#ifdef CYXWIZ_HAS_DUCKDB
Tensor DataLoader::ResultToTensor(void* result_ptr) {
    duckdb_result* result = static_cast<duckdb_result*>(result_ptr);

    idx_t row_count = duckdb_row_count(result);
    idx_t col_count = duckdb_column_count(result);

    if (row_count == 0 || col_count == 0) {
        return Tensor({0, 0});
    }

    // Check memory limit
    size_t estimated_mb = (row_count * col_count * sizeof(float)) / (1024 * 1024);
    if (estimated_mb > config_.memory_limit_mb) {
        spdlog::warn("DataLoader: Result size ({} MB) exceeds memory limit ({} MB). Consider using BatchIterator.",
                     estimated_mb, config_.memory_limit_mb);
    }

    // Allocate tensor
    std::vector<float> data(row_count * col_count, 0.0f);

    // Process result using chunks (modern API)
    idx_t current_row = 0;
    idx_t chunk_idx = 0;

    while (true) {
        duckdb_data_chunk chunk = duckdb_result_get_chunk(*result, chunk_idx++);
        if (!chunk) break;

        idx_t chunk_size = duckdb_data_chunk_get_size(chunk);

        for (idx_t col = 0; col < col_count; col++) {
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, col);
            duckdb_type col_type = duckdb_column_type(result, col);
            uint64_t* validity = duckdb_vector_get_validity(vec);

            void* col_data = duckdb_vector_get_data(vec);

            for (idx_t row = 0; row < chunk_size; row++) {
                // Check validity (NULL handling)
                bool is_valid = validity == nullptr ||
                               duckdb_validity_row_is_valid(validity, row);

                float value = 0.0f;
                if (is_valid && col_data) {
                    switch (col_type) {
                        case DUCKDB_TYPE_FLOAT:
                            value = static_cast<float*>(col_data)[row];
                            break;
                        case DUCKDB_TYPE_DOUBLE:
                            value = static_cast<float>(static_cast<double*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_INTEGER:
                            value = static_cast<float>(static_cast<int32_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_BIGINT:
                            value = static_cast<float>(static_cast<int64_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_SMALLINT:
                            value = static_cast<float>(static_cast<int16_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_TINYINT:
                            value = static_cast<float>(static_cast<int8_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_UINTEGER:
                            value = static_cast<float>(static_cast<uint32_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_UBIGINT:
                            value = static_cast<float>(static_cast<uint64_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_BOOLEAN:
                            value = static_cast<bool*>(col_data)[row] ? 1.0f : 0.0f;
                            break;
                        default:
                            // Unsupported type - leave as 0
                            if (config_.verbose && current_row == 0) {
                                spdlog::warn("DataLoader: Unsupported column type {} for column {}",
                                            static_cast<int>(col_type), col);
                            }
                            break;
                    }
                }

                // Row-major storage: data[row * cols + col]
                data[(current_row + row) * col_count + col] = value;
            }
        }

        current_row += chunk_size;
        duckdb_destroy_data_chunk(&chunk);
    }

    // Create tensor from data
    return Tensor({static_cast<size_t>(row_count), static_cast<size_t>(col_count)},
                  data.data(), DataType::Float32);
}
#else
Tensor DataLoader::ResultToTensor(void*) {
    throw std::runtime_error("DuckDB not available");
}
#endif

// ============ File Loading ============

Tensor DataLoader::LoadParquet(const std::string& path,
                                const std::vector<std::string>& columns) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string normalized = NormalizePath(path);

    std::string col_list = columns.empty() ? "*" : "";
    if (!columns.empty()) {
        for (size_t i = 0; i < columns.size(); i++) {
            if (i > 0) col_list += ", ";
            col_list += "\"" + columns[i] + "\"";
        }
    }

    std::string sql = "SELECT " + col_list + " FROM read_parquet('" + normalized + "')";

    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to load Parquet: " + error);
    }

    Tensor tensor = ResultToTensor(&result);
    duckdb_destroy_result(&result);
    return tensor;
#else
    (void)path;
    (void)columns;
    throw std::runtime_error("DuckDB not available");
#endif
}

Tensor DataLoader::LoadCSV(const std::string& path,
                           const std::vector<std::string>& columns,
                           char delimiter,
                           bool has_header) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string normalized = NormalizePath(path);

    std::string col_list = columns.empty() ? "*" : "";
    if (!columns.empty()) {
        for (size_t i = 0; i < columns.size(); i++) {
            if (i > 0) col_list += ", ";
            col_list += "\"" + columns[i] + "\"";
        }
    }

    std::string sql = "SELECT " + col_list + " FROM read_csv('" + normalized + "', "
                      "delim='" + std::string(1, delimiter) + "', "
                      "header=" + (has_header ? "true" : "false") + ")";

    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to load CSV: " + error);
    }

    Tensor tensor = ResultToTensor(&result);
    duckdb_destroy_result(&result);
    return tensor;
#else
    (void)path;
    (void)columns;
    (void)delimiter;
    (void)has_header;
    throw std::runtime_error("DuckDB not available");
#endif
}

Tensor DataLoader::LoadJSON(const std::string& path,
                            const std::vector<std::string>& columns) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string normalized = NormalizePath(path);

    std::string col_list = columns.empty() ? "*" : "";
    if (!columns.empty()) {
        for (size_t i = 0; i < columns.size(); i++) {
            if (i > 0) col_list += ", ";
            col_list += "\"" + columns[i] + "\"";
        }
    }

    std::string sql = "SELECT " + col_list + " FROM read_json('" + normalized + "')";

    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to load JSON: " + error);
    }

    Tensor tensor = ResultToTensor(&result);
    duckdb_destroy_result(&result);
    return tensor;
#else
    (void)path;
    (void)columns;
    throw std::runtime_error("DuckDB not available");
#endif
}

// ============ SQL Queries ============

Tensor DataLoader::Query(const std::string& sql) {
#ifdef CYXWIZ_HAS_DUCKDB
    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Query failed: " + error);
    }

    Tensor tensor = ResultToTensor(&result);
    duckdb_destroy_result(&result);
    return tensor;
#else
    (void)sql;
    throw std::runtime_error("DuckDB not available");
#endif
}

std::vector<Tensor> DataLoader::QueryColumns(const std::string& sql) {
#ifdef CYXWIZ_HAS_DUCKDB
    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Query failed: " + error);
    }

    idx_t row_count = duckdb_row_count(&result);
    idx_t col_count = duckdb_column_count(&result);

    std::vector<Tensor> tensors;
    tensors.reserve(col_count);

    for (idx_t col = 0; col < col_count; col++) {
        std::vector<float> col_data(row_count, 0.0f);

        idx_t current_row = 0;
        idx_t chunk_idx = 0;

        while (true) {
            duckdb_data_chunk chunk = duckdb_result_get_chunk(result, chunk_idx++);
            if (!chunk) break;

            idx_t chunk_size = duckdb_data_chunk_get_size(chunk);
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, col);
            duckdb_type col_type = duckdb_column_type(&result, col);
            uint64_t* validity = duckdb_vector_get_validity(vec);
            void* data = duckdb_vector_get_data(vec);

            for (idx_t row = 0; row < chunk_size; row++) {
                bool is_valid = validity == nullptr ||
                               duckdb_validity_row_is_valid(validity, row);

                if (is_valid && data) {
                    switch (col_type) {
                        case DUCKDB_TYPE_FLOAT:
                            col_data[current_row + row] = static_cast<float*>(data)[row];
                            break;
                        case DUCKDB_TYPE_DOUBLE:
                            col_data[current_row + row] = static_cast<float>(static_cast<double*>(data)[row]);
                            break;
                        case DUCKDB_TYPE_INTEGER:
                            col_data[current_row + row] = static_cast<float>(static_cast<int32_t*>(data)[row]);
                            break;
                        case DUCKDB_TYPE_BIGINT:
                            col_data[current_row + row] = static_cast<float>(static_cast<int64_t*>(data)[row]);
                            break;
                        default:
                            break;
                    }
                }
            }

            current_row += chunk_size;
            duckdb_destroy_data_chunk(&chunk);
        }

        tensors.push_back(Tensor({row_count}, col_data.data(), DataType::Float32));
    }

    duckdb_destroy_result(&result);
    return tensors;
#else
    (void)sql;
    throw std::runtime_error("DuckDB not available");
#endif
}

// ============ Schema Inspection ============

std::vector<ColumnInfo> DataLoader::GetSchema(const std::string& path) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string normalized = NormalizePath(path);
    std::string sql = "DESCRIBE SELECT * FROM '" + normalized + "'";

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to get schema: " + error);
    }

    std::vector<ColumnInfo> schema;
    idx_t row_count = duckdb_row_count(&result);

    for (idx_t i = 0; i < row_count; i++) {
        ColumnInfo info;
        info.index = i;

        // Get column name (column 0)
        char* name_val = duckdb_value_varchar(&result, 0, i);
        if (name_val) {
            info.name = name_val;
            duckdb_free(name_val);
        }

        // Get column type (column 1)
        char* type_val = duckdb_value_varchar(&result, 1, i);
        if (type_val) {
            info.type = type_val;
            duckdb_free(type_val);
        }

        // Get nullable (column 2) - YES or NO
        char* null_val = duckdb_value_varchar(&result, 2, i);
        if (null_val) {
            info.nullable = (std::string(null_val) == "YES");
            duckdb_free(null_val);
        }

        schema.push_back(info);
    }

    duckdb_destroy_result(&result);
    return schema;
#else
    (void)path;
    throw std::runtime_error("DuckDB not available");
#endif
}

std::vector<std::string> DataLoader::GetColumns(const std::string& path) {
    auto schema = GetSchema(path);
    std::vector<std::string> columns;
    columns.reserve(schema.size());
    for (const auto& col : schema) {
        columns.push_back(col.name);
    }
    return columns;
}

size_t DataLoader::GetRowCount(const std::string& path) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string normalized = NormalizePath(path);
    std::string sql = "SELECT COUNT(*) FROM '" + normalized + "'";

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to get row count: " + error);
    }

    int64_t count = duckdb_value_int64(&result, 0, 0);
    duckdb_destroy_result(&result);
    return static_cast<size_t>(count);
#else
    (void)path;
    throw std::runtime_error("DuckDB not available");
#endif
}

// ============ File Conversion ============

void DataLoader::ConvertCSVToParquet(const std::string& csv_path,
                                      const std::string& parquet_path,
                                      const std::string& compression) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string csv_norm = NormalizePath(csv_path);
    std::string parquet_norm = NormalizePath(parquet_path);

    std::string sql = "COPY (SELECT * FROM read_csv('" + csv_norm + "')) "
                      "TO '" + parquet_norm + "' (FORMAT PARQUET, COMPRESSION '" + compression + "')";

    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to convert CSV to Parquet: " + error);
    }

    duckdb_destroy_result(&result);
    spdlog::info("DataLoader: Converted {} to {}", csv_path, parquet_path);
#else
    (void)csv_path;
    (void)parquet_path;
    (void)compression;
    throw std::runtime_error("DuckDB not available");
#endif
}

void DataLoader::ConvertJSONToParquet(const std::string& json_path,
                                       const std::string& parquet_path,
                                       const std::string& compression) {
#ifdef CYXWIZ_HAS_DUCKDB
    std::string json_norm = NormalizePath(json_path);
    std::string parquet_norm = NormalizePath(parquet_path);

    std::string sql = "COPY (SELECT * FROM read_json('" + json_norm + "')) "
                      "TO '" + parquet_norm + "' (FORMAT PARQUET, COMPRESSION '" + compression + "')";

    if (config_.verbose) {
        spdlog::info("DataLoader: {}", sql);
    }

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("Failed to convert JSON to Parquet: " + error);
    }

    duckdb_destroy_result(&result);
    spdlog::info("DataLoader: Converted {} to {}", json_path, parquet_path);
#else
    (void)json_path;
    (void)parquet_path;
    (void)compression;
    throw std::runtime_error("DuckDB not available");
#endif
}

// ============ BatchIterator Implementation ============

DataLoader::BatchIterator::BatchIterator()
    : batch_size_(0)
    , current_batch_(0)
    , total_rows_(0)
    , total_rows_computed_(false)
    , connection_(nullptr)
{
}

DataLoader::BatchIterator::~BatchIterator() {
    // Connection is owned by DataLoader, don't clean up here
}

DataLoader::BatchIterator::BatchIterator(BatchIterator&& other) noexcept
    : sql_(std::move(other.sql_))
    , batch_size_(other.batch_size_)
    , current_batch_(other.current_batch_)
    , total_rows_(other.total_rows_)
    , total_rows_computed_(other.total_rows_computed_)
    , connection_(other.connection_)
{
    other.connection_ = nullptr;
}

DataLoader::BatchIterator& DataLoader::BatchIterator::operator=(BatchIterator&& other) noexcept {
    if (this != &other) {
        sql_ = std::move(other.sql_);
        batch_size_ = other.batch_size_;
        current_batch_ = other.current_batch_;
        total_rows_ = other.total_rows_;
        total_rows_computed_ = other.total_rows_computed_;
        connection_ = other.connection_;
        other.connection_ = nullptr;
    }
    return *this;
}

DataLoader::BatchIterator::BatchIterator(const std::string& sql, size_t batch_size, void* connection)
    : sql_(sql)
    , batch_size_(batch_size)
    , current_batch_(0)
    , total_rows_(0)
    , total_rows_computed_(false)
    , connection_(connection)
{
}

bool DataLoader::BatchIterator::HasNext() const {
#ifdef CYXWIZ_HAS_DUCKDB
    if (!connection_) return false;

    // Compute total rows if not done yet
    if (!total_rows_computed_) {
        // const_cast to allow computing total rows in const method
        auto* self = const_cast<BatchIterator*>(this);

        std::string count_sql = "SELECT COUNT(*) FROM (" + sql_ + ") AS subquery";
        duckdb_result result;
        if (duckdb_query(static_cast<duckdb_connection>(connection_), count_sql.c_str(), &result) == DuckDBSuccess) {
            self->total_rows_ = static_cast<size_t>(duckdb_value_int64(&result, 0, 0));
            duckdb_destroy_result(&result);
        }
        self->total_rows_computed_ = true;
    }

    return (current_batch_ * batch_size_) < total_rows_;
#else
    return false;
#endif
}

Tensor DataLoader::BatchIterator::Next() {
#ifdef CYXWIZ_HAS_DUCKDB
    if (!HasNext()) {
        throw std::runtime_error("BatchIterator: No more batches");
    }

    size_t offset = current_batch_ * batch_size_;
    std::string sql = sql_ + " LIMIT " + std::to_string(batch_size_) +
                      " OFFSET " + std::to_string(offset);

    duckdb_result result;
    if (duckdb_query(static_cast<duckdb_connection>(connection_), sql.c_str(), &result) != DuckDBSuccess) {
        std::string error = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("BatchIterator query failed: " + error);
    }

    // Convert result to tensor (reuse DataLoader's logic)
    idx_t row_count = duckdb_row_count(&result);
    idx_t col_count = duckdb_column_count(&result);

    std::vector<float> data(row_count * col_count, 0.0f);

    idx_t current_row = 0;
    idx_t chunk_idx = 0;

    while (true) {
        duckdb_data_chunk chunk = duckdb_result_get_chunk(result, chunk_idx++);
        if (!chunk) break;

        idx_t chunk_size = duckdb_data_chunk_get_size(chunk);

        for (idx_t col = 0; col < col_count; col++) {
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, col);
            duckdb_type col_type = duckdb_column_type(&result, col);
            uint64_t* validity = duckdb_vector_get_validity(vec);
            void* col_data = duckdb_vector_get_data(vec);

            for (idx_t row = 0; row < chunk_size; row++) {
                bool is_valid = validity == nullptr ||
                               duckdb_validity_row_is_valid(validity, row);

                float value = 0.0f;
                if (is_valid && col_data) {
                    switch (col_type) {
                        case DUCKDB_TYPE_FLOAT:
                            value = static_cast<float*>(col_data)[row];
                            break;
                        case DUCKDB_TYPE_DOUBLE:
                            value = static_cast<float>(static_cast<double*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_INTEGER:
                            value = static_cast<float>(static_cast<int32_t*>(col_data)[row]);
                            break;
                        case DUCKDB_TYPE_BIGINT:
                            value = static_cast<float>(static_cast<int64_t*>(col_data)[row]);
                            break;
                        default:
                            break;
                    }
                }

                data[(current_row + row) * col_count + col] = value;
            }
        }

        current_row += chunk_size;
        duckdb_destroy_data_chunk(&chunk);
    }

    duckdb_destroy_result(&result);
    current_batch_++;

    return Tensor({row_count, col_count}, data.data(), DataType::Float32);
#else
    throw std::runtime_error("DuckDB not available");
#endif
}

void DataLoader::BatchIterator::Reset() {
    current_batch_ = 0;
}

size_t DataLoader::BatchIterator::TotalRows() const {
    // Force computation of total rows
    HasNext();
    return total_rows_;
}

DataLoader::BatchIterator DataLoader::CreateBatchIterator(const std::string& sql,
                                                           size_t batch_size) {
#ifdef CYXWIZ_HAS_DUCKDB
    if (batch_size == 0) {
        batch_size = config_.batch_size;
    }
    return BatchIterator(sql, batch_size, connection_);
#else
    (void)sql;
    (void)batch_size;
    throw std::runtime_error("DuckDB not available");
#endif
}

} // namespace cyxwiz
