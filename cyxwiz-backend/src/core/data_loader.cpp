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

}  // namespace cyxwiz

