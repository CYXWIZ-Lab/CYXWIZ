#include "cyxwiz/data_loader.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {
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
}  // namespace cyxwiz
