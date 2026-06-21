#include "cyxwiz/data_loader.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {
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
}  // namespace cyxwiz
