#include "cyxwiz/data_loader.h"

#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {
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
}  // namespace cyxwiz
