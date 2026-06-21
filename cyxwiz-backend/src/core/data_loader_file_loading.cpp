#include "cyxwiz/data_loader.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {
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
}  // namespace cyxwiz
