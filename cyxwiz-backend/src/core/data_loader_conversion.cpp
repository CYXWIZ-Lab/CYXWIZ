#include "cyxwiz/data_loader.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {
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
}  // namespace cyxwiz
