#pragma once

#include <arrow/api.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz {

class DuckDBConnector;

// SQL transform step, contract version 1 (TOFIX101 package C, section 6).
//
// One read-only SELECT (WITH allowed) over named input tables, returning one
// table. Enforcement is DuckDB's, not text matching:
//   - the connection is opened with external access disabled, extension
//     auto-install/auto-load off and configuration locked, so read_parquet,
//     read_csv, COPY, ATTACH and file or network access fail inside DuckDB;
//   - the text must parse to exactly one SELECT statement;
//   - input and result column types must round-trip exactly (boolean,
//     integers, float, double, text). Anything else fails closed with the
//     column named, instead of being silently turned into text; cast in SQL.
// Inputs are copied into DuckDB (DuckDBConnector::RegisterTable); there is no
// zero-copy claim.
inline constexpr const char* kSqlContractParameter = "sql_contract_version";
inline constexpr const char* kSqlContractVersion = "1";
inline constexpr const char* kSqlQueryParameter = "query";
inline constexpr const char* kSqlInputAliasParameter = "input_alias";
inline constexpr const char* kSqlDefaultInputAlias = "input";

struct SqlTransformInput {
    std::string alias;
    std::shared_ptr<arrow::Table> table;
};

struct SqlTransformResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::shared_ptr<arrow::Table> table;
};

class SqlTransform {
public:
    SqlTransform();
    ~SqlTransform();
    SqlTransform(const SqlTransform&) = delete;
    SqlTransform& operator=(const SqlTransform&) = delete;

    SqlTransformResult Run(const std::string& query,
                           const std::vector<SqlTransformInput>& inputs);

    // Thread-safe: asks DuckDB to stop the running query. Run then returns
    // with cancelled = true.
    void Interrupt();

private:
    std::unique_ptr<DuckDBConnector> connector_;
    std::mutex interrupt_mutex_;
    bool interrupted_ = false;
};

// Why this alias cannot name an input table, or empty when it can.
std::string SqlInputAliasRejection(const std::string& alias);

}  // namespace cyxwiz
