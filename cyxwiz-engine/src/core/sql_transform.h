#pragma once

#include "sql_step_contract.h"

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
// An input is a table copied into DuckDB (DuckDBConnector::RegisterTable) or
// a Parquet file DuckDB reads in place (a view, TOFIX154): the pipeline gives
// files, so a step never holds its whole input in memory. The result comes
// back as a table, or DuckDB writes it to a Parquet file (WriteResultTo).
// Contract names and input-alias parsing: sql_step_contract.h.

struct SqlTransformInput {
    std::string alias;
    std::shared_ptr<arrow::Table> table;   // copied in, or
    std::string parquet_path;              // read in place (then `schema` is its schema)
    bool load_file = false;                // the file loaded into a DuckDB table (rowid order)
    std::shared_ptr<arrow::Schema> schema;
};

struct SqlTransformResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::shared_ptr<arrow::Table> table;
    std::string plan;       // Explain / Profile: DuckDB's plan as JSON
    std::string plan_text;  // Explain / Profile: DuckDB's plan as its text drawing
    int64_t rows_written = -1;  // Run with WriteResultTo: the rows in the file
};

// What Run does after the same checks (TOFIX146 SQL Workbench):
//   Run     - execute and return the table;
//   Check   - parse and bind against the inputs (tables, columns, result
//             types) without executing; ok means the query would run;
//   Explain - DuckDB's plan with its estimates (EXPLAIN), not executed;
//   Profile - run the query once and time every operator (EXPLAIN ANALYZE);
//             the result rows are discarded.
enum class SqlRunMode { Run, Check, Explain, Profile };

class SqlTransform {
public:
    // `readable_folders`: folders of the Parquet inputs and of the result
    // file (DuckDB may read and write nothing else).
    explicit SqlTransform(std::vector<std::string> readable_folders = {});
    ~SqlTransform();
    SqlTransform(const SqlTransform&) = delete;
    SqlTransform& operator=(const SqlTransform&) = delete;

    SqlTransformResult Run(const std::string& query,
                           const std::vector<SqlTransformInput>& inputs,
                           SqlRunMode mode = SqlRunMode::Run);

    // Thread-safe: asks DuckDB to stop the running query. Run then returns
    // with cancelled = true.
    void Interrupt();

    // Run writes the result to this Parquet file instead of returning it.
    void WriteResultTo(std::string parquet_path) { output_path_ = std::move(parquet_path); }
    // The pipeline's own SQL (FilterRows, GroupBy, ...): inputs and results
    // of any type DuckDB reads, as before contract 1 (not the SQL step).
    void AllowAnyTypes() { exact_types_ = false; }
private:
    std::string output_path_;
    bool exact_types_ = true;
    std::unique_ptr<DuckDBConnector> connector_;
    std::mutex interrupt_mutex_;
    bool interrupted_ = false;
};

// Why this alias cannot name an input table, or empty when it can.
std::string SqlInputAliasRejection(const std::string& alias);

}  // namespace cyxwiz
