#pragma once

// The dataset profiler (TOFIX134 P3 foundation, dashboard_architecture.md L3):
// one profile of a table, computed with SQL aggregates through the session
// query service off the UI thread, used by Data Studio's Profile tab, the
// dashboards' automatic layout, the SQL Workbench's result profile and
// (TOFIX148) the Data Input dialog and the Table Viewer. It is the one column
// profile: the per-screen profilers it replaces are deleted.
//
// Per column: type, rows, missing (nulls plus texts the user marked as
// missing, e.g. N/A), distinct values, min / max / mean / std / quartiles,
// outliers (1.5 x IQR), a 20-bin histogram, the top 10 values, text length,
// and the share of values that read as dates or file paths (for roles). Per
// table: duplicate rows and the correlations of the first 30 numbers.
// Wide tables (more than kExactColumns columns), and tables whose values
// take more than a quarter of the data budget (exact distinct counts,
// quartiles, top values and duplicate rows hold every value, and DuckDB
// cannot spill all of them; TOFIX154), are profiled approximately: DuckDB's
// approximate distinct counts and quartiles, top values found by
// approx_top_k and then counted exactly, duplicate rows not counted. The
// profile says so (exact = false).

#include "dataset_contract.h"
#include "session_query_engine.h"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace cyxwiz {

struct ProfiledColumn {
    ColumnFacts facts;                  // type, rows, non-null, distinct, text length, date / path share
    std::string sql_type;               // DuckDB's type name
    size_t missing = 0;                 // nulls + missing texts
    size_t missing_text = 0;            // of which texts like N/A
    size_t empty_text = 0;              // text columns: "" values (not counted as missing)
    bool numeric = false;
    double min = 0, max = 0, mean = 0, std = 0, q1 = 0, median = 0, q3 = 0;
    size_t outliers = 0;                // outside q1 - 1.5 IQR .. q3 + 1.5 IQR
    std::vector<double> hist_edges;     // 21 edges
    std::vector<size_t> hist_counts;    // 20 bins
    std::vector<std::pair<std::string, size_t>> top;  // the most frequent values (text, small numbers)
    std::string min_text, max_text;     // dates and text: first and last in order
};

// Pearson r of every pair of `columns` (rows where both are present), and
// how many rows each pair used. Row-major n x n; the diagonal is 1; NaN when
// a pair has no spread.
struct ColumnCorrelations {
    std::vector<std::string> columns;
    std::vector<double> r;
    std::vector<size_t> pairs;
    std::string error;
    bool ok() const { return error.empty(); }
    size_t size() const { return columns.size(); }
    double At(size_t i, size_t j) const { return r[i * columns.size() + j]; }
    size_t PairRows(size_t i, size_t j) const { return pairs[i * columns.size() + j]; }
};

struct DatasetProfile {
    std::string table;                  // the name it was queried by
    size_t rows = 0;
    std::vector<ProfiledColumn> columns;
    bool duplicates_known = false;
    size_t duplicate_rows = 0;
    std::vector<std::tuple<std::string, std::string, double>> correlations;  // strongest |r| first (10)
    ColumnCorrelations correlation;      // every pair of the first kCorrelationColumns numbers
    bool exact = true;
    double elapsed_ms = 0;
    std::string error;
    bool ok() const { return error.empty(); }
    std::vector<ColumnFacts> Facts() const;
    const ProfiledColumn* Find(const std::string& name) const;
    size_t MissingCells() const;
};

struct ProfileOptions {
    // Texts that mean "missing", per column (from the project's settings).
    std::map<std::string, std::vector<std::string>> missing_text;
    std::function<bool()> should_stop;            // cancellation
    std::function<void(float, const std::string&)> progress;
};

// Runs a query (the session query service's RunNow, or an engine in tests).
using QueryRunner = std::function<QueryResult(const QueryRequest&)>;

constexpr size_t kExactColumns = 64;      // beyond this many columns: approximate distinct counts and quartiles
constexpr size_t kExactBytesPerValue = 16; // what an exact count holds per value, plus a text's own bytes
constexpr size_t kDetailColumns = 200;    // histograms, top values: the first this many columns
constexpr size_t kCorrelationColumns = 30;

// The correlation matrix of chosen columns (Explore > Correlation asks for
// columns beyond the profile's first kCorrelationColumns).
ColumnCorrelations Correlate(const std::string& table, const std::vector<std::string>& columns, const QueryRunner& run,
                            bool cache = true);
// Words for |r|: "very strong" (>= .8), "strong" (.6), "moderate" (.4), "weak" (.2), "none".
const char* CorrelationStrength(double r);

// Its queries ask to be saved (QueryRequest::cache): through the session
// query service, a table profiled before (by Data Studio or a dashboard, in
// this session or an earlier one) is profiled from the saved answers.
DatasetProfile ProfileTable(const std::string& table, const QueryRunner& run, const ProfileOptions& options = {});

// Profiles a table held in memory (a query's result) on a private query
// engine: the same queries, nothing saved. Blocks; call it off the UI thread.
DatasetProfile ProfileArrow(const std::shared_ptr<arrow::Table>& table, const ProfileOptions& options = {});

}  // namespace cyxwiz
