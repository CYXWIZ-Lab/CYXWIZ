#pragma once

// The dataset profiler (TOFIX134 P3 foundation, dashboard_architecture.md L3):
// one profile of a table, computed with SQL aggregates through the session
// query service off the UI thread, used by Data Studio's Profile tab, the
// dashboards' automatic layout and (later) the Data Input dialog. Replaces
// the per-screen profilers (Data Studio Analyzer, ...).
//
// Per column: type, rows, missing (nulls plus texts the user marked as
// missing, e.g. N/A), distinct values, min / max / mean / std / quartiles,
// outliers (1.5 x IQR), a 20-bin histogram, the top 10 values, text length,
// and the share of values that read as dates or file paths (for roles). Per
// table: duplicate rows and the strongest correlations.
// Wide tables (more than kExactColumns columns) use DuckDB's approximate
// distinct counts and quartiles; the profile says so (exact = false).

#include "dataset_contract.h"
#include "session_query_engine.h"

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace cyxwiz {

struct ProfiledColumn {
    ColumnFacts facts;                  // type, rows, non-null, distinct, text length, date / path share
    std::string sql_type;               // DuckDB's type name
    size_t missing = 0;                 // nulls + missing texts
    size_t missing_text = 0;            // of which texts like N/A
    bool numeric = false;
    double min = 0, max = 0, mean = 0, std = 0, q1 = 0, median = 0, q3 = 0;
    size_t outliers = 0;                // outside q1 - 1.5 IQR .. q3 + 1.5 IQR
    std::vector<double> hist_edges;     // 21 edges
    std::vector<size_t> hist_counts;    // 20 bins
    std::vector<std::pair<std::string, size_t>> top;  // the most frequent values (text, small numbers)
    std::string min_text, max_text;     // dates and text: first and last in order
};

struct DatasetProfile {
    std::string table;                  // the name it was queried by
    size_t rows = 0;
    std::vector<ProfiledColumn> columns;
    bool duplicates_known = false;
    size_t duplicate_rows = 0;
    std::vector<std::tuple<std::string, std::string, double>> correlations;  // strongest |r| first
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
constexpr size_t kDetailColumns = 200;    // histograms, top values: the first this many columns
constexpr size_t kCorrelationColumns = 30;

DatasetProfile ProfileTable(const std::string& table, const QueryRunner& run, const ProfileOptions& options = {});

}  // namespace cyxwiz
