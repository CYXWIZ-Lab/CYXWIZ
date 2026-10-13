#pragma once

// Query plan model for the SQL Workbench's Plan tab (TOFIX146, mockup v5
// board 4). DuckDB's EXPLAIN (FORMAT JSON) and EXPLAIN (ANALYZE, FORMAT
// JSON) become operators in the order rows flow (input first), each tied
// to the query part it comes from, with plain notes on what is costly.
// Pure data: no ImGui, no DuckDB.

#include "sql_step_presentation.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cyxwiz::sqlplan {

struct PlanOperator {
    std::string name;      // DuckDB operator name, e.g. "FILTER", "HASH_JOIN"
    std::string summary;   // one line: what it does
    // DuckDB's extra info in its order ("Expression", "Projections", ...);
    // list values joined with ", ".
    std::vector<std::pair<std::string, std::string>> details;
    std::vector<std::string> columns;  // projected columns, when DuckDB lists them
    std::vector<std::string> made_columns;  // a projection's new columns (not from its input)
    int depth = 0;           // 1 for the inputs of a join, deeper when nested
    int64_t rows = -1;       // actual rows out (timed plans)
    int64_t rows_in = -1;    // rows from its inputs (timed plans)
    int64_t estimated = -1;  // DuckDB's estimate, -1 when it gives none
    double seconds = -1.0;   // time in this operator (timed plans)
    bool internal = false;   // DuckDB-internal reordering or compression
    std::vector<std::string> parts;  // query parts it comes from
    std::vector<std::string> notes;  // plain observations, most useful first
};

struct Plan {
    bool timed = false;
    std::vector<PlanOperator> ops;  // in the order rows flow
    double operator_seconds = 0.0;  // sum of operator times (timed)
    int64_t rows_out = -1;          // rows of the last operator (timed)
    int slowest = -1;               // index of the slowest operator (timed)
    int estimates_off = 0;          // operators whose estimate is 100x off (timed)
    int internal_count = 0;
    std::string error;              // when the JSON could not be read
};

// Reads DuckDB's plan JSON: the array EXPLAIN (FORMAT JSON) returns, or the
// object EXPLAIN (ANALYZE, FORMAT JSON) returns (timed).
Plan ParsePlanJson(std::string_view json);

// Ties operators to the query's parts and adds the notes. `sql` is the
// query the plan was made for, `outline` its outline.
void ExplainPlan(Plan& plan, std::string_view sql, const sqlstep::SqlOutline& outline);

// "1.81 s", "54.6 ms", "<0.1 ms".
std::string FormatSeconds(double seconds);

}  // namespace cyxwiz::sqlplan
