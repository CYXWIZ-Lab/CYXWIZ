#pragma once

// Presentation model for SQL steps in Preparation Recipes (TOFIX146): the
// Properties panel, the SQL Workbench dialog and guided mode render these
// results and hold no parsing logic. Pure data in, data out: no ImGui, no
// DuckDB, no registry.

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::sqlstep {

// One named part of a query: a WITH entry (`kept AS (...)`) or the final
// statement, reported as "result".
struct SqlPart {
    std::string name;
    int first_line = 0;        // 1-based line of the part's name (or of the final SELECT)
    int last_line = 0;         // 1-based line where the part ends
    std::string description;   // "-- ..." comment lines right above the part, joined; may be empty
    // Tables and earlier parts it reads (FROM / JOIN), in first-use order,
    // lower-cased; for a part without a comment this says where it starts.
    std::vector<std::string> reads;
    // Byte range in the query text: from the part's name (or the final
    // statement) to just after its closing bracket (or the end of the text).
    size_t begin_offset = 0;
    size_t end_offset = 0;
};

struct SqlOutline {
    // Leading "-- " comment lines before the first statement token, without
    // the "--" marker; the header's first line is the query's title.
    std::vector<std::string> header;
    std::vector<SqlPart> parts;  // WITH entries in order, then "result"
    int line_count = 0;
    // False when the text is not a single statement this outline can split
    // (unbalanced brackets or quotes); parts then holds one "query" part.
    bool structured = true;
};

// Splits a query into its named parts. Respects '...' strings, "..."
// identifiers, -- and /* */ comments and nested brackets, so commas,
// brackets and keywords inside them never split a part.
SqlOutline OutlineSqlQuery(std::string_view sql);

// The query cut after part `index`, returning that part's rows:
// "WITH ... <part> AS (...) SELECT * FROM <part>". The last part ("result")
// or an unstructured outline gives the whole query. Empty when `index` is
// out of range.
std::string QueryThroughPart(std::string_view sql, const SqlOutline& outline, size_t index);

// One query returning the rows after every part: columns `part` (text) and
// `row_count` (BIGINT), one row per part in order. Empty when the outline
// is not structured (nothing to split).
std::string PartCountsQuery(std::string_view sql, const SqlOutline& outline);

// The rows part `index` dropped from the part before it:
// "<prev> EXCEPT ALL <part>". DuckDB refuses it when the two parts have
// different columns. Empty for the first part, an out-of-range index or an
// unstructured outline.
std::string RemovedRowsQuery(std::string_view sql, const SqlOutline& outline, size_t index);

// The 1-based query line a DuckDB error points at ("LINE 57: ..."), or 0.
int ErrorLine(std::string_view error);

// "15349" -> "15,349"
std::string FormatCount(int64_t value);

// A Check's rule in plain words, from its parameters:
// "exactly 3,712 rows where source_id = pg41633",
// "between 10 and 20 rows", "at least 5 rows where training_admitted is true".
// Counts that are not whole numbers are quoted as written.
std::string DescribeRowCountRule(const std::map<std::string, std::string>& params);

// The step's input table names, in input-pin order (the SQL step contract).
std::vector<std::string> InputNames(const std::map<std::string, std::string>& params);

}  // namespace cyxwiz::sqlstep
