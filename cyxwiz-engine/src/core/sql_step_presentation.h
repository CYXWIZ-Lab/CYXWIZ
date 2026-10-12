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

// "15349" -> "15,349"
std::string FormatCount(int64_t value);

// A Row Count Check's rule in plain words, from its parameters:
// "exactly 3,712 rows where source_id = pg41633",
// "between 10 and 20 rows", "at least 5 rows where training_admitted is true".
// Counts that are not whole numbers are quoted as written.
std::string DescribeRowCountRule(const std::map<std::string, std::string>& params);

// The step's input table names, in input-pin order (the SQL step contract).
std::vector<std::string> InputNames(const std::map<std::string, std::string>& params);

}  // namespace cyxwiz::sqlstep
