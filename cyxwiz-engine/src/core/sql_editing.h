#pragma once

// Editing helpers for the SQL Workbench (TOFIX146 step 4b-2): completion
// candidates, a safe keyword formatter and a line diff against the saved
// query. Pure data: no ImGui, no DuckDB.

#include "sql_step_presentation.h"

#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::sqledit {

enum class CompletionKind { Column, Part, Name, Function, Keyword };

struct Completion {
    std::string text;
    CompletionKind kind = CompletionKind::Keyword;
    std::string detail;  // "input column", "part", "DuckDB function", ...
};

// Candidates starting with `prefix` (case-insensitive), best first: input
// columns, the query's parts, names it defines (AS x), DuckDB functions,
// keywords. At most `limit`. An empty prefix lists columns and parts.
std::vector<Completion> SqlCompletions(std::string_view prefix, const std::vector<std::string>& input_columns,
                                       std::string_view sql, const sqlstep::SqlOutline& outline, size_t limit = 40);

// The word being typed just before byte `column` of `line` (letters,
// digits, _), or empty.
std::string WordBefore(std::string_view line, size_t column);

// Upper-cases SQL keywords and removes trailing spaces; never touches text
// inside '...' strings, "..." names or comments. The rest of the layout is
// the author's and stays.
std::string FormatSqlKeywords(std::string_view sql);

enum class DiffKind { Same, Added, Removed };

struct DiffLine {
    DiffKind kind = DiffKind::Same;
    int old_line = 0;  // 1-based in the saved query, 0 when added
    int new_line = 0;  // 1-based in the draft, 0 when removed
    std::string text;
};

// Line diff (longest common subsequence) from `before` to `after`.
std::vector<DiffLine> DiffLines(std::string_view before, std::string_view after);

}  // namespace cyxwiz::sqledit
