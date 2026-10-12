// SQL colouring for the SQL step editor and the SQL Workbench (TOFIX146).
// Same spans and line states as the Python colouring, so CodeEditor draws
// both: a line ends inside a /* */ comment (state '*') or a '...' string
// (state '\'') when they run past it. Keywords and types follow DuckDB;
// a name followed by "(" is a function. No ImGui.
#pragma once

#include "python_highlight.h"

#include <string_view>
#include <vector>

namespace cyxwiz::editor {

LineState HighlightSqlLine(std::string_view line, LineState in, std::vector<Span>& spans);

}  // namespace cyxwiz::editor
