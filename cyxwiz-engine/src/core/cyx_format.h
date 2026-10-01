// The .cyx notebook text format (TOFIX133 P0 item 15). A line whose text
// starts with %% opens a cell: %%code, %%markdown, %%raw, or a bare %% (code;
// any title after %% is ignored). Text before the first marker is a code
// cell, except the header lines this Engine writes. Parse and Serialize are
// inverse: saving does not grow cells. No ImGui, no editor types.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::cyx {

enum class CellKind { Code, Markdown, Raw };

struct Cell {
    CellKind kind = CellKind::Code;
    std::string source;   // no trailing newline
};

bool HasCellMarkers(const std::string& content);
std::vector<Cell> Parse(const std::string& content);
std::string Serialize(const std::vector<Cell>& cells);

}  // namespace cyxwiz::cyx
