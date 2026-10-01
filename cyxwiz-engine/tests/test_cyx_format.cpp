// .cyx notebook format (TOFIX133 P0 item 15): bare %% markers split cells,
// comments before the first marker are kept, empty cells survive, and
// save -> load -> save is stable (no blank line added per save).
#include "../src/core/cyx_format.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::cyx;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    // Bare %% (MATLAB/Spyder style) splits cells.
    auto cells = Parse("%%\nx = 1\n%%\ny = 2\n");
    Check(cells.size() == 2 && cells[0].source == "x = 1" && cells[1].source == "y = 2", "bare %% markers");
    Check(HasCellMarkers("%%\nx = 1\n"), "bare %% counts as a marker");
    Check(!HasCellMarkers("s = 'use %%code here'\n"), "%% inside a line is not a marker");

    // Comments before the first marker are the user's: kept as a code cell.
    cells = Parse("# Train the model\nimport os\n\n%%markdown\n# Title\n");
    Check(cells.size() == 2 && cells[0].kind == CellKind::Code && cells[0].source == "# Train the model\nimport os",
          "preamble with comments kept");
    Check(cells[1].kind == CellKind::Markdown && cells[1].source == "# Title", "markdown cell (its # is text)");

    // Empty cells survive; marker titles and case are tolerated.
    cells = Parse("%%code\n%% Markdown\nnotes\n%%raw\n");
    Check(cells.size() == 3 && cells[0].source.empty() && cells[1].kind == CellKind::Markdown &&
              cells[2].kind == CellKind::Raw,
          "empty code cell, '%% Markdown', raw");

    // Round trip is stable: no blank line added per save, header not a cell.
    const std::vector<Cell> start = {{CellKind::Code, "import numpy as np\n\nx = np.ones(3)"},
                                     {CellKind::Markdown, "## Results"},
                                     {CellKind::Code, ""}};
    const std::string once = Serialize(start);
    const auto back = Parse(once);
    Check(back.size() == start.size(), "same number of cells after load");
    for (size_t i = 0; i < start.size(); ++i)
        Check(back[i].kind == start[i].kind && back[i].source == start[i].source, "cell " + std::to_string(i) + " unchanged");
    Check(Serialize(back) == once, "save -> load -> save is byte-identical");
    Check(Serialize(Parse(Serialize(Parse(once)))) == once, "stable over repeated saves");

    // CRLF files parse the same.
    cells = Parse("%%code\r\nx = 1\r\n%%code\r\ny = 2\r\n");
    Check(cells.size() == 2 && cells[0].source == "x = 1", "CRLF");

    // No markers: one code cell with the whole text.
    cells = Parse("print('hi')\n");
    Check(cells.size() == 1 && cells[0].source == "print('hi')", "no markers");

    std::cout << "cyx format: bare markers, preamble, empty cells, stable round trip. OK\n";
    return 0;
}
