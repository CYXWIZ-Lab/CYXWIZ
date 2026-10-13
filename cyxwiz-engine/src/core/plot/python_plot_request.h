#pragma once

// A plot asked for from Python (TOFIX134 P5, approved board 18): the
// matplotlib-like cyxwiz module (python_tools/cyxwiz.py) sends a plot spec,
// the names of the columns it sends with it, and where the call came from.
// Pure: checked here on the script thread, so a mistake is raised in Python
// with the Plot window's words before anything reaches the UI.

#include "plot_model.h"

#include <string>
#include <vector>

namespace cyxwiz::plot {

struct PythonPlotRequest {
    PlotSpec spec;
    std::vector<std::string> columns;  // the data's columns, in the order sent
    std::string title;                 // the window's key: the same title updates it
    std::string source;                // "df", "arrays"
    std::string file;                  // the calling script or notebook (may be empty)
    int line = 0;
};

// Reads {"spec": {...}, "columns": [...], "source": "", "file": "", "line": 0}.
// False with `error` set when the spec cannot be read, a column the spec uses
// is not in `columns`, or the kind's required columns are missing.
bool ParsePythonPlotRequest(const std::string& json, PythonPlotRequest& out, std::string* error);

// "from Python · df · p5_api.py line 7" (the window's source line).
std::string PythonPlotSourceText(const PythonPlotRequest& request, size_t rows);

}  // namespace cyxwiz::plot
