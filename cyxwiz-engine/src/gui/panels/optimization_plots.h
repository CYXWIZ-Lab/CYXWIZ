#pragma once

// The Optimization and Calculus window's plots (TOFIX129 D10), drawn straight
// onto the window in the theme's colours: a map of contour lines with paths
// and marks, a program's feasible region, a curve with a shaded area, and
// lines on log axes.

#include "../../core/optimization_presentation.h"

#include <imgui.h>

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::optplot {

struct Path {
    std::vector<std::vector<double>> points;   // (x, y) per step
    size_t shown = 0;                          // steps drawn (0: all)
    ImVec4 colour;
};

struct Mark {
    double x, y;
    std::string caption;
    bool diamond = false;                      // a known minimum
};

// Contour lines of `map` (darker low, lighter high), paths and marks, the
// axes labelled.
void Map(const optcalc::SurfaceMap& map, const std::vector<Path>& paths, const std::vector<Mark>& marks, float width,
         float height);

// A program of two variables: the feasible region, each row's line in a
// series colour with its name, the best point. `map` (a QP's objective) is
// drawn under it when given.
void Program(const optcalc::ProgramPlot& plot, const std::vector<std::string>& row_names,
             const std::vector<std::string>& variables, const optcalc::SurfaceMap* map, float width, float height);

// f over x with the area from `from` to `to` shaded (to the x axis).
void ShadedCurve(const std::vector<double>& xs, const std::vector<double>& ys, double from, double to, float width,
                 float height);

struct LogLine {
    std::string name;
    std::vector<double> ys;
    ImVec4 colour;
};
// Lines over x on log axes (log x when `log_x`), zeros and non-finite
// values skipped; the decades labelled.
void LogLines(const std::vector<double>& xs, const std::vector<LogLine>& lines, bool log_x, const char* x_label,
              const char* y_label, float width, float height);

}  // namespace cyxwiz::optplot
