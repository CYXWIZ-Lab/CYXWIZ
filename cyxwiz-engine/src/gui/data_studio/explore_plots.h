#pragma once

// Plots drawn straight onto the window (no ImPlot), shared by Explore >
// Regression and Results > Test's regression plots (TOFIX129 D3 / D4).

#include <cstdint>
#include <utility>
#include <vector>

namespace cyxwiz::explore_plot {

// A scatter of (x, y) on its own scale, with an optional curve, and a guide:
// 0 none, 1 the line y = 0, 2 the diagonal y = x. At most 4,000 points are
// drawn (every n-th beyond that).
void Scatter(const std::vector<double>& xs, const std::vector<double>& ys, float width, float height,
             const std::vector<std::pair<double, double>>* curve, int guide, const char* x_label, const char* y_label);

// Bars of counts over lo .. hi (a histogram's bins), with the ends labelled.
void Bars(const std::vector<int64_t>& bins, double lo, double hi, float width, float height);

}  // namespace cyxwiz::explore_plot
