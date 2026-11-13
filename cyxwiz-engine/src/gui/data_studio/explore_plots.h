#pragma once

// Plots drawn straight onto the window (no ImPlot), shared by Explore >
// Regression, Groups and Projection and Results > Test's regression plots
// (TOFIX129 D3 / D4 / D5).

#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::explore_plot {

// A scatter of (x, y) on its own scale, with an optional curve, and a guide:
// 0 none, 1 the line y = 0, 2 the diagonal y = x. At most 4,000 points are
// drawn (every n-th beyond that). With `groups` (one per point), each point
// takes its group's series colour; a group below 0 (noise) is drawn faint.
void Scatter(const std::vector<double>& xs, const std::vector<double>& ys, float width, float height,
             const std::vector<std::pair<double, double>>* curve, int guide, const char* x_label, const char* y_label,
             const std::vector<int>* groups = nullptr);

// Lines over x (one per series, in the series colours from `first_colour`),
// each on its own scale when `own_scale` (the scale of series 0 is labelled),
// with a dashed vertical mark at `mark_x` (NaN: none) captioned `mark`.
struct Line {
    std::string name;
    std::vector<double> ys;
};
void Lines(const std::vector<double>& xs, const std::vector<Line>& lines, float width, float height, bool own_scale,
           double mark_x, const std::string& mark, const char* x_label, size_t first_colour = 0);

// Lines whose x axis ends are labelled by the caller (dates: "1949-01" ..
// "1960-12"); otherwise as Lines.
void LinesOver(const std::vector<double>& xs, const std::vector<Line>& lines, float width, float height, bool own_scale,
               double mark_x, const std::string& mark, const char* x_label, const std::string& lo_text, const std::string& hi_text,
               size_t first_colour = 0);

// A series and a second one (a fit and its forecast) with a band `low` ..
// `high` around the second, over every x (NaN where absent); a dashed mark at
// `mark_x` (NaN: none) captioned `mark`; the axis ends labelled.
void Band(const std::vector<double>& xs, const Line& line, const Line& centre, const std::vector<double>& low,
          const std::vector<double>& high, float width, float height, double mark_x, const std::string& mark, const char* x_label,
          const std::string& lo_text, const std::string& hi_text);

// Stems from 0 at each lag (a correlogram) with the band `lower` .. `upper`
// (per lag; a stem beyond its half-width is drawn stronger); every `season`
// lags marked (0: none). Values are in -1 .. 1.
void Stems(const std::vector<double>& values, const std::vector<double>& lower, const std::vector<double>& upper, float width,
           float height, int season, const char* y_label);

// Bars of values by position (a season by month), labelled under each bar,
// positive up and negative down from a zero line.
void PositionBars(const std::vector<double>& values, const std::vector<std::string>& names, float width, float height);

// A curve over its index (a sorted k-distance curve) with a dashed level at
// `level` (NaN: none) captioned `caption`.
void Curve(const std::vector<double>& ys, float width, float height, double level, const std::string& caption,
           const char* x_label, const char* y_label);

// A dendrogram's links (scipy coordinates: x in leaf units, leaf i at 10 i +
// 5), a dashed cut at `cut` (NaN: none), and each leaf's row count below it.
void Dendrogram(const std::vector<std::array<double, 4>>& xs, const std::vector<std::array<double, 4>>& ys,
                const std::vector<int64_t>& leaf_rows, double cut, float width, float height);

// Bars of counts over lo .. hi (a histogram's bins), with the ends labelled,
// and a dashed mark at `mark` (NaN: none) captioned `caption`.
void Bars(const std::vector<int64_t>& bins, double lo, double hi, float width, float height, double mark = NAN,
          const std::string& caption = "");

// One bar across per name, in the order given (the first on top), each with
// its value at its end (`texts`, else the value in a few digits).
void NamedBars(const std::vector<std::string>& names, const std::vector<double>& values, float width,
               const std::vector<std::string>* texts = nullptr);

// A heatmap of `cells[column][row]` (a spectrogram: frames across, bins up,
// row 0 at the bottom), each value placed on the sequential scale from `lo`
// (and below: the floor) to `hi`; at most `rows` rows are drawn (0: all). A
// cell at or below the floor takes the plot background. The x axis ends are
// labelled `x_lo` .. `x_hi` with `x_label` between; the y ends `y_lo` ..
// `y_hi` in a gutter left of the cells.
void Heatmap(const std::vector<std::vector<double>>& cells, size_t rows, double lo, double hi, float width, float height,
             const std::string& x_lo, const char* x_label, const std::string& x_hi, const std::string& y_lo, const std::string& y_hi);

}  // namespace cyxwiz::explore_plot
