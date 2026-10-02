#pragma once

// Plot exports that need no screen (TOFIX134 P1 step 1.4): the plot's data
// as CSV and the plot as SVG. Reduced or sampled series export all values.

#include "plot_model.h"

#include <string>

namespace cyxwiz::plot {

std::string ToCsv(const Prepared& data);

struct SvgStyle {
    int width = 960;
    int height = 540;
    // "#rrggbb" colours from the theme tokens.
    std::string background = "#262626";
    std::string grid = "#3a3a3a";
    std::string text = "#d4d4d4";
    std::string text_dim = "#8f8f8f";
    std::string series[6] = {"#9d8fff", "#7fc8e8", "#e6bf4a", "#3dd68c", "#ff7a73", "#f27bc4"};
    std::string scale_low = "#3a2f7a";
    std::string scale_high = "#ebe6ff";
};

// The visible axis ranges (from the view; log_y draws the y axis in log10).
struct AxisRange {
    double x0 = 0, x1 = 1, y0 = 0, y1 = 1;
    bool log_y = false;
};

std::string ToSvg(const Prepared& data, const AxisRange& range, const SvgStyle& style);

// "#rrggbb" for an ImVec4-like colour (0..1 floats).
std::string HexColour(float r, float g, float b);

}  // namespace cyxwiz::plot
