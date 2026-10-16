#pragma once

// Plot exports that need no screen (TOFIX134 P1 step 1.4): the plot's data
// as CSV and the plot as SVG. Reduced or sampled series export all values.

#include "plot_model.h"
#include "plot_scales.h"

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
    // Two-sided scale (values on both sides of 0): low, middle (0), high.
    std::string diverging_low = "#4a8fd9";
    std::string diverging_mid = "#262626";
    std::string diverging_high = "#e0704f";
    // A scale picked in the plot (ToSvg sets these from the spec).
    const ScaleInfo* scale = nullptr;
    bool scale_reverse = false;
};

// The visible axis ranges (from the view; log_y draws the y axis in log10).
struct AxisRange {
    double x0 = 0, x1 = 1, y0 = 0, y1 = 1;
    bool log_y = false;
};

std::string ToSvg(const Prepared& data, const AxisRange& range, const SvgStyle& style);

// How an image plot lays its pictures out: columns of the grid (one row: 1;
// means: one line up to 12; a gallery: about square).
int PictureColumns(const Prepared& data);

// "#rrggbb" for an ImVec4-like colour (0..1 floats).
std::string HexColour(float r, float g, float b);

}  // namespace cyxwiz::plot
