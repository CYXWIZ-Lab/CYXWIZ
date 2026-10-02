#pragma once

#include <implot.h>

namespace cyxwiz::plot {

// Every plot takes its colours from the theme tokens (TOFIX134 P1 step 1.1):
// series colours, plot area, grid, axes, legend and tooltips. Called once per
// frame after the ImPlot context is current; cheap, so a theme change shows
// at once.
void ApplyPlotStyle();

// Colormaps registered from the tokens (light and dark variants; the one
// matching the current theme is returned).
ImPlotColormap SeriesColormap();
ImPlotColormap SequentialColormap();
ImPlotColormap DivergingColormap();

}  // namespace cyxwiz::plot
