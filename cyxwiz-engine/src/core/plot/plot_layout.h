#pragma once

// Layouts that depend on the drawn size (TOFIX134 P2b group 5): the
// treemap's rectangles. Pure; the view and the SVG export share them.

#include <optional>
#include <string>
#include <vector>

#include "plot_model.h"

namespace cyxwiz::plot {

struct TreeRect {
    double x = 0, y = 0, w = 0, h = 0;  // y from the top
    int depth = 0;                      // 0: the outer groups (of the zoomed level)
    bool leaf = false;
    std::vector<std::string> path{};    // outer group first
    double size = 0;
    double colour = NAN;                // leaves: the colour value (NaN without a colour column)
    int top = 0;                        // index into Prepared::tree_tops
};

// Squarified rectangles (Bruls, Huizing, van Wijk) for the values, largest
// first, filling x, y, w, h; areas proportional to the values.
std::vector<TreeRect> Squarify(const std::vector<double>& values, double x, double y, double w, double h);

// The treemap of the leaves under `zoom` (a path prefix; empty: all) in a
// w x h box: a rectangle per group with a header strip of `header` (when it
// fits), its children inside, then the leaves. Groups come before their
// children, so drawing in order paints the leaves on top.
std::vector<TreeRect> TreemapLayout(const Prepared& p, const std::vector<std::string>& zoom, double w, double h, double header);

// Where a line or scatter chart's legend hides the least: the corner of the
// data's box whose corner region (35% of each side) holds the fewest points.
// Ties keep the earlier corner (top right first), so a falling loss curve
// keeps its legend top right. `blocked`: corners not to use (bit per Corner,
// e.g. where a reference line's label sits). `current`: the corner in use,
// kept unless another holds clearly fewer points, so a live curve's legend
// does not jump between two nearly equal corners.
enum class Corner { NorthEast, NorthWest, SouthEast, SouthWest };
Corner QuietestCorner(const std::vector<Series>& series, unsigned blocked = 0,
                      std::optional<Corner> current = std::nullopt);

}  // namespace cyxwiz::plot
