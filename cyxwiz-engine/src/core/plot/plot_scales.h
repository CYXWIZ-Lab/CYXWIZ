#pragma once

// Colour scales the plot colour picker offers besides the theme's (TOFIX134
// P4.4, approved board 16): Viridis, Plasma, Magma, Cividis, Turbo, Greys and
// the two-sided Cool-warm, 16 stops each (tools/gen_plot_scales.py). The
// theme's scales stay the default; a plot that names one of these keeps it in
// its spec. Pure: no ImGui.

#include <array>
#include <string>
#include <vector>

namespace cyxwiz::plot {

struct ScaleInfo {
    const char* id;      // saved in plot specs ("viridis")
    const char* label;   // shown ("Viridis")
    bool two_sided;      // centred on 0 (Cool-warm)
    std::array<std::array<float, 3>, 16> stops;  // RGB 0..1, low to high
};

// The scales in the order the picker lists them.
const std::vector<ScaleInfo>& Scales();
// nullptr for "" (the theme) or an unknown id.
const ScaleInfo* FindScale(const std::string& id);
// RGB at t (0..1, clamped) on a scale; `reverse` runs it high to low.
std::array<float, 3> SampleScale(const ScaleInfo& scale, double t, bool reverse = false);

}  // namespace cyxwiz::plot
