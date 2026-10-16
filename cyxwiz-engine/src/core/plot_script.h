#pragma once

// The matplotlib script "Plot with Python" copies (TOFIX134 P0 item 4): valid
// Python whatever the data and title. Pure, no ImGui.

#include <string>
#include <vector>

namespace cyxwiz::plotscript {

enum class Kind { Histogram, Bar, Line, Scatter, Box, Pie, Stairs, Stem, Area };

// Values past this are left out, and the script says so in a comment.
constexpr size_t kMaxValues = 100000;

// `y` is used by Scatter (paired with `x` by index). Text is written as
// Python string literals, numbers with all their digits.
std::string MatplotlibScript(Kind kind, const std::string& title, const std::vector<double>& x,
                             const std::vector<double>& y = {});

}  // namespace cyxwiz::plotscript
