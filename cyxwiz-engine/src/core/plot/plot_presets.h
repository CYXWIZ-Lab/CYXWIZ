#pragma once

// First plots for model-evaluation tables (TOFIX134 P2, approved board 5):
// a Plot after Confusion Matrix opens as a heatmap (actual by predicted),
// after ROC Curve or PR Curve as a line with the AUC / average precision in
// the title. Recognised by the table's columns, so the same tables from a
// file or a script get the same first plot. The type can be changed like any
// plot. A trained tree model's rows (P4.7) open as a Tree, on a forest's
// first tree.

#include "plot_model.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::plot {

// `first_value(column)` gives the column's first number (NaN when there is
// none); it is read only for the title (AUC, average precision).
std::optional<PlotSpec> EvaluationPreset(const std::vector<std::string>& columns,
                                         const std::function<double(const std::string&)>& first_value);

}  // namespace cyxwiz::plot
