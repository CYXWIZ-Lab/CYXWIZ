#pragma once

// Checks shared by the classic ML models (TOFIX145 Group C).

#include <string>
#include <vector>

namespace cyxwiz::classic_detail {

// Empty when every label is in 0..n_classes-1 and every class has a row;
// `counts` gets the rows per class.
inline std::string LabelRefusal(const std::vector<int>& labels, size_t rows, int n_classes,
                                std::vector<double>& counts) {
    if (labels.size() != rows) return "Labels and rows differ in number";
    if (n_classes < 1) return "At least one class is needed";
    counts.assign(static_cast<size_t>(n_classes), 0.0);
    for (const int label : labels) {
        if (label < 0 || label >= n_classes) return "A label is outside 0..n_classes-1";
        counts[static_cast<size_t>(label)] += 1.0;
    }
    for (const double count : counts)
        if (count == 0.0) return "Every class needs at least one row";
    return {};
}

}  // namespace cyxwiz::classic_detail
