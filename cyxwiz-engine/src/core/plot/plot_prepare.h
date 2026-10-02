#pragma once

// Builds the drawn data of a plot from table columns (TOFIX134 P1 step 1.3).
// Pure and thread-safe: run it off the UI thread for large tables; the UI
// only draws the result. Nothing is cut silently: the result's DataLabel says
// when values were reduced, sampled or read with a row limit.

#include "plot_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::plot {

// One column of the source table. Numeric columns fill `numbers` (NaN for a
// missing value); other columns fill `text`.
struct SourceColumn {
    std::string name;
    bool numeric = true;
    std::vector<double> numbers;
    std::vector<std::string> text;
    size_t size() const { return numeric ? numbers.size() : text.size(); }
};

struct Source {
    std::vector<SourceColumn> columns;
    // The read was cut at this many rows (0: the whole table was read);
    // total_rows is the table's size when known.
    size_t row_limit = 0;
    size_t total_rows = 0;
    const SourceColumn* Find(const std::string& name) const;
};

// Limits: lines longer than this are reduced (min/max per step), scatters
// larger are sampled, categories beyond the cap are merged into "other".
constexpr size_t kMaxLinePoints = 4000;
constexpr size_t kMaxScatterPoints = 50000;
constexpr size_t kMaxColorGroups = 12;
constexpr size_t kMaxCategories = 200;
constexpr size_t kMaxPieSlices = 12;
constexpr int kViolinSteps = 64;

Prepared Prepare(const PlotSpec& spec, const Source& source);

}  // namespace cyxwiz::plot
