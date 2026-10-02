#pragma once

// Plot source columns from a DataTable (TOFIX134 P1 step 1.5). Rows stay
// aligned: a missing or text cell in a numeric column becomes NaN instead of
// being dropped (the old Quick Plot dropped it and shifted the column).

#include "plot_prepare.h"

#include <string>
#include <vector>

namespace cyxwiz {
class DataTable;
}

namespace cyxwiz::plot {

// A column is numeric when it has a number and every other non-empty cell
// is a number too (numbers stored as text count).
std::vector<bool> NumericColumns(const DataTable& table);

// The named columns (others are skipped); unknown names are ignored.
Source SourceFromTable(const DataTable& table, const std::vector<std::string>& columns,
                       const std::vector<bool>& numeric);

}  // namespace cyxwiz::plot
