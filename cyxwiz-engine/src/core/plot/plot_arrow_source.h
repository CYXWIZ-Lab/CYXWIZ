#pragma once

// Plot source columns from an Arrow table (TOFIX134 P2 step 2.2): the table a
// node result lane or a loaded Data Input gives. Rows stay aligned; a null is
// NaN in a numeric column and "" in a text column.

#include "plot_prepare.h"

#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz::plot {

struct ArrowColumnInfo {
    std::string name;
    bool numeric = false;
};

// Every column with whether it is numeric (integers, floats, booleans).
std::vector<ArrowColumnInfo> ArrowColumns(const arrow::Table& table);

// The named columns (unknown names skipped). max_rows > 0 reads at most that
// many rows and labels the source with the row limit.
Source SourceFromArrow(const arrow::Table& table, const std::vector<std::string>& columns, size_t max_rows = 0);

// Only the given rows (0-based, in this order) of the named columns: a wide
// table (MNIST's 784 pixel columns) is never read whole for a few pictures.
Source SourceFromArrowRows(const arrow::Table& table, const std::vector<std::string>& columns, const std::vector<size_t>& rows);

}  // namespace cyxwiz::plot
