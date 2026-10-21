#pragma once

// An Arrow table (a pipeline step's output) as a DataTable the Table Viewer
// opens (TOFIX146 "Open in Table Viewer"). Integers stay integers, floats and
// decimals are numbers (double), everything else is its text; nulls stay empty.

#include <cstdint>
#include <memory>
#include <string>

namespace arrow {
class Table;
}

namespace cyxwiz {

class DataTable;

// The first `max_rows` rows (all when max_rows < 0), named `name`.
std::shared_ptr<DataTable> DataTableFromArrow(const arrow::Table& table, const std::string& name,
                                              int64_t max_rows = -1);
// Appends the first `max_rows` rows of `table` (all when max_rows < 0) to
// `out`, whose headers are already the table's columns.
void AppendArrowRows(const arrow::Table& table, DataTable& out, int64_t max_rows = -1);

// The DataTable as Arrow, for profiling it (TOFIX148): a column whose cells
// are all integers is int64, all numbers double, anything else text (numbers
// written as text); empty cells are null. Blank or repeated headers get a
// unique name ("column 3", "price 2").
std::shared_ptr<arrow::Table> ArrowFromDataTable(const DataTable& table);

}  // namespace cyxwiz
