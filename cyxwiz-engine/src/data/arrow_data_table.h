#pragma once

// An Arrow table (a pipeline step's output) as a DataTable the Table Viewer
// opens (TOFIX146 "Open in Table Viewer"). Integers stay integers, floats
// stay numbers, everything else is its text; nulls stay empty.

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

}  // namespace cyxwiz
