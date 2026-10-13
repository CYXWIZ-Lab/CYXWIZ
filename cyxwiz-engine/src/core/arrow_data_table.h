#pragma once

// An Arrow table as a DataTable (the Table Viewer's model): numbers stay
// numbers, text stays text, nulls are empty. Used to open query results in
// the Table Viewer.

#include <arrow/api.h>

#include <memory>
#include <string>

namespace cyxwiz {

class DataTable;

// `max_rows` 0: all rows.
std::shared_ptr<DataTable> DataTableFromArrow(const arrow::Table& table, const std::string& name, size_t max_rows = 0);

// One cell as text ("" for null): integers without decimals, doubles to
// 6 significant digits, booleans true / false, everything else Arrow's text.
std::string ArrowCellText(const arrow::ChunkedArray& column, int64_t row);

}  // namespace cyxwiz
