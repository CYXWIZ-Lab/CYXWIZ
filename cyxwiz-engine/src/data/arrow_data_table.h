#pragma once

// An Arrow table (a pipeline step's output) as a DataTable the Table Viewer
// opens (TOFIX146 "Open in Table Viewer"). Integers stay integers, floats
// stay numbers, everything else is its text; nulls stay empty.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class ChunkedArray;
class Table;
}

namespace cyxwiz {

class DataTable;

// The first `max_rows` rows (all when max_rows < 0), named `name`.
std::shared_ptr<DataTable> DataTableFromArrow(const arrow::Table& table, const std::string& name,
                                              int64_t max_rows = -1);

// One cell as text ("" for null): integers without decimals, doubles to
// 6 significant digits, booleans true / false, everything else Arrow's text.
std::string ArrowCellText(const arrow::ChunkedArray& column, int64_t row);

// The first `max_rows` rows as one-line display text, for a preview grid:
// newlines and tabs become spaces, cells longer than `max_chars` end in
// "...", nulls are empty. rows[r][c].
struct ArrowPreview {
    std::vector<std::string> headers;
    std::vector<std::string> types;  // Arrow type names, e.g. "string", "int64"
    std::vector<std::vector<std::string>> rows;
};
ArrowPreview PreviewArrowRows(const arrow::Table& table, int64_t max_rows, size_t max_chars);

}  // namespace cyxwiz
