#pragma once

// The Table Viewer grid's work over a whole table (TOFIX159): a cell as text,
// each column's kind, the filter's match, and the display order (sorted, then
// filtered) with the Find counts. No ImGui: the grid runs it in the frame for
// a small table and in a task off the UI thread for a large one.

#include "../data/data_table.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

// A cell as text, "" when empty: a number in its shortest exact form (106.4),
// or to 10 significant digits when that is long (2/3).
std::string TableCellText(const DataTable& table, size_t row, size_t col);

// A column's kind and range, for drawing the grid: sorting, the header icon,
// number formats, data bars and colour maps. Column statistics are the column
// profile (core/dataset_profiler, TOFIX148).
struct TableColumnKind {
    std::string type = "Unknown";  // "Numeric" (any number in it) or "Text"
    double min_val = 0, max_val = 0;
};

// One column's kind: a scan of its cells.
TableColumnKind TableColumnKindOf(const DataTable& table, size_t col);
// Every column's kind.
std::vector<TableColumnKind> TableColumnKindsOf(const DataTable& table);

// The filter text and its options; matches a cell's text.
class TableFilter {
public:
    TableFilter() = default;
    TableFilter(std::string text, bool case_sensitive, bool whole_cell);
    bool empty() const { return text_.empty(); }
    // Never matches when the text is empty.
    bool Matches(const std::string& cell) const;

private:
    std::string text_;
    std::string folded_;  // lower case, for a case-insensitive match
    bool case_sensitive_ = false;
    bool whole_cell_ = false;
};

struct TableRowRequest {
    // The order: by this column (-1: the table's own order, reversed when
    // not ascending), unless `order` is given (the order kept, only the
    // filter work redone).
    int sort_column = -1;
    bool ascending = true;
    bool numeric = false;  // the sort column holds numbers (its kind)
    std::vector<size_t> order;
    TableFilter filter;
    bool filter_rows = false;    // `filtered`: the rows with a matching cell
    bool count_matches = false;  // the matching cells and rows (Find)
};

struct TableRowOrder {
    std::vector<size_t> sorted;
    std::vector<size_t> filtered;  // in `sorted` order
    int64_t match_cells = -1;      // -1: not counted
    int64_t match_rows = 0;
    bool stopped = false;          // `progress` said stop
};

// Called as the work goes with the share done (0..1); false stops it.
using TableRowProgress = std::function<bool(float fraction)>;

// The rows of `table` in display order. Missing numbers sort last.
TableRowOrder ComputeTableRowOrder(const DataTable& table, const TableRowRequest& request,
                                   const TableRowProgress& progress = {});

}  // namespace cyxwiz
