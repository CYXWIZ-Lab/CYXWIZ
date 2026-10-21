// The Table Viewer grid's work over a whole table (TOFIX159): cell text,
// column kinds, the filter, and the display order with the Find counts, as
// the grid runs it in a task off the UI thread.
#include "../src/core/table_grid_rows.h"
#include "../src/data/data_table.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

// id (int), name (text), score (double, missing in row 2).
DataTable Sample() {
    DataTable t;
    t.SetHeaders({"id", "name", "score"});
    t.AddRow({int64_t(0), std::string("Pear"), 2.5});
    t.AddRow({int64_t(1), std::string("apple"), 10.0});
    t.AddRow({int64_t(2), std::string("Apple pie"), std::monostate{}});
    t.AddRow({int64_t(3), std::string("fig"), -1.0});
    return t;
}

std::string Join(const std::vector<size_t>& v) {
    std::string s;
    for (size_t i : v) s += (s.empty() ? "" : ",") + std::to_string(i);
    return s;
}

}  // namespace

int main() {
    const DataTable t = Sample();

    // Cell text and column kinds.
    Check(TableCellText(t, 0, 2) == "2.5", "a double in its shortest form");
    Check(TableCellText(t, 2, 2).empty(), "an empty cell is empty text");
    Check(TableCellText(t, 3, 0) == "3", "an integer");
    const auto kinds = TableColumnKindsOf(t);
    Check(kinds.size() == 3 && kinds[0].type == "Numeric" && kinds[1].type == "Text", "column kinds");
    Check(kinds[2].min_val == -1.0 && kinds[2].max_val == 10.0, "a numeric column's range skips the empty cell");
    Check(TableColumnKindOf(t, 2).max_val == 10.0, "one column's kind");

    // The filter.
    Check(TableFilter("apple", false, false).Matches("Apple pie"), "case-insensitive part");
    Check(!TableFilter("apple", true, false).Matches("Apple pie"), "case-sensitive part");
    Check(!TableFilter("apple", false, true).Matches("Apple pie"), "whole cell");
    Check(TableFilter("APPLE", false, true).Matches("apple"), "whole cell, any case");
    Check(!TableFilter().Matches(""), "an empty filter matches nothing");

    // Orders.
    TableRowRequest r;
    Check(Join(ComputeTableRowOrder(t, r).sorted) == "0,1,2,3", "the table's own order");
    r.ascending = false;
    Check(Join(ComputeTableRowOrder(t, r).sorted) == "3,2,1,0", "the table's order reversed");
    r.sort_column = 2;
    r.numeric = true;
    r.ascending = true;
    Check(Join(ComputeTableRowOrder(t, r).sorted) == "3,0,1,2", "numbers ascending, the missing one last");
    r.ascending = false;
    Check(Join(ComputeTableRowOrder(t, r).sorted) == "1,0,3,2", "numbers descending, the missing one last");
    r.sort_column = 1;
    r.numeric = false;
    r.ascending = true;
    Check(Join(ComputeTableRowOrder(t, r).sorted) == "2,0,1,3", "text ascending (byte order)");

    // Filter rows and count matches, in the sorted order.
    r.filter = TableFilter("p", false, false);
    r.filter_rows = true;
    r.count_matches = true;
    auto o = ComputeTableRowOrder(t, r);
    Check(Join(o.filtered) == "2,0,1", "the matching rows in sorted order");
    Check(o.match_cells == 3 && o.match_rows == 3, "matching cells and rows counted");
    r.filter = TableFilter("1", false, false);
    o = ComputeTableRowOrder(t, r);
    Check(o.match_cells == 3 && o.match_rows == 2, "id 1 and score 10 in row 1, score -1 in row 3");
    r.count_matches = false;
    o = ComputeTableRowOrder(t, r);
    Check(o.match_cells == -1 && Join(o.filtered) == "1,3", "rows without counting");

    // A given order: kept, only the filter redone.
    r.order = {3, 1};
    r.filter = TableFilter("a", false, false);
    o = ComputeTableRowOrder(t, r);
    Check(Join(o.sorted) == "3,1" && Join(o.filtered) == "1", "the given order is kept");
    r.order.clear();

    // Stop: a large table stops when progress says so, and reports it.
    DataTable big;
    big.SetHeaders({"v"});
    for (int i = 0; i < 300000; ++i) big.AddRow({std::string("row ") + std::to_string(i)});
    TableRowRequest b;
    b.sort_column = 0;
    b.filter = TableFilter("row 1", false, false);
    b.filter_rows = true;
    int calls = 0;
    float last = 0.0f;
    bool rising = true;
    o = ComputeTableRowOrder(big, b, [&](float f) {
        rising = rising && f >= last;
        last = f;
        return ++calls < 3;
    });
    Check(o.stopped && calls == 3, "stopped on the third progress call");
    calls = 0;
    last = 0.0f;
    o = ComputeTableRowOrder(big, b, [&](float f) {
        rising = rising && f >= last && f <= 1.0f;
        last = f;
        ++calls;
        return true;
    });
    Check(!o.stopped && calls > 3 && rising, "progress rises to the end");
    Check(o.filtered.size() == 111111 && o.filtered.front() == 1, "row 1, row 10..19, ... match");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_table_grid_rows: all checks passed\n");
    return 0;
}
