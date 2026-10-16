// ReadTableClassLabels: numeric label columns keep their numbers (DBSCAN's -1
// for noise included); text label columns get ids in order of first
// appearance, shared across the columns read together; empty cells and NaN
// have no label.
#include "../src/core/table_class_labels.h"
#include "../src/data/data_table.h"

#include <cmath>
#include <cstdio>
#include <string>

using cyxwiz::DataTable;
using cyxwiz::ReadTableClassLabels;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

DataTable Table(const std::vector<DataTable::Row>& rows) {
    DataTable table;
    table.SetHeaders({"a", "b", "c"});
    for (const auto& row : rows) table.AddRow(row);
    return table;
}

}  // namespace

int main() {
    // Text: the customer-segments shape, with an empty cell and NaN.
    {
        const DataTable table = Table({
            {std::string("delta"), std::string("north"), 1.0},
            {std::string("north"), std::string("north"), 2.0},
            {std::string(""), std::string("east"), 3.0},
            {std::string("delta"), std::monostate{}, std::nan("")},
            {std::string("east"), std::string("delta"), 5.0},
        });
        const auto labels = ReadTableClassLabels(table, {0, 1});
        Check(labels.IsText(), "text columns read as text");
        Check(labels.names == std::vector<std::string>({"delta", "north", "east"}),
              "names in order of first appearance across both columns");
        const auto& a = labels.columns[0];
        const auto& b = labels.columns[1];
        Check(a[0] == 0 && a[1] == 1 && !a[2] && a[3] == 0 && a[4] == 2, "first column ids, empty string unlabelled");
        Check(b[0] == 1 && b[1] == 1 && b[2] == 2 && !b[3] && b[4] == 0, "second column shares the mapping, null unlabelled");
        Check(labels.Name(2) == "east" && labels.Name(7) == "7", "names by id");
    }

    // Numeric: numbers are the ids, text that parses counts as a number, NaN is empty.
    {
        const DataTable table = Table({
            {1.0, int64_t{0}, 0.0},
            {-1.0, std::string("2"), 0.0},
            {std::nan(""), int64_t{1}, 0.0},
            {3.7, std::monostate{}, 0.0},
        });
        const auto labels = ReadTableClassLabels(table, {0, 1});
        Check(!labels.IsText(), "numeric columns read as numbers");
        const auto& a = labels.columns[0];
        const auto& b = labels.columns[1];
        Check(a[0] == 1 && a[1] == -1 && !a[2] && a[3] == 3, "numeric ids kept, -1 kept, NaN unlabelled");
        Check(b[0] == 0 && b[1] == 2 && b[2] == 1 && !b[3], "int and numeric text cells");
        Check(labels.Name(-1) == "-1", "numeric names are the numbers");
    }

    // One text cell makes the whole set text, so ids stay consistent.
    {
        const DataTable table = Table({{1.0, 0.0, 0.0}, {std::string("x"), 0.0, 0.0}, {1.0, 0.0, 0.0}});
        const auto labels = ReadTableClassLabels(table, {0});
        Check(labels.IsText() && labels.names.size() == 2, "mixed column read as text");
        Check(labels.columns[0][0] == labels.columns[0][2] && labels.columns[0][0] != labels.columns[0][1],
              "equal cells share an id");
    }

    // A column outside the table yields no labels.
    {
        const DataTable table = Table({{1.0, 0.0, 0.0}});
        const auto labels = ReadTableClassLabels(table, {5});
        Check(labels.columns.size() == 1 && !labels.columns[0][0], "out-of-range column has no labels");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("table class labels: numeric ids kept, text ids shared in first-appearance order. OK\n");
    return 0;
}
