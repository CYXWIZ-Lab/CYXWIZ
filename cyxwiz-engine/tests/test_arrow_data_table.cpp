// Arrow <-> DataTable (TOFIX148): the Table Viewer profiles its DataTable as
// Arrow, and opens Arrow results as a DataTable.

#include "../src/data/arrow_data_table.h"
#include "../src/data/data_table.h"

#include <arrow/api.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    DataTable t;
    t.SetName("prices");
    t.SetHeaders({"id", "price", "", "id", "note"});
    t.AddRow({int64_t{1}, 2.5, std::string("a"), int64_t{7}, std::string("x")});
    t.AddRow({int64_t{2}, int64_t{3}, std::monostate{}, int64_t{8}, std::monostate{}});
    t.AddRow({std::monostate{}, 4.0, std::string("12"), int64_t{9}, std::string("")});

    const auto a = ArrowFromDataTable(t);
    Check(a && a->num_rows() == 3 && a->num_columns() == 5, "3 rows, 5 columns");
    const auto& s = *a->schema();
    Check(s.field(0)->name() == "id" && s.field(0)->type()->id() == arrow::Type::INT64, "integers stay int64");
    Check(s.field(1)->type()->id() == arrow::Type::DOUBLE, "integers mixed with doubles are double");
    Check(s.field(2)->name() == "column 3" && s.field(2)->type()->id() == arrow::Type::STRING, "a blank header gets a name; text");
    Check(s.field(3)->name() == "id 2", "a repeated header gets a number");
    Check(a->column(0)->null_count() == 1 && a->column(4)->null_count() == 1, "empty cells are null");

    // And back: numbers stay numbers, nulls stay empty.
    const auto back = DataTableFromArrow(*a, "back");
    Check(back->GetRowCount() == 3 && back->GetColumnCount() == 5, "back: same shape");
    Check(std::get<int64_t>(back->GetCell(0, 0)) == 1 && std::get<double>(back->GetCell(1, 1)) == 3.0, "back: numbers");
    Check(std::holds_alternative<std::monostate>(back->GetCell(2, 0)), "back: null");
    Check(std::get<std::string>(back->GetCell(2, 2)) == "12", "back: a number written as text stays text");
    Check(std::get<std::string>(back->GetCell(2, 4)).empty() && a->column(4)->null_count() == 1,
          "back: an empty text stays a value, not a null");
    Check(DataTableFromArrow(*a, "first", 2)->GetRowCount() == 2, "back: the first rows only");

    std::cout << "arrow data table: types, names, nulls, round trip. OK\n";
    return 0;
}
