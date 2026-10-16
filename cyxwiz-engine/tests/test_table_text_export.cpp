// The Table Viewer's Export and Copy (TOFIX148): CSV / TSV quoting, exact
// numbers, empty cells, JSON types and escaping, a subset of rows in order.

#include "../src/data/data_table.h"
#include "../src/data/table_text_export.h"

#include <nlohmann/json.hpp>

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
    t.SetName("notes");
    t.SetHeaders({"id", "price", "note"});
    t.AddRow({int64_t{1}, 106.4, std::string("plain")});
    t.AddRow({int64_t{2}, 0.1, std::string("a, \"quoted\"\nline")});
    t.AddRow({int64_t{3}, std::monostate{}, std::string("tab\there")});

    const std::string csv = TableText(t, {0, 1, 2}, TableTextFormat::Csv);
    Check(csv == "id,price,note\n1,106.4,plain\n2,0.1,\"a, \"\"quoted\"\"\nline\"\n3,,tab\there\n", "CSV: " + csv);

    const std::string tsv = TableText(t, {2, 0}, TableTextFormat::Tsv, false);
    Check(tsv == "3\t\t\"tab\there\"\n1\t106.4\tplain\n", "TSV, chosen rows in order, no header: " + tsv);

    const auto json = nlohmann::json::parse(TableText(t, {1, 2}, TableTextFormat::Json));
    Check(json.size() == 2 && json[0]["id"] == 2 && json[0]["price"] == 0.1, "JSON numbers stay numbers");
    Check(json[0]["note"] == "a, \"quoted\"\nline", "JSON text is escaped and comes back whole");
    Check(json[1]["price"].is_null(), "an empty cell is null in JSON");

    const auto subset = TableRows(t, {2, 0});
    Check(subset->GetRowCount() == 2 && subset->GetHeaders() == t.GetHeaders(), "subset: rows and columns");
    Check(std::get<int64_t>(subset->GetCell(0, 0)) == 3 && std::get<int64_t>(subset->GetCell(1, 0)) == 1,
          "subset keeps the given order");

    std::cout << "table text export: CSV / TSV quoting, exact numbers, empty cells, JSON, row subsets. OK\n";
    return 0;
}
