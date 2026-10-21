// Plot source from a DataTable (TOFIX134 P1 step 1.5): numeric detection and
// row alignment (a missing cell becomes NaN, it does not shift the column).
#include "../src/core/plot/plot_table_source.h"
#include "../src/data/data_table.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::plot;

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
    t.SetHeaders({"epoch", "loss", "label", "as_text", "empty"});
    t.AddRow({DataTable::CellValue(int64_t{1}), DataTable::CellValue(2.5), DataTable::CellValue(std::string("cat")),
              DataTable::CellValue(std::string("3.5")), DataTable::CellValue(std::monostate{})});
    t.AddRow({DataTable::CellValue(int64_t{2}), DataTable::CellValue(std::monostate{}), DataTable::CellValue(std::string("dog")),
              DataTable::CellValue(std::string(" 4 ")), DataTable::CellValue(std::monostate{})});
    t.AddRow({DataTable::CellValue(int64_t{3}), DataTable::CellValue(1.5), DataTable::CellValue(std::string("7")),
              DataTable::CellValue(std::string("")), DataTable::CellValue(std::monostate{})});

    const auto numeric = NumericColumns(t);
    Check(numeric.size() == 5, "one flag per column");
    Check(numeric[0] && numeric[1], "integers and doubles are numeric");
    Check(!numeric[2], "a text column with one number in it is text");
    Check(numeric[3], "numbers stored as text (' 4 ' with padding too) are numeric");
    Check(!numeric[4], "an all-empty column is not numeric");

    const Source src = SourceFromTable(t, {"loss", "label", "epoch", "loss", "missing"}, numeric);
    Check(src.columns.size() == 3, "named columns once each, unknown skipped");
    const SourceColumn* loss = src.Find("loss");
    Check(loss && loss->numeric && loss->numbers.size() == 3, "loss has 3 rows");
    Check(loss->numbers[0] == 2.5 && std::isnan(loss->numbers[1]) && loss->numbers[2] == 1.5,
          "the missing cell is NaN in its own row (no shift)");
    const SourceColumn* label = src.Find("label");
    Check(label && !label->numeric && label->text[2] == "7", "text column as text");
    const Source padded = SourceFromTable(t, {"as_text"}, numeric);
    Check(padded.columns[0].numbers[1] == 4.0 && std::isnan(padded.columns[0].numbers[2]), "' 4 ' is 4, '' is missing");

    // Prepared from it: a scatter pairs epoch 1 and 3 with their own losses.
    PlotSpec s;
    s.kind = Kind::Scatter;
    s.x_column = "epoch";
    s.y_columns = {"loss"};
    const Prepared p = Prepare(s, src);
    Check(p.series.size() == 1 && p.series[0].x == std::vector<double>({1, 3}) &&
              p.series[0].y == std::vector<double>({2.5, 1.5}),
          "pairs stay row-aligned");
    std::cout << "plot table source: numeric detection, aligned rows. OK\n";
    return 0;
}
