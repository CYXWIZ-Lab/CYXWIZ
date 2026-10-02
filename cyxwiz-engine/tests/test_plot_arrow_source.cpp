// Plot source from an Arrow table (TOFIX134 P2 step 2.2): numeric types,
// nulls, strings, chunks, row limit.
#include "../src/core/plot/plot_arrow_source.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Builder, typename T>
std::shared_ptr<arrow::Array> Build(const std::vector<T>& values, const std::vector<bool>& valid) {
    Builder b;
    for (size_t i = 0; i < values.size(); ++i) {
        if (valid[i]) (void)b.Append(values[i]);
        else (void)b.AppendNull();
    }
    std::shared_ptr<arrow::Array> out;
    (void)b.Finish(&out);
    return out;
}
}  // namespace

int main() {
    // Two chunks for "pixel" (uint8), a double with a null, a string label.
    auto pixel1 = Build<arrow::UInt8Builder, uint8_t>({0, 255}, {true, true});
    auto pixel2 = Build<arrow::UInt8Builder, uint8_t>({128}, {true});
    auto loss = Build<arrow::DoubleBuilder, double>({0.5, 0.0, 0.25}, {true, false, true});
    auto label = Build<arrow::StringBuilder, std::string>({"7", "", "3"}, {true, false, true});
    auto flag = Build<arrow::BooleanBuilder, bool>({true, false, true}, {true, true, true});
    auto schema = arrow::schema({arrow::field("pixel", arrow::uint8()), arrow::field("loss", arrow::float64()),
                                 arrow::field("class", arrow::utf8()), arrow::field("ok", arrow::boolean())});
    auto table = arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{pixel1, pixel2}),
                                             std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{loss}),
                                             std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{label}),
                                             std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{flag})});

    const auto info = ArrowColumns(*table);
    Check(info.size() == 4 && info[0].numeric && info[1].numeric && !info[2].numeric && info[3].numeric,
          "integers, floats and booleans are numeric; strings are not");

    const Source src = SourceFromArrow(*table, {"pixel", "loss", "class", "ok", "missing", "pixel"});
    Check(src.columns.size() == 4, "named columns once, unknown skipped");
    const SourceColumn* pixel = src.Find("pixel");
    Check(pixel && pixel->numbers == std::vector<double>({0, 255, 128}), "chunks joined in row order");
    const SourceColumn* l = src.Find("loss");
    Check(l->numbers[0] == 0.5 && std::isnan(l->numbers[1]) && l->numbers[2] == 0.25, "a null is NaN in its row");
    const SourceColumn* c = src.Find("class");
    Check(!c->numeric && c->text == std::vector<std::string>({"7", "", "3"}), "strings, a null is empty");
    Check(src.Find("ok")->numbers == std::vector<double>({1, 0, 1}), "booleans as 1/0");
    Check(src.row_limit == 0, "all rows: no limit label");

    const Source cut = SourceFromArrow(*table, {"pixel", "class"}, 2);
    Check(cut.Find("pixel")->numbers.size() == 2 && cut.Find("class")->text.size() == 2, "row limit");
    Check(cut.row_limit == 2 && cut.total_rows == 3, "the limit is labelled");

    // Prepared from it: a bar of class counts.
    PlotSpec s;
    s.kind = Kind::Bar;
    s.x_column = "class";
    const Prepared p = Prepare(s, src);
    Check(p.categories.size() == 3 && p.series[0].y == std::vector<double>({1, 1, 1}), "bar of class");
    std::cout << "plot arrow source: types, nulls, strings, chunks, row limit. OK\n";
    return 0;
}
