// Scatter and correlation pairs come from the same row (TOFIX134 P0 item 3).
#include "../src/core/paired_columns.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::chartdata;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    double v = 0.0;
    Check(ParseFinite(" 2.5 ", &v) && v == 2.5, "spaces around a number");
    Check(ParseFinite("-1e3", &v) && v == -1000.0, "exponent");
    Check(!ParseFinite("12abc", &v), "a number followed by text is not a number");
    Check(!ParseFinite("", &v) && !ParseFinite("  ", &v), "empty");
    Check(!ParseFinite("nan", &v) && !ParseFinite("inf", &v) && !ParseFinite("1e999", &v), "not finite");

    // Row 2 has no x, row 4 has no y: before, x skipped row 2 and y skipped
    // row 4, so (x of row 3) was drawn against (y of row 2) and so on.
    const std::vector<std::vector<std::string>> rows = {
        {"1", "10"}, {"", "20"}, {"3", "30"}, {"4", "n/a"}, {"5", "50"}, {"6"},
    };
    std::vector<double> xs, ys;
    PairedNumbers(rows, 0, 1, &xs, &ys);
    Check(xs == std::vector<double>({1, 3, 5}) && ys == std::vector<double>({10, 30, 50}), "pairs from the same row");
    PairedNumbers(rows, 0, 5, &xs, &ys);
    Check(xs.empty() && ys.empty(), "a column past the row ends gives nothing");
    PairedNumbers(rows, -1, 1, &xs, &ys);
    Check(xs.empty(), "no column");
    std::cout << "paired columns: finite parse, pairs from one row. OK\n";
    return 0;
}
