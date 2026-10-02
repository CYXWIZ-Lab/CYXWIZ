// "Plot with Python" writes valid Python (TOFIX134 P0 item 4). When
// CYXWIZ_TEST_PYTHON names a Python executable, every script is also
// compiled by it.
#include "../src/core/plot_script.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace cyxwiz::plotscript;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Compiles(const std::string& script, const std::string& what) {
    const char* python = std::getenv("CYXWIZ_TEST_PYTHON");
    if (!python) return;
    const auto file = std::filesystem::temp_directory_path() / "cyxwiz_plot_script_test.py";
    std::ofstream(file, std::ios::binary) << script;
    const std::string cmd = std::string("\"\"") + python + "\" -c \"import sys; compile(open(sys.argv[1], encoding='utf-8').read(), 'x', 'exec')\" \"" +
                            file.string() + "\"\"";
    Check(std::system(cmd.c_str()) == 0, what + ": Python could not compile the script");
    std::filesystem::remove(file);
}
}  // namespace

int main() {
    std::vector<double> big(250);
    for (size_t i = 0; i < big.size(); ++i) big[i] = 0.1 * static_cast<double>(i) + 1e-12;
    const std::string s = MatplotlibScript(Kind::Histogram, "O'Brien's \"loss\"\nper epoch", big);
    Check(s.find("...") == std::string::npos, "no '...' in the data (it was put after 101 values)");
    Check(s.find("plt.title('O\\'Brien\\'s \"loss\"\\nper epoch')") != std::string::npos, "title as a Python literal");
    Check(s.find("24.900000000001") != std::string::npos, "the last value, with its digits");
    Compiles(s, "histogram");

    const std::vector<double> xs = {1.5, 2.5, 3.5}, ys = {10.0, 20.0};
    const std::string sc = MatplotlibScript(Kind::Scatter, "xy", xs, ys);
    Check(sc.find("data = [1.5, 2.5]") != std::string::npos && sc.find("y_data = [10, 20]") != std::string::npos,
          "scatter pairs the same count of x and y");
    Check(sc.find("plt.scatter(data, y_data") != std::string::npos, "scatter call");
    Compiles(sc, "scatter");

    std::vector<double> huge(kMaxValues + 5, 1.0);
    huge.back() = NAN;
    const std::string h = MatplotlibScript(Kind::Line, "big", huge);
    Check(h.find("the first 100000 of 100005 values") != std::string::npos, "a cut is stated");
    const std::string odd = MatplotlibScript(Kind::Line, "odd", {1.0, NAN, INFINITY, -INFINITY});
    Check(odd.find("float('nan'), float('inf'), float('-inf')") != std::string::npos, "nan and inf as Python");
    Compiles(odd, "nan and inf");
    for (Kind k : {Kind::Bar, Kind::Box, Kind::Pie, Kind::Stairs, Kind::Stem, Kind::Area}) Compiles(MatplotlibScript(k, "t", xs), "kind");
    std::cout << "plot script: no '...', literal title, digits, scatter pairs, stated cut, nan/inf. OK\n";
    return 0;
}
