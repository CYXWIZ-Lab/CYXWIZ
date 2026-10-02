// Training Dashboard series drawn reduced (TOFIX134 P0 item 8): at most the
// requested points, extremes and the newest value kept, x in order.
#include "../src/core/series_decimation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::series;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    // 100k batch losses with one spike and one dip.
    std::vector<double> x(100000), y(100000);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i) / 1000.0;
        y[i] = 1.0 / (1.0 + x[i]) + 0.01 * std::sin(static_cast<double>(i));
    }
    y[31337] = 9.0;
    y[77777] = -2.0;
    const Decimated d = MinMaxDecimate(x, y, 2000);
    Check(d.x.size() == d.y.size() && d.x.size() <= 2002, "at most the requested points (plus first and newest)");
    Check(d.x.front() == x.front() && d.y.front() == y.front(), "the first value is drawn");
    Check(std::is_sorted(d.x.begin(), d.x.end()), "x stays in order");
    Check(*std::max_element(d.y.begin(), d.y.end()) == 9.0, "the spike is kept");
    Check(*std::min_element(d.y.begin(), d.y.end()) == -2.0, "the dip is kept");
    Check(d.x.back() == x.back() && d.y.back() == y.back(), "the newest value is drawn");

    const Decimated small = MinMaxDecimate({1, 2, 3}, {5, 4}, 2000);
    Check(small.x.size() == 2 && small.y[1] == 4, "a short series is kept, paired to the shorter side");

    const auto avg = MovingAverage({2, 4, 6, 8}, 2);
    Check(avg.size() == 4 && avg[0] == 2 && avg[1] == 3 && avg[3] == 7, "moving average");
    std::cout << "series decimation: bounded, extremes and newest kept, ordered; moving average. OK\n";
    return 0;
}
