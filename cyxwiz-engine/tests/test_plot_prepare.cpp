// Plot data preparation (TOFIX134 P1 step 1.3): every P1 kind from table
// columns, with reduction, sampling, colour groups and honest labels.
#include "../src/core/plot/plot_prepare.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

SourceColumn Numbers(const std::string& name, std::vector<double> v) {
    SourceColumn c;
    c.name = name;
    c.numbers = std::move(v);
    return c;
}

SourceColumn Text(const std::string& name, std::vector<std::string> v) {
    SourceColumn c;
    c.name = name;
    c.numeric = false;
    c.text = std::move(v);
    return c;
}

PlotSpec Spec(Kind kind, std::string x, std::vector<std::string> y = {}, std::string colour = "") {
    PlotSpec s;
    s.kind = kind;
    s.x_column = std::move(x);
    s.y_columns = std::move(y);
    s.color_column = std::move(colour);
    return s;
}
}  // namespace

int main() {
    // Histogram: 10 values in 5 bins, the maximum in the last bin.
    Source h;
    h.columns.push_back(Numbers("v", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    PlotSpec hs = Spec(Kind::Histogram, "v");
    hs.bins = 5;
    Prepared p = Prepare(hs, h);
    Check(p.problem.empty() && p.series.size() == 1, "histogram series");
    Check(p.series[0].y == std::vector<double>({2, 2, 2, 2, 2}), "two per bin, max in the last");
    Check(p.edges.size() == 6 && p.edges.front() == 0 && p.edges.back() == 9, "edges span min..max");
    Check(p.label.state == DataLabel::State::Exact && p.label.shown == 10, "exact, 10 values");
    hs.density = true;
    p = Prepare(hs, h);
    double area = 0;
    for (size_t i = 0; i < p.series[0].y.size(); ++i) area += p.series[0].y[i] * (p.edges[i + 1] - p.edges[i]);
    Check(std::fabs(area - 1.0) < 1e-9, "density integrates to 1");

    // Long line: reduced, smoothed curve kept, row number as X.
    Source big;
    std::vector<double> loss(120000);
    for (size_t i = 0; i < loss.size(); ++i) loss[i] = 2.3 * std::exp(-static_cast<double>(i) / 24000.0) + 0.05 * std::sin(i * 0.7);
    loss[77777] = 9.0;  // a spike that must stay visible
    big.columns.push_back(Numbers("loss", loss));
    PlotSpec ls = Spec(Kind::Line, "", {"loss"});
    ls.smooth = 50;
    p = Prepare(ls, big);
    Check(p.label.state == DataLabel::State::Reduced && p.label.total == 120000 && p.label.shown <= kMaxLinePoints + 2,
          "reduced: " + p.label.Text());
    Check(p.series[0].x.front() == 0 && p.series[0].x.back() == 119999, "row numbers as X, newest kept: " + std::to_string(p.series[0].x.front()) + " .. " + std::to_string(p.series[0].x.back()) + " n=" + std::to_string(p.series.size()));
    double top = 0;
    for (double v : p.series[0].y) top = std::max(top, v);
    Check(top == 9.0, "the spike is drawn");
    Check(!p.series[0].smooth_y.empty() && p.series[0].smooth_y.size() <= kMaxLinePoints + 2, "smoothed curve");
    Check(p.series[0].all_y.size() == 120000 && p.series[0].x_sorted, "all values kept for hover and export");

    // Scatter: sampled evenly and the same way each time.
    Source sc;
    std::vector<double> xs(120000), ys(120000);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = static_cast<double>(i);
        ys[i] = std::sin(static_cast<double>(i));
    }
    sc.columns.push_back(Numbers("x", xs));
    sc.columns.push_back(Numbers("y", ys));
    const Prepared s1 = Prepare(Spec(Kind::Scatter, "x", {"y"}), sc);
    const Prepared s2 = Prepare(Spec(Kind::Scatter, "x", {"y"}), sc);
    Check(s1.label.state == DataLabel::State::Sampled && s1.label.shown == kMaxScatterPoints && s1.label.total == 120000,
          "sampled: " + s1.label.Text());
    Check(s1.series[0].x == s2.series[0].x, "the same sample each time");
    Check(s1.series[0].x.front() < 1000 && s1.series[0].x.back() > 119000, "spread over all rows");
    Check(s1.series[0].all_x.size() == 120000, "all rows kept for export");

    // Colour groups: 15 values -> 11 largest + other.
    Source g;
    std::vector<double> gx, gy;
    std::vector<std::string> gc;
    for (int k = 0; k < 15; ++k)
        for (int r = 0; r <= k; ++r) {
            gx.push_back(static_cast<double>(gx.size()));
            gy.push_back(k);
            gc.push_back("g" + std::to_string(k));
        }
    g.columns.push_back(Numbers("x", gx));
    g.columns.push_back(Numbers("y", gy));
    g.columns.push_back(Text("group", gc));
    p = Prepare(Spec(Kind::Line, "x", {"y"}, "group"), g);
    Check(p.series.size() == kMaxColorGroups && p.series.back().label == "other", "12 series, the rest in other");
    Check(p.series.front().label == "g4", "smallest groups merged first (g0..g3 are in other)");

    // Bar: rows per category, or the mean of Y.
    Source b;
    b.columns.push_back(Text("cls", {"a", "b", "a", "c", "a"}));
    b.columns.push_back(Numbers("score", {1, 10, 3, 7, NAN}));
    p = Prepare(Spec(Kind::Bar, "cls"), b);
    Check(p.categories == std::vector<std::string>({"a", "b", "c"}) && p.series[0].y == std::vector<double>({3, 1, 1}),
          "row counts per category");
    p = Prepare(Spec(Kind::Bar, "cls", {"score"}), b);
    Check(p.series[0].y == std::vector<double>({2, 10, 7}) && p.series[0].label == "mean of score", "mean of Y, NaN skipped");
    p = Prepare(Spec(Kind::Pie, "cls", {"score"}), b);
    Check(p.series[0].y == std::vector<double>({4, 10, 7}), "pie: sum of Y");
    Source many;
    std::vector<std::string> cats;
    for (int k = 0; k < 20; ++k) cats.push_back("c" + std::to_string(k));
    many.columns.push_back(Text("c", cats));
    p = Prepare(Spec(Kind::Pie, "c"), many);
    Check(p.categories.size() == kMaxPieSlices && p.categories.back() == "other" && p.series[0].y.back() == 9,
          "pie: 11 slices + other (9 rows)");

    // Box: whiskers stop at the last value inside 1.5 IQR.
    Source bx;
    bx.columns.push_back(Numbers("v", {1, 2, 3, 4, 5, 6, 7, 8, 9, 100}));
    p = Prepare(Spec(Kind::Box, "", {"v"}), bx);
    Check(p.boxes.size() == 1 && p.boxes[0].high == 9 && p.boxes[0].low == 1 && p.boxes[0].median == 5.5, "box whiskers");
    p = Prepare(Spec(Kind::Violin, "", {"v"}), bx);
    Check(p.series[0].y.size() == kViolinSteps && p.series[0].low.size() == kViolinSteps, "violin outline");
    for (size_t k = 0; k < p.series[0].low.size(); ++k)
        Check(p.series[0].low[k] <= 0.0 && p.series[0].high[k] >= 0.0 && p.series[0].high[k] <= 0.4, "violin half-width");

    // Error bars: mean and sd per group.
    p = Prepare(Spec(Kind::ErrorBars, "cls", {"score"}), b);
    Check(p.series[0].y == std::vector<double>({2, 10, 7}) && std::fabs(p.series[0].low[0] - std::sqrt(2.0)) < 1e-12 &&
              p.series[0].low[1] == 0,
          "mean +- sd per group");

    // Heatmap: counts of category pairs.
    Source hm;
    hm.columns.push_back(Text("pred", {"cat", "dog", "cat", "cat"}));
    hm.columns.push_back(Text("true", {"cat", "dog", "dog", "cat"}));
    p = Prepare(Spec(Kind::Heatmap, "pred", {"true"}), hm);
    Check(p.grid_rows == 2 && p.grid_cols == 2 && p.grid == std::vector<double>({2, 0, 1, 1}), "confusion counts");

    // 2D histogram: all points counted, highest y in the top row.
    Source h2;
    h2.columns.push_back(Numbers("x", {0, 1, 0, 1}));
    h2.columns.push_back(Numbers("y", {0, 0, 1, 1}));
    PlotSpec h2s = Spec(Kind::Histogram2D, "x", {"y"});
    h2s.bins = 2;
    p = Prepare(h2s, h2);
    Check(std::accumulate(p.grid.begin(), p.grid.end(), 0.0) == 4 && p.grid[0] == 1 && p.y_max == 1, "2D bins");

    // Problems are said, not drawn.
    Check(Prepare(Spec(Kind::Histogram, "nope"), h).problem == "Column 'nope' is not in the table.", "missing column");
    Check(Prepare(Spec(Kind::Scatter, "x", {"group"}), g).problem == "group is not numeric.", "text as Y");
    Check(Prepare(Spec(Kind::Scatter, "x"), g).problem == "Choose Y values.", "missing encoding");

    // A source read with a row limit says so.
    Source cut = h;
    cut.row_limit = 10;
    cut.total_rows = 2500;
    Check(Prepare(Spec(Kind::Histogram, "v"), cut).label.Text() == "first 10 of 2,500 rows", "row limit label");
    std::cout << "plot prepare: 13 kinds, reduce, sample, colour groups, categories, box/violin, grids, problems. OK\n";
    return 0;
}
