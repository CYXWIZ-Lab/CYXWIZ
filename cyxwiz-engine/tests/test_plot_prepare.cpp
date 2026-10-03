// Plot data preparation (TOFIX134 P1 step 1.3): every P1 kind from table
// columns, with reduction, sampling, colour groups and honest labels.
#include "../src/core/plot/plot_layout.h"
#include "../src/core/plot/plot_prepare.h"
#include "../src/core/plot/world_map.h"
#include "../src/core/plot/plot_presets.h"
#include "../src/core/plot/plot_image.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <set>
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

    // Rows (P2 board 6): 100 rows, class 0..9 repeating, v = row, name text.
    Source rs;
    std::vector<double> cls, v;
    std::vector<std::string> names;
    for (int i = 0; i < 100; ++i) {
        cls.push_back(i % 10);
        v.push_back(i);
        names.push_back(i % 2 ? "odd" : "even");
    }
    rs.columns.push_back(Numbers("class", cls));
    rs.columns.push_back(Numbers("v", v));
    rs.columns.push_back(Text("name", names));
    PlotSpec rsp = Spec(Kind::Line, "", {"v"});
    p = Prepare(rsp, rs);
    Check(p.rows_selected == 100 && p.rows_total == 100 && p.label.selection.empty(), "all rows by default");
    rsp.rows = RowMode::First;
    rsp.first_rows = 10;
    p = Prepare(rsp, rs);
    Check(p.rows_selected == 10 && p.series[0].y.back() == 9 && p.label.Text() == "first 10 of 100 rows", "first 10 rows");
    rsp.first_rows = 500;
    Check(Prepare(rsp, rs).label.selection.empty(), "first N beyond the table is all rows");
    rsp.rows = RowMode::Range;
    rsp.row_from = 21;
    rsp.row_to = 30;
    p = Prepare(rsp, rs);
    Check(p.rows_selected == 10 && p.series[0].x.front() == 20 && p.series[0].y.front() == 20,
          "range 21..30 keeps its row numbers (0-based x = 20)");
    Check(p.label.Text() == "rows 21 to 30 of 100 rows", "range label: " + p.label.Text());
    rsp.row_from = 101;
    rsp.row_to = 200;
    Check(Prepare(rsp, rs).problem == "The table has 100 rows; the range starts at row 101.", "range past the end");
    rsp.rows = RowMode::Filter;
    rsp.conditions = {{"class", "=", " 7 "}};
    p = Prepare(rsp, rs);
    Check(p.rows_selected == 10 && p.series[0].x[1] == 17 && p.label.Text() == "filtered \xC2\xB7 10 of 100 rows",
          "class = 7 (number, spaces trimmed): " + p.label.Text());
    rsp.conditions = {{"class", ">=", "8"}, {"name", "=", "odd"}};
    Check(Prepare(rsp, rs).rows_selected == 10, "class >= 8 and name = odd (9, 19, ... 99)");
    rsp.conditions = {{"name", "contains", "ve"}};
    Check(Prepare(rsp, rs).rows_selected == 50, "text contains");
    rsp.conditions = {{"class", "!=", "0"}};
    Check(Prepare(rsp, rs).rows_selected == 90, "not equal");
    rsp.conditions = {{"class", "=", "70"}};
    Check(Prepare(rsp, rs).problem == "No rows match class = 70.", "no match is said");
    rsp.conditions = {{"nope", "=", "1"}};
    Check(Prepare(rsp, rs).problem == "Filter column 'nope' is not in the table.", "unknown filter column");
    rsp.conditions = {{"", "=", ""}};
    Check(Prepare(rsp, rs).rows_selected == 100, "an empty condition filters nothing yet");
    rsp.conditions = {{"class", "=", "7"}};
    Check(ColumnsNeeded(rsp) == std::vector<std::string>({"v", "class"}), "filter columns are read too");
    Source nan_rows;
    nan_rows.columns.push_back(Numbers("a", {1, NAN, 3}));
    PlotSpec nsp = Spec(Kind::Histogram, "a");
    nsp.rows = RowMode::Filter;
    nsp.conditions = {{"a", "!=", "1"}};
    Check(Prepare(nsp, nan_rows).rows_selected == 1, "a missing value never matches");

    // Colour by a number column with many values: a scatter gets a scale,
    // other kinds get equal ranges.
    Source cs;
    std::vector<double> cx, cy, cc;
    for (int i = 0; i < 40; ++i) {
        cx.push_back(i);
        cy.push_back(i * 2);
        cc.push_back(i - 10);  // -10..29: both sides of 0
    }
    cs.columns.push_back(Numbers("x", cx));
    cs.columns.push_back(Numbers("y", cy));
    cs.columns.push_back(Numbers("c", cc));
    p = Prepare(Spec(Kind::Scatter, "x", {"y"}, "c"), cs);
    Check(p.colour_scale && p.series.size() == 1 && p.series[0].c.size() == 40 && p.series[0].c[0] == -10,
          "scatter: one series, a colour value per point");
    Check(p.colour_diverging && p.colour_min == -29 && p.colour_max == 29 && p.colour_label == "c",
          "both sides of 0: two-sided scale centred on 0");
    PlotSpec groups_spec = Spec(Kind::Scatter, "x", {"y"}, "c");
    groups_spec.color_mode = ColourMode::Groups;
    p = Prepare(groups_spec, cs);
    Check(!p.colour_scale && p.series.size() == static_cast<size_t>(kColourRanges) && p.series[0].label == "-10 to -3.5",
          "groups of a many-valued number column are ranges: " + (p.series.empty() ? std::string() : p.series[0].label));
    p = Prepare(Spec(Kind::Line, "x", {"y"}, "c"), cs);
    Check(!p.colour_scale && p.series.size() == static_cast<size_t>(kColourRanges), "a line coloured by a number: ranges");
    p = Prepare(Spec(Kind::Scatter, "class", {"v"}, "class"), rs);
    Check(!p.colour_scale && p.series.size() == 10, "10 class values stay groups (auto)");
    PlotSpec scale_spec = Spec(Kind::Scatter, "class", {"v"}, "v");
    p = Prepare(scale_spec, rs);
    Check(p.colour_scale && !p.colour_diverging && p.colour_min == 0 && p.colour_max == 99, "0..99: one-sided scale");
    {
        Source few;
        few.columns.push_back(Numbers("x", {0, 1, 2, 3}));
        few.columns.push_back(Numbers("y", {0, 1, 2, 3}));
        few.columns.push_back(Numbers("depth", {-3, 10, 300, 600}));
        PlotSpec fs = Spec(Kind::Scatter, "x", {"y"}, "depth");
        fs.color_mode = ColourMode::Scale;
        p = Prepare(fs, few);
        Check(p.colour_scale && !p.colour_diverging && p.colour_min == -3 && p.colour_max == 600, "a few values just below 0: one-sided scale");
    }

    // Evaluation tables (P2 presets): a Confusion Matrix node's long table
    // opens as a heatmap of actual by predicted with its counts, labels in
    // numeric order; ROC and PR open as lines with the score in the title.
    Source cm;
    cm.columns.push_back(Text("actual_label", {"1", "1", "0", "0", "10", "2"}));
    cm.columns.push_back(Text("predicted_label", {"1", "0", "1", "0", "2", "10"}));
    cm.columns.push_back(Numbers("count", {40, 3, 5, 52, 7, 1}));
    cm.columns.push_back(Numbers("value", {40, 3, 5, 52, 7, 1}));
    const auto none = [](const std::string&) { return NAN; };
    const auto cm_preset = EvaluationPreset({"actual_label", "predicted_label", "count", "value"}, none);
    Check(cm_preset && cm_preset->kind == Kind::Heatmap && cm_preset->x_column == "predicted_label" &&
              cm_preset->y_columns == std::vector<std::string>({"actual_label"}) && cm_preset->value_column == "value" &&
              cm_preset->x_label == "Predicted" && cm_preset->y_label == "Actual",
          "confusion matrix preset");
    p = Prepare(*cm_preset, cm);
    Check(p.problem.empty() && p.row_names == std::vector<std::string>({"0", "1", "2", "10"}) &&
              p.col_names == std::vector<std::string>({"0", "1", "2", "10"}),
          "numeric labels in numeric order on both axes");
    // Rows actual 0,1,2,10; columns predicted 0,1,2,10.
    Check(p.grid == std::vector<double>({52, 5, 0, 0, 3, 40, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0}), "cells hold the summed counts");
    const auto roc = EvaluationPreset({"threshold", "fpr", "tpr", "auc"}, [](const std::string& c) { return c == "auc" ? 0.9731 : NAN; });
    Check(roc && roc->kind == Kind::Line && roc->x_column == "fpr" && roc->y_columns.front() == "tpr" && roc->show_diagonal &&
              roc->title == "ROC curve \xC2\xB7 AUC 0.973",
          "ROC preset: " + (roc ? roc->title : std::string()));
    const auto pr = EvaluationPreset({"threshold", "precision", "recall", "average_precision"},
                                     [](const std::string&) { return 0.81234; });
    Check(pr && pr->x_column == "recall" && pr->y_columns.front() == "precision" && !pr->show_diagonal &&
              pr->title == "Precision-recall curve \xC2\xB7 AP 0.812",
          "PR preset: " + (pr ? pr->title : std::string()));
    Check(!EvaluationPreset({"class", "pixel1"}, none), "other tables get no preset");
    PlotSpec back_spec;
    Check(SpecFromJson(SpecToJson(*roc), back_spec) && back_spec.show_diagonal &&
              SpecFromJson(SpecToJson(*cm_preset), back_spec) && back_spec.value_column == "value",
          "diagonal and value column saved");

    // ---- P2b group 1 (board 8) ----
    // KDE: integrates to 1, symmetric data peaks in the middle; one curve per group.
    Source kd;
    kd.columns.push_back(Numbers("v", {-1, 0, 1, -1, 0, 1, 10, 11, 12}));
    kd.columns.push_back(Text("g", {"a", "a", "a", "a", "a", "a", "b", "b", "b"}));
    PlotSpec ks = Spec(Kind::Kde, "", {"v"}, "g");
    p = Prepare(ks, kd);
    Check(p.problem.empty() && p.series.size() == 2 && p.series[0].label == "a" && p.series[0].x.size() == 128, "KDE: a curve per group");
    for (const auto& s : p.series) {
        double kde_area = 0;
        for (size_t k = 0; k + 1 < s.x.size(); ++k) kde_area += (s.y[k] + s.y[k + 1]) / 2 * (s.x[k + 1] - s.x[k]);
        Check(std::fabs(kde_area - 1.0) < 0.02, "KDE integrates to 1 (" + std::to_string(kde_area) + ")");
    }
    {
        const auto& a = p.series[0];
        size_t peak = 0;
        for (size_t k = 1; k < a.y.size(); ++k)
            if (a.y[k] > a.y[peak]) peak = k;
        Check(std::fabs(a.x[peak]) < 0.3, "symmetric data peaks at 0");
        ks.kde_bandwidth = 2.0;
        const Prepared wide = Prepare(ks, kd);
        Check(wide.series[0].y[peak] < a.y[peak], "a wider bandwidth flattens the peak");
    }

    // Matrix: Pearson 1, -1 and 0.8 (a = 1..5, d = 1 3 2 5 4); Spearman of the same ranks.
    Source mx;
    mx.columns.push_back(Numbers("a", {1, 2, 3, 4, 5}));
    mx.columns.push_back(Numbers("b", {2, 4, 6, 8, 10}));
    mx.columns.push_back(Numbers("c", {-1, -2, -3, -4, -5}));
    mx.columns.push_back(Numbers("d", {1, 3, 2, 5, 4}));
    PlotSpec ms = Spec(Kind::Matrix, "", {"a", "b", "c", "d"});
    p = Prepare(ms, mx);
    const auto cell = [&](int r, int c) { return p.grid[static_cast<size_t>(r * p.grid_cols + c)]; };
    Check(p.problem.empty() && p.grid_rows == 4 && p.grid_cols == 4 && p.grid_diverging && p.grid_lo == -1 && p.grid_hi == 1,
          "matrix: 4 x 4 on -1..1");
    Check(std::fabs(cell(0, 1) - 1) < 1e-12 && std::fabs(cell(0, 2) + 1) < 1e-12 && std::fabs(cell(0, 3) - 0.8) < 1e-12 &&
              cell(3, 0) == cell(0, 3) && cell(2, 2) == 1,
          "Pearson 1, -1, 0.8, symmetric, 1 on the diagonal");
    ms.matrix_values = PlotSpec::MatrixValues::Spearman;
    mx.columns[3].numbers = {1, 30, 20, 500, 40};  // same ranks as 1 3 2 5 4
    p = Prepare(ms, mx);
    Check(std::fabs(cell(0, 3) - 0.8) < 1e-12, "Spearman uses the ranks");
    ms.matrix_values = PlotSpec::MatrixValues::Values;
    p = Prepare(ms, mx);
    Check(p.grid_rows == 5 && p.grid_cols == 4 && cell(1, 1) == 4 && p.grid_diverging && p.grid_lo == -500,
          "values: the columns as a grid, two-sided across 0");
    Check(Prepare(Spec(Kind::Matrix, "", {"a"}), mx).problem == "Choose two or more number columns.", "one column is not a matrix");

    // Hexbin: every point in one hexagon, totals kept; mean of a column.
    Source hx;
    std::vector<double> hxs, hys, hvs;
    for (int i = 0; i < 400; ++i) {
        hxs.push_back((i * 37) % 100);
        hys.push_back((i * 53) % 100);
        hvs.push_back(i % 2 ? 1.0 : 3.0);
    }
    hx.columns.push_back(Numbers("x", hxs));
    hx.columns.push_back(Numbers("y", hys));
    hx.columns.push_back(Numbers("v", hvs));
    PlotSpec hs2 = Spec(Kind::Hexbin, "x", {"y"});
    hs2.bins = 10;
    p = Prepare(hs2, hx);
    double hex_total = 0;
    for (double hv : p.hex_v) hex_total += hv;
    Check(p.problem.empty() && hex_total == 400 && p.hex_x.size() == p.hex_v.size() && p.hex_sx > 0 && p.grid_lo == 0,
          "hexbin: every row in one hexagon");
    hs2.value_column = "v";
    p = Prepare(hs2, hx);
    for (double hv : p.hex_v) Check(hv >= 1.0 && hv <= 3.0, "hexbin mean stays within the values");

    // Contour: z = x + y on a grid; the level lines lie on x + y = level.
    Source ct;
    std::vector<double> cxs, cys, czs;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j) {
            cxs.push_back(i / 39.0);
            cys.push_back(j / 39.0);
            czs.push_back(i / 39.0 + j / 39.0);
        }
    ct.columns.push_back(Numbers("x", cxs));
    ct.columns.push_back(Numbers("y", cys));
    ct.columns.push_back(Numbers("z", czs));
    PlotSpec cs2 = Spec(Kind::Contour, "x", {"y"});
    cs2.value_column = "z";
    cs2.bins = 20;
    cs2.levels = 3;
    p = Prepare(cs2, ct);
    Check(p.problem.empty() && p.contour_levels.size() == 3 && p.contour_segments.size() == 3, "contour: 3 levels");
    for (size_t l = 0; l < 3; ++l) {
        Check(!p.contour_segments[l].empty(), "every level has segments");
        for (size_t k = 0; k + 1 < p.contour_segments[l].size(); k += 2)
            Check(std::fabs(p.contour_segments[l][k] + p.contour_segments[l][k + 1] - p.contour_levels[l]) < 0.06,
                  "segment points lie on x + y = level");
    }
    cs2.kind = Kind::FilledContour;
    p = Prepare(cs2, ct);
    Check(p.band_rows == 80 && p.band_cols == 80, "filled contour: bands upsampled 4x");
    std::set<double> bands(p.band_grid.begin(), p.band_grid.end());
    Check(bands.size() == 4, "4 bands for 3 levels (" + std::to_string(bands.size()) + ")");

    // Bars with Colour by: grouped counts, stacked the same, 100% sums.
    Source gb;
    gb.columns.push_back(Text("type", {"album", "album", "album", "single", "single", "album"}));
    gb.columns.push_back(Text("explicit", {"no", "yes", "no", "no", "yes", "no"}));
    PlotSpec bs = Spec(Kind::Bar, "type", {}, "explicit");
    p = Prepare(bs, gb);
    Check(p.categories == std::vector<std::string>({"album", "single"}) && p.series.size() == 2 && p.series[0].label == "no" &&
              p.series[0].y == std::vector<double>({3, 1}) && p.series[1].y == std::vector<double>({1, 1}),
          "grouped bars: counts per category and group");
    bs.bar_layout = PlotSpec::BarLayout::Percent;
    p = Prepare(bs, gb);
    Check(p.series[0].y[0] == 75 && p.series[1].y[0] == 25 && p.series[0].y[1] == 50, "100%: shares per category");
    PlotSpec back_g1;
    bs.donut = true;
    bs.kde_bandwidth = 1.5;
    bs.levels = 9;
    bs.log_colour = true;
    bs.matrix_values = PlotSpec::MatrixValues::Spearman;
    Check(SpecFromJson(SpecToJson(bs), back_g1) && back_g1.bar_layout == PlotSpec::BarLayout::Percent && back_g1.donut &&
              back_g1.kde_bandwidth == 1.5 && back_g1.levels == 9 && back_g1.log_colour &&
              back_g1.matrix_values == PlotSpec::MatrixValues::Spearman,
          "group 1 options saved");

    // ---- P2b group 2 (board 9) ----
    // Polar, categories: 12 months share the turn in numeric order and close the line.
    Source pl;
    std::vector<std::string> months;
    std::vector<double> radius;
    for (int m = 12; m >= 1; --m) {
        months.push_back(std::to_string(m));
        radius.push_back(100.0 + m);
    }
    pl.columns.push_back(Text("month", months));
    pl.columns.push_back(Numbers("passengers", radius));
    p = Prepare(Spec(Kind::Polar, "month", {"passengers"}), pl);
    Check(p.problem.empty() && p.polar_closed && p.polar_names.size() == 12 && p.polar_names.front() == "1" &&
              p.series.size() == 1 && p.series[0].x.size() == 12,
          "polar categories: 12 names in numeric order, closed");
    Check(std::fabs(p.series[0].x[3] - 3.0 / 12 * 6.283185307179586) < 1e-12 && p.series[0].y[3] == 104 && p.polar_rmax == 112,
          "April at a quarter turn with its radius");
    Source pd;
    pd.columns.push_back(Numbers("deg", {0, 90, 180}));
    pd.columns.push_back(Numbers("r", {1, 2, 3}));
    p = Prepare(Spec(Kind::Polar, "deg", {"r"}), pd);
    Check(!p.polar_closed && std::fabs(p.series[0].x[1] - 1.5707963267948966) < 1e-12, "polar numbers: degrees by default");
    PlotSpec rad_spec = Spec(Kind::Polar, "deg", {"r"});
    rad_spec.angle_unit = PlotSpec::AngleUnit::Radians;
    Check(Prepare(rad_spec, pd).series[0].x[1] == 90, "radians as given");

    // Quiver: direction + length (90 degrees = east), the wind turned round, and every Nth.
    Source qd;
    qd.columns.push_back(Numbers("x", {0, 1}));
    qd.columns.push_back(Numbers("y", {0, 0}));
    qd.columns.push_back(Numbers("dir", {90, 0}));
    qd.columns.push_back(Numbers("speed", {2, 3}));
    PlotSpec qs = Spec(Kind::Quiver, "x", {"y"});
    qs.u_column = "dir";
    qs.v_column = "speed";
    qs.vector_from = PlotSpec::VectorFrom::DirectionLength;
    p = Prepare(qs, qd);
    Check(p.problem.empty() && p.qx.size() == 2 && std::fabs(p.qu[0] - 2) < 1e-12 && std::fabs(p.qv[0]) < 1e-12 &&
              std::fabs(p.qv[1] - 3) < 1e-12 && p.grid_hi == 3,
          "quiver: east 2, north 3, coloured up to the longest");
    qs.wind_from = true;
    p = Prepare(qs, qd);
    Check(std::fabs(p.qu[0] + 2) < 1e-12 && std::fabs(p.qv[1] + 3) < 1e-12, "wind: where it comes from, so the arrow turns round");
    Check(Prepare(Spec(Kind::Quiver, "x", {"y"}), qd).problem == "Choose the arrow columns (u and v).", "quiver asks for its arrows");
    Source qg;
    std::vector<double> qgx, qgy, qgu, qgv;
    for (int j = 0; j < 60; ++j)
        for (int i = 0; i < 80; ++i) {
            qgx.push_back(i * 0.25);
            qgy.push_back(j * 0.25);
            qgu.push_back(1.0);
            qgv.push_back(0.5);
        }
    qg.columns.push_back(Numbers("lon", qgx));
    qg.columns.push_back(Numbers("lat", qgy));
    qg.columns.push_back(Numbers("u", qgu));
    qg.columns.push_back(Numbers("v", qgv));
    PlotSpec qgs = Spec(Kind::Quiver, "lon", {"lat"});
    qgs.u_column = "u";
    qgs.v_column = "v";
    qgs.arrow_every = 4;
    p = Prepare(qgs, qg);
    Check(p.qx.size() == 4800 && p.q_drawn.size() == 300 && p.q_every == 4 && p.label.state == DataLabel::State::Sampled,
          "an 80 x 60 grid shows every 4th each way: 300 of 4,800");
    qgs.arrow_every = 0;
    p = Prepare(qgs, qg);
    Check(p.q_drawn.size() <= 600 && p.q_drawn.size() >= 200, "as many as fit: about 600 at most");

    // Stream: a rotating field (u = -y, v = x) keeps every line at its radius.
    Source sf;
    std::vector<double> sx, sy, su, sv;
    for (int j = 0; j <= 40; ++j)
        for (int i = 0; i <= 40; ++i) {
            const double x = -1.0 + i * 0.05, y = -1.0 + j * 0.05;
            sx.push_back(x);
            sy.push_back(y);
            su.push_back(-y);
            sv.push_back(x);
        }
    sf.columns.push_back(Numbers("x", sx));
    sf.columns.push_back(Numbers("y", sy));
    sf.columns.push_back(Numbers("u", su));
    sf.columns.push_back(Numbers("v", sv));
    PlotSpec ss = Spec(Kind::Stream, "x", {"y"});
    ss.u_column = "u";
    ss.v_column = "v";
    p = Prepare(ss, sf);
    Check(p.problem.empty() && p.stream_lines.size() >= 5 && p.grid_rows == 41 && p.grid_cols == 41, "stream: lines over a 41 x 41 field");
    for (const auto& line : p.stream_lines) {
        const double r0 = std::hypot(line[0], line[1]);
        if (r0 < 0.2 || r0 > 0.9) continue;  // away from the centre and the corners
        for (size_t k = 0; k + 1 < line.size(); k += 2)
            Check(std::fabs(std::hypot(line[k], line[k + 1]) - r0) < 0.06, "a streamline of a rotation stays on its circle");
    }
    PlotSpec back_g2;
    qs.arrow_every = 3;
    qs.stream_density = 1.5;
    qs.angle_unit = PlotSpec::AngleUnit::Categories;
    qs.polar_points = true;
    Check(SpecFromJson(SpecToJson(qs), back_g2) && back_g2.u_column == "dir" && back_g2.v_column == "speed" &&
              back_g2.vector_from == PlotSpec::VectorFrom::DirectionLength && back_g2.wind_from && back_g2.arrow_every == 3 &&
              back_g2.stream_density == 1.5 && back_g2.angle_unit == PlotSpec::AngleUnit::Categories && back_g2.polar_points,
          "group 2 options saved");

    // ---- P2b group 3 (board 10): any table, not only MNIST ----
    PlotSpec lay;
    auto l = ImageLayoutFor(lay, 784);
    Check(l.problem.empty() && l.width == 28 && l.height == 28 && l.channels == 1, "784 columns: 28 x 28 grey");
    l = ImageLayoutFor(lay, 3072);
    Check(l.width == 32 && l.height == 32 && l.channels == 3, "3,072 columns: 32 x 32 RGB (CIFAR)");
    l = ImageLayoutFor(lay, 10);
    Check(l.width == 4 && l.height == 3 && l.channels == 1, "10 columns: 4 wide, rows to fit");
    lay.image_width = 5;
    l = ImageLayoutFor(lay, 10);
    Check(l.width == 5 && l.height == 2, "a chosen width");
    lay.image_width = 0;
    lay.image_channels = 3;
    Check(!ImageLayoutFor(lay, 10).problem.empty(), "10 columns do not split into 3 channels");

    // 2 x 2 grey pictures: rows a = (0, 1, 2, 3), b = (4, 5, 6, 7) and class x, y, x.
    Source im;
    im.columns.push_back(Numbers("p1", {0, 4, 8}));
    im.columns.push_back(Numbers("p2", {1, 5, 9}));
    im.columns.push_back(Numbers("p3", {2, 6, 10}));
    im.columns.push_back(Numbers("p4", {3, 7, 11}));
    im.columns.push_back(Text("class", {"x", "y", "x"}));
    PlotSpec ims = Spec(Kind::Image, "", {"p1", "p2", "p3", "p4"}, "class");
    p = Prepare(ims, im);
    Check(p.problem.empty() && p.img_w == 2 && p.img_h == 2 && p.pictures.size() == 3 && p.pictures[1].label == "y" &&
              p.pictures[2].row == 3 && p.img_lo == 0 && p.img_hi == 11,
          "gallery: three pictures, labels and rows, range 0..11");
    Check(std::fabs(p.pictures[2].pix[3] - 1.0f) < 1e-6 && p.pictures[0].pix[0] == 0.0f, "values scaled to 0..1");
    ims.image_mode = PlotSpec::ImageMode::OneRow;
    ims.image_row = 2;
    p = Prepare(ims, im);
    Check(p.pictures.size() == 1 && p.pictures[0].row == 2 && p.img_rows == 3, "one row: row 2 of 3");
    ims.image_mode = PlotSpec::ImageMode::MeanPerClass;
    ims.image_range = PlotSpec::ImageRange::Byte;
    p = Prepare(ims, im);
    Check(p.pictures.size() == 2 && p.pictures[0].label == "x" && p.pictures[0].count == 2 &&
              std::fabs(p.pictures[0].pix[0] - 4.0f / 255.0f) < 1e-6 && p.img_hi == 255,
          "mean per class: x averages rows 1 and 3 (4 at the first pixel), on 0..255");
    ims.color_column.clear();
    Check(Prepare(ims, im).problem == "Choose a label column for the mean per class.", "mean per class needs a label");
    // Planar RGB 2 x 2 (12 columns R R R R G G G G B B B B): pixel 0 is (r0, g0, b0).
    Source rgb;
    std::vector<std::string> rgb_cols;
    for (int k = 0; k < 12; ++k) {
        rgb_cols.push_back("c" + std::to_string(k));
        rgb.columns.push_back(Numbers(rgb_cols.back(), {static_cast<double>(k)}));
    }
    PlotSpec rs2 = Spec(Kind::Image, "", rgb_cols);
    rs2.image_channels = 3;
    rs2.image_planar = true;
    p = Prepare(rs2, rgb);
    Check(p.img_channels == 3 && p.img_w == 2 && std::fabs(p.pictures[0].pix[1] * 11.0f - 4.0f) < 1e-5 &&
              std::fabs(p.pictures[0].pix[2] * 11.0f - 8.0f) < 1e-5,
          "planar RGB: pixel 0 takes c0, c4, c8");
    rs2.image_planar = false;
    p = Prepare(rs2, rgb);
    Check(std::fabs(p.pictures[0].pix[1] * 11.0f - 1.0f) < 1e-5, "interleaved RGB: pixel 0 takes c0, c1, c2");

    // Pair plot: diagonals integrate to 1, scatters keep every row here.
    Source pp;
    std::vector<double> pa, pb, pc;
    std::vector<std::string> pg;
    for (int i = 0; i < 300; ++i) {
        pa.push_back(std::sin(i * 0.1));
        pb.push_back(std::cos(i * 0.07) * 2);
        pc.push_back(i % 17);
        pg.push_back(i % 3 ? "a" : "b");
    }
    pp.columns.push_back(Numbers("a", pa));
    pp.columns.push_back(Numbers("b", pb));
    pp.columns.push_back(Numbers("c", pc));
    pp.columns.push_back(Text("g", pg));
    p = Prepare(Spec(Kind::PairPlot, "", {"a", "b", "c"}, "g"), pp);
    Check(p.problem.empty() && p.multi_cols.size() == 3 && p.multi_values[0].size() == 300 && p.multi_groups.size() == 2 &&
              p.pair_diag.size() == 3 && p.pair_diag[0].size() == 2 && p.multi_lo[2] == 0 && p.multi_hi[2] == 16,
          "pair plot: 3 columns, 2 groups, ranges");
    {
        const auto& d0 = p.pair_diag[0][0];
        double diag_area = 0;
        const double w = (p.multi_hi[0] - p.multi_lo[0]) / (p.pair_steps - 1);
        for (size_t k = 0; k + 1 < d0.size(); ++k) diag_area += (d0[k] + d0[k + 1]) / 2 * w;
        // A KDE spreads past the column's range (a sine piles up at its ends),
        // so less than 1 falls inside; a histogram keeps all of it.
        Check(diag_area > 0.7 && diag_area < 1.02, "a KDE diagonal: most of it in the range (" + std::to_string(diag_area) + ")");
        PlotSpec hist_pair = Spec(Kind::PairPlot, "", {"a", "b", "c"}, "g");
        hist_pair.pair_histogram = true;
        const Prepared hp2 = Prepare(hist_pair, pp);
        const double bw2 = (hp2.multi_hi[0] - hp2.multi_lo[0]) / hp2.pair_steps;
        double hist_area = 0;
        for (double hv : hp2.pair_diag[0][0]) hist_area += hv * bw2;
        Check(std::fabs(hist_area - 1.0) < 1e-9, "a histogram diagonal holds every row (density sums to 1)");
    }
    Check(Prepare(Spec(Kind::PairPlot, "", {"a"}), pp).problem == "Choose 2 to 6 number columns (1 chosen).", "one column is not a pair plot");
    // Parallel coordinates: sampled to 1,000 lines.
    Source par_src;
    std::vector<double> ra, rb;
    for (int i = 0; i < 5000; ++i) {
        ra.push_back(i);
        rb.push_back(-i);
    }
    par_src.columns.push_back(Numbers("a", ra));
    par_src.columns.push_back(Numbers("b", rb));
    p = Prepare(Spec(Kind::Parallel, "", {"a", "b"}), par_src);
    Check(p.multi_group.size() == 1000 && p.label.state == DataLabel::State::Sampled && p.multi_hi[0] == 4999 && p.multi_lo[1] == -4999,
          "parallel: 1,000 of 5,000 lines on each column's range");

    // ---- Model results (P2b group 4) ----
    {
    // Confusion: rows actual, columns predicted, shares by actual.
    Source conf_src;
    conf_src.columns.push_back(Text("actual", {"no", "no", "no", "yes", "yes", "yes", "yes"}));
    conf_src.columns.push_back(Text("pred", {"no", "no", "yes", "yes", "yes", "yes", "no"}));
    p = Prepare(Spec(Kind::Confusion, "actual", {"pred"}), conf_src);
    Check(p.problem.empty() && p.row_names == std::vector<std::string>({"no", "yes"}) &&
              p.grid_counts == std::vector<double>({2, 1, 1, 3}), "confusion counts: " + p.problem);
    Check(std::fabs(p.grid[0] - 2.0 / 3.0) < 1e-12 && std::fabs(p.grid[3] - 0.75) < 1e-12, "confusion shares by actual");
    Check(p.metrics.size() == 2 && std::fabs(p.metrics[0].second - 5.0 / 7.0) < 1e-12, "confusion accuracy");
    // ROC: a perfect score has AUC 1, a reversed one 0, all tied 0.5.
    Source roc_src;
    roc_src.columns.push_back(Numbers("y", {0, 0, 0, 1, 1}));
    roc_src.columns.push_back(Numbers("s", {0.1, 0.2, 0.3, 0.8, 0.9}));
    roc_src.columns.push_back(Numbers("r", {0.9, 0.8, 0.7, 0.2, 0.1}));
    roc_src.columns.push_back(Numbers("t", {0.5, 0.5, 0.5, 0.5, 0.5}));
    p = Prepare(Spec(Kind::Roc, "y", {"s"}), roc_src);
    Check(p.problem.empty() && p.positive_label == "1" && std::fabs(p.metrics[0].second - 1.0) < 1e-12, "ROC: perfect AUC 1 (" + p.problem + ")");
    Check(p.series[0].x.front() == 0 && p.series[0].x.back() == 1 && p.series[0].y.back() == 1, "ROC from (0,0) to (1,1)");
    Check(std::fabs(Prepare(Spec(Kind::Roc, "y", {"r"}), roc_src).metrics[0].second) < 1e-12, "ROC: reversed AUC 0");
    Check(std::fabs(Prepare(Spec(Kind::Roc, "y", {"t"}), roc_src).metrics[0].second - 0.5) < 1e-12, "ROC: all tied AUC 0.5");
    PlotSpec neg = Spec(Kind::Roc, "y", {"s"});
    neg.positive_class = "0";
    Check(std::fabs(Prepare(neg, roc_src).metrics[0].second) < 1e-12, "ROC: the chosen positive class");
    // PR: perfect AP 1, baseline the positive share.
    p = Prepare(Spec(Kind::PrCurve, "y", {"s"}), roc_src);
    Check(std::fabs(p.metrics[0].second - 1.0) < 1e-12 && std::fabs(p.baseline - 0.4) < 1e-12, "PR: AP 1, baseline 0.4");
    // AP on a known order: positives at ranks 1 and 3 -> (1 + 2/3) / 2.
    Source ap;
    ap.columns.push_back(Text("y", {"yes", "no", "yes", "no"}));
    ap.columns.push_back(Numbers("s", {0.9, 0.8, 0.7, 0.6}));
    p = Prepare(Spec(Kind::PrCurve, "y", {"s"}), ap);
    Check(p.positive_label == "yes" && std::fabs(p.metrics[0].second - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12, "PR: average precision");
    // Calibration: bin b holds p = b/10 + 0.05 with b + 1 positives of 10,
    // so every bin is 0.05 off the diagonal.
    Source cal;
    std::vector<double> cal_y, cp;
    for (int bin = 0; bin < 10; ++bin)
        for (int i = 0; i < 10; ++i) {
            cp.push_back(bin / 10.0 + 0.05);
            cal_y.push_back(i <= bin ? 1 : 0);
        }
    cal.columns.push_back(Numbers("y", cal_y));
    cal.columns.push_back(Numbers("p", cp));
    p = Prepare(Spec(Kind::Calibration, "y", {"p"}), cal);
    Check(p.problem.empty() && p.series[0].x.size() == 10 && std::fabs(p.series[0].x[3] - 0.35) < 1e-9 &&
              std::fabs(p.series[0].y[3] - 0.4) < 1e-12 && p.series[0].low[3] == 10, "calibration bins: " + p.problem);
    Check(std::fabs(p.metrics[1].second - 0.05) < 1e-9, "calibration ECE 0.05");
    Source over;
    over.columns.push_back(Numbers("y", {0, 1}));
    over.columns.push_back(Numbers("s", {0.5, 2}));
    Check(!Prepare(Spec(Kind::Calibration, "y", {"s"}), over).problem.empty(), "calibration refuses values above 1");
    // Residuals: RMSE, MAE and R squared.
    Source res;
    res.columns.push_back(Numbers("a", {1, 2, 3, 4}));
    res.columns.push_back(Numbers("f", {1, 3, 3, 3}));
    p = Prepare(Spec(Kind::Residuals, "a", {"f"}), res);
    Check(p.series[0].y == std::vector<double>({0, -1, 0, 1}) && std::fabs(p.metrics[0].second - std::sqrt(0.5)) < 1e-12 &&
              std::fabs(p.metrics[1].second - 0.5) < 1e-12 && std::fabs(p.metrics[2].second - 0.6) < 1e-12,
          "residuals: RMSE, MAE, R squared");
    // Learning curve: bands from the spread, the best validation point.
    Source lc;
    lc.columns.push_back(Numbers("n", {100, 200, 300, 400}));
    lc.columns.push_back(Numbers("train", {0.9, 0.88, 0.87, 0.86}));
    lc.columns.push_back(Numbers("val_loss", {0.6, 0.4, 0.35, 0.38}));
    lc.columns.push_back(Numbers("sd", {0.1, 0.1, 0.1, 0.1}));
    PlotSpec lcs = Spec(Kind::LearningCurve, "n", {"train", "val_loss"});
    lcs.spread_columns = {"", "sd"};
    p = Prepare(lcs, lc);
    Check(p.series.size() == 2 && p.series[0].low.empty() && p.series[1].low.size() == 4 && std::fabs(p.series[1].high[0] - 0.7) < 1e-12,
          "learning curve: band on the second curve");
    Check(p.best_series == 1 && p.best_index == 2, "learning curve: a loss is best at its lowest");
    lcs.best = PlotSpec::Best::Highest;
    Check(Prepare(lcs, lc).best_index == 0, "learning curve: highest when asked");
    // Importance: sorted largest first, top N.
    Source imp;
    imp.columns.push_back(Text("feature", {"a", "b", "c", "d"}));
    imp.columns.push_back(Numbers("gain", {0.1, 0.4, 0.2, 0.3}));
    PlotSpec imp_spec = Spec(Kind::Importance, "feature", {"gain"});
    imp_spec.top_n = 3;
    p = Prepare(imp_spec, imp);
    Check(p.categories == std::vector<std::string>({"b", "d", "c"}) && p.series[0].y == std::vector<double>({0.4, 0.3, 0.2}) &&
              p.label.state == DataLabel::State::Truncated, "importance: top 3, largest first");
    }

    // A fixed histogram range (dashboards align filtered bins with all rows).
    {
        Source hr;
        hr.columns.push_back(Numbers("v", {1, 2, 3, 4, 5, 6, 7, 8, 9, 30}));
        PlotSpec fixed_spec = Spec(Kind::Histogram, "v");
        fixed_spec.bins = 4;
        fixed_spec.range_lo = 0;
        fixed_spec.range_hi = 20;
        p = Prepare(fixed_spec, hr);
        Check(p.edges.size() == 5 && p.edges.front() == 0 && p.edges.back() == 20 && p.series[0].y == std::vector<double>({4, 5, 0, 0}),
              "fixed range: edges 0..20, 30 not counted");
        PlotSpec back;
        Check(SpecFromJson(SpecToJson(fixed_spec), back) && back.range_lo == 0 && back.range_hi == 20, "range round trip");
    }

    // ---- Flows, hierarchies and maps (P2b group 5) ----
    {
        // Sankey: 6 rows over two steps; flows add up and the layout stacks.
        Source sk;
        sk.columns.push_back(Text("from", {"a", "a", "a", "b", "b", "c"}));
        sk.columns.push_back(Text("to", {"x", "x", "y", "y", "y", "y"}));
        sk.columns.push_back(Numbers("w", {1, 1, 2, 1, 1, 4}));
        p = Prepare(Spec(Kind::Sankey, "", {"from", "to"}), sk);
        Check(p.problem.empty() && p.sankey_nodes.size() == 5 && p.sankey_links.size() == 4 && p.sankey_total == 6,
              "sankey: 5 nodes, 4 bands, 6 rows (" + p.problem + ")");
        Check(p.sankey_nodes[0].name == "a" && p.sankey_nodes[0].value == 3 && p.sankey_nodes[3].name == "y" && p.sankey_nodes[3].value == 4,
              "sankey: largest first per step");
        double into_y = 0;
        for (const auto& lk : p.sankey_links)
            if (p.sankey_nodes[static_cast<size_t>(lk.to)].name == "y") into_y += lk.value;
        Check(into_y == 4, "sankey: what flows into y is its value");
        const auto& ny = p.sankey_nodes[3];
        const auto& nx = p.sankey_nodes[4];
        Check(nx.name == "x" && std::fabs((ny.y1 - ny.y0) / (nx.y1 - nx.y0) - 2.0) < 1e-9 && nx.y0 > ny.y1, "sankey: heights follow the values, stacked");
        Check(std::fabs(p.sankey_links[0].y_from - p.sankey_nodes[0].y0) < 1e-12, "sankey: the first band starts at the top of its node");
        PlotSpec skw = Spec(Kind::Sankey, "", {"from", "to"});
        skw.value_column = "w";
        p = Prepare(skw, sk);
        Check(p.sankey_total == 10 && p.sankey_nodes[0].name == "a" && p.sankey_nodes[1].name == "c" && p.sankey_nodes[1].value == 4, "sankey: summed value, ties by name");
        skw.sankey_top = 1;
        p = Prepare(skw, sk);
        Check(p.sankey_nodes[1].name == "other" && p.sankey_nodes[1].value == 6, "sankey: the rest as other");
        Check(Prepare(Spec(Kind::Sankey, "", {"from"}), sk).problem == "Choose two or more step columns.", "sankey: one step refused");

        // Treemap: areas follow the sizes and fill the box.
        Source tm;
        tm.columns.push_back(Text("continent", {"A", "A", "B", "B", "B"}));
        tm.columns.push_back(Text("country", {"a1", "a2", "b1", "b2", "b3"}));
        tm.columns.push_back(Numbers("pop", {10, 30, 20, 25, 15}));
        tm.columns.push_back(Numbers("gdp", {1, 2, 3, 4, 5}));
        PlotSpec ts = Spec(Kind::Treemap, "", {"continent", "country"});
        ts.value_column = "pop";
        ts.color_column = "gdp";
        p = Prepare(ts, tm);
        Check(p.problem.empty() && p.tree_leaves.size() == 5 && p.tree_tops == std::vector<std::string>({"B", "A"}) && p.colour_scale &&
                  p.colour_min == 1 && p.colour_max == 5, "treemap: leaves, tops by size, colour scale (" + p.problem + ")");
        const auto flat = TreemapLayout(p, {}, 100, 60, 0.0);
        double leaf_area = 0;
        for (const auto& r : flat)
            if (r.leaf) {
                leaf_area += r.w * r.h;
                Check(r.x >= -1e-9 && r.y >= -1e-9 && r.x + r.w <= 100 + 1e-9 && r.y + r.h <= 60 + 1e-9, "treemap: inside the box");
            }
        Check(std::fabs(leaf_area - 6000) < 1e-6, "treemap: the leaves fill the box");
        for (const auto& r : flat)
            if (r.leaf && r.path.back() == "a2") Check(std::fabs(r.w * r.h - 6000.0 * 30 / 100) < 1e-6, "treemap: area follows size");
        const auto zoomed = TreemapLayout(p, {"A"}, 100, 60, 0.0);
        Check(zoomed.size() == 2 && zoomed[0].path.back() == "a2", "treemap: zoom into A");
        const auto sq = Squarify({6, 6, 4, 3, 2, 2, 1}, 0, 0, 6, 4);
        Check(std::fabs(sq[0].w * sq[0].h - 6) < 1e-9 && std::fabs(sq[6].w * sq[6].h - 1) < 1e-9, "squarify: the classic example");

        // Map regions: names and ISO codes, unmatched listed.
        Check(FindCountry("France") >= 0 && FindCountry("FRA") == FindCountry("france") && FindCountry("United States") == FindCountry("USA") &&
                  FindCountry("UK") == FindCountry("GBR") && FindCountry("Atlantis") < 0, "world map: names, codes, aliases");
        Check(CountryAt(2.35, 48.85) == FindCountry("France") && CountryAt(-30, 30) < 0, "world map: Paris is in France, the Atlantic is sea");
        Check(CountryAt(28.2, -29.5) == FindCountry("Lesotho"), "world map: Lesotho inside South Africa");
        Source rg;
        rg.columns.push_back(Text("country", {"France", "FRA", "Germany", "Atlantis", "Atlantis"}));
        rg.columns.push_back(Numbers("v", {1, 2, 5, 7, 8}));
        PlotSpec region_spec = Spec(Kind::MapRegions, "country", {"v"});
        p = Prepare(region_spec, rg);
        const size_t fr = static_cast<size_t>(FindCountry("France"));
        Check(p.problem.empty() && p.region_value[fr] == 3 && p.region_rows[fr] == 2 && p.unmatched.size() == 1 &&
                  p.unmatched[0].first == "Atlantis" && p.unmatched[0].second == 2, "regions: summed, unmatched listed (" + p.problem + ")");
        region_spec.region_agg = PlotSpec::RegionAgg::Mean;
        Check(Prepare(region_spec, rg).region_value[fr] == 1.5, "regions: mean");
        Source no_match;
        no_match.columns.push_back(Text("country", {"Atlantis"}));
        no_match.columns.push_back(Numbers("v", {1}));
        Check(Prepare(Spec(Kind::MapRegions, "country", {"v"}), no_match).problem.find("Atlantis") != std::string::npos, "regions: nothing matched");

        // Map points: sizes kept beside the points; out-of-range refused.
        Source mp;
        mp.columns.push_back(Numbers("lon", {2.35, -0.13, 13.4}));
        mp.columns.push_back(Numbers("lat", {48.85, 51.5, 52.5}));
        mp.columns.push_back(Numbers("mag", {3, 5, 4}));
        PlotSpec map_spec = Spec(Kind::MapPoints, "lon", {"lat"});
        map_spec.value_column = "mag";
        p = Prepare(map_spec, mp);
        Check(p.problem.empty() && p.series[0].z == std::vector<double>({3, 5, 4}) && p.size_min == 3 && p.size_max == 5, "map points: sizes");
        Check(Prepare(Spec(Kind::MapPoints, "lat", {"lon"}), mp).problem.empty(), "map points: swapped but in range is allowed");
        Source bad;
        bad.columns.push_back(Numbers("lon", {200}));
        bad.columns.push_back(Numbers("lat", {10}));
        Check(Prepare(Spec(Kind::MapPoints, "lon", {"lat"}), bad).problem.find("between -180 and 180") != std::string::npos, "map points: longitude range");
    }

    // Column summaries for the picker.
    ColumnSummary cs1 = SummarizeColumn(Numbers("pixel1", {0, 0, 0}));
    Check(cs1.OneValue() && cs1.Text() == "always 0", "a one-value column: " + cs1.Text());
    cs1 = SummarizeColumn(Numbers("pixel407", {0, 0, 255, 128}));
    Check(!cs1.OneValue() && cs1.distinct == 3 && cs1.Text() == "0 to 255 \xC2\xB7 50.0% not 0",
          "range and share not 0: " + cs1.Text());
    cs1 = SummarizeColumn(Text("name", names));
    Check(cs1.Text() == "2 values" && !cs1.numeric, "text: " + cs1.Text());
    Check(SummarizeColumn(Numbers("v", v)).distinct == kMaxColorGroups + 1, "distinct counted up to 13");
    std::cout << "plot prepare: 35 kinds, reduce, sample, colour groups, categories, box/violin, grids, problems, rows "
                 "(first, range, filter), colour scale and ranges, column summaries, KDE, matrix, hexbin, contours, grouped bars, polar, quiver, stream, image, pair plot, parallel, confusion, ROC, PR, calibration, residuals, learning curve, importance, sankey, treemap, map regions, map points. OK\n";
    return 0;
}
