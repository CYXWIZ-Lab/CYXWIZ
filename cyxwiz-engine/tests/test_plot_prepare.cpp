// Plot data preparation (TOFIX134 P1 step 1.3): every P1 kind from table
// columns, with reduction, sampling, colour groups and honest labels.
#include "../src/core/plot/plot_prepare.h"
#include "../src/core/plot/plot_presets.h"

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

    // Column summaries for the picker.
    ColumnSummary cs1 = SummarizeColumn(Numbers("pixel1", {0, 0, 0}));
    Check(cs1.OneValue() && cs1.Text() == "always 0", "a one-value column: " + cs1.Text());
    cs1 = SummarizeColumn(Numbers("pixel407", {0, 0, 255, 128}));
    Check(!cs1.OneValue() && cs1.distinct == 3 && cs1.Text() == "0 to 255 \xC2\xB7 50.0% not 0",
          "range and share not 0: " + cs1.Text());
    cs1 = SummarizeColumn(Text("name", names));
    Check(cs1.Text() == "2 values" && !cs1.numeric, "text: " + cs1.Text());
    Check(SummarizeColumn(Numbers("v", v)).distinct == kMaxColorGroups + 1, "distinct counted up to 13");
    std::cout << "plot prepare: 13 kinds, reduce, sample, colour groups, categories, box/violin, grids, problems, rows "
                 "(first, range, filter), colour scale and ranges, column summaries. OK\n";
    return 0;
}
