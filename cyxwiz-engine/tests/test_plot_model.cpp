// Plot model (TOFIX134 P1 step 1.2): kind registry, spec JSON round trip,
// missing-encoding messages, data label wording, column statistics.
#include "../src/core/plot/plot_model.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
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
}  // namespace

int main() {
    // Registry: unique ids, every kind reachable by its id, groups in order.
    std::set<std::string> ids;
    for (const auto& k : Kinds()) {
        Check(ids.insert(k.id).second, std::string("unique id ") + k.id);
        Check(FindKind(k.id) && FindKind(k.id)->kind == k.kind, std::string("find by id ") + k.id);
        Check(k.required != 0, std::string("kind needs at least one column: ") + k.id);
    }
    Check(Kinds().size() == 21, "21 kinds (P1 13 + P2b groups 1 and 2)");
    Check(!FindKind("sankey"), "unknown kind not found");
    Check(std::string(Info(Kind::Histogram).label) == "Histogram" && Info(Kind::Histogram).group == Group::Basic,
          "histogram is a basic kind");
    Check(std::string(GroupLabel(Group::GridDensity)) == "Grid and density", "group label");

    // Spec JSON round trip, including every field.
    PlotSpec s;
    s.kind = Kind::Histogram;
    s.x_column = "loss";
    s.y_columns = {"a", "b"};
    s.color_column = "label";
    s.title = "Histogram of \"loss\"";
    s.bins = 40;
    s.smooth = 7;
    s.density = true;
    s.show_mean = s.show_median = s.log_y = true;
    s.legend = false;
    PlotSpec back;
    Check(SpecFromJson(SpecToJson(s), back), "round trip parses");
    Check(back.kind == Kind::Histogram && back.x_column == "loss" && back.y_columns.size() == 2 &&
              back.color_column == "label" && back.title == s.title && back.bins == 40 && back.smooth == 7 &&
              back.density && back.show_mean && back.show_median && back.log_y && !back.log_x && !back.legend,
          "round trip keeps every field");
    std::string problem;
    PlotSpec untouched;
    Check(!SpecFromJson("{\"version\":2,\"kind\":\"line\"}", untouched, &problem) && problem.find("version 2") != std::string::npos,
          "a newer version is refused with a reason");
    Check(!SpecFromJson("{\"version\":1,\"kind\":\"sankey\"}", untouched, &problem) && problem.find("sankey") != std::string::npos,
          "an unknown kind is refused with its name");
    Check(!SpecFromJson("not json", untouched, &problem), "text that is not JSON is refused");
    Check(SpecFromJson("{\"version\":1,\"kind\":\"scatter\",\"bins\":0,\"extra\":1}", back) && back.bins == 1 &&
              back.legend,
          "out-of-range values clamped, missing keys default, unknown keys ignored");
    Check(back.rows == RowMode::All && back.conditions.empty() && back.color_mode == ColourMode::Auto,
          "a spec without rows plots all rows, colour mode auto");

    // Rows and colour mode (P2 board 6) round trip; bad values are refused.
    PlotSpec r;
    r.kind = Kind::Scatter;
    r.rows = RowMode::Filter;
    r.conditions = {{"class", "=", "7"}, {"pixel407", ">", "0"}};
    r.first_rows = 250;
    r.row_from = 10000;
    r.row_to = 10999;
    r.color_mode = ColourMode::Scale;
    Check(SpecFromJson(SpecToJson(r), back), "rows round trip parses");
    Check(back.rows == RowMode::Filter && back.conditions.size() == 2 && back.conditions[1].column == "pixel407" &&
              back.conditions[1].op == ">" && back.conditions[1].value == "0" && back.first_rows == 250 &&
              back.row_from == 10000 && back.row_to == 10999 && back.color_mode == ColourMode::Scale,
          "rows, conditions and colour mode kept");
    Check(ConditionsText(r.conditions) == "class = 7 and pixel407 > 0", "conditions in words");
    Check(!SpecFromJson("{\"version\":1,\"kind\":\"line\",\"rows\":{\"mode\":\"filter\",\"conditions\":[{\"column\":\"a\",\"op\":\"~\"}]}}",
                        untouched, &problem) &&
              problem.find("'~'") != std::string::npos,
          "an unknown condition is refused");
    Check(!SpecFromJson("{\"version\":1,\"kind\":\"line\",\"rows\":{\"mode\":\"some\"}}", untouched, &problem) &&
              problem.find("'some'") != std::string::npos,
          "an unknown row selection is refused");
    Check(SpecFromJson("{\"version\":1,\"kind\":\"line\",\"rows\":{\"mode\":\"range\",\"from\":0,\"to\":0}}", back) &&
              back.row_from == 1 && back.row_to == 1,
          "a range starts at row 1 and never ends before it starts");

    // What is missing.
    PlotSpec scatter;
    scatter.kind = Kind::Scatter;
    Check(MissingEncoding(scatter) == "Choose X values.", "scatter asks for X first");
    scatter.x_column = "epoch";
    Check(MissingEncoding(scatter) == "Choose Y values.", "then Y");
    scatter.y_columns = {"loss"};
    Check(MissingEncoding(scatter).empty(), "complete");
    PlotSpec line;
    line.kind = Kind::Line;
    line.y_columns = {"loss"};
    Check(MissingEncoding(line).empty(), "a line needs no X (row number)");

    // Data label wording (the mockup's four states).
    DataLabel label;
    label.shown = label.total = 2000;
    Check(label.Text() == "exact \xC2\xB7 all 2,000 values", "exact: " + label.Text());
    label = {DataLabel::State::Reduced, 4000, 120000};
    Check(label.Text() == "reduced \xC2\xB7 4,000 of 120,000 points", "reduced: " + label.Text());
    label = {DataLabel::State::Sampled, 50000, 1200000};
    Check(label.Text() == "sampled \xC2\xB7 50,000 of 1,200,000 rows", "sampled: " + label.Text());
    label = {DataLabel::State::Truncated, 100000, 0};
    Check(label.Text() == "first 100,000 rows", "truncated, total unknown: " + label.Text());
    label.total = 250000;
    Check(label.Text() == "first 100,000 of 250,000 rows", "truncated, total known: " + label.Text());
    label = {DataLabel::State::Exact, 7293, 7293, "filtered \xC2\xB7 7,293 of 70,000 rows"};
    Check(label.Text() == "filtered \xC2\xB7 7,293 of 70,000 rows", "a selection says it alone when exact: " + label.Text());
    label.state = DataLabel::State::Sampled;
    label.shown = 5000;
    Check(label.Text() == "filtered \xC2\xB7 7,293 of 70,000 rows \xC2\xB7 sampled \xC2\xB7 5,000 of 7,293 rows",
          "then the sampling: " + label.Text());
    Check(Thousands(-1234567) == "-1,234,567" && Thousands(999) == "999", "thousands");

    // Statistics skip NaN and infinity and count them as missing.
    const ColumnStats st = Summarize({4, 1, NAN, 3, 2, INFINITY});
    Check(st.count == 4 && st.missing == 2, "count and missing");
    Check(st.min == 1 && st.max == 4 && st.mean == 2.5 && st.median == 2.5, "min max mean median");
    Check(std::fabs(st.q1 - 1.75) < 1e-12 && std::fabs(st.q3 - 3.25) < 1e-12, "quartiles (linear)");
    Check(Summarize({}).count == 0, "empty column");
    std::cout << "plot model: 21 kinds, spec JSON round trip and refusals, rows and colour mode, missing columns, labels, "
                 "stats. OK\n";
    return 0;
}
