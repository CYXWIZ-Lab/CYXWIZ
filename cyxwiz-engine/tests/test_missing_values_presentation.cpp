// The one-click fix for missing values (TOFIX223 0.3, mockup approved
// 2026-11-21): the card both screens show and the steps each action adds.
// The main scenario is the mockup's (training reads number features only, so
// the mockup's text column is a number here): Patients CSV, 1,200 rows; age 37
// empty, cholesterol 9 empty and 3 NaN, bmi 5 empty; label diagnosis 4 empty.

#include "core/column_list.h"
#include "core/missing_values_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

using namespace cyxwiz::missing_values_fix;

cyxwiz::MissingColumn Column(const std::string& name, int64_t rows, int64_t nan, bool text, bool label) {
    cyxwiz::MissingColumn c;
    c.name = name;
    c.rows = rows;
    c.nan_rows = nan;
    c.text = text;
    c.label = label;
    return c;
}

Input Patients() {
    Input in;
    in.input_name = "Patients CSV";
    in.total_rows = 1200;
    in.columns = {Column("age", 37, 0, false, false), Column("cholesterol", 12, 3, false, false),
                  Column("bmi", 5, 0, false, false), Column("diagnosis", 4, 0, true, true)};
    return in;
}

const Step* Find(const std::vector<Step>& steps, const std::string& name) {
    for (const auto& s : steps)
        if (s.name == name) return &s;
    return nullptr;
}

}  // namespace

int main() {
    // Before any fix.
    {
        const Card card = BuildCard(Patients());
        Check(card.verdict == "Training did not start: 4 columns have missing values", "verdict: " + card.verdict);
        Check(!card.resolved && card.open_columns == 4 && card.input_name == "Patients CSV", "not resolved before a fix");
        Check(GroupCount(card) == 2, "two groups");
        const Group& f = card.features;
        Check(f.present && f.title == "3 feature columns have missing values", "feature title: " + f.title);
        Check(f.subtitle == "Patients CSV · 1,200 rows", "feature subtitle: " + f.subtitle);
        Check(f.fix_header == "Fill would use", "header before a fix: " + f.fix_header);
        Check(f.rows.size() == 3 && f.rows[0].column == "age" && f.rows[0].missing == "37 empty" &&
                  f.rows[0].type == "number" && f.rows[0].fix == "mean of Train",
              "age row");
        Check(f.rows.size() == 3 && f.rows[1].missing == "9 empty, 3 NaN", "cholesterol counts empty and NaN apart");
        Check(f.can_fill && f.can_drop && f.can_leave_out.empty(), "Fill and Drop; nothing mostly empty");

        // Fill: one Missing Value step over the number columns; the label is never filled.
        Check(f.fill_steps.size() == 1, "one Missing Value step");
        const Step* numbers = Find(f.fill_steps, "Fill age, cholesterol, bmi (mean)");
        Check(numbers && numbers->type == gui::NodeType::FillMissingValues &&
                  numbers->params.at("columns") == "age,cholesterol,bmi" && numbers->params.at("strategy") == "mean" &&
                  numbers->params.at("label_col") == "diagnosis",
              "the fill step: columns, mean, the label kept out");
        Check(f.note.find("most frequent") == std::string::npos, "no text fill is promised: " + f.note);

        // Drop: one Row Filter over the three columns.
        Check(f.drop_steps.size() == 1 && f.drop_steps[0].type == gui::NodeType::FilterRows &&
                  f.drop_steps[0].name == "Keep rows with age, cholesterol and bmi" &&
                  f.drop_steps[0].params.at("condition") ==
                      "\"age\" IS NOT EMPTY AND \"cholesterol\" IS NOT EMPTY AND \"bmi\" IS NOT EMPTY",
              "drop: " + (f.drop_steps.empty() ? std::string("none") : f.drop_steps[0].params.at("condition")));

        const Group& l = card.label;
        Check(l.present && l.title == "4 rows have no label", "label title: " + l.title);
        Check(!l.can_fill && l.can_drop, "a label is only dropped (M-5)");
        Check(l.drop_steps.size() == 1 && l.drop_steps[0].name == "Keep rows with a diagnosis" &&
                  l.drop_steps[0].params.at("condition") == "\"diagnosis\" IS NOT EMPTY",
              "the label's Row Filter");

        // Fix and train: fill the features, then drop the rows without a label (the card's order).
        Check(card.fix_all_steps.size() == 2 && card.fix_all_steps[0].name == "Fill age, cholesterol, bmi (mean)" &&
                  card.fix_all_steps[1].name == "Keep rows with a diagnosis",
              "Fix and train adds the fill, then the label's filter");
        Check(card.fix_all_hint == "Fills the feature columns with the chosen value and drops the rows without a "
                                   "label, then trains.",
              card.fix_all_hint);
    }

    // The numbers' value: median and 0.
    {
        Input in = Patients();
        in.number_fill = NumberFill::Median;
        const Card median = BuildCard(in);
        const Step* step = Find(median.features.fill_steps, "Fill age, cholesterol, bmi (median)");
        Check(step && step->params.at("strategy") == "median" && median.features.rows[0].fix == "median of Train",
              "median");
        in.number_fill = NumberFill::Zero;
        const Card zero = BuildCard(in);
        step = Find(zero.features.fill_steps, "Fill age, cholesterol, bmi (0)");
        Check(step && step->params.at("strategy") == "constant" && step->params.at("value") == "0" &&
                  zero.features.rows[0].fix == "0",
              "0 is a constant fill");
    }

    // After the actions: the done texts, the resolved verdict, and Fix and
    // train adds only what is still open.
    {
        Input in = Patients();
        in.features = Done::Filled;
        Card card = BuildCard(in);
        Check(card.features.resolved && !card.label.resolved && !card.resolved && card.open_columns == 1,
              "features filled, label still open");
        Check(card.features.fix_header == "Now", "header after a fix");
        Check(card.features.done_text == "Added \"Fill age, cholesterol, bmi (mean)\" (Missing Value step).",
              "fill done: " + card.features.done_text);
        Check(card.fix_all_steps.size() == 1 && card.fix_all_steps[0].name == "Keep rows with a diagnosis",
              "Fix and train adds the open group's step only");
        in.label = Done::Dropped;
        card = BuildCard(in);
        Check(card.resolved && card.resolved_verdict == "Fixed: 2 steps added after Patients CSV",
              "resolved: " + card.resolved_verdict);
        Check(card.label.done_text == "Added \"Keep rows with a diagnosis\" (Row Filter: \"diagnosis\" IS NOT EMPTY).",
              "label done: " + card.label.done_text);
        in.features = Done::Dropped;
        card = BuildCard(in);
        Check(card.resolved_verdict == "Fixed: 2 steps added after Patients CSV" &&
                  card.features.rows[0].fix == "rows dropped" &&
                  card.features.done_text ==
                      "Added \"Keep rows with age, cholesterol and bmi\" (Row Filter: each IS NOT EMPTY).",
              "drop done: " + card.features.done_text);
    }

    // A supplied Validation / Test input is read from its own Data Input: no
    // step, only the file (M-6, owner 2026-11-21), and nothing to leave out there.
    {
        Input in = MakeInput(Patients().columns, 100, "Patients test CSV", "Test");
        Check(in.supplied_role && in.role_name == "Test" && !in.steps_reach_input, "a supplied input reaches no step");
        in.columns.push_back(Column("mostly", 80, 0, false, false));
        const Card card = BuildCard(in);
        Check(!card.features.can_fill && !card.features.can_drop && !card.label.can_drop,
              "no step is offered where a step would not reach training");
        Check(card.fix_all_steps.empty(), "Fix and train has nothing to add there");
        Check(card.features.can_leave_out.empty(), "no Leave out on a supplied input");
        Check(card.features.note.find("in the file") != std::string::npos &&
                  card.label.note.find("in the file") != std::string::npos && card.features.rows[0].fix == "fix the file",
              "the notes say to fix the file: " + card.features.note);
        const Input train = MakeInput(Patients().columns, 100, "");
        Check(!train.supplied_role && train.steps_reach_input && train.input_name == "the training data",
              "the Train input reaches its steps");
    }

    // A column more than half empty can be left out (M-7); once left out it is
    // not filled or dropped; with every missing column left out the card is fixed.
    {
        Input in;
        in.input_name = "Sensors";
        in.total_rows = 100;
        in.columns = {Column("pressure", 70, 0, false, false), Column("temp", 2, 2, false, false)};
        Card card = BuildCard(in);
        Check(card.features.can_leave_out == std::vector<std::string>{"pressure"} &&
                  card.features.rows[0].mostly_missing && !card.features.rows[1].mostly_missing,
              "pressure is mostly empty");
        Check(card.features.note.find("more than half empty") != std::string::npos, card.features.note);
        Check(!card.label.present && card.label.resolved, "no label group when the label is complete");
        Check(card.verdict == "Training did not start: 2 columns have missing values", card.verdict);
        in.left_out = {"pressure"};
        card = BuildCard(in);
        Check(card.features.rows.size() == 1 && card.features.rows[0].column == "temp" &&
                  card.features.title == "1 feature column has missing values" &&
                  card.features.fill_steps.size() == 1 && card.features.fill_steps[0].name == "Fill temp (mean)" &&
                  card.features.rows[0].missing == "2 NaN",
              "left out: only temp remains");
        Check(card.fix_all_hint.rfind("Fills", 0) == 0 && card.fix_all_steps.size() == 1, card.fix_all_hint);

        in.columns = {Column("pressure", 70, 0, false, false)};
        card = BuildCard(in);
        Check(card.resolved && !card.features.present && card.open_columns == 0,
              "every missing column left out: fixed, not open forever");
        Check(card.resolved_verdict == "Fixed: pressure is left out of the features", card.resolved_verdict);
    }

    // A regression target of width 2: every target column is a label, never
    // filled; one Row Filter drops the rows missing any of them.
    {
        Input in;
        in.input_name = "Forecast";
        in.total_rows = 500;
        in.columns = {Column("load", 3, 0, false, false), Column("y", 2, 0, false, true),
                      Column("y_1", 4, 4, false, true)};
        const Card card = BuildCard(in);
        Check(card.features.rows.size() == 1 && card.features.rows[0].column == "load", "y_1 is not a feature");
        Check(card.features.fill_steps.size() == 1 && card.features.fill_steps[0].params.at("columns") == "load",
              "the fill never names a target");
        Check(card.label.present && card.label.title == "2 target columns have missing values" &&
                  card.label.rows.size() == 2 && card.label.rows[1].column == "y_1" &&
                  card.label.rows[1].missing == "4 NaN",
              "the targets' group: " + card.label.title);
        Check(card.label.drop_steps.size() == 1 && card.label.drop_steps[0].name == "Keep rows with y and y_1" &&
                  card.label.drop_steps[0].params.at("condition") == "\"y\" IS NOT EMPTY AND \"y_1\" IS NOT EMPTY",
              "one Row Filter over every target");
        Check(card.verdict == "Training did not start: 3 columns have missing values", card.verdict);
        Check(card.fix_all_steps.size() == 2 && card.fix_all_steps[1].name == "Keep rows with y and y_1",
              "Fix and train drops the rows without every target");
    }

    // Column names with a comma, edge spaces or a leading quote: the Missing
    // Value step's list quotes them, so Fill is offered as for any name.
    {
        Input in;
        in.input_name = "Odd";
        in.total_rows = 10;
        in.columns = {Column("x, y", 2, 0, false, false), Column(" age", 1, 0, false, false),
                      Column("\"q\"", 1, 0, false, false)};
        const Card card = BuildCard(in);
        Check(card.features.can_fill && card.features.can_drop && card.features.fill_steps.size() == 1,
              "Fill for names with a comma or edge spaces");
        const std::string list = card.features.fill_steps[0].params.at("columns");
        Check(list == "\"x, y\",\" age\",\"\"\"q\"\"\"", "the fill lists the names quoted: " + list);
        Check(cyxwiz::ParseColumnList(list) == std::vector<std::string>{"x, y", " age", "\"q\""},
              "the step reads the same names back");
        Check(card.features.note.rfind("Fill learns its values from the Train rows only", 0) == 0, card.features.note);
        Check(card.features.rows[0].fix == "mean of Train", card.features.rows[0].fix);
        Check(card.features.drop_steps[0].params.at("condition") ==
                  "\"x, y\" IS NOT EMPTY AND \" age\" IS NOT EMPTY AND \"\"\"q\"\"\" IS NOT EMPTY",
              "the Row Filter quotes the names: " + card.features.drop_steps[0].params.at("condition"));
        Check(card.fix_all_steps.size() == 1 && card.fix_all_steps[0].type == gui::NodeType::FillMissingValues &&
                  card.fix_all_hint.rfind("Fills the feature columns", 0) == 0,
              "Fix and train fills them");
    }

    // The column list: names read back as written, quoted where needed.
    {
        using cyxwiz::FormatColumnList;
        using cyxwiz::ParseColumnList;
        using Names = std::vector<std::string>;
        Check(ParseColumnList(" age , sex,,bmi ") == Names{"age", "sex", "bmi"}, "plain names are trimmed");
        Check(ParseColumnList("") == Names{} && ParseColumnList(" , ") == Names{}, "an empty list has no names");
        Check(ParseColumnList("\"a,b\", c") == Names{"a,b", "c"}, "a quoted name keeps its comma");
        Check(ParseColumnList("\" pad \"") == Names{" pad "}, "a quoted name keeps its spaces");
        Check(ParseColumnList("\"say \"\"hi\"\"\"") == Names{"say \"hi\""}, "a doubled quote is one quote");
        Check(ParseColumnList("x\"y, z") == Names{"x\"y", "z"}, "a quote inside a plain name is kept");
        Check(ParseColumnList("\"open, end") == Names{"open, end"}, "an open quote runs to the end");
        const Names odd{"age", "a,b", " lead", "trail ", "\"q", "x\"y", "ok name"};
        Check(ParseColumnList(FormatColumnList(odd)) == odd, "every name comes back: " + FormatColumnList(odd));
        Check(FormatColumnList({"age", "sex"}) == "age,sex", "plain names are not quoted");
        // A column with an empty name is "" and comes back (an empty entry does not).
        Check(FormatColumnList({"a", ""}) == "a,\"\"" && ParseColumnList("a,\"\"") == Names{"a", ""},
              "an empty name is quoted and comes back");
        Check(ParseColumnList(FormatColumnList({""})) == Names{""}, "an empty name alone comes back");
        Check(FormatColumnList({"a", "b, c"}, ", ") == "a, \"b, c\"" &&
                  ParseColumnList(FormatColumnList({"a", "b, c"}, ", ")) == Names{"a", "b, c"},
              "the list can be written with spaces after the commas");
    }

    // Only the label; and counts read with thousands.
    {
        Input in;
        in.input_name = "Big";
        in.total_rows = 2500000;
        cyxwiz::MissingColumn label = Column("y", 1, 0, false, true);
        label.at_least = true;
        in.columns = {label};
        const Card card = BuildCard(in);
        Check(!card.features.present && card.features.resolved && card.label.present, "label only");
        Check(card.label.title == "At least 1 row has no label", card.label.title);
        Check(card.fix_all_hint == "Drops the rows without a label, then trains.", card.fix_all_hint);
        Check(card.copy_lines.size() == 3, "copy lines: verdict, the label, the fix");
    }

    if (failures == 0) std::cout << "test_missing_values_presentation: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
