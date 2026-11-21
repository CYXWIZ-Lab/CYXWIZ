// The one-click fix for missing values (TOFIX223 0.3, mockup approved
// 2026-11-21): the card both screens show and the steps each action adds.
// The main scenario is the mockup's: Patients CSV, 1,200 rows; age 37 empty,
// cholesterol 9 empty and 3 NaN, smoker (text) 5 empty; label diagnosis 4 empty.

#include "core/missing_values_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

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
                  Column("smoker", 5, 0, true, false), Column("diagnosis", 4, 0, true, true)};
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
        Check(!card.resolved, "not resolved before a fix");
        const Group& f = card.features;
        Check(f.present && f.title == "3 feature columns have missing values", "feature title: " + f.title);
        Check(f.subtitle == "Patients CSV · 1,200 rows", "feature subtitle: " + f.subtitle);
        Check(f.fix_header == "Fill would use", "header before a fix: " + f.fix_header);
        Check(f.rows.size() == 3 && f.rows[0].column == "age" && f.rows[0].missing == "37 empty" &&
                  f.rows[0].type == "number" && f.rows[0].fix == "mean of Train",
              "age row");
        Check(f.rows.size() == 3 && f.rows[1].missing == "9 empty, 3 NaN", "cholesterol counts empty and NaN apart");
        Check(f.rows.size() == 3 && f.rows[2].type == "text" && f.rows[2].fix == "most frequent of Train",
              "a text column fills with its most frequent value");
        Check(f.can_fill && f.can_drop && f.can_leave_out.empty(), "Fill and Drop; nothing mostly empty");

        // Fill: one step for the numbers, one for the text; the label is never filled.
        Check(f.fill_steps.size() == 2, "two Missing Value steps");
        const Step* numbers = Find(f.fill_steps, "Fill age, cholesterol (mean)");
        const Step* texts = Find(f.fill_steps, "Fill smoker (most frequent)");
        Check(numbers && numbers->type == gui::NodeType::FillMissingValues && numbers->params.at("columns") == "age,cholesterol" &&
                  numbers->params.at("strategy") == "mean" && numbers->params.at("label_col") == "diagnosis",
              "the numbers' step: columns, mean, the label kept out");
        Check(texts && texts->params.at("columns") == "smoker" && texts->params.at("strategy") == "mode",
              "the text step: most frequent");

        // Drop: one Row Filter over the three columns.
        Check(f.drop_steps.size() == 1 && f.drop_steps[0].type == gui::NodeType::FilterRows &&
                  f.drop_steps[0].name == "Keep rows with age, cholesterol and smoker" &&
                  f.drop_steps[0].params.at("condition") ==
                      "\"age\" IS NOT EMPTY AND \"cholesterol\" IS NOT EMPTY AND \"smoker\" IS NOT EMPTY",
              "drop: " + (f.drop_steps.empty() ? std::string("none") : f.drop_steps[0].params.at("condition")));

        const Group& l = card.label;
        Check(l.present && l.title == "4 rows have no label", "label title: " + l.title);
        Check(!l.can_fill && l.can_drop, "a label is only dropped (M-5)");
        Check(l.drop_steps.size() == 1 && l.drop_steps[0].name == "Keep rows with a diagnosis" &&
                  l.drop_steps[0].params.at("condition") == "\"diagnosis\" IS NOT EMPTY",
              "the label's Row Filter");

        // Fix and train: fill the features, drop the rows without a label.
        Check(card.fix_all_steps.size() == 3 && card.fix_all_steps[0].name == "Fill age, cholesterol (mean)" &&
                  card.fix_all_steps[2].name == "Keep rows with a diagnosis",
              "Fix and train adds the two fills and the label's filter");
        Check(card.fix_all_hint.find("drops the rows without a label") != std::string::npos, card.fix_all_hint);
    }

    // The numbers' value: median and 0.
    {
        Input in = Patients();
        in.number_fill = NumberFill::Median;
        const Card median = BuildCard(in);
        const Step* step = Find(median.features.fill_steps, "Fill age, cholesterol (median)");
        Check(step && step->params.at("strategy") == "median" && median.features.rows[0].fix == "median of Train",
              "median");
        in.number_fill = NumberFill::Zero;
        const Card zero = BuildCard(in);
        step = Find(zero.features.fill_steps, "Fill age, cholesterol (0)");
        Check(step && step->params.at("strategy") == "constant" && step->params.at("value") == "0" &&
                  zero.features.rows[0].fix == "0",
              "0 is a constant fill");
    }

    // After the actions: the done texts and the resolved verdict.
    {
        Input in = Patients();
        in.features = Done::Filled;
        Card card = BuildCard(in);
        Check(card.features.resolved && !card.label.resolved && !card.resolved, "features filled, label still open");
        Check(card.features.fix_header == "Now", "header after a fix");
        Check(card.features.done_text ==
                  "Added \"Fill age, cholesterol (mean)\" and \"Fill smoker (most frequent)\" (Missing Value steps).",
              "fill done: " + card.features.done_text);
        in.label = Done::Dropped;
        card = BuildCard(in);
        Check(card.resolved && card.resolved_verdict == "Fixed: 3 steps added after Patients CSV",
              "resolved: " + card.resolved_verdict);
        Check(card.label.done_text == "Added \"Keep rows with a diagnosis\" (Row Filter: \"diagnosis\" IS NOT EMPTY).",
              "label done: " + card.label.done_text);
        in.features = Done::Dropped;
        card = BuildCard(in);
        Check(card.resolved_verdict == "Fixed: 2 steps added after Patients CSV" &&
                  card.features.rows[0].fix == "rows dropped" &&
                  card.features.done_text ==
                      "Added \"Keep rows with age, cholesterol and smoker\" (Row Filter: each IS NOT EMPTY).",
              "drop done: " + card.features.done_text);
    }

    // A supplied Validation input: Drop only (M-6).
    {
        Input in = Patients();
        in.supplied_role = true;
        in.role_name = "Validation";
        in.input_name = "Patients validation CSV";
        const Card card = BuildCard(in);
        Check(!card.features.can_fill && card.features.can_drop, "a supplied input offers Drop only");
        Check(card.features.note.find("Fill is not offered here") != std::string::npos, card.features.note);
        Check(card.fix_all_steps.size() == 2 && card.fix_all_steps[0].type == gui::NodeType::FilterRows,
              "Fix and train drops on a supplied input");
    }

    // A supplied input training reads from its own file: a step after it would
    // not apply, so the card offers no step, only the file.
    {
        Input in = Patients();
        in.supplied_role = true;
        in.role_name = "Test";
        in.input_name = "Patients test CSV";
        in.steps_reach_input = false;
        const Card card = BuildCard(in);
        Check(!card.features.can_fill && !card.features.can_drop && !card.label.can_drop,
              "no step is offered where a step would not reach training");
        Check(card.fix_all_steps.empty(), "Fix and train has nothing to add there");
        Check(card.features.note.find("in the file") != std::string::npos &&
                  card.label.note.find("in the file") != std::string::npos,
              "the notes say to fix the file: " + card.features.note);
    }

    // A column more than half empty can be left out (M-7); once left out it is
    // not filled or dropped.
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
        Check(Thousands(2500000) == "2,500,000" && Thousands(1200) == "1,200" && Thousands(999) == "999",
              "thousands");
    }

    if (failures == 0) std::cout << "test_missing_values_presentation: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
