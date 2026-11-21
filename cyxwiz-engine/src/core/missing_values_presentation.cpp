#include "missing_values_presentation.h"

#include "data_input_presentation.h"
#include "filter_condition.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::missing_values_fix {

namespace {

using data_input_view::FormatCount;

// "age", "age and sex", "age, sex and bmi", "age, sex, bmi and 2 more".
std::string NameList(const std::vector<std::string>& names, size_t shown = 3) {
    if (names.empty()) return "";
    if (names.size() > shown + 1) {
        std::string out;
        for (size_t i = 0; i < shown; ++i) out += (i ? ", " : "") + names[i];
        return out + " and " + std::to_string(names.size() - shown) + " more";
    }
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out += i + 1 == names.size() ? " and " : ", ";
        out += names[i];
    }
    return out;
}

// A fill step's name: "age, cholesterol", "a, b, c and 2 more".
std::string CommaNames(const std::vector<std::string>& names, size_t shown = 3) {
    std::string out;
    for (size_t i = 0; i < names.size() && i < shown; ++i) out += (i ? ", " : "") + names[i];
    if (names.size() > shown) out += " and " + std::to_string(names.size() - shown) + " more";
    return out;
}

// The comma list the Missing Value step reads (its `columns`).
std::string CommaList(const std::vector<std::string>& names) {
    std::string out;
    for (const auto& n : names) out += (out.empty() ? "" : ",") + n;
    return out;
}

// A name that list cannot carry: it splits on commas and trims each name.
bool FitsCommaList(const std::string& name) {
    return !name.empty() && name.find(',') == std::string::npos &&
           !std::isspace(static_cast<unsigned char>(name.front())) &&
           !std::isspace(static_cast<unsigned char>(name.back()));
}

std::string Plural(int64_t n, const char* one, const char* many) {
    return FormatCount(n) + " " + (n == 1 ? one : many);
}

std::string MissingText(const MissingColumn& c) {
    const int64_t empty = c.rows - c.nan_rows;
    std::string text = c.at_least ? "at least " : "";
    if (empty > 0 && c.nan_rows > 0) return text + FormatCount(empty) + " empty, " + FormatCount(c.nan_rows) + " NaN";
    if (c.nan_rows > 0) return text + FormatCount(c.nan_rows) + " NaN";
    return text + FormatCount(empty) + " empty";
}

std::string QuotedSteps(const std::vector<Step>& steps) {
    std::vector<std::string> names;
    for (const auto& s : steps) names.push_back("\"" + s.name + "\"");
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out += i + 1 == names.size() ? " and " : ", ";
        out += names[i];
    }
    return out;
}

Step KeepRowsWith(const std::vector<std::string>& columns, const std::string& name) {
    std::vector<FilterRow> rows;
    for (const auto& column : columns) {
        FilterRow row;
        row.column = column;
        row.op = FilterOp::IsNotEmpty;
        rows.push_back(std::move(row));
    }
    Step step;
    step.type = gui::NodeType::FilterRows;
    step.name = name;
    step.params["condition"] = BuildFilterCondition(rows);
    return step;
}

}  // namespace

const char* NumberFillWord(NumberFill fill) {
    switch (fill) {
        case NumberFill::Mean: return "mean";
        case NumberFill::Median: return "median";
        case NumberFill::Zero: return "0";
    }
    return "mean";
}

Input MakeInput(const std::vector<MissingColumn>& columns, int64_t total_rows, const std::string& input_name,
                const std::string& supplied_role) {
    Input in;
    in.columns = columns;
    in.total_rows = total_rows;
    in.input_name = input_name.empty() ? "the training data" : input_name;
    in.supplied_role = !supplied_role.empty();
    in.role_name = supplied_role;
    // A supplied Validation / Test is read from its own Data Input: a step
    // after it does not reach training (M-6, owner 2026-11-21: fix the file).
    in.steps_reach_input = !in.supplied_role;
    return in;
}

size_t GroupCount(const Card& card) { return (card.features.present ? 1 : 0) + (card.label.present ? 1 : 0); }

Card BuildCard(const Input& in) {
    Card card;
    card.input_name = in.input_name;
    std::vector<const MissingColumn*> features, labels;
    bool any_missing = false;
    for (const auto& c : in.columns) {
        if (c.rows <= 0) continue;
        any_missing = true;
        if (c.label) {
            labels.push_back(&c);
            continue;
        }
        if (std::find(in.left_out.begin(), in.left_out.end(), c.name) != in.left_out.end()) continue;
        features.push_back(&c);
    }
    const std::string where = in.supplied_role ? "the supplied " + in.role_name + " data (" + in.input_name + ")"
                                               : in.input_name;
    std::vector<std::string> label_names;
    for (const auto* c : labels) label_names.push_back(c->name);

    // Feature columns (training reads number columns only).
    Group& f = card.features;
    f.present = !features.empty();
    if (f.present) {
        const size_t n = features.size();
        f.title = std::to_string(n) + (n == 1 ? " feature column has" : " feature columns have") + " missing values";
        f.subtitle = in.input_name + (in.total_rows >= 0 ? " · " + Plural(in.total_rows, "row", "rows") : "");
        if (in.supplied_role) f.subtitle += " · supplied " + in.role_name;
        std::vector<std::string> names, unlisted;
        int64_t most = 0;
        for (const auto* c : features) {
            names.push_back(c->name);
            if (!FitsCommaList(c->name)) unlisted.push_back("'" + c->name + "'");
        }
        f.can_drop = in.steps_reach_input;
        f.can_fill = in.steps_reach_input && unlisted.empty();
        f.fix_header = in.features != Done::No ? "Now" : f.can_fill ? "Fill would use" : "Fix";
        for (const auto* c : features) {
            Row row;
            row.column = c->name;
            row.type = c->text ? "text" : "number";
            row.missing = MissingText(*c);
            row.mostly_missing = in.total_rows > 0 && c->rows * 2 > in.total_rows;
            if (row.mostly_missing && !in.supplied_role) f.can_leave_out.push_back(c->name);
            if (in.features == Done::Dropped) row.fix = "rows dropped";
            else if (!in.steps_reach_input) row.fix = "fix the file";
            else if (!f.can_fill) row.fix = "drop the rows";
            else row.fix = in.number_fill == NumberFill::Zero ? "0" : std::string(NumberFillWord(in.number_fill)) + " of Train";
            f.rows.push_back(std::move(row));
            most += c->rows;
        }
        const std::string fill_word = NumberFillWord(in.number_fill);
        Step fill;
        fill.type = gui::NodeType::FillMissingValues;
        fill.name = "Fill " + CommaNames(names) + " (" + fill_word + ")";
        fill.params["columns"] = CommaList(names);
        fill.params["strategy"] = in.number_fill == NumberFill::Zero ? "constant" : fill_word;
        if (in.number_fill == NumberFill::Zero) fill.params["value"] = "0";
        if (!label_names.empty()) fill.params["label_col"] = label_names.front();
        if (f.can_fill) f.fill_steps.push_back(std::move(fill));
        f.drop_steps.push_back(KeepRowsWith(names, "Keep rows with " + NameList(names)));
        if (!in.steps_reach_input) {
            f.note = "Training reads " + where + " as its file holds it, so a step here would not apply: fill or "
                     "remove these cells in the file, then Apply its Data Input again.";
        } else if (!f.can_fill) {
            f.note = "Fill is not offered: the Missing Value step lists its columns separated by commas, and " +
                     NameList(unlisted) + (unlisted.size() == 1 ? " has" : " have") +
                     " a comma or spaces at the ends of the name. Drop removes every row with an empty cell or NaN "
                     "in these columns (at most " + FormatCount(most) + "), or rename the column in the file.";
        } else {
            f.note = "Fill learns its values from the Train rows only. Drop removes every row with an empty cell or "
                     "NaN in these columns (at most " + FormatCount(most) + ").";
        }
        if (!f.can_leave_out.empty())
            f.note += " " + NameList(f.can_leave_out) + (f.can_leave_out.size() == 1 ? " is" : " are") +
                      " more than half empty: filling would make up most of " +
                      (f.can_leave_out.size() == 1 ? "its" : "their") + " values; it can be left out of the "
                      "features instead (Actions).";
        f.resolved = in.features != Done::No;
        if (in.features == Done::Filled && !f.fill_steps.empty())
            f.done_text = "Added " + QuotedSteps(f.fill_steps) + " (Missing Value step).";
        else if (in.features == Done::Dropped)
            f.done_text = "Added " + QuotedSteps(f.drop_steps) + " (Row Filter: " +
                          (names.size() == 1 ? "IS NOT EMPTY" : "each IS NOT EMPTY") + ").";
    } else {
        f.resolved = true;
    }

    // The label, and a regression target's further columns: dropped, never filled.
    Group& l = card.label;
    l.present = !labels.empty();
    if (l.present) {
        const MissingColumn& first = *labels.front();
        if (labels.size() == 1) {
            l.title = Plural(first.rows, "row has", "rows have") + " no label";
            if (first.at_least) l.title = "At least " + l.title;
            l.subtitle = "label column " + first.name + " · " + MissingText(first);
        } else {
            l.title = std::to_string(labels.size()) + " target columns have missing values";
            l.subtitle = "a row is trained on only with every target";
            l.fix_header = in.label == Done::No ? "Fix" : "Now";
            for (const auto* c : labels) {
                Row row;
                row.column = c->name;
                row.type = c->text ? "text" : "number";
                row.missing = MissingText(*c);
                row.fix = in.label == Done::Dropped ? "rows dropped"
                          : in.steps_reach_input   ? "drop the rows"
                                                   : "fix the file";
                l.rows.push_back(std::move(row));
            }
        }
        if (in.supplied_role) l.subtitle += " · supplied " + in.role_name;
        l.can_drop = in.steps_reach_input;
        l.can_fill = false;
        l.note = in.steps_reach_input
                     ? "A label is never filled in: a guessed value would be trained on as the truth."
                     : "Training reads " + where + " as its file holds it: remove these rows in the file, then "
                       "Apply its Data Input again. A label is never filled in.";
        l.drop_steps.push_back(KeepRowsWith(
            label_names, labels.size() == 1 ? "Keep rows with a " + first.name : "Keep rows with " + NameList(label_names)));
        l.drop_steps.back().label = true;
        l.resolved = in.label != Done::No;
        if (in.label == Done::Dropped)
            l.done_text = "Added \"" + l.drop_steps.front().name + "\" (Row Filter: " +
                          (labels.size() == 1 ? l.drop_steps.front().params["condition"] : "each IS NOT EMPTY") + ").";
    } else {
        l.resolved = true;
    }

    card.open_columns = (f.resolved ? 0 : features.size()) + (l.resolved ? 0 : labels.size());
    const size_t columns = features.size() + labels.size();
    card.verdict = "Training did not start: " + std::to_string(columns) +
                   (columns == 1 ? " column has" : " columns have") + " missing values";
    card.subtitle = "Training never uses an empty cell or NaN as 0 or as class 0. Fix them here in one click, or "
                    "in the file. Nothing was trained and nothing changed in the graph.";
    // Resolved: every group has its step, or its columns were left out.
    card.resolved = any_missing && f.resolved && l.resolved;
    size_t steps = 0;
    if (in.features == Done::Filled) steps += f.fill_steps.size();
    if (in.features == Done::Dropped) steps += f.drop_steps.size();
    if (in.label == Done::Dropped) steps += l.drop_steps.size();
    if (steps > 0) {
        card.resolved_verdict = "Fixed: " + std::to_string(steps) + (steps == 1 ? " step" : " steps") +
                                " added after " + in.input_name;
        card.resolved_subtitle = "Train runs them first, then trains. Undo here, or Ctrl+Z on the canvas, removes "
                                 "a step again.";
    } else {
        card.resolved_verdict = "Fixed: " + NameList(in.left_out) + (in.left_out.size() == 1 ? " is" : " are") +
                                " left out of the features";
        card.resolved_subtitle = "Training reads the other columns.";
    }

    // Fix and train: fill the features (or drop their rows when they cannot
    // be filled), drop the rows without a label.
    if (f.present && !f.resolved && in.steps_reach_input) {
        const auto& steps_for_features = f.can_fill ? f.fill_steps : f.drop_steps;
        card.fix_all_steps.insert(card.fix_all_steps.end(), steps_for_features.begin(), steps_for_features.end());
    }
    if (l.present && !l.resolved && in.steps_reach_input)
        card.fix_all_steps.insert(card.fix_all_steps.end(), l.drop_steps.begin(), l.drop_steps.end());
    std::string hint;
    if (f.present && !f.resolved)
        hint = f.can_fill ? "Fills the feature columns with the chosen value" : "Drops the rows with an empty feature cell";
    if (l.present && !l.resolved)
        hint += hint.empty() ? "Drops the rows without a label" : " and drops the rows without a label";
    card.fix_all_hint = hint.empty() ? "" : hint + ", then trains.";

    card.copy_lines.push_back(card.verdict);
    for (const auto* c : features) card.copy_lines.push_back("  feature " + c->name + ": " + MissingText(*c));
    for (const auto* c : labels) card.copy_lines.push_back("  label " + c->name + ": " + MissingText(*c));
    for (const auto& s : card.fix_all_steps) card.copy_lines.push_back("  fix: " + s.name);
    return card;
}

}  // namespace cyxwiz::missing_values_fix
