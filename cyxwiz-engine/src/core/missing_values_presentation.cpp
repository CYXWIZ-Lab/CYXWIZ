#include "missing_values_presentation.h"

#include "filter_condition.h"

#include <algorithm>

namespace cyxwiz::missing_values_fix {

namespace {

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

// The comma list the Missing Value step reads.
std::string CommaList(const std::vector<std::string>& names) {
    std::string out;
    for (const auto& n : names) out += (out.empty() ? "" : ",") + n;
    return out;
}

std::string Plural(int64_t n, const char* one, const char* many) {
    return Thousands(n) + " " + (n == 1 ? one : many);
}

std::string MissingText(const MissingColumn& c) {
    const int64_t empty = c.rows - c.nan_rows;
    std::string text = c.at_least ? "at least " : "";
    if (empty > 0 && c.nan_rows > 0) return text + Thousands(empty) + " empty, " + Thousands(c.nan_rows) + " NaN";
    if (c.nan_rows > 0) return text + Thousands(c.nan_rows) + " NaN";
    return text + Thousands(empty) + " empty";
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

std::string Thousands(int64_t value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return negative ? "-" + out : out;
}

Card BuildCard(const Input& in) {
    Card card;
    std::vector<const MissingColumn*> features;
    const MissingColumn* label = nullptr;
    for (const auto& c : in.columns) {
        if (c.rows <= 0) continue;
        if (c.label) {
            if (!label) label = &c;
            continue;
        }
        if (std::find(in.left_out.begin(), in.left_out.end(), c.name) != in.left_out.end()) continue;
        features.push_back(&c);
    }
    const std::string where = in.supplied_role ? "the supplied " + in.role_name + " data (" + in.input_name + ")"
                                               : in.input_name;

    // Feature columns.
    Group& f = card.features;
    f.present = !features.empty();
    if (f.present) {
        const size_t n = features.size();
        f.title = std::to_string(n) + (n == 1 ? " feature column has" : " feature columns have") + " missing values";
        f.subtitle = in.input_name + (in.total_rows >= 0 ? " · " + Plural(in.total_rows, "row", "rows") : "");
        if (in.supplied_role) f.subtitle += " · supplied " + in.role_name;
        f.fix_header = in.features == Done::No ? (in.supplied_role ? "Drop would" : "Fill would use") : "Now";
        std::vector<std::string> numbers, texts, all;
        int64_t most = 0;
        for (const auto* c : features) {
            Row row;
            row.column = c->name;
            row.type = c->text ? "text" : "number";
            row.missing = MissingText(*c);
            row.mostly_missing = in.total_rows > 0 && c->rows * 2 > in.total_rows;
            if (row.mostly_missing) f.can_leave_out.push_back(c->name);
            if (in.features == Done::Dropped || (in.supplied_role && in.features == Done::No))
                row.fix = in.features == Done::Dropped ? "rows dropped" : "drop the rows";
            else if (c->text)
                row.fix = "most frequent of Train";
            else
                row.fix = in.number_fill == NumberFill::Zero ? "0"
                                                              : std::string(NumberFillWord(in.number_fill)) + " of Train";
            f.rows.push_back(std::move(row));
            (c->text ? texts : numbers).push_back(c->name);
            all.push_back(c->name);
            most += c->rows;
        }
        f.can_drop = in.steps_reach_input;
        f.can_fill = !in.supplied_role && in.steps_reach_input;
        const std::string fill_word = NumberFillWord(in.number_fill);
        if (!numbers.empty()) {
            Step step;
            step.type = gui::NodeType::FillMissingValues;
            step.name = "Fill " + CommaNames(numbers) + " (" + fill_word + ")";
            step.params["columns"] = CommaList(numbers);
            step.params["strategy"] = in.number_fill == NumberFill::Zero ? "constant" : fill_word;
            if (in.number_fill == NumberFill::Zero) step.params["value"] = "0";
            if (label) step.params["label_col"] = label->name;
            f.fill_steps.push_back(std::move(step));
        }
        if (!texts.empty()) {
            Step step;
            step.type = gui::NodeType::FillMissingValues;
            step.name = "Fill " + CommaNames(texts) + " (most frequent)";
            step.params["columns"] = CommaList(texts);
            step.params["strategy"] = "mode";
            if (label) step.params["label_col"] = label->name;
            f.fill_steps.push_back(std::move(step));
        }
        f.drop_steps.push_back(KeepRowsWith(all, "Keep rows with " + NameList(all)));
        if (!in.steps_reach_input) {
            f.note = "Training reads " + where + " as its file holds it, so a step here would not apply: fill or "
                     "remove these cells in the file, then Apply its Data Input again.";
        } else if (in.supplied_role) {
            f.note = "Drop removes the rows of " + where + " with an empty cell or NaN in these columns. Fill is "
                     "not offered here: it would need the values learned from the Train rows.";
        } else {
            f.note = std::string("Fill learns its values from the Train rows only.") +
                     (texts.empty() ? "" : " Text columns are filled with their most frequent value.") +
                     " Drop removes every row with an empty cell or NaN in these columns (at most " + Thousands(most) +
                     ").";
        }
        if (!f.can_leave_out.empty())
            f.note += " " + NameList(f.can_leave_out) + (f.can_leave_out.size() == 1 ? " is" : " are") +
                      " more than half empty: filling would make up most of " +
                      (f.can_leave_out.size() == 1 ? "its" : "their") + " values; it can be left out of the "
                      "features instead (Actions).";
        f.resolved = in.features != Done::No;
        if (in.features == Done::Filled)
            f.done_text = "Added " + QuotedSteps(f.fill_steps) + " (Missing Value " +
                          (f.fill_steps.size() == 1 ? "step" : "steps") + ").";
        else if (in.features == Done::Dropped)
            f.done_text = "Added " + QuotedSteps(f.drop_steps) + " (Row Filter: " +
                          (all.size() == 1 ? "IS NOT EMPTY" : "each IS NOT EMPTY") + ").";
    } else {
        f.resolved = true;
    }

    // The label.
    Group& l = card.label;
    l.present = label != nullptr;
    if (l.present) {
        l.title = Plural(label->rows, "row has", "rows have") + " no label";
        if (label->at_least) l.title = "At least " + l.title;
        l.subtitle = "label column " + label->name + " · " + MissingText(*label);
        if (in.supplied_role) l.subtitle += " · supplied " + in.role_name;
        l.can_drop = in.steps_reach_input;
        l.can_fill = false;
        l.note = in.steps_reach_input
                     ? "A label is never filled in: a guessed class would be trained on as the truth."
                     : "Training reads " + where + " as its file holds it: remove these rows in the file, then "
                       "Apply its Data Input again. A label is never filled in.";
        l.drop_steps.push_back(KeepRowsWith({label->name}, "Keep rows with a " + label->name));
        l.resolved = in.label != Done::No;
        if (in.label == Done::Dropped)
            l.done_text = "Added \"" + l.drop_steps.front().name + "\" (Row Filter: " +
                          l.drop_steps.front().params["condition"] + ").";
    } else {
        l.resolved = true;
    }

    const size_t columns = features.size() + (label ? 1 : 0);
    card.verdict = "Training did not start: " + std::to_string(columns) +
                   (columns == 1 ? " column has" : " columns have") + " missing values";
    card.subtitle = "Training never uses an empty cell or NaN as 0 or as class 0. Fix them here in one click, or "
                    "in the file. Nothing was trained and nothing changed in the graph.";
    card.resolved = f.resolved && l.resolved && (f.present || l.present);
    size_t steps = 0;
    if (in.features == Done::Filled) steps += f.fill_steps.size();
    if (in.features == Done::Dropped) steps += f.drop_steps.size();
    if (in.label == Done::Dropped) steps += l.drop_steps.size();
    card.resolved_verdict = "Fixed: " + std::to_string(steps) + (steps == 1 ? " step" : " steps") +
                            " added after " + in.input_name;
    card.resolved_subtitle = "Train runs them first, then trains. Ctrl+Z on the canvas removes them.";

    // Fix and train: fill the features (drop on a supplied input), drop the
    // rows without a label.
    if (f.present && in.steps_reach_input) {
        const auto& steps_for_features = f.can_fill ? f.fill_steps : f.drop_steps;
        card.fix_all_steps.insert(card.fix_all_steps.end(), steps_for_features.begin(), steps_for_features.end());
    }
    if (l.present && in.steps_reach_input)
        card.fix_all_steps.insert(card.fix_all_steps.end(), l.drop_steps.begin(), l.drop_steps.end());
    card.fix_all_hint = std::string(f.present ? (f.can_fill ? "Fills the feature columns (numbers with the chosen "
                                                              "value, text with the most frequent)"
                                                            : "Drops the rows with an empty feature cell")
                                              : "") +
                        (f.present && l.present ? " and drops the rows without a label" : "") +
                        (!f.present && l.present ? "Drops the rows without a label" : "") + ", then trains.";

    card.copy_lines.push_back(card.verdict);
    for (const auto* c : features)
        card.copy_lines.push_back("  feature " + c->name + ": " + MissingText(*c));
    if (label) card.copy_lines.push_back("  label " + label->name + ": " + MissingText(*label));
    for (const auto& s : card.fix_all_steps) card.copy_lines.push_back("  fix: " + s.name);
    return card;
}

}  // namespace cyxwiz::missing_values_fix
