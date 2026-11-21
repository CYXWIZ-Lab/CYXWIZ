#pragma once

// The one-click fix for missing values (owner decision D4; mockup approved
// 2026-11-21, decisions M-1 to M-7): what the "Cannot start training" report
// and the Training Dashboard show when training refused empty cells or NaN,
// and the exact steps each action adds. Pure data in and out (no ImGui, no
// registry), so both screens say the same words and add the same steps.
//
// - Feature columns: Fill (numbers with the mean, median or 0; text with its
//   most frequent value), or Drop the rows; a column more than half empty
//   can also be left out of the features (M-7).
// - The label: rows without one are dropped, never filled (M-5).
// - A supplied Validation / Test input offers Drop only (M-6): its fill would
//   need the Train rows' values.

#include "graph_model.h"
#include "missing_columns.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::missing_values_fix {

enum class NumberFill { Mean, Median, Zero };

// "mean", "median", "0".
const char* NumberFillWord(NumberFill fill);

// What the user already did with a group.
enum class Done { No, Filled, Dropped };

struct Input {
    std::vector<MissingColumn> columns;  // the refusal's columns (those with a missing cell)
    int64_t total_rows = -1;             // the rows they are out of; -1 when not known
    std::string input_name;              // the Data Input the data comes from ("Patients CSV")
    bool supplied_role = false;          // a supplied Validation / Test input (M-6)
    std::string role_name;               // "Validation" or "Test" for a supplied input
    // Whether a step after the input reaches what training reads. A supplied
    // Validation / Test is read from its own Data Input today, so a step
    // there would not apply: the card then offers no step, only the file.
    bool steps_reach_input = true;
    NumberFill number_fill = NumberFill::Mean;
    Done features = Done::No;
    Done label = Done::No;
    std::vector<std::string> left_out;   // feature columns the user left out (M-7)
};

// A step the fix adds after the step that feeds the Data Split.
struct Step {
    gui::NodeType type = gui::NodeType::FillMissingValues;  // or FilterRows
    std::string name;
    std::map<std::string, std::string> params;
};

struct Row {
    std::string column;
    std::string type;     // "number" or "text"
    std::string missing;  // "37 empty", "9 empty, 3 NaN", "at least 3 NaN"
    std::string fix;      // "mean of Train", "most frequent of Train", "rows dropped"
    bool mostly_missing = false;  // more than half of the rows: can be left out (M-7)
};

struct Group {
    bool present = false;
    bool resolved = false;
    std::string title;          // "3 feature columns have missing values", "4 rows have no label"
    std::string subtitle;       // "Patients CSV · 1,200 rows", "label column diagnosis"
    std::vector<Row> rows;
    std::string fix_header;     // "Fill would use", then "Now"
    bool can_fill = false;
    bool can_drop = false;
    std::string note;           // what Fill / Drop do, or why Fill is not offered
    std::vector<Step> fill_steps;
    std::vector<Step> drop_steps;
    std::vector<std::string> can_leave_out;  // mostly missing feature columns (M-7)
    std::string done_text;      // after an action: what was added
};

struct Card {
    std::string verdict;        // "Training did not start: 4 columns have missing values"
    std::string subtitle;
    bool resolved = false;
    std::string resolved_verdict;  // "Fixed: 3 steps added after Patients CSV"
    std::string resolved_subtitle;
    Group features;
    Group label;
    std::vector<Step> fix_all_steps;  // "Fix and train": fill the features, drop the rows without a label
    std::string fix_all_hint;
    std::vector<std::string> copy_lines;  // for Copy report
};

Card BuildCard(const Input& input);

// "1,200".
std::string Thousands(int64_t value);

}  // namespace cyxwiz::missing_values_fix
