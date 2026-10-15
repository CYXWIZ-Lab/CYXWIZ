#pragma once

// Data Studio > Validate (TOFIX148 D4, approved mockup v8): the graph's
// checks as a table (the check node, the data it reads, its rule in words,
// its last run) and the form that adds or edits one. A check is a Row Count
// Check node (the one check model: row count, allowed values, unique, not
// empty, range); a Data Validator is listed with its rules too. Pure data in
// and out: the panel reads the nodes, Validate draws the rows and the form.

#include "data_studio_presentation.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz::studio {

enum class CheckKind { RowCount, AllowedValues, Unique, NotEmpty, Range };
constexpr int kCheckKindCount = 5;
const char* CheckKindName(CheckKind kind);       // "Row count", "Allowed values", ...
const char* CheckKindParameter(CheckKind kind);  // "row_count", "allowed_values", ...

// The form's fields. Row count: `count_mode` 0 exactly, 1 at least, 2 at
// most, 3 between (`count` to `count_max`); `where` 0 all rows, 1 rows whose
// `where_column` equals `where_value`, 2 rows where the boolean
// `where_column` is true.
struct CheckForm {
    CheckKind kind = CheckKind::RowCount;
    std::string name;
    int count_mode = 0;
    std::string count, count_max;
    int where = 0;
    std::string where_column, where_value;
    std::string column;               // allowed values, not empty, range
    std::string columns;              // unique (a comma list)
    std::string allowed;              // allowed values (a comma list)
    std::string min_value, max_value; // range
};

// The node's parameters for the form, every check key set (unused ones
// empty) so an edit clears what the old kind used. False, with `error` saying
// what is missing, when the form is not complete.
bool CheckParameters(const CheckForm& form, std::map<std::string, std::string>& parameters, std::string& error);
// The form for a node's parameters (Edit).
CheckForm CheckFormFrom(const std::map<std::string, std::string>& parameters);
// The rule in words: "exactly 15,349 rows", "exactly 0 rows where
// training_admitted is true", "source_id is one of a, b", "id is unique".
std::string CheckRule(const std::map<std::string, std::string>& parameters);
// The name a check gets when none is given: "Check: " + its rule, cut to 60
// characters.
std::string CheckDefaultName(const std::map<std::string, std::string>& parameters);
// A Data Validator's rules: "required: a, b · not null: c · unique: d".
std::string ValidatorRule(const std::map<std::string, std::string>& parameters);

// A Data Validator's form: comma lists of the columns that must be there,
// must have no missing value, and must not repeat.
struct ValidatorForm {
    std::string required, not_null, unique;
};
// Its node parameters; false, with `error`, when no rule is set.
bool ValidatorParameters(const ValidatorForm& form, std::map<std::string, std::string>& parameters, std::string& error);
ValidatorForm ValidatorFormFrom(const std::map<std::string, std::string>& parameters);

struct CheckRow {
    int id = -1;
    bool validator = false;    // a Data Validator (edited in its dialog)
    std::string name;
    std::string reads;         // the node whose table it checks ("tier1_paragraphs"); "" when nothing is linked
    int reads_id = -1;
    std::string rule;
    std::string result;        // "pass · 15,349 rows", "pass", "fail", "could not run", "not run", "3 issues"
    std::string detail;        // the failure (without the executor's prefix), or the pass outcome in full
    Tone tone = Tone::Plain;
};
// The graph's checks and validators in graph order. `nodes` carry their
// parameters and input_from (pin 0 first); a check after a check reads what
// the first one reads.
std::vector<CheckRow> CheckRows(const std::vector<GraphNode>& nodes);

}  // namespace cyxwiz::studio
