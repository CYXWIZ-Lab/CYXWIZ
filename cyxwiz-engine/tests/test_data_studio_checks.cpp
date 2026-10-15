// Data Studio > Validate's model (TOFIX148 D4): the checks table and the
// "Add a check" form, on the Berean tier1 recipe's four checks as they run in
// the Engine (tier1_paragraphs SQL step -> 4 Row Count Checks in a chain).

#include "../src/core/data_studio_checks.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::studio;
using gui::NodeType;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Same(const std::string& got, const std::string& want, const std::string& what) {
    Check(got == want, what + ": got \"" + got + "\", want \"" + want + "\"");
}

GraphNode CheckNode(int id, std::string name, int from, std::map<std::string, std::string> parameters) {
    GraphNode n;
    n.id = id;
    n.type = NodeType::RowCountCheck;
    n.name = std::move(name);
    n.input_from = {from};
    parameters.emplace("check", "row_count");
    n.parameters = std::move(parameters);
    return n;
}

std::map<std::string, std::string> Params(const CheckForm& form) {
    std::map<std::string, std::string> p;
    std::string error;
    Check(CheckParameters(form, p, error), "form should be complete: " + error);
    return p;
}

std::string FormError(const CheckForm& form) {
    std::map<std::string, std::string> p;
    std::string error;
    Check(!CheckParameters(form, p, error), "form should be refused");
    return error;
}
}  // namespace

int main() {
    // The tier1 recipe: the SQL step 162, then its checks 163 -> 164 -> 165 -> 166.
    GraphNode step;
    step.id = 162;
    step.type = NodeType::SQLQuery;
    step.name = "tier1_paragraphs";
    std::vector<GraphNode> nodes{
        step,
        CheckNode(163, "Check: paragraphs", 162, {{"expected_rows", "15349"}}),
        CheckNode(164, "Check: Baxter Part 1", 163, {{"expected_rows", "3712"}, {"count_column", "source_id"}, {"count_value", "pg41633"}}),
        CheckNode(165, "Check: Law", 164, {{"expected_rows", "1942"}, {"count_column", "source_id"}, {"count_value", "ccel_law_serious_call"}}),
        CheckNode(166, "Check: nothing admitted", 165, {{"expected_rows", "0"}, {"count_true_column", "training_admitted"}}),
    };

    // Before a run: the rules in words, each reads the SQL step (past the checks before it).
    {
        const auto rows = CheckRows(nodes);
        Check(rows.size() == 4, "four checks");
        Same(rows[0].rule, "exactly 15,349 rows", "paragraphs rule");
        Same(rows[1].rule, "exactly 3,712 rows where source_id = pg41633", "Baxter rule");
        Same(rows[2].rule, "exactly 1,942 rows where source_id = ccel_law_serious_call", "Law rule");
        Same(rows[3].rule, "exactly 0 rows where training_admitted is true", "admitted rule");
        for (const auto& r : rows) {
            Same(r.reads, "tier1_paragraphs", r.name + " reads");
            Check(r.reads_id == 162, r.name + " reads the step");
            Same(r.result, "not run", r.name + " before a run");
        }
    }

    // After the run: the counts the operator reported.
    {
        auto ran = nodes;
        const char* outcomes[] = {"passed: 15349 rows (expected exactly 15349)",
                                  "passed: 3712 rows where source_id = 'pg41633' (expected exactly 3712)",
                                  "passed: 1942 rows where source_id = 'ccel_law_serious_call' (expected exactly 1942)",
                                  "passed: 0 rows where training_admitted is true (expected exactly 0)"};
        for (size_t i = 1; i < ran.size(); ++i) {
            ran[i].ran = true;
            ran[i].rows = 15349;
            ran[i].outcome = outcomes[i - 1];
        }
        const auto rows = CheckRows(ran);
        Same(rows[0].result, "pass \xC2\xB7 15,349 rows", "paragraphs passed");
        Same(rows[3].result, "pass \xC2\xB7 0 rows", "nothing admitted passed");
        Check(rows[0].tone == Tone::Ok, "a pass is ok");
        Same(rows[1].detail, outcomes[1], "the outcome in full");

        // A failure, and a check that could not look at the data.
        ran[2].outcome.clear();
        ran[2].error = "'Check: Baxter Part 1' failed: expected exactly 3712 rows where source_id = 'pg41633', found 3700.";
        ran[3].error = "RowCountCheck: column 'source_id' not found";
        ran[3].outcome.clear();
        ran[4].ran = false;
        ran[4].outcome.clear();
        ran[2].error = "RowCountCheck: operator execution failed: Invalid: " + ran[2].error;
        const auto failed = CheckRows(ran);
        Same(failed[1].result, "fail", "a check that found bad data");
        Check(failed[1].detail.rfind("'Check: Baxter Part 1' failed: ", 0) == 0, "the executor's prefix is dropped");
        Check(failed[1].tone == Tone::Error, "a fail is an error");
        Same(failed[2].result, "could not run", "a check stopped before it could look");
        Same(failed[3].result, "not run", "a check after the failure");
    }

    // A Data Validator is listed with its rules and its issues.
    {
        GraphNode v;
        v.id = 200;
        v.type = NodeType::DataValidator;
        v.name = "Data Validator";
        v.input_from = {166};
        v.parameters = {{"required_columns", "text,source_id"}, {"unique_columns", " paragraph_id "}};
        auto with = nodes;
        with.push_back(v);
        auto rows = CheckRows(with);
        Check(rows.size() == 5 && rows[4].validator, "the validator is a row");
        Same(rows[4].reads, "tier1_paragraphs", "the validator reads past the checks");
        Same(rows[4].rule, "required: text, source_id \xC2\xB7 unique: paragraph_id", "validator rules");
        with.back().ran = true;
        with.back().rows = 0;
        Same(CheckRows(with)[4].result, "no issues", "no issues");
        with.back().rows = 3;
        rows = CheckRows(with);
        Same(rows[4].result, "3 issues", "issues");
        Check(rows[4].tone == Tone::Warn, "issues warn");
    }

    // A Data Validator's form.
    {
        ValidatorForm v;
        std::map<std::string, std::string> p;
        std::string error;
        Check(!ValidatorParameters(v, p, error) && error.find("at least one") != std::string::npos, "a validator needs a rule");
        v.required = "text , source_id";
        v.unique = "paragraph_id";
        Check(ValidatorParameters(v, p, error), "validator form complete");
        Same(p["required_columns"], "text, source_id", "required tidied");
        Same(p["not_null_columns"], "", "unused rule cleared");
        Same(ValidatorRule(p), "required: text, source_id \xC2\xB7 unique: paragraph_id", "validator rule from the form");
        const ValidatorForm back = ValidatorFormFrom(p);
        Check(back.required == "text, source_id" && back.unique == "paragraph_id" && back.not_null.empty(), "validator round trip");
    }

    // The form -> the node's parameters -> the form again.
    {
        CheckForm f;
        f.name = "Check: Baxter Part 1";
        f.count = "3712";
        f.where = 1;
        f.where_column = " source_id ";
        f.where_value = "pg41633";
        auto p = Params(f);
        Same(p["check"], "row_count", "kind");
        Same(p["expected_rows"], "3712", "exact count");
        Same(p["count_column"], "source_id", "trimmed column");
        Same(p["count_value"], "pg41633", "value");
        Same(p["min_rows"], "", "unused keys are cleared");
        Same(p["on_failure"], "stop", "stops the run");
        Same(CheckRule(p), "exactly 3,712 rows where source_id = pg41633", "rule from the form");
        Same(CheckDefaultName(p), "Check: exactly 3,712 rows where source_id = pg41633", "default name");
        p["allowed"] = std::string(80, 'x');
        p["check"] = "allowed_values";
        p["column"] = "role";
        Check(CheckDefaultName(p).size() == 60, "a long default name is cut");
        p = Params(f);
        const CheckForm back = CheckFormFrom(p);
        Check(back.kind == CheckKind::RowCount && back.where == 1 && back.count == "3712" && back.name == f.name, "form round trip");

        f = {};
        f.count_mode = 3;
        f.count = "10";
        f.count_max = "20";
        p = Params(f);
        Same(p["min_rows"] + "/" + p["max_rows"], "10/20", "between");
        Same(CheckRule(p), "10 to 20 rows", "between rule");
        Check(CheckFormFrom(p).count_mode == 3, "between round trip");
        f.count_mode = 1;
        f.count = "1";
        Same(CheckRule(Params(f)), "at least 1 row", "one row");
        f.count_mode = 3;
        f.count = "30";
        Check(FormError(f).find("larger") != std::string::npos, "between upside down");
        f = {};
        Check(FormError(f).find("number of rows") != std::string::npos, "no count");
        f.count = "-1";
        Check(!FormError(f).empty(), "a negative count");
        f.count = "5";
        f.where = 2;
        Check(FormError(f).find("column") != std::string::npos, "where without a column");
        f.where_column = "training_admitted";
        Same(CheckRule(Params(f)), "exactly 5 rows where training_admitted is true", "where true");

        f = {};
        f.kind = CheckKind::AllowedValues;
        f.column = "role";
        Check(FormError(f).find("allowed values") != std::string::npos, "no values");
        f.allowed = "book, chapter ,verse,,";
        p = Params(f);
        Same(p["check"], "allowed_values", "allowed kind");
        Same(p["allowed"], "book, chapter, verse", "the list tidied");
        Same(CheckRule(p), "role is one of book, chapter, verse", "allowed rule");

        f = {};
        f.kind = CheckKind::Unique;
        Check(!FormError(f).empty(), "unique needs columns");
        f.columns = "book,chapter";
        p = Params(f);
        Same(CheckRule(p), "book, chapter together are unique", "unique pair rule");
        f.columns = "paragraph_id";
        Same(CheckRule(Params(f)), "paragraph_id is unique", "unique rule");
        Same(CheckFormFrom({{"check", "unique"}, {"column", "id"}}).columns, "id", "unique on `column` edits as columns");

        f = {};
        f.kind = CheckKind::NotEmpty;
        Check(FormError(f).find("column") != std::string::npos, "not empty needs a column");
        f.column = "text";
        Same(CheckRule(Params(f)), "text is never empty", "not empty rule");

        f = {};
        f.kind = CheckKind::Range;
        f.column = "word_count";
        Check(FormError(f).find("smallest") != std::string::npos, "a range needs a bound");
        f.min_value = "1";
        Same(CheckRule(Params(f)), "word_count at least 1", "lower bound");
        f.max_value = "900";
        p = Params(f);
        Same(CheckRule(p), "word_count from 1 to 900", "both bounds");
        Same(p["min_value"] + "/" + p["max_value"], "1/900", "range keys");
        f.max_value = "abc";
        Check(FormError(f).find("numbers") != std::string::npos, "a bound must be a number");
        f.max_value = "0.5";
        Check(FormError(f).find("larger") != std::string::npos, "range upside down");

        // Editing a range check into a row count clears the range keys.
        f.max_value = "900";
        p = Params(f);
        CheckForm edit = CheckFormFrom(p);
        edit.kind = CheckKind::RowCount;
        edit.count = "7";
        p = Params(edit);
        Same(p["column"] + p["min_value"] + p["max_value"], "", "old kind's keys cleared");
        Same(p["expected_rows"], "7", "new count");
    }

    std::cout << "test_data_studio_checks: all checks passed\n";
    return 0;
}
