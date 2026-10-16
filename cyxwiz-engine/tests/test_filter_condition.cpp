// The Row Filter's condition as dialog rows (TOFIX129 B-5): rows build the
// text the executor runs, and a saved condition reads back into rows.
#include "core/filter_condition.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cyxwiz;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

FilterRow Row(std::string column, FilterOp op, FilterValueKind kind, std::string value, bool or_join = false) {
    FilterRow row;
    row.column = std::move(column);
    row.op = op;
    row.kind = kind;
    row.value = std::move(value);
    row.or_join = or_join;
    return row;
}

void CheckBuild() {
    const std::vector<FilterRow> rows = {
        Row("amount", FilterOp::AtLeast, FilterValueKind::Number, "100"),
        Row("status", FilterOp::Is, FilterValueKind::Text, "Anxiety"),
        Row("is test", FilterOp::IsNot, FilterValueKind::TrueFalse, "true", true),
    };
    const std::string condition = BuildFilterCondition(rows);
    Check(condition == "\"amount\" >= 100 AND \"status\" = 'Anxiety' OR \"is test\" != TRUE",
          "rows build the executor's condition: " + condition);

    // Quotes inside names and text are doubled.
    const std::string quoted = BuildFilterCondition({Row("say \"hi\"", FilterOp::Is, FilterValueKind::Text, "it's")});
    Check(quoted == "\"say \"\"hi\"\"\" = 'it''s'", "quotes are doubled: " + quoted);

    std::vector<FilterConditionToken> tokens;
    std::string error;
    Check(TokenizeFilterCondition(condition, tokens, error) && tokens.size() == 12,
          "the built condition tokenizes: " + error);
}

void CheckRoundTrip() {
    const std::vector<FilterRow> rows = {
        Row("say \"hi\"", FilterOp::LessThan, FilterValueKind::Number, "-2.5"),
        Row("label", FilterOp::IsNot, FilterValueKind::Text, "it's", true),
        Row("ok", FilterOp::Is, FilterValueKind::TrueFalse, "false"),
    };
    const auto back = ParseFilterCondition(BuildFilterCondition(rows));
    Check(back.has_value() && back->size() == 3, "a built condition reads back into three rows");
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& a = rows[i];
        const auto& b = (*back)[i];
        Check(a.column == b.column && a.op == b.op && a.kind == b.kind && a.value == b.value &&
                  (i == 0 || a.or_join == b.or_join),
              "row " + std::to_string(i) + " survives the round trip");
    }
}

void CheckHandWritten() {
    const auto rows = ParseFilterCondition("amount >= 100 and city == 'Paris' OR flag <> false");
    Check(rows && rows->size() == 3, "a hand-written flat condition reads as rows");
    Check((*rows)[0].column == "amount" && (*rows)[0].op == FilterOp::AtLeast &&
              (*rows)[0].kind == FilterValueKind::Number && (*rows)[0].value == "100",
          "a bare column and a number");
    Check((*rows)[1].op == FilterOp::Is && (*rows)[1].kind == FilterValueKind::Text && !(*rows)[1].or_join,
          "== is 'is', and joins");
    Check((*rows)[2].op == FilterOp::IsNot && (*rows)[2].kind == FilterValueKind::TrueFalse &&
              (*rows)[2].value == "false" && (*rows)[2].or_join,
          "<> is 'is not', or joins, booleans read back");

    Check(!ParseFilterCondition("(a = 1 OR b = 2) AND c = 3"), "brackets do not fit rows");
    Check(!ParseFilterCondition("a = 'open"), "an unclosed text does not parse");
    Check(!ParseFilterCondition("a = b"), "a column compared with a column does not fit rows");
    Check(!ParseFilterCondition("a = 1 b = 2"), "two comparisons need and/or between them");
    const auto none = ParseFilterCondition("   ");
    Check(none && none->empty(), "an empty condition gives no rows");
}

void CheckEmptyTests() {
    const std::vector<FilterRow> rows = {
        Row("status", FilterOp::IsNot, FilterValueKind::Text, "Normal"),
        Row("statement", FilterOp::IsNotEmpty, FilterValueKind::Text, ""),
        Row("score", FilterOp::IsEmpty, FilterValueKind::Number, "", true),
    };
    const std::string condition = BuildFilterCondition(rows);
    Check(condition == "\"status\" != 'Normal' AND \"statement\" IS NOT EMPTY OR \"score\" IS EMPTY",
          "the empty tests take no value: " + condition);
    const auto back = ParseFilterCondition(condition);
    Check(back && back->size() == 3 && (*back)[1].op == FilterOp::IsNotEmpty &&
              (*back)[1].column == "statement" && (*back)[2].op == FilterOp::IsEmpty && (*back)[2].or_join,
          "the empty tests read back");
    const auto typed = ParseFilterCondition("name is not empty and x IS empty");
    Check(typed && typed->size() == 2 && (*typed)[0].op == FilterOp::IsNotEmpty && (*typed)[1].op == FilterOp::IsEmpty,
          "is [not] empty in any case");
    Check(!ParseFilterCondition("name IS NOT 5"), "IS needs EMPTY");
    Check(FilterRowProblem(Row("ok", FilterOp::IsEmpty, FilterValueKind::TrueFalse, "")).empty(),
          "any column can be tested for empty, with no value");
    Check(FilterOpAllowed(FilterOp::IsNotEmpty, FilterValueKind::Text) && !FilterOpTakesValue(FilterOp::IsEmpty),
          "the empty tests suit text and take no value");
}

void CheckProblems() {
    Check(FilterRowProblem(Row("", FilterOp::Is, FilterValueKind::Text, "x")) == "Pick a column", "a column is needed");
    Check(!FilterRowProblem(Row("label", FilterOp::MoreThan, FilterValueKind::Text, "x")).empty(),
          "text compares only with is / is not");
    Check(!FilterRowProblem(Row("amount", FilterOp::Is, FilterValueKind::Number, "ten")).empty(),
          "a number column needs a number");
    Check(!FilterRowProblem(Row("amount", FilterOp::Is, FilterValueKind::Number, "")).empty(),
          "an empty number is refused");
    Check(FilterRowProblem(Row("label", FilterOp::Is, FilterValueKind::Text, "")).empty(),
          "empty text is a real value");
    Check(FilterRowProblem(Row("amount", FilterOp::AtMost, FilterValueKind::Number, "1e3")).empty(),
          "1e3 is a number");
    Check(std::string(FilterOpWords(FilterOp::MoreThan)) == "is more than", "operators read as words");
    Check(FilterOpAllowed(FilterOp::AtLeast, FilterValueKind::Number) &&
              !FilterOpAllowed(FilterOp::AtLeast, FilterValueKind::TrueFalse),
          "ordering only on numbers");
}

}  // namespace

int main() {
    CheckBuild();
    CheckRoundTrip();
    CheckHandWritten();
    CheckEmptyTests();
    CheckProblems();
    std::cout << "test_filter_condition: all checks passed\n";
    return 0;
}
