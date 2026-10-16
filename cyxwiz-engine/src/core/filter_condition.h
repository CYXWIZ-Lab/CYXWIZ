#pragma once

// The Row Filter's condition language, shared by the executor (which turns it
// into SQL) and the Row Filter dialog (which edits it as rows).
//
//   condition  := term (OR term)*
//   term       := factor (AND factor)*
//   factor     := '(' condition ')' | column op literal
//   column     := name | "quoted name"
//   op         := = | == | != | <> | < | > | <= | >=
//   literal    := 'text' | number | true | false

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class FilterConditionTokenKind {
    Identifier,
    StringLiteral,
    NumericLiteral,
    BooleanLiteral,
    ComparisonOperator,
    And,
    Or,
    LeftParen,
    RightParen,
    End,
};

struct FilterConditionToken {
    FilterConditionTokenKind kind = FilterConditionTokenKind::End;
    std::string value;  // operators normalised: "==" -> "="; booleans upper case
};

// Splits a condition into tokens ending with End; false with `error` set on
// a malformed condition.
bool TokenizeFilterCondition(const std::string& condition,
                             std::vector<FilterConditionToken>& tokens,
                             std::string& error);

// ---- The dialog's rows ----

enum class FilterOp { Is, IsNot, MoreThan, LessThan, AtLeast, AtMost };
enum class FilterValueKind { Number, Text, TrueFalse };

struct FilterRow {
    bool or_join = false;  // joins this row to the one above: false = and
    std::string column;
    FilterOp op = FilterOp::Is;
    FilterValueKind kind = FilterValueKind::Text;
    std::string value;     // the number, the text, or "true" / "false"
};

// "is", "is not", "is more than", ...
const char* FilterOpWords(FilterOp op);
// Text and true/false values compare only with is / is not.
bool FilterOpAllowed(FilterOp op, FilterValueKind kind);

// What is wrong with a row, or empty when it can run.
std::string FilterRowProblem(const FilterRow& row);

// The condition the rows describe (column names always "quoted").
std::string BuildFilterCondition(const std::vector<FilterRow>& rows);

// The rows of a saved condition, or nullopt when it does not fit rows
// (brackets, or anything the tokenizer refuses). An empty condition gives no
// rows.
std::optional<std::vector<FilterRow>> ParseFilterCondition(const std::string& condition);

}  // namespace cyxwiz
