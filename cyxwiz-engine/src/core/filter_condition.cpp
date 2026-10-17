#include "filter_condition.h"

#include <cctype>
#include <cmath>
#include <cstdlib>

namespace cyxwiz {

namespace {

std::string ToUpperAscii(std::string value) {
    for (char& c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return value;
}

bool IsValidNumericLiteral(const std::string& value) {
    if (value.empty()) return false;
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    return end == value.c_str() + value.size() && std::isfinite(parsed);
}

std::string QuoteName(const std::string& name) {
    std::string out = "\"";
    for (char c : name) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}

std::string QuoteText(const std::string& text) {
    std::string out = "'";
    for (char c : text) {
        if (c == '\'') out += '\'';
        out += c;
    }
    return out + '\'';
}

const char* OpSymbol(FilterOp op) {
    switch (op) {
        case FilterOp::Is: return "=";
        case FilterOp::IsNot: return "!=";
        case FilterOp::MoreThan: return ">";
        case FilterOp::LessThan: return "<";
        case FilterOp::AtLeast: return ">=";
        case FilterOp::AtMost: return "<=";
        case FilterOp::IsEmpty: return "IS EMPTY";
        case FilterOp::IsNotEmpty: return "IS NOT EMPTY";
    }
    return "=";
}

std::optional<FilterOp> OpFromSymbol(const std::string& symbol) {
    if (symbol == "=") return FilterOp::Is;
    if (symbol == "!=" || symbol == "<>") return FilterOp::IsNot;
    if (symbol == ">") return FilterOp::MoreThan;
    if (symbol == "<") return FilterOp::LessThan;
    if (symbol == ">=") return FilterOp::AtLeast;
    if (symbol == "<=") return FilterOp::AtMost;
    return std::nullopt;
}

}  // namespace

bool TokenizeFilterCondition(const std::string& condition,
                             std::vector<FilterConditionToken>& tokens,
                             std::string& error) {
    size_t index = 0;
    while (index < condition.size()) {
        const unsigned char ch =
            static_cast<unsigned char>(condition[index]);
        if (std::isspace(ch)) {
            ++index;
            continue;
        }

        if (std::isalpha(ch) || condition[index] == '_') {
            const size_t start = index;
            while (index < condition.size()) {
                const unsigned char current =
                    static_cast<unsigned char>(condition[index]);
                if (!std::isalnum(current) && condition[index] != '_') {
                    break;
                }
                ++index;
            }
            const std::string value = condition.substr(start, index - start);
            const std::string upper_value = ToUpperAscii(value);
            if (upper_value == "AND") {
                tokens.push_back({FilterConditionTokenKind::And, value});
            } else if (upper_value == "IS") {
                tokens.push_back({FilterConditionTokenKind::Is, upper_value});
            } else if (upper_value == "NOT") {
                tokens.push_back({FilterConditionTokenKind::Not, upper_value});
            } else if (upper_value == "EMPTY") {
                tokens.push_back({FilterConditionTokenKind::Empty, upper_value});
            } else if (upper_value == "TRUE" || upper_value == "FALSE") {
                tokens.push_back({FilterConditionTokenKind::BooleanLiteral, upper_value});
            } else if (upper_value == "OR") {
                tokens.push_back({FilterConditionTokenKind::Or, value});
            } else {
                tokens.push_back({FilterConditionTokenKind::Identifier, value});
            }
            continue;
        }

        if (condition[index] == '"') {
            ++index;
            std::string value;
            bool closed = false;
            while (index < condition.size()) {
                if (condition[index] == '"') {
                    if (index + 1 < condition.size() &&
                        condition[index + 1] == '"') {
                        value += '"';
                        index += 2;
                        continue;
                    }
                    ++index;
                    closed = true;
                    break;
                }
                value += condition[index++];
            }
            if (!closed) {
                error = "FilterRows: unterminated quoted column name";
                return false;
            }
            tokens.push_back({FilterConditionTokenKind::Identifier, value});
            continue;
        }

        if (condition[index] == '\'') {
            ++index;
            std::string value;
            bool closed = false;
            while (index < condition.size()) {
                if (condition[index] == '\'') {
                    if (index + 1 < condition.size() &&
                        condition[index + 1] == '\'') {
                        value += '\'';
                        index += 2;
                        continue;
                    }
                    ++index;
                    closed = true;
                    break;
                }
                value += condition[index++];
            }
            if (!closed) {
                error = "FilterRows: unterminated string literal";
                return false;
            }
            tokens.push_back(
                {FilterConditionTokenKind::StringLiteral, value});
            continue;
        }

        const bool starts_numeric =
            std::isdigit(ch) || condition[index] == '.' ||
            ((condition[index] == '-' || condition[index] == '+') &&
             index + 1 < condition.size() &&
             (std::isdigit(static_cast<unsigned char>(condition[index + 1])) ||
              condition[index + 1] == '.'));
        if (starts_numeric) {
            const size_t start = index;
            while (index < condition.size()) {
                const char current = condition[index];
                if (!std::isdigit(static_cast<unsigned char>(current)) &&
                    current != '.' && current != '-' && current != '+' &&
                    current != 'e' && current != 'E') {
                    break;
                }
                ++index;
            }
            const std::string value = condition.substr(start, index - start);
            if (!IsValidNumericLiteral(value)) {
                error = "FilterRows: invalid numeric literal '" + value + "'";
                return false;
            }
            tokens.push_back(
                {FilterConditionTokenKind::NumericLiteral, value});
            continue;
        }

        if (condition[index] == '(') {
            tokens.push_back({FilterConditionTokenKind::LeftParen, "("});
            ++index;
            continue;
        }
        if (condition[index] == ')') {
            tokens.push_back({FilterConditionTokenKind::RightParen, ")"});
            ++index;
            continue;
        }

        std::string op;
        if (condition[index] == '=' || condition[index] == '!' ||
            condition[index] == '<' || condition[index] == '>') {
            op += condition[index++];
            if (index < condition.size()) {
                const char next = condition[index];
                if ((op == "=" && next == '=') ||
                    (op == "!" && next == '=') ||
                    (op == "<" && (next == '=' || next == '>')) ||
                    (op == ">" && next == '=')) {
                    op += next;
                    ++index;
                }
            }
            if (op == "==") {
                op = "=";
            }
            if (op == "!") {
                error = "FilterRows: unsupported comparison operator '!'";
                return false;
            }
            tokens.push_back(
                {FilterConditionTokenKind::ComparisonOperator, op});
            continue;
        }

        error = "FilterRows: unsupported token '" +
                std::string(1, condition[index]) + "'";
        return false;
    }

    tokens.push_back({FilterConditionTokenKind::End, {}});
    return true;
}


const char* FilterOpWords(FilterOp op) {
    switch (op) {
        case FilterOp::Is: return "is";
        case FilterOp::IsNot: return "is not";
        case FilterOp::MoreThan: return "is more than";
        case FilterOp::LessThan: return "is less than";
        case FilterOp::AtLeast: return "is at least";
        case FilterOp::AtMost: return "is at most";
        case FilterOp::IsEmpty: return "is empty";
        case FilterOp::IsNotEmpty: return "is not empty";
    }
    return "is";
}

bool FilterOpTakesValue(FilterOp op) {
    return op != FilterOp::IsEmpty && op != FilterOp::IsNotEmpty;
}

bool FilterOpAllowed(FilterOp op, FilterValueKind kind) {
    return kind == FilterValueKind::Number || !FilterOpTakesValue(op) || op == FilterOp::Is ||
           op == FilterOp::IsNot;
}

std::string FilterRowProblem(const FilterRow& row) {
    if (row.column.empty()) return "Pick a column";
    if (!FilterOpAllowed(row.op, row.kind))
        return "\"" + row.column + "\" holds " +
               (row.kind == FilterValueKind::Text ? "text" : "true/false") +
               ": it can only be compared with is, is not or the empty tests";
    if (!FilterOpTakesValue(row.op)) return {};
    if (row.kind == FilterValueKind::Number && !IsValidNumericLiteral(row.value))
        return "\"" + row.column + "\" holds numbers: enter a number";
    if (row.kind == FilterValueKind::TrueFalse && row.value != "true" && row.value != "false")
        return "\"" + row.column + "\" holds true/false: pick true or false";
    return {};
}

std::string BuildFilterCondition(const std::vector<FilterRow>& rows) {
    std::string out;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0) out += row.or_join ? " OR " : " AND ";
        out += QuoteName(row.column) + " " + OpSymbol(row.op);
        if (!FilterOpTakesValue(row.op)) continue;
        out += " ";
        switch (row.kind) {
            case FilterValueKind::Number: out += row.value; break;
            case FilterValueKind::Text: out += QuoteText(row.value); break;
            case FilterValueKind::TrueFalse: out += row.value == "true" ? "TRUE" : "FALSE"; break;
        }
    }
    return out;
}

std::optional<std::vector<FilterRow>> ParseFilterCondition(const std::string& condition) {
    std::vector<FilterConditionToken> tokens;
    std::string error;
    if (!TokenizeFilterCondition(condition, tokens, error)) return std::nullopt;
    std::vector<FilterRow> rows;
    size_t i = 0;
    while (tokens[i].kind != FilterConditionTokenKind::End) {
        FilterRow row;
        if (!rows.empty()) {
            if (tokens[i].kind == FilterConditionTokenKind::Or) row.or_join = true;
            else if (tokens[i].kind != FilterConditionTokenKind::And) return std::nullopt;
            ++i;
        }
        // End closes the list, so a column and an operator have a token after them.
        if (tokens[i].kind != FilterConditionTokenKind::Identifier) return std::nullopt;
        row.column = tokens[i].value;
        if (tokens[i + 1].kind == FilterConditionTokenKind::Is) {
            const bool negated = tokens[i + 2].kind == FilterConditionTokenKind::Not;
            const size_t empty = i + (negated ? 3 : 2);
            if (tokens[empty].kind != FilterConditionTokenKind::Empty) return std::nullopt;
            row.op = negated ? FilterOp::IsNotEmpty : FilterOp::IsEmpty;
            rows.push_back(std::move(row));
            i = empty + 1;
            continue;
        }
        if (tokens[i + 1].kind != FilterConditionTokenKind::ComparisonOperator) return std::nullopt;
        const auto op = OpFromSymbol(tokens[i + 1].value);
        if (!op) return std::nullopt;
        row.op = *op;
        const auto& literal = tokens[i + 2];
        switch (literal.kind) {
            case FilterConditionTokenKind::NumericLiteral: row.kind = FilterValueKind::Number; row.value = literal.value; break;
            case FilterConditionTokenKind::StringLiteral: row.kind = FilterValueKind::Text; row.value = literal.value; break;
            case FilterConditionTokenKind::BooleanLiteral:
                row.kind = FilterValueKind::TrueFalse;
                row.value = literal.value == "TRUE" ? "true" : "false";
                break;
            default: return std::nullopt;
        }
        rows.push_back(std::move(row));
        i += 3;
    }
    return rows;
}

}  // namespace cyxwiz
