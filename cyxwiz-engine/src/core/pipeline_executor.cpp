#include "pipeline_executor.h"
#include "duckdb_connector.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include "node_executors/pipeline_operator_factory.h"
#include "pipeline_runtime_capabilities.h"
#include <arrow/table.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <queue>
#include <sstream>
#include <future>
#include <thread>
#include <chrono>
#include <set>
#include <mutex>

namespace cyxwiz {

namespace {

bool HasNonEmptyParameter(const std::map<std::string, std::string>& parameters,
                          const std::string& name) {
    auto it = parameters.find(name);
    return it != parameters.end() && !it->second.empty();
}

std::string TrimString(const std::string& value) {
    auto begin = std::find_if_not(value.begin(), value.end(),
                                  [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(value.rbegin(), value.rend(),
                                [](unsigned char c) { return std::isspace(c); }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

std::string ToUpperAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::toupper(c));
                   });
    return value;
}

std::map<std::string, std::string> ParseRenameMapping(const std::string& mapping) {
    std::map<std::string, std::string> result;
    std::stringstream pairs(mapping);
    std::string pair;
    while (std::getline(pairs, pair, ',')) {
        std::stringstream semicolon_pairs(pair);
        std::string semicolon_pair;
        while (std::getline(semicolon_pairs, semicolon_pair, ';')) {
            const auto delimiter = semicolon_pair.find(':');
            if (delimiter == std::string::npos) {
                continue;
            }
            const std::string old_name = TrimString(semicolon_pair.substr(0, delimiter));
            const std::string new_name = TrimString(semicolon_pair.substr(delimiter + 1));
            if (!old_name.empty() && !new_name.empty()) {
                result[old_name] = new_name;
            }
        }
    }
    return result;
}

std::string ScalarToColumnName(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return {};
    }
    std::string value = TrimString(scalar->ToString());
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }
    return TrimString(value);
}

std::string QuoteSqlIdentifier(const std::string& identifier) {
    std::string quoted = "\"";
    for (char c : identifier) {
        if (c == '"') {
            quoted += "\"\"";
        } else {
            quoted += c;
        }
    }
    quoted += '"';
    return quoted;
}

std::string QuoteSqlStringLiteral(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::vector<std::string> ParseCommaSeparatedNames(const std::string& value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        name = TrimString(name);
        if (!name.empty()) {
            result.push_back(name);
        }
    }
    return result;
}

bool ResolveExistingColumns(const std::shared_ptr<arrow::Table>& table,
                            const std::string& node_type,
                            const std::string& columns,
                            std::vector<std::string>& column_names,
                            std::string& error) {
    if (!table || !table->schema()) {
        error = node_type + ": input table schema is unavailable";
        return false;
    }

    column_names = ParseCommaSeparatedNames(columns);
    if (column_names.empty()) {
        error = node_type + ": no columns were provided";
        return false;
    }

    for (const auto& column : column_names) {
        if (table->schema()->GetFieldIndex(column) < 0) {
            error = node_type + ": column '" + column + "' not found";
            return false;
        }
    }
    return true;
}

std::string JoinQuotedColumns(const std::vector<std::string>& column_names) {
    std::string result;
    for (size_t i = 0; i < column_names.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += QuoteSqlIdentifier(column_names[i]);
    }
    return result;
}

bool RequireColumnExists(const std::shared_ptr<arrow::Table>& table,
                         const std::string& node_type,
                         const std::string& column,
                         const std::string& table_role,
                         std::string& error) {
    if (!table || !table->schema()) {
        error = node_type + ": input table schema is unavailable";
        return false;
    }
    if (table->schema()->GetFieldIndex(column) < 0) {
        error = node_type + ": column '" + column + "' not found in " +
                table_role;
        return false;
    }
    return true;
}

bool IsNumericArrowType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) {
        return false;
    }
    switch (type->id()) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

bool IsStringArrowType(const std::shared_ptr<arrow::DataType>& type) {
    return type &&
           (type->id() == arrow::Type::STRING ||
            type->id() == arrow::Type::LARGE_STRING);
}

bool IsValidNumericLiteral(const std::string& value) {
    const std::string trimmed = TrimString(value);
    if (trimmed.empty()) {
        return false;
    }
    double parsed = 0.0;
    const char* begin = trimmed.data();
    const char* end = trimmed.data() + trimmed.size();
    const auto result = std::from_chars(begin, end, parsed);
    return result.ec == std::errc{} && result.ptr == end;
}

bool BuildFillMissingConstantExpression(
    const std::shared_ptr<arrow::Field>& field,
    const std::string& raw_value,
    std::string& expression,
    std::string& error) {
    if (!field || !field->type()) {
        error = "FillMissing: input field type is unavailable";
        return false;
    }

    const std::string trimmed_value = TrimString(raw_value);
    if (IsNumericArrowType(field->type())) {
        if (!IsValidNumericLiteral(trimmed_value)) {
            error = "FillMissing: constant value '" + raw_value +
                    "' is not numeric for column '" + field->name() + "'";
            return false;
        }
        expression = trimmed_value;
        return true;
    }

    if (IsStringArrowType(field->type())) {
        expression = QuoteSqlStringLiteral(raw_value);
        return true;
    }

    error = "FillMissing: constant fill is not supported for column '" +
            field->name() + "' of type " + field->type()->ToString();
    return false;
}

enum class FilterConditionTokenKind {
    Identifier,
    StringLiteral,
    NumericLiteral,
    ComparisonOperator,
    And,
    Or,
    LeftParen,
    RightParen,
    End,
};

struct FilterConditionToken {
    FilterConditionTokenKind kind = FilterConditionTokenKind::End;
    std::string value;
};

bool TokenizeFilterRowsCondition(const std::string& condition,
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

class FilterRowsConditionParser {
public:
    FilterRowsConditionParser(const std::shared_ptr<arrow::Table>& table,
                              const std::vector<FilterConditionToken>& tokens)
        : table_(table), tokens_(tokens) {}

    bool Parse(std::string& expression, std::string& error) {
        if (!table_ || !table_->schema()) {
            error = "FilterRows: input table schema is unavailable";
            return false;
        }
        if (!ParseExpression(expression, error)) {
            return false;
        }
        if (Peek().kind != FilterConditionTokenKind::End) {
            error = "FilterRows: unexpected token '" + Peek().value + "'";
            return false;
        }
        return true;
    }

private:
    bool ParseExpression(std::string& expression, std::string& error) {
        if (!ParseTerm(expression, error)) {
            return false;
        }
        while (Peek().kind == FilterConditionTokenKind::Or) {
            Consume();
            std::string rhs;
            if (!ParseTerm(rhs, error)) {
                return false;
            }
            expression = "(" + expression + " OR " + rhs + ")";
        }
        return true;
    }

    bool ParseTerm(std::string& expression, std::string& error) {
        if (!ParseFactor(expression, error)) {
            return false;
        }
        while (Peek().kind == FilterConditionTokenKind::And) {
            Consume();
            std::string rhs;
            if (!ParseFactor(rhs, error)) {
                return false;
            }
            expression = "(" + expression + " AND " + rhs + ")";
        }
        return true;
    }

    bool ParseFactor(std::string& expression, std::string& error) {
        if (Peek().kind == FilterConditionTokenKind::LeftParen) {
            Consume();
            if (!ParseExpression(expression, error)) {
                return false;
            }
            if (Peek().kind != FilterConditionTokenKind::RightParen) {
                error = "FilterRows: expected ')'";
                return false;
            }
            Consume();
            expression = "(" + expression + ")";
            return true;
        }
        return ParseComparison(expression, error);
    }

    bool ParseComparison(std::string& expression, std::string& error) {
        if (Peek().kind != FilterConditionTokenKind::Identifier) {
            error = "FilterRows: expected a column name";
            return false;
        }
        const std::string column = Consume().value;
        const int column_index = table_->schema()->GetFieldIndex(column);
        if (column_index < 0) {
            error = "FilterRows: column '" + column + "' not found";
            return false;
        }

        if (Peek().kind != FilterConditionTokenKind::ComparisonOperator) {
            error = "FilterRows: expected a comparison operator after column '" +
                    column + "'";
            return false;
        }
        const std::string op = Consume().value;

        const auto field = table_->schema()->field(column_index);
        if (IsNumericArrowType(field->type())) {
            if (Peek().kind != FilterConditionTokenKind::NumericLiteral) {
                error = "FilterRows: column '" + column +
                        "' requires a numeric literal";
                return false;
            }
            const std::string literal = Consume().value;
            expression = QuoteSqlIdentifier(column) + " " + op + " " + literal;
            return true;
        }

        if (IsStringArrowType(field->type())) {
            if (op != "=" && op != "!=" && op != "<>") {
                error = "FilterRows: string column '" + column +
                        "' only supports equality comparisons";
                return false;
            }
            if (Peek().kind != FilterConditionTokenKind::StringLiteral) {
                error = "FilterRows: column '" + column +
                        "' requires a quoted string literal";
                return false;
            }
            const std::string literal = Consume().value;
            expression = QuoteSqlIdentifier(column) + " " + op + " " +
                         QuoteSqlStringLiteral(literal);
            return true;
        }

        error = "FilterRows: column '" + column +
                "' has unsupported type " + field->type()->ToString();
        return false;
    }

    const FilterConditionToken& Peek() const {
        return tokens_[index_];
    }

    const FilterConditionToken& Consume() {
        return tokens_[index_++];
    }

    std::shared_ptr<arrow::Table> table_;
    const std::vector<FilterConditionToken>& tokens_;
    size_t index_ = 0;
};

bool BuildFilterRowsConditionExpression(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& condition,
    std::string& expression,
    std::string& error) {
    std::vector<FilterConditionToken> tokens;
    if (!TokenizeFilterRowsCondition(condition, tokens, error)) {
        return false;
    }
    FilterRowsConditionParser parser(table, tokens);
    return parser.Parse(expression, error);
}

bool RequireColumnKind(const std::shared_ptr<arrow::Table>& table,
                       const std::string& node_type,
                       const std::string& column,
                       const std::string& kind,
                       bool (*predicate)(const std::shared_ptr<arrow::DataType>&),
                       std::string& error) {
    if (!table || !table->schema()) {
        error = node_type + ": input table schema is unavailable";
        return false;
    }

    const int column_index = table->schema()->GetFieldIndex(column);
    if (column_index < 0) {
        error = node_type + ": column '" + column + "' not found";
        return false;
    }

    const auto field = table->schema()->field(column_index);
    if (!field || !predicate(field->type())) {
        const std::string found =
            field && field->type() ? field->type()->ToString() : "unknown";
        error = node_type + ": column '" + column + "' must be " + kind +
                " (found " + found + ")";
        return false;
    }
    return true;
}

bool IsSimpleSqlIdentifier(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    const auto is_identifier_char = [](unsigned char c) {
        return std::isalnum(c) || c == '_';
    };
    if (!(std::isalpha(static_cast<unsigned char>(value.front())) ||
          value.front() == '_')) {
        return false;
    }
    return std::all_of(value.begin(), value.end(),
                       [is_identifier_char](unsigned char c) {
                           return is_identifier_char(c);
                       });
}

bool IsNumericAggregateFunction(const std::string& function_name) {
    return function_name == "SUM" || function_name == "AVG" ||
           function_name == "MEDIAN";
}

bool IsAllowedAggregateFunction(const std::string& function_name) {
    return function_name == "COUNT" || function_name == "SUM" ||
           function_name == "AVG" || function_name == "MIN" ||
           function_name == "MAX" || function_name == "MEDIAN" ||
           function_name == "MODE";
}

bool BuildGroupByAggregationExpression(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& aggregation,
    std::string& expression,
    std::string& error) {
    const std::string spec = TrimString(aggregation);
    const auto open = spec.find('(');
    const auto close = spec.find(')', open == std::string::npos ? 0 : open + 1);
    if (open == std::string::npos || close == std::string::npos ||
        spec.find('(', open + 1) != std::string::npos ||
        spec.find(')', close + 1) != std::string::npos) {
        error = "GroupBy: unsupported aggregation '" + spec +
                "'; expected FUNC(column) or COUNT(*)";
        return false;
    }

    const std::string function_name =
        ToUpperAscii(TrimString(spec.substr(0, open)));
    if (!IsAllowedAggregateFunction(function_name)) {
        error = "GroupBy: aggregation function '" + function_name +
                "' is not supported";
        return false;
    }

    const std::string argument =
        TrimString(spec.substr(open + 1, close - open - 1));
    if (argument.empty()) {
        error = "GroupBy: aggregation '" + spec + "' has no column argument";
        return false;
    }

    std::string quoted_argument;
    if (argument == "*") {
        if (function_name != "COUNT") {
            error = "GroupBy: only COUNT(*) can use '*'";
            return false;
        }
        quoted_argument = "*";
    } else {
        if (!table || !table->schema()) {
            error = "GroupBy: input table schema is unavailable";
            return false;
        }
        const int column_index = table->schema()->GetFieldIndex(argument);
        if (column_index < 0) {
            error = "GroupBy: aggregation column '" + argument + "' not found";
            return false;
        }
        const auto field = table->schema()->field(column_index);
        if (IsNumericAggregateFunction(function_name) &&
            (!field || !IsNumericArrowType(field->type()))) {
            const std::string found =
                field && field->type() ? field->type()->ToString() : "unknown";
            error = "GroupBy: aggregation column '" + argument +
                    "' must be numeric for " + function_name +
                    " (found " + found + ")";
            return false;
        }
        quoted_argument = QuoteSqlIdentifier(argument);
    }

    expression = function_name + "(" + quoted_argument + ")";

    const std::string suffix = TrimString(spec.substr(close + 1));
    if (!suffix.empty()) {
        const std::string lowered_suffix = ToLowerAscii(suffix);
        if (lowered_suffix.rfind("as ", 0) != 0) {
            error = "GroupBy: aggregation alias must use AS";
            return false;
        }
        const std::string alias = TrimString(suffix.substr(3));
        if (!IsSimpleSqlIdentifier(alias)) {
            error = "GroupBy: aggregation alias '" + alias +
                    "' is not a valid identifier";
            return false;
        }
        expression += " AS " + QuoteSqlIdentifier(alias);
    }
    return true;
}

bool BuildGroupByAggregationExpressions(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& aggregations,
    std::string& expressions,
    std::string& error) {
    const std::vector<std::string> specs = ParseCommaSeparatedNames(aggregations);
    if (specs.empty()) {
        error = "GroupBy: no aggregations were provided";
        return false;
    }

    std::vector<std::string> built;
    built.reserve(specs.size());
    for (const auto& spec : specs) {
        std::string expression;
        if (!BuildGroupByAggregationExpression(table, spec, expression, error)) {
            return false;
        }
        built.push_back(expression);
    }

    for (size_t i = 0; i < built.size(); ++i) {
        if (i > 0) {
            expressions += ", ";
        }
        expressions += built[i];
    }
    return true;
}

bool BuildMathFormulaExpression(const std::shared_ptr<arrow::Table>& table,
                                const std::string& formula,
                                std::string& expression,
                                std::string& error) {
    if (!table || !table->schema()) {
        error = "MathFormula: input table schema is unavailable";
        return false;
    }

    expression.clear();
    bool saw_token = false;
    size_t index = 0;
    while (index < formula.size()) {
        const unsigned char c = static_cast<unsigned char>(formula[index]);
        if (std::isspace(c)) {
            expression.push_back(' ');
            ++index;
            continue;
        }

        if (std::isalpha(c) || c == '_') {
            const size_t start = index;
            ++index;
            while (index < formula.size()) {
                const unsigned char id_char =
                    static_cast<unsigned char>(formula[index]);
                if (!std::isalnum(id_char) && id_char != '_') {
                    break;
                }
                ++index;
            }

            const std::string column = formula.substr(start, index - start);
            const int column_index = table->schema()->GetFieldIndex(column);
            if (column_index < 0) {
                error = "MathFormula: formula references unknown column '" +
                        column + "'";
                return false;
            }
            const auto field = table->schema()->field(column_index);
            if (!field || !IsNumericArrowType(field->type())) {
                const std::string found =
                    field && field->type() ? field->type()->ToString()
                                           : "unknown";
                error = "MathFormula: formula column '" + column +
                        "' must be numeric (found " + found + ")";
                return false;
            }
            expression += QuoteSqlIdentifier(column);
            saw_token = true;
            continue;
        }

        if (std::isdigit(c) ||
            (formula[index] == '.' && index + 1 < formula.size() &&
             std::isdigit(static_cast<unsigned char>(formula[index + 1])))) {
            const size_t start = index;
            bool saw_digit = false;
            bool saw_dot = false;
            while (index < formula.size()) {
                const unsigned char number_char =
                    static_cast<unsigned char>(formula[index]);
                if (std::isdigit(number_char)) {
                    saw_digit = true;
                    ++index;
                    continue;
                }
                if (formula[index] == '.' && !saw_dot) {
                    saw_dot = true;
                    ++index;
                    continue;
                }
                break;
            }
            if (index < formula.size() &&
                (formula[index] == 'e' || formula[index] == 'E')) {
                const size_t exponent = index;
                ++index;
                if (index < formula.size() &&
                    (formula[index] == '+' || formula[index] == '-')) {
                    ++index;
                }
                const size_t exponent_digits = index;
                while (index < formula.size() &&
                       std::isdigit(static_cast<unsigned char>(formula[index]))) {
                    ++index;
                }
                if (exponent_digits == index) {
                    index = exponent;
                }
            }
            if (!saw_digit) {
                error = "MathFormula: invalid numeric literal";
                return false;
            }
            expression += formula.substr(start, index - start);
            saw_token = true;
            continue;
        }

        if (formula[index] == '+' || formula[index] == '-' ||
            formula[index] == '*' || formula[index] == '/' ||
            formula[index] == '(' || formula[index] == ')') {
            expression.push_back(formula[index]);
            ++index;
            saw_token = true;
            continue;
        }

        error = "MathFormula: formula contains unsupported token '" +
                std::string(1, formula[index]) + "'";
        return false;
    }

    if (!saw_token) {
        error = "MathFormula: Formula required";
        return false;
    }
    return true;
}

bool IsIntegerAtLeast(const std::string& value, int64_t minimum) {
    if (value.empty()) {
        return false;
    }
    int64_t parsed = 0;
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    auto [ptr, ec] = std::from_chars(begin, end, parsed);
    return ec == std::errc() && ptr == end && parsed >= minimum;
}

bool ValidateIntegerParameterAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    if (IsIntegerAtLeast(it->second, minimum)) {
        return true;
    }
    error = node_type + " " + parameter_name + " must be an integer >= " +
            std::to_string(minimum);
    return false;
}

bool ValidateCommaSeparatedIntegersAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end()) {
        return true;
    }
    if (it->second.empty()) {
        error = node_type + " " + parameter_name +
                " must be a comma-separated list of integers >= " +
                std::to_string(minimum);
        return false;
    }

    std::stringstream values(it->second);
    std::string value;
    while (std::getline(values, value, ',')) {
        if (!IsIntegerAtLeast(value, minimum)) {
            error = node_type + " " + parameter_name +
                    " must be a comma-separated list of integers >= " +
                    std::to_string(minimum);
            return false;
        }
    }
    return true;
}

bool IsAllowedParameterValue(
    const PipelineAllowedParameterValuesRuntimeCapability& capability,
    const std::string& value) {
    const std::string normalized_value = ToLowerAscii(TrimString(value));
    return std::find_if(capability.allowed_values.begin(),
                        capability.allowed_values.end(),
                        [&normalized_value](const char* allowed) {
                            return allowed != nullptr &&
                                   normalized_value ==
                                       ToLowerAscii(TrimString(allowed));
                        }) != capability.allowed_values.end();
}

bool TryParseBooleanParameterValue(const std::string& value, bool& out) {
    const std::string normalized_value = ToLowerAscii(TrimString(value));
    if (normalized_value == "true") {
        out = true;
        return true;
    }
    if (normalized_value == "false") {
        out = false;
        return true;
    }
    return false;
}

bool ValidateOptionalBooleanParameter(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const char* parameter_name,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }

    bool parsed = false;
    if (!TryParseBooleanParameterValue(it->second, parsed)) {
        error = node_type + ": '" + parameter_name +
                "' must be 'true' or 'false'";
        return false;
    }
    return true;
}

bool OptionalBooleanParameterIsTrue(
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return false;
    }

    bool parsed = false;
    return TryParseBooleanParameterValue(it->second, parsed) && parsed;
}

std::string NormalizeSortOrder(
    const std::map<std::string, std::string>& parameters) {
    auto order_it = parameters.find("order");
    if (order_it != parameters.end() && !order_it->second.empty()) {
        return ToUpperAscii(TrimString(order_it->second));
    }

    auto ascending_it = parameters.find("ascending");
    if (ascending_it != parameters.end() && !ascending_it->second.empty() &&
        ToLowerAscii(TrimString(ascending_it->second)) == "false") {
        return "DESC";
    }
    return "ASC";
}

std::string NormalizeJoinTypeSql(const std::string& value) {
    const std::string join_type = ToLowerAscii(TrimString(value));
    if (join_type == "left") {
        return "LEFT";
    }
    if (join_type == "right") {
        return "RIGHT";
    }
    if (join_type == "outer") {
        return "FULL OUTER";
    }
    return "INNER";
}

std::string NormalizeDataInputFileType(const std::string& value) {
    const std::string file_type = ToLowerAscii(TrimString(value));
    return file_type.empty() ? "auto" : file_type;
}

std::string NormalizeBinningMethod(const std::string& value) {
    const std::string method = ToLowerAscii(TrimString(value));
    if (method == "equal_frequency") {
        return "equal_freq";
    }
    return method.empty() ? "equal_width" : method;
}

const char* MissingRequiredParameter(
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const std::vector<const char*>& required_parameters) {
    if (node_type == "DataInput") {
        const auto source_it = parameters.find("source_type");
        const std::string source_type =
            (source_it != parameters.end() && !source_it->second.empty())
                ? ToLowerAscii(TrimString(source_it->second))
                : "file";
        if (source_type == "file") {
            return HasNonEmptyParameter(parameters, "file_path") ? nullptr : "file_path";
        }
        if (source_type == "folder") {
            return HasNonEmptyParameter(parameters, "folder_path") ? nullptr : "folder_path";
        }
        return nullptr;
    }

    if (node_type == "RenameColumns") {
        return (HasNonEmptyParameter(parameters, "mapping") ||
                HasNonEmptyParameter(parameters, "rename_map"))
            ? nullptr
            : "mapping";
    }

    if (node_type == "ExportCSV") {
        return (HasNonEmptyParameter(parameters, "file_path") ||
                HasNonEmptyParameter(parameters, "path"))
            ? nullptr
            : "file_path";
    }

    for (const char* parameter : required_parameters) {
        if (!HasNonEmptyParameter(parameters, parameter)) {
            return parameter;
        }
    }

    return nullptr;
}

bool HasSupportedParameterValues(
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const std::vector<PipelineAllowedParameterValuesRuntimeCapability>&
        allowed_parameter_values,
    const std::vector<PipelineIntegerParameterRuntimeCapability>&
        integer_parameters,
    std::string& error) {
    for (const auto& capability : allowed_parameter_values) {
        auto it = parameters.find(capability.parameter_name);
        const std::string value =
            (it != parameters.end() && !it->second.empty())
                ? it->second
                : capability.default_value;
        if (!IsAllowedParameterValue(capability, value)) {
            error = node_type + " " + capability.parameter_name + " '" +
                    value + "' is not supported by PipelineExecutor";
            return false;
        }
    }

    for (const auto& capability : integer_parameters) {
        if (capability.comma_separated) {
            if (!ValidateCommaSeparatedIntegersAtLeast(
                    parameters, node_type, capability.parameter_name,
                    capability.minimum, error)) {
                return false;
            }
        } else if (!ValidateIntegerParameterAtLeast(
                       parameters, node_type, capability.parameter_name,
                       capability.minimum, error)) {
            return false;
        }
    }

    if (node_type == "StringManipulation") {
        const auto operation_it = parameters.find("operation");
        const std::string operation =
            (operation_it != parameters.end() && !operation_it->second.empty())
                ? ToLowerAscii(TrimString(operation_it->second))
                : "trim";
        if (operation == "replace" &&
            !HasNonEmptyParameter(parameters, "param1")) {
            error = "StringManipulation replace requires param1";
            return false;
        }
        if (operation == "substring") {
            const auto start_it = parameters.find("param1");
            const auto length_it = parameters.find("param2");
            if (start_it == parameters.end() || start_it->second.empty() ||
                !IsIntegerAtLeast(start_it->second, 1)) {
                error = "StringManipulation substring param1 must be an integer >= 1";
                return false;
            }
            if (length_it == parameters.end() || length_it->second.empty() ||
                !IsIntegerAtLeast(length_it->second, 0)) {
                error = "StringManipulation substring param2 must be an integer >= 0";
                return false;
            }
        }
    }

    if (node_type == "TextClean") {
        if (!ValidateOptionalBooleanParameter(
                parameters, node_type, "lowercase", error) ||
            !ValidateOptionalBooleanParameter(
                parameters, node_type, "remove_html", error) ||
            !ValidateOptionalBooleanParameter(
                parameters, node_type, "remove_special_chars", error) ||
            !ValidateOptionalBooleanParameter(
                parameters, node_type, "remove_stopwords", error)) {
            return false;
        }
        const auto remove_stopwords_it = parameters.find("remove_stopwords");
        if (remove_stopwords_it != parameters.end() &&
            OptionalBooleanParameterIsTrue(parameters, "remove_stopwords")) {
            error = "TextClean remove_stopwords is not supported by PipelineExecutor";
            return false;
        }
    }

    if (node_type == "TextVectorize" &&
        HasNonEmptyParameter(parameters, "max_features")) {
        error = "TextVectorize max_features is not supported by the legacy PipelineExecutor path; use CountVectorizer or TFIDFVectorizer";
        return false;
    }

    if (node_type == "TSWindow") {
        const auto stride_it = parameters.find("stride");
        if (stride_it != parameters.end() && !stride_it->second.empty() &&
            std::stoi(stride_it->second) != 1) {
            error = "TSWindow stride values other than 1 are not supported by PipelineExecutor";
            return false;
        }
    }

    if (node_type == "Binning") {
        const auto columns_it = parameters.find("columns");
        if (columns_it != parameters.end() &&
            columns_it->second.find(',') != std::string::npos) {
            error = "Binning columns supports exactly one column";
            return false;
        }
    }

    if (node_type == "PolynomialFeatures") {
        const auto columns_it = parameters.find("columns");
        if (columns_it != parameters.end() &&
            columns_it->second.find(',') != std::string::npos) {
            error = "PolynomialFeatures columns supports exactly one column";
            return false;
        }
    }

    if (node_type == "DataInput") {
        if (!ValidateOptionalBooleanParameter(
                parameters, node_type, "has_header", error) ||
            !ValidateOptionalBooleanParameter(
                parameters, node_type, "json_lines", error)) {
            return false;
        }

        const auto source_it = parameters.find("source_type");
        const std::string source_type =
            (source_it != parameters.end() && !source_it->second.empty())
                ? ToLowerAscii(TrimString(source_it->second))
                : "file";

        const auto type_it = parameters.find("type");
        const std::string file_type =
            (type_it != parameters.end())
                ? NormalizeDataInputFileType(type_it->second)
                : "auto";
        const auto skip_rows_it = parameters.find("skip_rows");
        if (source_type == "file" && skip_rows_it != parameters.end() &&
            !skip_rows_it->second.empty() &&
            !IsIntegerAtLeast(skip_rows_it->second, 0)) {
            error = "DataInput skip_rows must be a non-negative integer";
            return false;
        }
        const auto sheet_idx_it = parameters.find("sheet_idx");
        if (source_type == "file" && file_type == "excel" &&
            sheet_idx_it != parameters.end() && !sheet_idx_it->second.empty() &&
            !IsIntegerAtLeast(sheet_idx_it->second, 0)) {
            error = "DataInput sheet_idx must be a non-negative integer";
            return false;
        }
    }

    return true;
}

} // namespace

PipelineExecutor::PipelineExecutor()
    : executing_(false)
    , progress_(0.0f)
    , stop_requested_(false)
    , cancel_requested_(false)
    , deployment_ready_(false)
{
    // Create DuckDB connector for SQL transformations
    duckdb_ = std::make_unique<DuckDBConnector>();
    spdlog::info("[Data Studio] PipelineExecutor initialized with DuckDB");
}

PipelineExecutor::~PipelineExecutor() = default;

bool PipelineExecutor::ExecutePipeline(const std::string& pipeline_json) {
    if (executing_) {
        last_error_ = "Pipeline is already executing";
        return false;
    }

    executing_ = true;
    progress_ = 0.0f;
    stop_requested_ = false;
    cancel_requested_ = false;
    last_error_ = "";
    current_status_ = "Starting pipeline execution...";
    deployment_ready_ = false;
    deployment_dataset_.clear();

    spdlog::info("[Data Studio] Starting pipeline execution");

    // Parse pipeline
    std::vector<Node> nodes;
    if (!ParsePipeline(pipeline_json, nodes)) {
        const std::string parse_error = last_error_.empty()
            ? "Failed to parse pipeline"
            : "Failed to parse pipeline: " + last_error_;
        ReportError(parse_error);
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.1f, "Pipeline parsed successfully");

    // Validate pipeline
    if (!ValidatePipeline(nodes)) {
        const std::string validation_error = last_error_.empty()
            ? "Pipeline validation failed"
            : "Pipeline validation failed: " + last_error_;
        ReportError(validation_error);
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.2f, "Pipeline validated");

    // Phase 8: Mark nodes that need execution (lazy evaluation)
    MarkDirtyNodes(nodes);

    int nodes_to_execute = 0;
    for (const auto& node : nodes) {
        if (node.needs_execution) {
            nodes_to_execute++;
        }
    }

    spdlog::info("[Data Studio] Lazy evaluation: {} of {} nodes need execution",
                 nodes_to_execute, nodes.size());
    UpdateProgress(0.25f, "Marked " + std::to_string(nodes_to_execute) + " nodes for execution");

    // Phase 8: Use parallel execution instead of sequential
    if (!ExecuteParallel(nodes)) {
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    executing_ = false;
    UpdateProgress(1.0f, "Pipeline execution completed");

    // Deployment status is set inside ExecuteParallel
    NotifyCompletion(true);

    spdlog::info("[Data Studio] Pipeline execution completed successfully");
    return true;
}

void PipelineExecutor::StopExecution() {
    if (executing_) {
        stop_requested_ = true;
        spdlog::info("[Data Studio] Stop requested for pipeline execution");
    }
}

void PipelineExecutor::SetProgressCallback(std::function<void(float, const std::string&)> callback) {
    progress_callback_ = callback;
}

void PipelineExecutor::SetCompletionCallback(std::function<void(bool)> callback) {
    completion_callback_ = callback;
}

void PipelineExecutor::RequestCancel() {
    cancel_requested_ = true;
    spdlog::info("[Data Studio] Cancellation requested");
}

bool PipelineExecutor::ParsePipeline(const std::string& pipeline_json,
                                    std::vector<Node>& nodes) {
    try {
        auto j = nlohmann::json::parse(pipeline_json);

        for (const auto& node_json : j["nodes"]) {
            Node node;
            node.id = node_json["id"];
            node.type = node_json["type"];
            node.runtime_type = ResolvePipelineRuntimeNodeType(node.type);
            node.name = node_json["name"];
            node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            nodes.push_back(node);
        }

        // Build input/output connections
        for (const auto& link_json : j["links"]) {
            int start_node = link_json["start_node"];
            int end_node = link_json["end_node"];

            auto start_it = std::find_if(nodes.begin(), nodes.end(),
                                        [start_node](const Node& n) { return n.id == start_node; });
            auto end_it = std::find_if(nodes.begin(), nodes.end(),
                                      [end_node](const Node& n) { return n.id == end_node; });

            if (start_it == nodes.end()) {
                last_error_ = "Link references missing start node id: " +
                              std::to_string(start_node);
                return false;
            }
            if (end_it == nodes.end()) {
                last_error_ = "Link references missing end node id: " +
                              std::to_string(end_node);
                return false;
            }

            start_it->outputs.push_back(end_node);
            end_it->inputs.push_back(start_node);
        }

        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("JSON parse error: ") + e.what();
        return false;
    }
}

PipelineRuntimeSupport
PipelineExecutor::ResolveNodeRuntimeSupport(const Node& node) const {
    if (node.runtime_type.has_value()) {
        auto support = ResolvePipelineRuntimeSupport(*node.runtime_type);
        if (support.mode != PipelineRuntimeSupportMode::Unknown) {
            return support;
        }
    }
    return ResolvePipelineRuntimeSupport(node.type);
}

bool PipelineExecutor::ValidatePipeline(const std::vector<Node>& nodes) {
    // Check that there's at least one node
    if (nodes.empty()) {
        last_error_ = "Pipeline is empty";
        return false;
    }

    std::set<int> ids;
    for (const auto& node : nodes) {
        if (!ids.insert(node.id).second) {
            last_error_ = "Pipeline contains duplicate node id: " + std::to_string(node.id);
            return false;
        }
    }

    for (const auto& node : nodes) {
        for (int input_id : node.inputs) {
            if (ids.find(input_id) == ids.end()) {
                last_error_ = "Node '" + node.name + "' has missing input node id: " +
                              std::to_string(input_id);
                return false;
            }
            if (input_id == node.id) {
                last_error_ = "Node '" + node.name + "' cannot connect to itself";
                return false;
            }
        }

        for (int output_id : node.outputs) {
            if (ids.find(output_id) == ids.end()) {
                last_error_ = "Node '" + node.name + "' has missing output node id: " +
                              std::to_string(output_id);
                return false;
            }
            if (output_id == node.id) {
                last_error_ = "Node '" + node.name + "' cannot connect to itself";
                return false;
            }
        }

        const auto runtime_support = ResolveNodeRuntimeSupport(node);
        const bool is_source = runtime_support.source_node;
        const auto required_input_count =
            runtime_support.required_input_count;

        if (is_source && !node.inputs.empty()) {
            last_error_ = "Source node '" + node.name + "' must not have input connections";
            return false;
        }

        if (!is_source && node.inputs.empty()) {
            last_error_ = "Node '" + node.name + "' requires an input connection";
            return false;
        }

        if (required_input_count.has_value() &&
            static_cast<int>(node.inputs.size()) != *required_input_count) {
            last_error_ = "Node '" + node.name + "' requires exactly " +
                          std::to_string(*required_input_count) +
                          " input connections";
            return false;
        }

        if (!required_input_count.has_value() && node.inputs.size() > 1) {
            last_error_ = "Node '" + node.name + "' has multiple inputs, but node type '" +
                          node.type + "' does not define multi-input execution";
            return false;
        }

        if (const char* missing_parameter =
                MissingRequiredParameter(node.type, node.parameters,
                                         runtime_support.required_parameters);
            missing_parameter != nullptr) {
            last_error_ = "Node '" + node.name + "' of type '" + node.type +
                          "' is missing required parameter '" + missing_parameter + "'";
            return false;
        }

        std::string parameter_error;
        if (!HasSupportedParameterValues(
                node.type,
                node.parameters,
                runtime_support.allowed_parameter_values,
                runtime_support.integer_parameters,
                parameter_error)) {
            last_error_ = "Node '" + node.name + "': " + parameter_error;
            return false;
        }

        if (runtime_support.mode == PipelineRuntimeSupportMode::Unknown) {
            last_error_ = "Node '" + node.name + "' has unsupported node type '" +
                          node.type + "' for PipelineExecutor";
            return false;
        }

        if (!runtime_support.pipeline_executor_supported) {
            last_error_ = "Node '" + node.name + "' of type '" + node.type +
                          "' is not supported by PipelineExecutor";
            if (runtime_support.fail_closed_reason != nullptr) {
                last_error_ += ": ";
                last_error_ += runtime_support.fail_closed_reason;
            }
            return false;
        }
    }

    if (nodes.size() > 1) {
        std::set<int> visited;
        std::queue<int> pending;
        pending.push(nodes.front().id);
        visited.insert(nodes.front().id);

        while (!pending.empty()) {
            const int current = pending.front();
            pending.pop();

            auto current_it = std::find_if(
                nodes.begin(), nodes.end(),
                [current](const Node& node) { return node.id == current; });
            if (current_it == nodes.end()) {
                continue;
            }

            for (int input_id : current_it->inputs) {
                if (visited.insert(input_id).second) {
                    pending.push(input_id);
                }
            }
            for (int output_id : current_it->outputs) {
                if (visited.insert(output_id).second) {
                    pending.push(output_id);
                }
            }
        }

        if (visited.size() != nodes.size()) {
            last_error_ = "Pipeline contains disconnected nodes";
            return false;
        }
    }

    if (TopologicalSort(nodes).empty() && !nodes.empty()) {
        last_error_ = "Pipeline contains a cycle";
        return false;
    }

    return true;
}

std::vector<int> PipelineExecutor::TopologicalSort(const std::vector<Node>& nodes) {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Build adjacency list and in-degree map
    for (const auto& node : nodes) {
        in_degree[node.id] = static_cast<int>(node.inputs.size());
        adj_list[node.id] = node.outputs;
    }

    // Queue for nodes with no dependencies
    std::queue<int> q;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            q.push(id);
        }
    }

    // Process nodes
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);

        // Reduce in-degree for neighbors
        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // Check if all nodes were processed (cycle detection)
    if (result.size() != nodes.size()) {
        return {};  // Cycle detected
    }

    return result;
}

bool PipelineExecutor::ExecuteNode(const Node& node, ExecutionContext& ctx) {
    spdlog::debug("[Data Studio] Executing node: {} (type: {})", node.name, node.type);

    bool typed_legacy_handled = false;
    if (ExecuteTypedLegacyNode(node, ctx, typed_legacy_handled)) {
        return true;
    }
    if (typed_legacy_handled) {
        return false;
    }

    const auto support = ResolveNodeRuntimeSupport(node);
    if (support.mode == PipelineRuntimeSupportMode::OperatorBacked &&
        support.operator_type.has_value()) {
        return ExecutePipelineOperatorNode(node, ctx, *support.operator_type);
    } else if (support.mode == PipelineRuntimeSupportMode::FailClosed &&
               support.fail_closed_reason != nullptr) {
        return FailUnsupportedNode(node, support.fail_closed_reason);
    }

    if (support.mode == PipelineRuntimeSupportMode::LegacyExecutor &&
        support.legacy_dispatch_kind != PipelineLegacyDispatchKind::Unknown) {
        return ExecuteLegacyDispatchKind(node, ctx,
                                         support.legacy_dispatch_kind);
    }

    ReportError("Unknown node type: " + node.type);
    return false;
}

bool PipelineExecutor::ExecuteTypedLegacyNode(const Node& node,
                                              ExecutionContext& ctx,
                                              bool& handled) {
    handled = false;
    if (!node.runtime_type.has_value()) {
        return false;
    }

    handled = true;
    switch (*node.runtime_type) {
    case gui::NodeType::CSVFile:
        return ExecuteFileInput(node, ctx);
    case gui::NodeType::ExcelFile:
        return ExecuteExcelInput(node, ctx);
    case gui::NodeType::DataInput:
        return ExecuteDataInput(node, ctx);
    case gui::NodeType::DataOutput:
        return ExecuteDataOutput(node, ctx);
    case gui::NodeType::FilterRows:
        return ExecuteFilterRows(node, ctx);
    case gui::NodeType::SelectColumns:
        return ExecuteSelectColumns(node, ctx);
    case gui::NodeType::RemoveDuplicateRows:
        return ExecuteRemoveDuplicates(node, ctx);
    case gui::NodeType::FillMissingValues:
        return ExecuteFillMissing(node, ctx);
    case gui::NodeType::SortRows:
        return ExecuteSortRows(node, ctx);
    case gui::NodeType::JoinTables:
        return ExecuteJoin(node, ctx);
    case gui::NodeType::GroupByAggregate:
        return ExecuteGroupBy(node, ctx);
    case gui::NodeType::ExportCSV:
        return ExecuteExportCSV(node, ctx);
    case gui::NodeType::RowToColumnNames:
        return ExecuteRowToColumnNames(node, ctx);
    case gui::NodeType::TableCropper:
        return ExecuteTableCropper(node, ctx);
    case gui::NodeType::StringManipulation:
        return ExecuteStringManipulation(node, ctx);
    case gui::NodeType::MathFormula:
        return ExecuteMathFormula(node, ctx);
    case gui::NodeType::RenameColumns:
        return ExecuteRenameColumns(node, ctx);
    default:
        handled = false;
        return false;
    }
}

bool PipelineExecutor::ExecuteLegacyDispatchKind(
    const Node& node,
    ExecutionContext& ctx,
    PipelineLegacyDispatchKind dispatch_kind) {
    switch (dispatch_kind) {
    case PipelineLegacyDispatchKind::SaveDataset:
        return ExecuteSaveDataset(node, ctx);
    case PipelineLegacyDispatchKind::DeployToNodeEditor:
        return ExecuteDeployToNodeEditor(node, ctx);
    case PipelineLegacyDispatchKind::TextClean:
        return ExecuteTextClean(node, ctx);
    case PipelineLegacyDispatchKind::TextTokenize:
        return ExecuteTextTokenize(node, ctx);
    case PipelineLegacyDispatchKind::TextVectorize:
        return ExecuteTextVectorize(node, ctx);
    case PipelineLegacyDispatchKind::TSWindow:
        return ExecuteTSWindow(node, ctx);
    case PipelineLegacyDispatchKind::TSFeatures:
        return ExecuteTSFeatures(node, ctx);
    case PipelineLegacyDispatchKind::TSLag:
        return ExecuteTSLag(node, ctx);
    case PipelineLegacyDispatchKind::TSDiff:
        return ExecuteTSDiff(node, ctx);
    case PipelineLegacyDispatchKind::PolynomialFeatures:
        return ExecutePolynomialFeatures(node, ctx);
    case PipelineLegacyDispatchKind::Binning:
        return ExecuteBinning(node, ctx);
    case PipelineLegacyDispatchKind::Unknown:
        break;
    }

    ReportError("Unknown legacy dispatch kind for node type: " + node.type);
    return false;
}

bool PipelineExecutor::ExecuteFileInput(const Node& node, ExecutionContext& ctx) {
    auto path_it = node.parameters.find("path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError(GetImprovedErrorMessage("FileInput", "missing_parameter", "path"));
        return false;
    }

    const std::string& file_path = path_it->second;
    std::string dataset_name = "ds_input_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Loading file: {} as dataset '{}'", file_path, dataset_name);

    try {
        // Use DataRegistry's Arrow support to load the file
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);

        if (!arrow_dataset) {
            ReportError(GetImprovedErrorMessage("FileInput", "invalid_path", file_path));
            return false;
        }

        // Store the dataset name for downstream nodes
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) {
            ctx.input_dataset = dataset_name;
        }

        spdlog::info("[Data Studio] FileInput loaded {} rows, {} columns",
                    arrow_dataset->GetNumRows(), arrow_dataset->GetNumColumns());
        return true;

    } catch (const std::exception& e) {
        ReportError("FileInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteDataInput(const Node& node, ExecutionContext& ctx) {
    // Universal DataInput node - supports multiple source types from DataInputDialog
    // Parameters: source_type, file_path, folder_path, file_category, type, etc.

    auto source_type_it = node.parameters.find("source_type");
    std::string source_type =
        (source_type_it != node.parameters.end() && !source_type_it->second.empty())
            ? ToLowerAscii(TrimString(source_type_it->second))
            : "file";

    std::string dataset_name = "ds_datainput_" + std::to_string(node.id);

    try {
        auto& registry = DataRegistry::Instance();
        std::shared_ptr<ArrowDataset> arrow_dataset;

        if (source_type == "file") {
            // File input mode
            auto path_it = node.parameters.find("file_path");
            if (path_it == node.parameters.end() || path_it->second.empty()) {
                ReportError(GetImprovedErrorMessage("DataInput", "missing_parameter", "file_path"));
                return false;
            }
            const std::string& file_path = path_it->second;
            spdlog::info("[Pipeline] DataInput loading file: {}", file_path);

            // Get file type and options from parameters
            auto type_it = node.parameters.find("type");
            std::string file_type =
                (type_it != node.parameters.end())
                    ? NormalizeDataInputFileType(type_it->second)
                    : "auto";

            // Load based on file type
            if (file_type == "csv" || file_type == "tsv") {
                bool has_header =
                    OptionalBooleanParameterIsTrue(node.parameters,
                                                   "has_header");
                std::string delimiter = (file_type == "tsv") ? "\t" : ",";
                auto delim_it = node.parameters.find("delimiter");
                if (delim_it != node.parameters.end() && !delim_it->second.empty()) {
                    delimiter = delim_it->second;
                }
                int skip_rows = 0;
                auto skip_it = node.parameters.find("skip_rows");
                if (skip_it != node.parameters.end()) {
                    skip_rows = std::stoi(skip_it->second);
                }

                arrow_dataset = registry.LoadCSVToArrow(file_path, dataset_name, has_header, delimiter[0], skip_rows);
            } else if (file_type == "parquet") {
                arrow_dataset = registry.LoadParquetToArrow(file_path, dataset_name);
            } else if (file_type == "json") {
                bool json_lines =
                    OptionalBooleanParameterIsTrue(node.parameters,
                                                   "json_lines");
                arrow_dataset = registry.LoadJSONToArrow(file_path, dataset_name, json_lines);
            } else if (file_type == "excel") {
                int sheet_idx = 0;
                auto sheet_it = node.parameters.find("sheet_idx");
                if (sheet_it != node.parameters.end()) {
                    sheet_idx = std::stoi(sheet_it->second);
                }
                arrow_dataset = registry.LoadExcelToArrow(file_path, dataset_name, sheet_idx);
            } else if (file_type == "auto" || file_type == "feather" ||
                       file_type == "arrow" || file_type == "ipc") {
                // Default: try auto-detect via LoadArrowTable
                arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);
            } else {
                ReportError("DataInput: file type '" + file_type +
                            "' is not supported by PipelineExecutor");
                return false;
            }

        } else if (source_type == "folder") {
            // Image folder mode
            auto path_it = node.parameters.find("folder_path");
            if (path_it == node.parameters.end() || path_it->second.empty()) {
                ReportError(GetImprovedErrorMessage("DataInput", "missing_parameter", "folder_path"));
                return false;
            }
            const std::string& folder_path = path_it->second;
            spdlog::info("[Pipeline] DataInput loading folder: {}", folder_path);

            // For image folders, create a table with file paths and labels
            arrow_dataset = registry.LoadImageFolderToArrow(folder_path, dataset_name);

        } else if (source_type == "ml_dataset") {
            // ML dataset (MNIST, CIFAR, etc.)
            auto ml_type_it = node.parameters.find("ml_dataset_type");
            std::string ml_type = (ml_type_it != node.parameters.end()) ? ml_type_it->second : "mnist";
            spdlog::info("[Pipeline] DataInput loading ML dataset: {}", ml_type);

            arrow_dataset = registry.LoadMLDatasetToArrow(ml_type, dataset_name);
        } else {
            ReportError("DataInput: Unknown source type: " + source_type);
            return false;
        }

        if (!arrow_dataset) {
            ReportError("DataInput: Failed to load dataset");
            return false;
        }

        // Store result for downstream nodes
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) {
            ctx.input_dataset = dataset_name;
        }

        spdlog::info("[Pipeline] DataInput loaded {} rows, {} columns",
                    arrow_dataset->GetNumRows(), arrow_dataset->GetNumColumns());
        return true;

    } catch (const std::exception& e) {
        ReportError("DataInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteDataOutput(const Node& node, ExecutionContext& ctx) {
    // Universal DataOutput node - exports data to various formats

    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("DataOutput: No input connection or dataset not found");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError(GetImprovedErrorMessage("DataOutput", "missing_parameter", "file_path"));
        return false;
    }
    const std::string& output_path = path_it->second;

    auto format_it = node.parameters.find("format");
    std::string format =
        (format_it != node.parameters.end() && !format_it->second.empty())
            ? ToLowerAscii(TrimString(format_it->second))
            : "csv";

    spdlog::info("[Pipeline] DataOutput exporting to {} (format: {})", output_path, format);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("DataOutput: Input dataset not found in registry");
            return false;
        }

        bool success = false;
        if (format == "csv") {
            success = registry.ExportArrowToCSV(input_dataset_name, output_path);
        } else if (format == "parquet") {
            success = registry.ExportArrowToParquet(input_dataset_name, output_path);
        } else if (format == "json") {
            success = registry.ExportArrowToJSON(input_dataset_name, output_path);
        } else {
            ReportError("DataOutput: Unsupported export format: " + format);
            return false;
        }

        if (!success) {
            ReportError("DataOutput: Export failed");
            return false;
        }

        // Pass through the dataset for any downstream nodes
        ctx.node_results[node.id] = input_dataset_name;
        ctx.output_dataset = output_path;

        spdlog::info("[Pipeline] DataOutput successfully exported to {}", output_path);
        return true;

    } catch (const std::exception& e) {
        ReportError("DataOutput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteFilterRows(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get filter condition from parameters
    auto condition_it = node.parameters.find("condition");
    if (condition_it == node.parameters.end() || condition_it->second.empty()) {
        ReportError("FilterRows: Missing 'condition' parameter");
        return false;
    }

    const std::string& condition = condition_it->second;
    std::string output_dataset_name = "ds_filter_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Filtering rows from '{}' with condition: {}",
                input_dataset_name, condition);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("FilterRows: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string filter_expression;
        std::string filter_error;
        if (!BuildFilterRowsConditionExpression(input_table, condition,
                                                filter_expression,
                                                filter_error)) {
            ReportError(filter_error);
            return false;
        }

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("FilterRows: Failed to register table with DuckDB");
            return false;
        }

        // Execute WHERE query
        std::string sql = "SELECT * FROM " + temp_table + " WHERE " +
                          filter_expression;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("FilterRows: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] FilterRows: {} -> {} rows",
                    input_table->num_rows(), result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("FilterRows error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSelectColumns(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get columns from parameters
    auto columns_it = node.parameters.find("columns");
    if (columns_it == node.parameters.end() || columns_it->second.empty()) {
        ReportError("SelectColumns: Missing 'columns' parameter");
        return false;
    }

    const std::string& columns = columns_it->second;
    std::string output_dataset_name = "ds_select_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Selecting columns from '{}': {}",
                input_dataset_name, columns);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("SelectColumns: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::vector<std::string> selected_columns;
        std::string schema_error;
        if (!ResolveExistingColumns(input_table, "SelectColumns", columns,
                                    selected_columns, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_columns = JoinQuotedColumns(selected_columns);

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("SelectColumns: Failed to register table with DuckDB");
            return false;
        }

        // Execute SELECT columns query
        std::string sql = "SELECT " + quoted_columns + " FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("SelectColumns: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] SelectColumns: {} -> {} columns",
                    input_table->num_columns(), result_table->num_columns());
        return true;

    } catch (const std::exception& e) {
        ReportError("SelectColumns error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRemoveDuplicates(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    std::string output_dataset_name = "ds_dedup_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Removing duplicates from '{}'", input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RemoveDuplicates: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("RemoveDuplicates: Input table schema is unavailable");
            return false;
        }

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("RemoveDuplicates: Failed to register table with DuckDB");
            return false;
        }

        // Execute DISTINCT query
        std::string sql = "SELECT DISTINCT * FROM " + temp_table;
        auto columns_it = node.parameters.find("columns");
        if (columns_it != node.parameters.end() &&
            !TrimString(columns_it->second).empty()) {
            std::vector<std::string> dedupe_columns;
            std::string column_error;
            if (!ResolveExistingColumns(input_table, "RemoveDuplicates",
                                        columns_it->second, dedupe_columns,
                                        column_error)) {
                duckdb_->UnregisterTable(temp_table);
                ReportError(column_error);
                return false;
            }

            std::vector<std::string> output_columns;
            output_columns.reserve(input_table->schema()->num_fields());
            for (const auto& field : input_table->schema()->fields()) {
                output_columns.push_back(field->name());
            }

            std::string rank_column = "__cyxwiz_dedup_rank";
            while (input_table->schema()->GetFieldIndex(rank_column) >= 0) {
                rank_column += "_";
            }

            sql = "SELECT " + JoinQuotedColumns(output_columns) +
                  " FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY " +
                  JoinQuotedColumns(dedupe_columns) + ") AS " +
                  QuoteSqlIdentifier(rank_column) + " FROM " + temp_table +
                  ") WHERE " + QuoteSqlIdentifier(rank_column) + " = 1";
        }
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("RemoveDuplicates: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] RemoveDuplicates: {} -> {} rows",
                    input_table->num_rows(), result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("RemoveDuplicates error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSaveDataset(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get the desired output name from parameters
    auto name_it = node.parameters.find("name");
    std::string output_name = (name_it != node.parameters.end() && !name_it->second.empty())
                              ? name_it->second
                              : "ds_output_" + std::to_string(node.id);
    auto path_it = node.parameters.find("path");
    const std::string output_path =
        (path_it != node.parameters.end()) ? path_it->second : "";
    auto format_it = node.parameters.find("format");
    const std::string format =
        (format_it != node.parameters.end() && !format_it->second.empty())
            ? ToLowerAscii(TrimString(format_it->second))
            : "csv";

    spdlog::info("[Data Studio] Saving dataset '{}' as '{}'", input_dataset_name, output_name);

    try {
        auto& registry = DataRegistry::Instance();

        // Get the Arrow dataset from the input
        auto arrow_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!arrow_dataset) {
            ReportError("SaveDataset: Input dataset not found in registry: " + input_dataset_name);
            return false;
        }

        // If the user specified a different name, register it again with the new name
        if (output_name != input_dataset_name) {
            auto arrow_table = arrow_dataset->GetArrowTable();
            registry.RegisterArrowTable(arrow_table, output_name);
        }

        if (!output_path.empty()) {
            bool success = false;
            if (format == "csv") {
                success = registry.ExportArrowToCSV(input_dataset_name, output_path);
            } else if (format == "parquet") {
                success = registry.ExportArrowToParquet(input_dataset_name, output_path);
            } else if (format == "json") {
                success = registry.ExportArrowToJSON(input_dataset_name, output_path);
            } else {
                ReportError("SaveDataset: Unsupported export format: " + format);
                return false;
            }

            if (!success) {
                ReportError("SaveDataset: Export failed");
                return false;
            }
        }

        // Store the output dataset name in context
        ctx.output_dataset = output_path.empty() ? output_name : output_path;

        spdlog::info("[Data Studio] Dataset saved successfully as '{}'",
                     ctx.output_dataset);
        return true;

    } catch (const std::exception& e) {
        ReportError("SaveDataset error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Phase 2 Week 4 - Additional Tabular Transformation Nodes
// ============================================================================

bool PipelineExecutor::ExecuteFillMissing(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get parameters
    auto strategy_it = node.parameters.find("strategy");
    std::string strategy =
        (strategy_it != node.parameters.end() && !strategy_it->second.empty())
            ? ToLowerAscii(TrimString(strategy_it->second))
            : "mean";

    auto value_it = node.parameters.find("value");
    std::string fill_value = (value_it != node.parameters.end()) ? value_it->second : "0";

    std::string output_dataset_name = "ds_fillmissing_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Filling missing values in '{}' with strategy: {}",
                input_dataset_name, strategy);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("FillMissing: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table) {
            ReportError("FillMissing: Input table is null");
            return false;
        }

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("FillMissing: Failed to register table with DuckDB");
            return false;
        }

        std::vector<std::string> select_exprs;
        select_exprs.reserve(input_table->num_columns());
        for (int i = 0; i < input_table->num_columns(); ++i) {
            const auto& field = input_table->schema()->field(i);
            const std::string quoted_column = QuoteSqlIdentifier(field->name());
            std::string expression = quoted_column;

            if (strategy == "constant") {
                std::string constant_expression;
                std::string constant_error;
                if (!BuildFillMissingConstantExpression(
                        field, fill_value, constant_expression, constant_error)) {
                    duckdb_->UnregisterTable(temp_table);
                    ReportError(constant_error);
                    return false;
                }
                expression = "COALESCE(" + quoted_column + ", " +
                             constant_expression + ")";
            } else if (strategy == "mean") {
                if (IsNumericArrowType(field->type())) {
                    expression = "COALESCE(" + quoted_column + ", (SELECT AVG(" +
                                 quoted_column + ") FROM " + temp_table + "))";
                }
            } else if (strategy == "median") {
                if (IsNumericArrowType(field->type())) {
                    expression = "COALESCE(" + quoted_column + ", (SELECT MEDIAN(" +
                                 quoted_column + ") FROM " + temp_table + "))";
                }
            } else if (strategy == "mode") {
                expression = "COALESCE(" + quoted_column + ", (SELECT MODE(" +
                             quoted_column + ") FROM " + temp_table + "))";
            } else {
                duckdb_->UnregisterTable(temp_table);
                ReportError("FillMissing: Unsupported strategy '" + strategy + "'");
                return false;
            }

            select_exprs.push_back(expression + " AS " + quoted_column);
        }

        std::string sql = "SELECT ";
        for (size_t i = 0; i < select_exprs.size(); ++i) {
            if (i > 0) {
                sql += ", ";
            }
            sql += select_exprs[i];
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("FillMissing: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] FillMissing completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("FillMissing error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSortRows(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get parameters
    auto columns_it = node.parameters.find("columns");
    if (columns_it == node.parameters.end() || columns_it->second.empty()) {
        ReportError("SortRows: Missing 'columns' parameter");
        return false;
    }

    const std::string order = NormalizeSortOrder(node.parameters);

    const std::string& sort_columns = columns_it->second;
    std::string output_dataset_name = "ds_sort_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Sorting '{}' by columns: {} {}",
                input_dataset_name, sort_columns, order);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("SortRows: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::vector<std::string> selected_columns;
        std::string schema_error;
        if (!ResolveExistingColumns(input_table, "SortRows", sort_columns,
                                    selected_columns, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_columns = JoinQuotedColumns(selected_columns);

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("SortRows: Failed to register table with DuckDB");
            return false;
        }

        // Execute ORDER BY query
        std::string sql = "SELECT * FROM " + temp_table +
                          " ORDER BY " + quoted_columns + " " + order;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("SortRows: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] SortRows completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("SortRows error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteJoin(const Node& node, ExecutionContext& ctx) {
    const auto input_dataset_names = GetInputDatasetNames(node, ctx, 2);
    if (input_dataset_names.empty()) {
        return false;
    }

    // Get parameters
    auto join_type_it = node.parameters.find("join_type");
    const std::string raw_join_type =
        (join_type_it != node.parameters.end()) ? join_type_it->second : "inner";
    const std::string join_type = NormalizeJoinTypeSql(raw_join_type);

    auto on_column_it = node.parameters.find("on_column");
    if (on_column_it == node.parameters.end() || on_column_it->second.empty()) {
        ReportError("Join: Missing 'on_column' parameter");
        return false;
    }

    const std::string& left_dataset_name = input_dataset_names[0];
    const std::string& right_dataset_name = input_dataset_names[1];
    const std::string on_column = TrimString(on_column_it->second);
    if (on_column.empty()) {
        ReportError("Join: Missing 'on_column' parameter");
        return false;
    }
    std::string output_dataset_name = "ds_join_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Joining '{}' and '{}' on column: {} ({})",
                left_dataset_name, right_dataset_name, on_column, join_type);

    try {
        auto& registry = DataRegistry::Instance();
        auto left_dataset = registry.GetArrowDataset(left_dataset_name);
        auto right_dataset = registry.GetArrowDataset(right_dataset_name);

        if (!left_dataset || !right_dataset) {
            ReportError("Join: Input datasets not found in registry");
            return false;
        }

        auto left_table = left_dataset->GetArrowTable();
        auto right_table = right_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnExists(left_table, "Join", on_column,
                                 "left input", schema_error) ||
            !RequireColumnExists(right_table, "Join", on_column,
                                 "right input", schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_on_column = QuoteSqlIdentifier(on_column);

        // Register both tables with DuckDB
        std::string left_temp = "temp_left_" + std::to_string(node.id);
        std::string right_temp = "temp_right_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(left_temp, left_table) ||
            !duckdb_->RegisterTable(right_temp, right_table)) {
            ReportError("Join: Failed to register tables with DuckDB");
            return false;
        }

        // Execute JOIN query
        std::string sql = "SELECT * FROM " + left_temp + " " + join_type + " JOIN " +
                         right_temp + " ON " + left_temp + "." + quoted_on_column +
                         " = " + right_temp + "." + quoted_on_column;

        auto result_table = duckdb_->Query(sql);

        // Unregister temp tables
        duckdb_->UnregisterTable(left_temp);
        duckdb_->UnregisterTable(right_temp);

        if (!result_table) {
            ReportError("Join: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] Join completed: {} rows, {} columns",
                    result_table->num_rows(), result_table->num_columns());
        return true;

    } catch (const std::exception& e) {
        ReportError("Join error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteGroupBy(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get parameters
    auto group_columns_it = node.parameters.find("group_columns");
    if (group_columns_it == node.parameters.end() || group_columns_it->second.empty()) {
        ReportError("GroupBy: Missing 'group_columns' parameter");
        return false;
    }

    auto agg_it = node.parameters.find("aggregations");
    if (agg_it == node.parameters.end() || agg_it->second.empty()) {
        ReportError("GroupBy: Missing 'aggregations' parameter");
        return false;
    }

    const std::string& group_columns = group_columns_it->second;
    const std::string& aggregations = agg_it->second;
    std::string output_dataset_name = "ds_groupby_" + std::to_string(node.id);

    spdlog::info("[Data Studio] GroupBy on '{}': columns={}, agg={}",
                input_dataset_name, group_columns, aggregations);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("GroupBy: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::vector<std::string> selected_columns;
        std::string schema_error;
        if (!ResolveExistingColumns(input_table, "GroupBy", group_columns,
                                    selected_columns, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_group_columns = JoinQuotedColumns(selected_columns);
        std::string aggregation_expressions;
        if (!BuildGroupByAggregationExpressions(input_table, aggregations,
                                                aggregation_expressions,
                                                schema_error)) {
            ReportError(schema_error);
            return false;
        }

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("GroupBy: Failed to register table with DuckDB");
            return false;
        }

        // Execute GROUP BY query
        // Aggregations format: "COUNT(*) as count, SUM(amount) as total"
        std::string sql = "SELECT " + quoted_group_columns + ", " +
                         aggregation_expressions +
                         " FROM " + temp_table + " GROUP BY " + quoted_group_columns;

        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("GroupBy: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] GroupBy completed: {} groups", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("GroupBy error: " + std::string(e.what()));
        return false;
    }
}

void PipelineExecutor::UpdateProgress(float progress, const std::string& status) {
    progress_ = progress;
    if (!status.empty()) {
        current_status_ = status;
    }
    if (progress_callback_) {
        progress_callback_(progress, current_status_);
    }
}

void PipelineExecutor::ReportError(const std::string& error) {
    last_error_ = error;
    spdlog::error("[Data Studio] Pipeline execution error: {}", error);
}

// Phase 7: Improved error message helper with actionable suggestions
std::string PipelineExecutor::GetImprovedErrorMessage(const std::string& node_type, const std::string& error_category, const std::string& details) {
    std::string message;

    if (error_category == "no_input") {
        message = node_type + ": No input dataset connected\n"
                  "Suggestion: Connect a FileInput or upstream transformation node";
    } else if (error_category == "dataset_not_found") {
        message = node_type + ": Input dataset not found\n"
                  "Suggestion: Ensure the upstream node executed successfully";
    } else if (error_category == "missing_parameter") {
        message = node_type + ": Missing required parameter '" + details + "'\n"
                  "Suggestion: Configure the node by right-clicking and selecting 'Configure'";
    } else if (error_category == "column_not_found") {
        message = node_type + ": Column '" + details + "' not found in dataset\n"
                  "Suggestion: Check dataset schema or use SelectColumns node first";
    } else if (error_category == "query_failed") {
        message = node_type + ": SQL query execution failed\n"
                  "Details: " + details + "\n"
                  "Suggestion: Check your filter conditions, column names, or SQL syntax";
    } else if (error_category == "empty_result") {
        message = node_type + ": Query returned 0 rows\n"
                  "Suggestion: Check filter conditions or use less restrictive thresholds";
    } else if (error_category == "type_mismatch") {
        message = node_type + ": Cannot apply numeric operation to text column\n"
                  "Suggestion: Use TextVectorize node to convert text to numbers first";
    } else if (error_category == "invalid_path") {
        message = node_type + ": File not found at path: " + details + "\n"
                  "Suggestion: Check file path, ensure file exists, and has correct permissions";
    } else if (error_category == "register_failed") {
        message = node_type + ": Failed to register table with DuckDB\n"
                  "Details: " + details + "\n"
                  "Suggestion: Check dataset format and memory availability";
    } else {
        // Fallback to generic message
        message = node_type + ": " + details;
    }

    return message;
}

void PipelineExecutor::NotifyCompletion(bool success) {
    if (completion_callback_) {
        completion_callback_(success);
    }
}

// ============================================================================
// Phase 5 Week 7 - Node Editor Handoff
// ============================================================================

bool PipelineExecutor::ExecuteDeployToNodeEditor(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        return false;
    }

    // Get the desired output name from parameters (optional)
    auto name_it = node.parameters.find("name");
    std::string deployment_name = (name_it != node.parameters.end() && !name_it->second.empty())
                                  ? name_it->second
                                  : "deployed_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Preparing dataset '{}' for Node Editor deployment as '{}'",
                input_dataset_name, deployment_name);

    try {
        auto& registry = DataRegistry::Instance();

        // Get the Arrow dataset from the input
        auto arrow_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!arrow_dataset) {
            ReportError("DeployToNodeEditor: Input dataset not found in registry: " + input_dataset_name);
            return false;
        }

        // If the user specified a different name, register it again with the new name
        if (deployment_name != input_dataset_name) {
            auto arrow_table = arrow_dataset->GetArrowTable();
            registry.RegisterArrowTable(arrow_table, deployment_name);
        }

        // Tag dataset for deployment
        ctx.deployment_dataset = deployment_name;
        ctx.deployment_ready = true;

        // Also store in output_dataset for consistency
        ctx.output_dataset = deployment_name;

        spdlog::info("[Data Studio] Dataset ready for deployment: '{}'", deployment_name);
        return true;

    } catch (const std::exception& e) {
        ReportError("DeployToNodeEditor error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Phase 6 Week 8-9: Advanced Nodes Implementation
// ============================================================================

std::string PipelineExecutor::GetInputDatasetName(const Node& node, ExecutionContext& ctx) {
    const auto input_dataset_names = GetInputDatasetNames(node, ctx, 1);
    if (input_dataset_names.empty()) {
        return "";
    }
    return input_dataset_names.front();
}

std::vector<std::string> PipelineExecutor::GetInputDatasetNames(
    const Node& node,
    ExecutionContext& ctx,
    size_t expected_count) {
    if (node.inputs.size() != expected_count) {
        ReportError(node.type + ": expected " + std::to_string(expected_count) +
                    " input dataset(s), got " + std::to_string(node.inputs.size()));
        return {};
    }

    std::vector<std::string> dataset_names;
    dataset_names.reserve(expected_count);
    for (int input_node_id : node.inputs) {
        auto result_it = ctx.node_results.find(input_node_id);
        if (result_it == ctx.node_results.end() || result_it->second.empty()) {
            ReportError(node.type + ": input dataset from node " +
                        std::to_string(input_node_id) + " not found");
            return {};
        }
        dataset_names.push_back(result_it->second);
    }
    return dataset_names;
}

bool PipelineExecutor::FailUnsupportedNode(const Node& node, const std::string& reason) {
    ReportError(node.type + " is not executable in the legacy Data Studio pipeline path: " +
                reason);
    return false;
}

bool PipelineExecutor::ExecutePipelineOperatorNode(
    const Node& node,
    ExecutionContext& ctx,
    gui::NodeType type) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(node.type + ": No input connection or dataset not found");
        return false;
    }

    auto& registry = DataRegistry::Instance();
    auto input_dataset = registry.GetArrowDataset(input_dataset_name);
    if (!input_dataset) {
        ReportError(node.type + ": input dataset '" + input_dataset_name +
                    "' is not an in-memory Arrow dataset");
        return false;
    }

    auto input_table = input_dataset->GetArrowTable();
    if (!input_table) {
        ReportError(node.type + ": input Arrow table is null");
        return false;
    }

    auto& factory = PipelineOperatorFactory::Instance();
    if (!factory.HasOperator(type)) {
        ReportError(node.type + ": no PipelineOperatorFactory registration exists");
        return false;
    }

    auto op = factory.Create(type);
    if (!op) {
        ReportError(node.type + ": PipelineOperatorFactory returned null operator");
        return false;
    }

    std::string configure_error;
    if (!op->Configure(node.parameters, configure_error)) {
        ReportError(configure_error.empty()
                        ? node.type + ": operator configuration failed"
                        : configure_error);
        return false;
    }

    auto result = op->Apply(input_table);
    if (!result.ok()) {
        ReportError(node.type + ": operator execution failed: " +
                    result.status().ToString());
        return false;
    }

    auto output_table = result.ValueOrDie();
    const std::string output_dataset_name =
        "ds_operator_" + node.type + "_" + std::to_string(node.id);
    auto output_dataset = registry.RegisterArrowTable(output_table, output_dataset_name);
    if (!output_dataset) {
        ReportError(node.type + ": failed to register operator output dataset");
        return false;
    }

    ctx.node_results[node.id] = output_dataset_name;
    ctx.output_dataset = output_dataset_name;

    spdlog::info("[Data Studio] {} routed through PipelineOperatorFactory: {} rows, {} columns",
                 node.type, output_table->num_rows(), output_table->num_columns());
    return true;
}

// ============================================================================
// Text Processing Nodes
// ============================================================================

bool PipelineExecutor::ExecuteTextClean(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextClean: No input connection or dataset not found");
        return false;
    }

    // Get parameters
    bool lowercase = OptionalBooleanParameterIsTrue(node.parameters, "lowercase");
    bool remove_html = OptionalBooleanParameterIsTrue(node.parameters, "remove_html");
    bool remove_special_chars =
        OptionalBooleanParameterIsTrue(node.parameters,
                                       "remove_special_chars");
    // Note: remove_stopwords would require dictionary integration - not implemented in MVP

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    std::string output_dataset_name = "ds_textclean_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextClean on column '{}' from '{}'", text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextClean: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TextClean", text_column,
                               "string", IsStringArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_text_column = QuoteSqlIdentifier(text_column);
        const std::string quoted_output_column =
            QuoteSqlIdentifier(text_column + "_cleaned");
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextClean: Failed to register table");
            return false;
        }

        // Build SQL transformation chain
        std::string sql = "SELECT *, ";
        std::string transform = quoted_text_column;

        if (remove_html) {
            transform = "regexp_replace(" + transform + ", '<[^>]*>', '', 'g')";
        }
        if (remove_special_chars) {
            transform = "regexp_replace(" + transform + ", '[^a-zA-Z0-9\\s]', '', 'g')";
        }
        if (lowercase) {
            transform = "lower(" + transform + ")";
        }
        // Remove extra whitespace
        transform = "regexp_replace(" + transform + ", '\\s+', ' ', 'g')";
        transform = "trim(" + transform + ")";

        sql += transform + " AS " + quoted_output_column + " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextClean: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextClean completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextClean error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTextTokenize(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextTokenize: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    auto method_it = node.parameters.find("method");
    std::string method =
        (method_it != node.parameters.end() && !method_it->second.empty())
            ? ToLowerAscii(TrimString(method_it->second))
            : "word";

    std::string output_dataset_name = "ds_texttokenize_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextTokenize ({}) on column '{}' from '{}'",
                method, text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextTokenize: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TextTokenize", text_column,
                               "string", IsStringArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_text_column = QuoteSqlIdentifier(text_column);
        const std::string quoted_tokens_column =
            QuoteSqlIdentifier(text_column + "_tokens");
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextTokenize: Failed to register table");
            return false;
        }

        std::string sql;
        if (method == "word") {
            // Split on whitespace and punctuation
            sql = "SELECT *, string_split_regex(" + quoted_text_column +
                  ", '\\s+') AS " + quoted_tokens_column + " FROM " + temp_table;
        } else if (method == "sentence") {
            // Split on sentence boundaries
            sql = "SELECT *, string_split_regex(" + quoted_text_column +
                  ", '[.!?]+') AS " + quoted_tokens_column + " FROM " + temp_table;
        } else if (method == "character") {
            // Split into characters (list of individual chars)
            sql = "SELECT *, string_split(" + quoted_text_column +
                  ", '') AS " + quoted_tokens_column + " FROM " + temp_table;
        } else {
            sql = "SELECT *, string_split(" + quoted_text_column +
                  ", ' ') AS " + quoted_tokens_column + " FROM " + temp_table;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextTokenize: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextTokenize completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextTokenize error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTextVectorize(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextVectorize: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    auto method_it = node.parameters.find("method");
    std::string method =
        (method_it != node.parameters.end() && !method_it->second.empty())
            ? ToLowerAscii(TrimString(method_it->second))
            : "count";

    std::string output_dataset_name = "ds_textvectorize_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextVectorize ({}) on column '{}' from '{}'",
                method, text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextVectorize: Input dataset not found");
            return false;
        }

        // For MVP: Create simple word count features
        // Full implementation would use TF-IDF or embeddings from backend
        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TextVectorize", text_column,
                               "string", IsStringArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_text_column = QuoteSqlIdentifier(text_column);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextVectorize: Failed to register table");
            return false;
        }

        // Create simple features: text length, word count
        std::string sql = "SELECT *, "
                         "length(" + quoted_text_column + ") AS " +
                         QuoteSqlIdentifier("text_length") + ", "
                         "length(" + quoted_text_column + ") - length(replace(" +
                         quoted_text_column + ", ' ', '')) + 1 AS " +
                         QuoteSqlIdentifier("word_count") + " "
                         "FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextVectorize: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextVectorize completed: {} rows, basic features added",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextVectorize error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Time-Series Nodes
// ============================================================================

bool PipelineExecutor::ExecuteTSWindow(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSWindow: No input connection or dataset not found");
        return false;
    }

    auto window_it = node.parameters.find("window_size");
    int window_size = (window_it != node.parameters.end()) ? std::stoi(window_it->second) : 10;

    auto stride_it = node.parameters.find("stride");
    int stride = (stride_it != node.parameters.end()) ? std::stoi(stride_it->second) : 1;

    auto target_it = node.parameters.find("target_column");
    std::string target_column = (target_it != node.parameters.end()) ? target_it->second : "value";

    std::string output_dataset_name = "ds_tswindow_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSWindow (size={}, stride={}) on '{}' from '{}'",
                window_size, stride, target_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSWindow: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TSWindow", target_column,
                               "numeric", IsNumericArrowType,
                               schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_target_column = QuoteSqlIdentifier(target_column);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSWindow: Failed to register table");
            return false;
        }

        // Create windows using LAG function
        std::string sql = "SELECT *, ";
        for (int i = 0; i < window_size; i++) {
            if (i > 0) sql += ", ";
            sql += "LAG(" + quoted_target_column + ", " +
                   std::to_string(i) + ") OVER (ORDER BY rowid) AS " +
                   QuoteSqlIdentifier("window_t" + std::to_string(i));
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSWindow: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSWindow completed: {} rows with {} timestep windows",
                    result_table->num_rows(), window_size);
        return true;

    } catch (const std::exception& e) {
        ReportError("TSWindow error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSFeatures(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSFeatures: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto window_it = node.parameters.find("rolling_window");
    int rolling_window = (window_it != node.parameters.end()) ? std::stoi(window_it->second) : 7;

    std::string output_dataset_name = "ds_tsfeatures_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSFeatures (window={}) on '{}' from '{}'",
                rolling_window, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSFeatures: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TSFeatures", columns,
                               "numeric", IsNumericArrowType,
                               schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_column = QuoteSqlIdentifier(columns);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSFeatures: Failed to register table");
            return false;
        }

        // Create rolling statistics
        std::string sql = "SELECT *, "
                         "AVG(" + quoted_column + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         QuoteSqlIdentifier(columns + "_rolling_mean") + ", "
                         "STDDEV(" + quoted_column + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         QuoteSqlIdentifier(columns + "_rolling_std") + ", "
                         "MIN(" + quoted_column + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         QuoteSqlIdentifier(columns + "_rolling_min") + ", "
                         "MAX(" + quoted_column + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         QuoteSqlIdentifier(columns + "_rolling_max") + " "
                         "FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSFeatures: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSFeatures completed: {} rows with rolling statistics",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSFeatures error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSLag(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSLag: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto lags_it = node.parameters.find("lag_periods");
    std::string lag_periods = (lags_it != node.parameters.end()) ? lags_it->second : "1,7,30";

    std::string output_dataset_name = "ds_tslag_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSLag (periods={}) on '{}' from '{}'",
                lag_periods, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSLag: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TSLag", columns,
                               "numeric", IsNumericArrowType,
                               schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_column = QuoteSqlIdentifier(columns);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSLag: Failed to register table");
            return false;
        }

        // Parse lag periods (comma-separated)
        std::string sql = "SELECT *, ";
        std::stringstream ss(lag_periods);
        std::string lag_str;
        bool first = true;

        while (std::getline(ss, lag_str, ',')) {
            int lag = std::stoi(lag_str);
            if (!first) sql += ", ";
            sql += "LAG(" + quoted_column + ", " + std::to_string(lag) +
                   ") OVER (ORDER BY rowid) AS " +
                   QuoteSqlIdentifier(columns + "_lag" + std::to_string(lag));
            first = false;
        }

        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSLag: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSLag completed: {} rows with lag features",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSLag error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSDiff(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSDiff: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto order_it = node.parameters.find("order");
    int order = (order_it != node.parameters.end()) ? std::stoi(order_it->second) : 1;

    std::string output_dataset_name = "ds_tsdiff_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSDiff (order={}) on '{}' from '{}'",
                order, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSDiff: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "TSDiff", columns,
                               "numeric", IsNumericArrowType,
                               schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_column = QuoteSqlIdentifier(columns);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSDiff: Failed to register table");
            return false;
        }

        // Create difference features
        std::string sql = "SELECT *, ";
        for (int i = 1; i <= order; i++) {
            if (i > 1) sql += ", ";
            sql += quoted_column + " - LAG(" + quoted_column + ", " +
                   std::to_string(i) + ") OVER (ORDER BY rowid) AS " +
                   QuoteSqlIdentifier(columns + "_diff" + std::to_string(i));
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSDiff: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSDiff completed: {} rows with difference features",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSDiff error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Feature Engineering Nodes
// ============================================================================

bool PipelineExecutor::ExecutePolynomialFeatures(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("PolynomialFeatures: No input connection or dataset not found");
        return false;
    }

    auto degree_it = node.parameters.find("degree");
    int degree = (degree_it != node.parameters.end()) ? std::stoi(degree_it->second) : 2;

    auto columns_it = node.parameters.find("columns");
    std::string column = (columns_it != node.parameters.end()) ? columns_it->second : "";
    if (column.empty()) {
        ReportError("PolynomialFeatures: Column name required");
        return false;
    }

    std::string output_dataset_name = "ds_poly_" + std::to_string(node.id);

    spdlog::info("[Data Studio] PolynomialFeatures (degree={}) on '{}' from '{}'",
                degree, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("PolynomialFeatures: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "PolynomialFeatures", column,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("PolynomialFeatures: Failed to register table");
            return false;
        }

        const std::string quoted_column = QuoteSqlIdentifier(column);
        std::string sql = "SELECT *";
        for (int power = 2; power <= degree; ++power) {
            std::string expr;
            for (int factor = 0; factor < power; ++factor) {
                if (!expr.empty()) {
                    expr += " * ";
                }
                expr += quoted_column;
            }

            std::string suffix;
            if (power == 2) {
                suffix = "_squared";
            } else if (power == 3) {
                suffix = "_cubed";
            } else {
                suffix = "_pow" + std::to_string(power);
            }
            sql += ", " + expr + " AS " + QuoteSqlIdentifier(column + suffix);
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("PolynomialFeatures: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] PolynomialFeatures completed: {} rows",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("PolynomialFeatures error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteBinning(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("Binning: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string column = (column_it != node.parameters.end()) ? column_it->second : "";
    if (column.empty()) {
        ReportError("Binning: Column name required");
        return false;
    }

    auto n_bins_it = node.parameters.find("n_bins");
    int n_bins = (n_bins_it != node.parameters.end()) ? std::stoi(n_bins_it->second) : 10;

    auto method_it = node.parameters.find("method");
    std::string method =
        (method_it != node.parameters.end())
            ? NormalizeBinningMethod(method_it->second)
            : "equal_width";

    std::string output_dataset_name = "ds_binning_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Binning (method={}, bins={}) on '{}' from '{}'",
                method, n_bins, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("Binning: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "Binning", column,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("Binning: Failed to register table");
            return false;
        }

        const std::string quoted_column = QuoteSqlIdentifier(column);
        const std::string quoted_bin_column = QuoteSqlIdentifier(column + "_bin");
        const std::string base_column = "base." + quoted_column;

        std::string sql;
        if (method == "equal_freq") {
            sql = "SELECT *, NTILE(" + std::to_string(n_bins) + ") OVER (ORDER BY " +
                  quoted_column + ") AS " + quoted_bin_column + " FROM " + temp_table;
        } else if (method == "equal_width") {
            const std::string n_bins_text = std::to_string(n_bins);
            sql = "SELECT base.*, CASE "
                  "WHEN stats.min_value = stats.max_value THEN 1 "
                  "ELSE LEAST(" + n_bins_text + ", GREATEST(1, CAST(FLOOR(((CAST(" +
                  base_column + " AS DOUBLE) - stats.min_value) / "
                  "NULLIF(stats.max_value - stats.min_value, 0)) * " +
                  n_bins_text + ") + 1 AS BIGINT))) END AS " + quoted_bin_column +
                  " FROM " + temp_table + " base CROSS JOIN (SELECT CAST(MIN(" +
                  quoted_column + ") AS DOUBLE) AS min_value, CAST(MAX(" +
                  quoted_column + ") AS DOUBLE) AS max_value FROM " + temp_table +
                  ") stats";
        } else {
            ReportError("Binning: Unsupported method '" + method + "'");
            return false;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("Binning: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] Binning completed: {} rows with {} bins",
                    result_table->num_rows(), n_bins);
        return true;

    } catch (const std::exception& e) {
        ReportError("Binning error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Phase 8: Performance Optimization Implementation
// ============================================================================

/* Memory Optimization Strategy (Streaming Mode)
 *
 * For large datasets (>1GB), we can implement chunk-based processing:
 *
 * 1. Check dataset size before loading:
 *    if (file_size > memory_limit_) { streaming_mode_ = true; }
 *
 * 2. Use Arrow RecordBatch API instead of full Table:
 *    auto reader = arrow::ipc::RecordBatchFileReader::Open(file);
 *    for (int i = 0; i < reader->num_record_batches(); i++) {
 *        auto batch = reader->ReadRecordBatch(i);
 *        ProcessChunk(batch);  // Process in chunks
 *    }
 *
 * 3. Example: Streaming RemoveDuplicates
 *    std::unordered_set<std::string> seen_hashes;
 *    for (int64_t offset = 0; offset < total_rows; offset += chunk_size_) {
 *        auto batch = table->Slice(offset, chunk_size_);
 *        for (int64_t i = 0; i < batch->num_rows(); i++) {
 *            std::string row_hash = ComputeRowHash(batch, i);
 *            if (seen_hashes.insert(row_hash).second) {
 *                output_batches.push_back(batch->Slice(i, 1));
 *            }
 *        }
 *        ReportProgress((float)offset / total_rows, "Deduplicating chunk...");
 *    }
 *
 * 4. Combine output batches:
 *    auto result = arrow::Table::FromRecordBatches(output_batches);
 *
 * This approach keeps memory usage bounded even for TB-scale datasets.
 */

uint64_t PipelineExecutor::ComputeNodeHash(const Node& node) const {
    // Simple hash combining node type and parameters
    // Using FNV-1a hash algorithm for fast hashing
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    const uint64_t prime = 1099511628211ULL;  // FNV prime

    // Hash node type
    for (char c : node.type) {
        hash ^= static_cast<uint64_t>(c);
        hash *= prime;
    }

    // Hash all parameters (sorted for consistency)
    std::vector<std::pair<std::string, std::string>> sorted_params(
        node.parameters.begin(), node.parameters.end());
    std::sort(sorted_params.begin(), sorted_params.end());

    for (const auto& [key, value] : sorted_params) {
        for (char c : key) {
            hash ^= static_cast<uint64_t>(c);
            hash *= prime;
        }
        for (char c : value) {
            hash ^= static_cast<uint64_t>(c);
            hash *= prime;
        }
    }

    return hash;
}

void PipelineExecutor::MarkDirtyNodes(std::vector<Node>& nodes) {
    // Step 1: Mark nodes whose parameters changed
    for (auto& node : nodes) {
        uint64_t current_hash = ComputeNodeHash(node);
        if (current_hash != node.last_execution_hash) {
            node.needs_execution = true;
            node.last_execution_hash = current_hash;
            spdlog::debug("[Data Studio] Node {} marked dirty (parameters changed)", node.name);
        } else {
            node.needs_execution = false;
            spdlog::debug("[Data Studio] Node {} cache valid", node.name);
        }
    }

    // Step 2: Propagate dirty flag to downstream nodes
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& node : nodes) {
            if (!node.needs_execution) {
                // Check if any input node is dirty
                for (int input_id : node.inputs) {
                    const auto* input_node = FindNodeById(nodes, input_id);
                    if (input_node && input_node->needs_execution) {
                        node.needs_execution = true;
                        changed = true;
                        spdlog::debug("[Data Studio] Node {} marked dirty (upstream dependency changed)",
                                     node.name);
                        break;
                    }
                }
            }
        }
    }
}

std::vector<int> PipelineExecutor::FindReadyNodes(
    const std::vector<Node>& nodes,
    const std::set<int>& completed) const {

    std::vector<int> ready;

    for (const auto& node : nodes) {
        // Skip if already completed
        if (completed.find(node.id) != completed.end()) {
            continue;
        }

        // Skip if doesn't need execution (cached)
        if (!node.needs_execution) {
            continue;
        }

        // Check if all input nodes are completed
        bool all_inputs_ready = true;
        for (int input_id : node.inputs) {
            if (completed.find(input_id) == completed.end()) {
                all_inputs_ready = false;
                break;
            }
        }

        if (all_inputs_ready) {
            ready.push_back(node.id);
        }
    }

    return ready;
}

bool PipelineExecutor::ExecuteParallel(std::vector<Node>& nodes) {
    std::set<int> completed;
    std::set<int> executing;
    ExecutionContext ctx;
    std::mutex execution_mutex;

    struct NodeExecutionResult {
        bool success = false;
        ExecutionContext ctx;
    };

    int total_nodes_to_execute = 0;
    for (const auto& node : nodes) {
        if (node.needs_execution) {
            total_nodes_to_execute++;
        } else {
            // Node doesn't need execution, use cached result
            if (!node.cached_output_dataset.empty()) {
                ctx.node_results[node.id] = node.cached_output_dataset;
                spdlog::info("[Data Studio] Using cached result for node: {}", node.name);
                completed.insert(node.id);
            } else {
                ReportError("Cached node '" + node.name +
                            "' has no cached output dataset and cannot be marked complete");
                return false;
            }
        }
    }

    if (total_nodes_to_execute == 0) {
        spdlog::info("[Data Studio] All nodes up-to-date, using cached results");
        UpdateProgress(1.0f, "All nodes up-to-date");
        return true;
    }

    int nodes_executed = 0;
    float base_progress = 0.3f;
    float progress_range = 0.7f;

    while (completed.size() < nodes.size()) {
        // Check for cancellation
        if (cancel_requested_) {
            ReportError("Pipeline execution cancelled by user");
            return false;
        }

        // Find nodes ready to execute
        auto ready = FindReadyNodes(nodes, completed);

        if (ready.empty() && executing.empty()) {
            // Deadlock or cycle detected
            ReportError("Pipeline execution stuck (possible cycle or missing dependencies)");
            return false;
        }

        if (ready.empty()) {
            // Wait a bit for executing nodes to complete
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Execute ready nodes in parallel (up to 4 concurrent for now)
        const size_t max_parallel = std::min<size_t>(4, ready.size());
        std::vector<std::future<NodeExecutionResult>> futures;
        std::vector<int> batch_node_ids;

        for (size_t i = 0; i < max_parallel && i < ready.size(); i++) {
            int node_id = ready[i];
            auto* node = FindNodeById(nodes, node_id);
            if (!node) continue;

            batch_node_ids.push_back(node_id);
            executing.insert(node_id);

            // Execute node asynchronously
            futures.push_back(std::async(std::launch::async,
                [this, node, &ctx, &execution_mutex]() mutable {
                    NodeExecutionResult result;
                    result.ctx.node_results = ctx.node_results;
                    std::lock_guard<std::mutex> lock(execution_mutex);
                    result.success = ExecuteNode(*node, result.ctx);
                    return result;
                }
            ));

            spdlog::info("[Data Studio] Started executing node: {} (parallel batch)", node->name);
        }

        // Wait for batch to complete
        for (size_t i = 0; i < futures.size(); i++) {
            NodeExecutionResult result = futures[i].get();
            int node_id = batch_node_ids[i];
            auto* node = FindNodeById(nodes, node_id);

            executing.erase(node_id);

            if (result.success && node) {
                for (const auto& [result_node_id, dataset_name] : result.ctx.node_results) {
                    ctx.node_results[result_node_id] = dataset_name;
                }
                if (result.ctx.deployment_ready) {
                    ctx.deployment_ready = true;
                    ctx.deployment_dataset = result.ctx.deployment_dataset;
                }
                if (!result.ctx.output_dataset.empty()) {
                    ctx.output_dataset = result.ctx.output_dataset;
                }
                if (!result.ctx.deployment_dataset.empty()) {
                    ctx.deployment_dataset = result.ctx.deployment_dataset;
                }

                completed.insert(node_id);
                nodes_executed++;

                // Cache the output dataset name
                auto result_it = ctx.node_results.find(node_id);
                if (result_it != ctx.node_results.end()) {
                    node->cached_output_dataset = result_it->second;
                }

                float progress = base_progress +
                    (progress_range * nodes_executed / total_nodes_to_execute);
                UpdateProgress(progress,
                    "Completed " + node->name + " (" +
                    std::to_string(nodes_executed) + "/" +
                    std::to_string(total_nodes_to_execute) + ")");

                spdlog::info("[Data Studio] Completed node: {} ({}/{})",
                           node->name, nodes_executed, total_nodes_to_execute);
            } else {
                // Execution failed
                return false;
            }
        }
    }

    // Transfer deployment status from context to executor state
    if (ctx.deployment_ready) {
        deployment_ready_ = true;
        deployment_dataset_ = ctx.deployment_dataset;
        spdlog::info("[Data Studio] Deployment ready: '{}'", deployment_dataset_);
    }

    return true;
}

PipelineExecutor::Node* PipelineExecutor::FindNodeById(std::vector<Node>& nodes, int node_id) {
    auto it = std::find_if(nodes.begin(), nodes.end(),
                          [node_id](const Node& n) { return n.id == node_id; });
    return (it != nodes.end()) ? &(*it) : nullptr;
}

const PipelineExecutor::Node* PipelineExecutor::FindNodeById(
    const std::vector<Node>& nodes, int node_id) const {
    auto it = std::find_if(nodes.begin(), nodes.end(),
                          [node_id](const Node& n) { return n.id == node_id; });
    return (it != nodes.end()) ? &(*it) : nullptr;
}


// ============================================================================
// KNIME-Style Table Manipulation Nodes
// ============================================================================

bool PipelineExecutor::ExecuteExcelInput(const Node& node, ExecutionContext& ctx) {
    auto path_it = node.parameters.find("path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExcelInput: Missing file path parameter");
        return false;
    }

    const std::string& file_path = path_it->second;
    std::string dataset_name = "ds_excel_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Loading Excel file: {} as dataset '{}'", file_path, dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);
        if (!arrow_dataset) {
            ReportError("ExcelInput: Failed to load file");
            return false;
        }
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) ctx.input_dataset = dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("ExcelInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteExportCSV(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportCSV: No input dataset");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        path_it = node.parameters.find("path");
    }
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExportCSV: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to CSV: {}", path_it->second);
    auto& registry = DataRegistry::Instance();
    if (!registry.GetArrowDataset(input_dataset_name)) {
        ReportError("ExportCSV: Input dataset not found in registry");
        return false;
    }
    if (!registry.ExportArrowToCSV(input_dataset_name, path_it->second)) {
        ReportError("ExportCSV: Export failed");
        return false;
    }

    ctx.node_results[node.id] = input_dataset_name;
    ctx.output_dataset = path_it->second;
    return true;
}

bool PipelineExecutor::ExecuteRenameColumns(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RenameColumns: No input dataset");
        return false;
    }

    auto rename_map_it = node.parameters.find("rename_map");
    auto mapping_it = node.parameters.find("mapping");
    const std::string mapping =
        (mapping_it != node.parameters.end() && !mapping_it->second.empty())
            ? mapping_it->second
            : ((rename_map_it != node.parameters.end()) ? rename_map_it->second : "");
    const auto rename_map = ParseRenameMapping(mapping);
    if (rename_map.empty()) {
        ReportError("RenameColumns: mapping must contain old_name:new_name pairs");
        return false;
    }

    std::string output_dataset_name = "ds_renamed_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Renaming columns in '{}'", input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RenameColumns: Input dataset not found");
            return false;
        }
        auto input_table = input_dataset->GetArrowTable();
        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(input_table->num_columns());
        std::set<std::string> output_names;
        std::set<std::string> matched_input_names;
        for (int i = 0; i < input_table->num_columns(); ++i) {
            const auto field = input_table->schema()->field(i);
            auto rename_it = rename_map.find(field->name());
            const std::string output_name =
                rename_it == rename_map.end() ? field->name() : rename_it->second;
            if (!output_names.insert(output_name).second) {
                ReportError("RenameColumns: duplicate output column name '" +
                            output_name + "'");
                return false;
            }
            if (rename_it != rename_map.end()) {
                matched_input_names.insert(field->name());
            }
            fields.push_back(rename_it == rename_map.end()
                                 ? field
                                 : field->WithName(output_name));
        }
        if (matched_input_names.size() != rename_map.size()) {
            for (const auto& [old_name, _] : rename_map) {
                if (matched_input_names.find(old_name) == matched_input_names.end()) {
                    ReportError("RenameColumns: input column '" + old_name +
                                "' does not exist");
                    return false;
                }
            }
        }
        auto renamed_table = arrow::Table::Make(
            arrow::schema(fields), input_table->columns(), input_table->num_rows());
        registry.RegisterArrowTable(renamed_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RenameColumns error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRowToColumnNames(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RowToColumnNames: No input dataset");
        return false;
    }

    auto row_idx_it = node.parameters.find("row_index");
    int row_index = (row_idx_it != node.parameters.end()) ? std::stoi(row_idx_it->second) : 0;

    std::string output_dataset_name = "ds_newheaders_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Promoting row {} to column names", row_index);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RowToColumnNames: Input dataset not found");
            return false;
        }
        auto input_table = input_dataset->GetArrowTable();
        if (!input_table) {
            ReportError("RowToColumnNames: Input table is null");
            return false;
        }
        if (row_index < 0 || row_index >= input_table->num_rows()) {
            ReportError("RowToColumnNames: Row index out of bounds");
            return false;
        }

        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(input_table->num_columns());
        std::set<std::string> output_names;
        for (int i = 0; i < input_table->num_columns(); ++i) {
            auto scalar_result = input_table->column(i)->GetScalar(row_index);
            if (!scalar_result.ok()) {
                ReportError("RowToColumnNames: Failed to read header cell");
                return false;
            }
            const std::string column_name = ScalarToColumnName(scalar_result.ValueOrDie());
            if (column_name.empty()) {
                ReportError("RowToColumnNames: Header row contains an empty column name");
                return false;
            }
            if (!output_names.insert(column_name).second) {
                ReportError("RowToColumnNames: duplicate output column name '" +
                            column_name + "'");
                return false;
            }
            fields.push_back(input_table->schema()->field(i)->WithName(column_name));
        }

        std::shared_ptr<arrow::Table> data_table;
        if (row_index == 0) {
            data_table = input_table->Slice(1);
        } else if (row_index == input_table->num_rows() - 1) {
            data_table = input_table->Slice(0, row_index);
        } else {
            auto top = input_table->Slice(0, row_index);
            auto bottom = input_table->Slice(row_index + 1);
            auto concat_result = arrow::ConcatenateTables({top, bottom});
            if (!concat_result.ok()) {
                ReportError("RowToColumnNames: Failed to remove promoted row");
                return false;
            }
            data_table = concat_result.ValueOrDie();
        }

        auto output_table = arrow::Table::Make(
            arrow::schema(fields), data_table->columns(), data_table->num_rows());
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RowToColumnNames error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTableCropper(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TableCropper: No input dataset");
        return false;
    }

    auto start_row_it = node.parameters.find("start_row");
    auto end_row_it = node.parameters.find("end_row");

    int64_t start_row =
        (start_row_it != node.parameters.end() && !start_row_it->second.empty())
            ? std::stoll(start_row_it->second)
            : 0;
    int64_t end_row =
        (end_row_it != node.parameters.end() && !end_row_it->second.empty())
            ? std::stoll(end_row_it->second)
            : -1;

    std::string output_dataset_name = "ds_cropped_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Cropping table rows {}:{}", start_row, end_row);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TableCropper: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table) {
            ReportError("TableCropper: Input table is null");
            return false;
        }
        int64_t num_rows = input_table->num_rows();

        if (end_row < 0) end_row = num_rows;
        if (start_row > num_rows) {
            ReportError("TableCropper: start_row out of bounds");
            return false;
        }
        if (end_row > num_rows) {
            ReportError("TableCropper: end_row out of bounds");
            return false;
        }
        if (end_row < start_row) {
            ReportError("TableCropper: end_row must be >= start_row");
            return false;
        }
        int64_t length = end_row - start_row;

        auto cropped_table = input_table->Slice(start_row, length);
        registry.RegisterArrowTable(cropped_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("TableCropper error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteStringManipulation(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("StringManipulation: No input dataset");
        return false;
    }

    auto column_it = node.parameters.find("column");
    auto operation_it = node.parameters.find("operation");

    std::string column = (column_it != node.parameters.end()) ? column_it->second : "";
    std::string operation =
        (operation_it != node.parameters.end() && !operation_it->second.empty())
            ? ToLowerAscii(TrimString(operation_it->second))
            : "trim";

    if (column.empty()) {
        ReportError("StringManipulation: Column name required");
        return false;
    }

    std::string output_dataset_name = "ds_string_" + std::to_string(node.id);
    spdlog::info("[Data Studio] String {} on column '{}'", operation, column);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("StringManipulation: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, "StringManipulation", column,
                               "string", IsStringArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("StringManipulation: Failed to register table");
            return false;
        }

        const std::string quoted_column = QuoteSqlIdentifier(column);
        const std::string quoted_output = QuoteSqlIdentifier(column + "_modified");

        std::string expr;
        if (operation == "trim") {
            expr = "TRIM(" + quoted_column + ")";
        } else if (operation == "upper") {
            expr = "UPPER(" + quoted_column + ")";
        } else if (operation == "lower") {
            expr = "LOWER(" + quoted_column + ")";
        } else if (operation == "replace") {
            const auto find_it = node.parameters.find("param1");
            const auto replacement_it = node.parameters.find("param2");
            const std::string find_value =
                (find_it != node.parameters.end()) ? find_it->second : "";
            const std::string replacement =
                (replacement_it != node.parameters.end()) ? replacement_it->second : "";
            if (find_value.empty()) {
                ReportError("StringManipulation: replace requires param1");
                return false;
            }
            expr = "REPLACE(" + quoted_column + ", " +
                   QuoteSqlStringLiteral(find_value) + ", " +
                   QuoteSqlStringLiteral(replacement) + ")";
        } else if (operation == "substring") {
            const auto start_it = node.parameters.find("param1");
            const auto length_it = node.parameters.find("param2");
            if (start_it == node.parameters.end() || length_it == node.parameters.end() ||
                start_it->second.empty() || length_it->second.empty()) {
                ReportError("StringManipulation: substring requires param1 and param2");
                return false;
            }
            expr = "SUBSTRING(" + quoted_column + ", " + start_it->second +
                   ", " + length_it->second + ")";
        } else {
            ReportError("StringManipulation: Unsupported operation '" + operation + "'");
            return false;
        }

        std::string sql =
            "SELECT *, " + expr + " AS " + quoted_output + " FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("StringManipulation: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("StringManipulation error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteMathFormula(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("MathFormula: No input dataset");
        return false;
    }

    auto output_col_it = node.parameters.find("output_column");
    auto formula_it = node.parameters.find("formula");

    std::string output_column = (output_col_it != node.parameters.end()) ? output_col_it->second : "result";
    std::string formula = (formula_it != node.parameters.end()) ? formula_it->second : "";
    if (formula.empty()) {
        ReportError("MathFormula: Formula required");
        return false;
    }

    std::string output_dataset_name = "ds_math_" + std::to_string(node.id);
    spdlog::info("[Data Studio] MathFormula: {} = {}", output_column, formula);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("MathFormula: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string formula_expression;
        std::string formula_error;
        if (!BuildMathFormulaExpression(input_table, formula, formula_expression,
                                        formula_error)) {
            ReportError(formula_error);
            return false;
        }

        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("MathFormula: Failed to register table");
            return false;
        }

        std::string sql = "SELECT *, (" + formula_expression + ") AS " +
                          QuoteSqlIdentifier(output_column) + " FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("MathFormula: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("MathFormula error: " + std::string(e.what()));
        return false;
    }
}

} // namespace cyxwiz
