#include "pipeline_executor.h"
#include "duckdb_connector.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include "data_convert_service.h"
#include "node_executors/pipeline_operator_factory.h"
#include "pipeline_runtime_capabilities.h"
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/scalar.h>
#include <arrow/table.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
#include <queue>
#include <regex>
#include <sstream>
#include <future>
#include <thread>
#include <chrono>
#include <set>
#include <mutex>
#include <optional>

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

std::string DetectFormatNameFromPath(const std::string& path) {
    std::filesystem::path file_path(path);
    std::string extension = file_path.extension().string();
    if (!extension.empty() && extension.front() == '.') {
        extension.erase(extension.begin());
    }
    return ToLowerAscii(extension);
}

std::shared_ptr<ArrowDataset> LoadDataConvertOutputDataset(
    const std::string& output_path,
    const std::string& dataset_name,
    const std::string& output_format) {
    std::string format = ToLowerAscii(TrimString(output_format));
    if (format.empty() || format == "auto") {
        format = DetectFormatNameFromPath(output_path);
    }
    if (format == "tsv") {
        auto read_options = arrow::csv::ReadOptions::Defaults();
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        parse_options.delimiter = '\t';
        return ArrowDataset::FromCSV(output_path, dataset_name,
                                     read_options, parse_options,
                                     arrow::csv::ConvertOptions::Defaults());
    }
    return ArrowDataset::FromFile(output_path, dataset_name);
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

std::string QuoteJsonString(const std::string& value) {
    std::string quoted = "\"";
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            quoted += "\\\"";
            break;
        case '\\':
            quoted += "\\\\";
            break;
        case '\b':
            quoted += "\\b";
            break;
        case '\f':
            quoted += "\\f";
            break;
        case '\n':
            quoted += "\\n";
            break;
        case '\r':
            quoted += "\\r";
            break;
        case '\t':
            quoted += "\\t";
            break;
        default:
            if (c < 0x20) {
                quoted += "\\u00";
                quoted.push_back(hex[(c >> 4) & 0x0f]);
                quoted.push_back(hex[c & 0x0f]);
            } else {
                quoted.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    quoted += '"';
    return quoted;
}

std::string ScalarToJsonValue(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return "null";
    }

    switch (scalar->type->id()) {
    case arrow::Type::BOOL:
        return std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value
                   ? "true"
                   : "false";
    case arrow::Type::STRING:
        return QuoteJsonString(
            std::static_pointer_cast<arrow::StringScalar>(scalar)->value);
    case arrow::Type::LARGE_STRING:
        return QuoteJsonString(
            std::static_pointer_cast<arrow::LargeStringScalar>(scalar)->value);
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
        return scalar->ToString();
    default:
        return QuoteJsonString(scalar->ToString());
    }
}

struct RuleEngineLiteral {
    std::string sql;
    bool numeric = false;
    bool null_value = false;
};

bool BuildRuleEngineLiteral(const std::string& raw_value,
                            RuleEngineLiteral& literal,
                            std::string& error) {
    const std::string value = TrimString(raw_value);
    if (value.empty()) {
        error = "RuleEngine: rule value is required";
        return false;
    }

    if (ToLowerAscii(value) == "null") {
        literal.sql = "NULL";
        literal.null_value = true;
        return true;
    }

    if (value.size() >= 2 &&
        ((value.front() == '\'' && value.back() == '\'') ||
         (value.front() == '"' && value.back() == '"'))) {
        literal.sql = QuoteSqlStringLiteral(value.substr(1, value.size() - 2));
        return true;
    }

    double numeric_value = 0.0;
    if (TryParseFiniteDouble(value, numeric_value)) {
        literal.sql = value;
        literal.numeric = true;
        return true;
    }

    literal.sql = QuoteSqlStringLiteral(value);
    return true;
}

bool BuildRuleEngineCondition(const std::shared_ptr<arrow::Table>& table,
                              const std::string& condition,
                              std::string& expression,
                              std::string& error) {
    static const std::vector<std::string> operators = {
        ">=", "<=", "!=", "==", "=", ">", "<"};
    for (const auto& op : operators) {
        const auto pos = condition.find(op);
        if (pos == std::string::npos) {
            continue;
        }

        const std::string column = TrimString(condition.substr(0, pos));
        const std::string raw_value = TrimString(condition.substr(pos + op.size()));
        if (column.empty() || raw_value.empty()) {
            error = "RuleEngine: condition must use column operator value";
            return false;
        }
        if (!table || !table->schema()) {
            error = "RuleEngine: input table schema is unavailable";
            return false;
        }

        const int column_index = table->schema()->GetFieldIndex(column);
        if (column_index < 0) {
            error = "RuleEngine: condition column '" + column + "' not found";
            return false;
        }

        RuleEngineLiteral literal;
        if (!BuildRuleEngineLiteral(raw_value, literal, error)) {
            return false;
        }
        const bool ordered_compare = op == ">" || op == "<" || op == ">=" || op == "<=";
        if (ordered_compare) {
            const auto field = table->schema()->field(column_index);
            if (!field || !IsNumericArrowType(field->type()) || !literal.numeric) {
                error = "RuleEngine: ordered comparisons require numeric columns and values";
                return false;
            }
        }

        const std::string sql_op = (op == "==" || op == "=") ? "=" : op;
        expression = QuoteSqlIdentifier(column) + " " + sql_op + " " + literal.sql;
        return true;
    }

    error = "RuleEngine: condition must contain a supported comparison operator";
    return false;
}

bool BuildRuleEngineCaseExpression(const std::shared_ptr<arrow::Table>& table,
                                   const std::string& rules,
                                   const std::string& default_value,
                                   std::string& expression,
                                   std::string& error) {
    RuleEngineLiteral default_literal;
    if (!BuildRuleEngineLiteral(default_value.empty() ? "NULL" : default_value,
                                default_literal, error)) {
        return false;
    }

    expression = "CASE";
    bool saw_rule = false;
    std::stringstream lines(rules);
    std::string line;
    while (std::getline(lines, line)) {
        line = TrimString(line);
        if (line.empty()) {
            continue;
        }

        const auto delimiter = line.find("=>");
        if (delimiter == std::string::npos) {
            error = "RuleEngine: rules must use condition => value";
            return false;
        }

        std::string condition_expression;
        if (!BuildRuleEngineCondition(table, TrimString(line.substr(0, delimiter)),
                                      condition_expression, error)) {
            return false;
        }

        RuleEngineLiteral value_literal;
        if (!BuildRuleEngineLiteral(TrimString(line.substr(delimiter + 2)),
                                    value_literal, error)) {
            return false;
        }

        expression += " WHEN " + condition_expression + " THEN " +
                      value_literal.sql;
        saw_rule = true;
    }

    if (!saw_rule) {
        error = "RuleEngine: rules are required";
        return false;
    }

    expression += " ELSE " + default_literal.sql + " END";
    return true;
}

std::string NormalizeUnitName(std::string unit) {
    unit = ToLowerAscii(TrimString(unit));
    unit.erase(std::remove_if(unit.begin(), unit.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               unit.end());
    return unit;
}

bool UnitScaleToBase(const std::string& category,
                     const std::string& unit,
                     double& scale,
                     std::string& error) {
    const std::string normalized = NormalizeUnitName(unit);
    const auto fail = [&]() {
        error = "UnitConverter: unsupported " + category + " unit '" + unit + "'";
        return false;
    };

    if (category == "length") {
        if (normalized == "m" || normalized == "meter" || normalized == "meters") scale = 1.0;
        else if (normalized == "km" || normalized == "kilometer" || normalized == "kilometers") scale = 1000.0;
        else if (normalized == "cm" || normalized == "centimeter" || normalized == "centimeters") scale = 0.01;
        else if (normalized == "mm" || normalized == "millimeter" || normalized == "millimeters") scale = 0.001;
        else if (normalized == "ft" || normalized == "foot" || normalized == "feet") scale = 0.3048;
        else if (normalized == "in" || normalized == "inch" || normalized == "inches") scale = 0.0254;
        else if (normalized == "mi" || normalized == "mile" || normalized == "miles") scale = 1609.344;
        else return fail();
        return true;
    }

    if (category == "mass") {
        if (normalized == "kg" || normalized == "kilogram" || normalized == "kilograms") scale = 1.0;
        else if (normalized == "g" || normalized == "gram" || normalized == "grams") scale = 0.001;
        else if (normalized == "mg" || normalized == "milligram" || normalized == "milligrams") scale = 0.000001;
        else if (normalized == "lb" || normalized == "lbs" || normalized == "pound" || normalized == "pounds") scale = 0.45359237;
        else if (normalized == "oz" || normalized == "ounce" || normalized == "ounces") scale = 0.028349523125;
        else return fail();
        return true;
    }

    if (category == "time") {
        if (normalized == "s" || normalized == "sec" || normalized == "second" || normalized == "seconds") scale = 1.0;
        else if (normalized == "min" || normalized == "minute" || normalized == "minutes") scale = 60.0;
        else if (normalized == "h" || normalized == "hr" || normalized == "hour" || normalized == "hours") scale = 3600.0;
        else if (normalized == "day" || normalized == "days" || normalized == "d") scale = 86400.0;
        else return fail();
        return true;
    }

    if (category == "area") {
        if (normalized == "m2" || normalized == "m^2" || normalized == "sqm") scale = 1.0;
        else if (normalized == "cm2" || normalized == "cm^2") scale = 0.0001;
        else if (normalized == "ft2" || normalized == "ft^2" || normalized == "sqft") scale = 0.09290304;
        else if (normalized == "acre" || normalized == "acres") scale = 4046.8564224;
        else return fail();
        return true;
    }

    if (category == "volume") {
        if (normalized == "l" || normalized == "liter" || normalized == "liters") scale = 1.0;
        else if (normalized == "ml" || normalized == "milliliter" || normalized == "milliliters") scale = 0.001;
        else if (normalized == "m3" || normalized == "m^3") scale = 1000.0;
        else if (normalized == "ft3" || normalized == "ft^3") scale = 28.316846592;
        else if (normalized == "gal" || normalized == "gallon" || normalized == "gallons") scale = 3.785411784;
        else return fail();
        return true;
    }

    error = "UnitConverter: unsupported category '" + category + "'";
    return false;
}

bool BuildUnitConverterExpression(const std::string& category,
                                  const std::string& from_unit,
                                  const std::string& to_unit,
                                  const std::string& quoted_column,
                                  std::string& expression,
                                  std::string& error) {
    const std::string normalized_category = ToLowerAscii(TrimString(category));
    if (normalized_category == "temperature") {
        const std::string from = NormalizeUnitName(from_unit);
        const std::string to = NormalizeUnitName(to_unit);
        std::string celsius_expression;
        if (from == "c" || from == "celsius") {
            celsius_expression = quoted_column;
        } else if (from == "f" || from == "fahrenheit") {
            celsius_expression = "((" + quoted_column + " - 32.0) * 5.0 / 9.0)";
        } else if (from == "k" || from == "kelvin") {
            celsius_expression = "(" + quoted_column + " - 273.15)";
        } else {
            error = "UnitConverter: unsupported temperature unit '" + from_unit + "'";
            return false;
        }

        if (to == "c" || to == "celsius") {
            expression = celsius_expression;
        } else if (to == "f" || to == "fahrenheit") {
            expression = "((" + celsius_expression + ") * 9.0 / 5.0 + 32.0)";
        } else if (to == "k" || to == "kelvin") {
            expression = "((" + celsius_expression + ") + 273.15)";
        } else {
            error = "UnitConverter: unsupported temperature unit '" + to_unit + "'";
            return false;
        }
        return true;
    }

    double from_scale = 1.0;
    double to_scale = 1.0;
    if (!UnitScaleToBase(normalized_category, from_unit, from_scale, error) ||
        !UnitScaleToBase(normalized_category, to_unit, to_scale, error)) {
        return false;
    }

    expression = "((" + quoted_column + ") * " + std::to_string(from_scale) +
                 " / " + std::to_string(to_scale) + ")";
    return true;
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

bool IsTextLabelArrowType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) {
        return false;
    }
    switch (type->id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
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

bool RequireRoleColumnKind(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::string& column,
    const std::string& role,
    const std::string& kind,
    bool (*predicate)(const std::shared_ptr<arrow::DataType>&),
    std::string& error) {
    if (!table || !table->schema()) {
        error = node_type + ": input table schema is unavailable";
        return false;
    }

    const int column_index = table->schema()->GetFieldIndex(column);
    const std::string column_role =
        (role == "column") ? "column" : role + " column";
    if (column_index < 0) {
        error = node_type + ": " + column_role + " '" + column +
                "' not found";
        return false;
    }

    const auto field = table->schema()->field(column_index);
    if (!field || !predicate(field->type())) {
        const std::string found =
            field && field->type() ? field->type()->ToString() : "unknown";
        error = node_type + ": " + column_role + " '" + column +
                "' must be " + kind + " (found " + found + ")";
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

bool BuildCalculatorExpression(const std::string& formula,
                               std::string& expression,
                               std::string& error) {
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
                error = "CalculatorNode: invalid numeric literal";
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

        error = "CalculatorNode: expression contains unsupported token '" +
                std::string(1, formula[index]) + "'";
        return false;
    }

    if (!saw_token) {
        error = "CalculatorNode: expression is required";
        return false;
    }
    return true;
}

bool ParseSimpleJsonPath(const std::string& path,
                         std::vector<std::string>& segments,
                         std::string& error) {
    segments.clear();
    const std::string trimmed = TrimString(path);
    if (trimmed.empty()) {
        error = "JSONPathExtractor: path is required";
        return false;
    }
    if (trimmed == "$") {
        return true;
    }
    if (trimmed.rfind("$.", 0) != 0) {
        error = "JSONPathExtractor: only simple $.field paths are supported";
        return false;
    }

    std::stringstream tokens(trimmed.substr(2));
    std::string token;
    while (std::getline(tokens, token, '.')) {
        token = TrimString(token);
        if (token.empty() || token.find('[') != std::string::npos ||
            token.find(']') != std::string::npos) {
            error = "JSONPathExtractor: only dot-separated object fields are supported";
            return false;
        }
        segments.push_back(token);
    }
    return true;
}

bool ExtractJsonPathValue(const nlohmann::json& document,
                          const std::vector<std::string>& segments,
                          nlohmann::json& value) {
    const nlohmann::json* current = &document;
    for (const auto& segment : segments) {
        if (!current->is_object()) {
            return false;
        }
        const auto it = current->find(segment);
        if (it == current->end()) {
            return false;
        }
        current = &(*it);
    }
    value = *current;
    return true;
}

std::string JsonValueToDatasetString(const nlohmann::json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return value.dump();
}

bool BuildRegexOptions(const std::string& flags,
                       std::regex::flag_type& options,
                       std::string& error) {
    options = std::regex::ECMAScript;
    for (char flag : flags) {
        if (std::isspace(static_cast<unsigned char>(flag))) {
            continue;
        }
        if (flag == 'i' || flag == 'I') {
            options |= std::regex::icase;
            continue;
        }
        if (flag == 'm' || flag == 'M') {
            continue;
        }
        error = "RegexTester: unsupported regex flag '" + std::string(1, flag) + "'";
        return false;
    }
    return true;
}

bool TryParseInteger(const std::string& value, int64_t& parsed) {
    if (value.empty()) {
        return false;
    }
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    auto [ptr, ec] = std::from_chars(begin, end, parsed);
    return ec == std::errc() && ptr == end;
}

bool IsForbiddenIntegerValue(
    int64_t value,
    const std::vector<int64_t>& forbidden_values) {
    return std::find(forbidden_values.begin(), forbidden_values.end(), value) !=
           forbidden_values.end();
}

bool IsIntegerAtLeastExcept(const std::string& value,
                            int64_t minimum,
                            const std::vector<int64_t>& forbidden_values) {
    int64_t parsed = 0;
    return TryParseInteger(value, parsed) && parsed >= minimum &&
           !IsForbiddenIntegerValue(parsed, forbidden_values);
}

bool IsIntegerAtLeast(const std::string& value, int64_t minimum) {
    return IsIntegerAtLeastExcept(value, minimum, {});
}

std::string DescribeIntegerParameterBounds(
    int64_t minimum,
    const std::vector<int64_t>& forbidden_values) {
    std::string description = "integer >= " + std::to_string(minimum);
    if (!forbidden_values.empty()) {
        description += " except";
        for (size_t i = 0; i < forbidden_values.size(); ++i) {
            description += (i == 0 ? " " : ", ") +
                           std::to_string(forbidden_values[i]);
        }
    }
    return description;
}

bool ValidateIntegerParameterAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    const std::vector<int64_t>& forbidden_values,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    if (IsIntegerAtLeastExcept(it->second, minimum, forbidden_values)) {
        return true;
    }
    error = node_type + " " + parameter_name + " must be an " +
            DescribeIntegerParameterBounds(minimum, forbidden_values);
    return false;
}

bool ValidateCommaSeparatedIntegersAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    const std::vector<int64_t>& forbidden_values,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end()) {
        return true;
    }
    if (it->second.empty()) {
        error = node_type + " " + parameter_name +
                " must be a comma-separated list of ";
        if (forbidden_values.empty()) {
            error += "integers >= " + std::to_string(minimum);
        } else {
            error += DescribeIntegerParameterBounds(minimum, forbidden_values);
        }
        return false;
    }

    std::stringstream values(it->second);
    std::string value;
    while (std::getline(values, value, ',')) {
        if (!IsIntegerAtLeastExcept(value, minimum, forbidden_values)) {
            error = node_type + " " + parameter_name +
                    " must be a comma-separated list of ";
            if (forbidden_values.empty()) {
                error += "integers >= " + std::to_string(minimum);
            } else {
                error += DescribeIntegerParameterBounds(minimum, forbidden_values);
            }
            return false;
        }
    }
    return true;
}

bool IsFloatInRange(const std::string& value,
                    const std::optional<double>& minimum,
                    const std::optional<double>& maximum,
                    bool minimum_inclusive,
                    bool maximum_inclusive) {
    const std::string trimmed = TrimString(value);
    if (trimmed.empty()) {
        return false;
    }

    double parsed = 0.0;
    const char* begin = trimmed.data();
    const char* end = trimmed.data() + trimmed.size();
    auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc() || ptr != end || !std::isfinite(parsed)) {
        return false;
    }

    if (minimum.has_value()) {
        if (minimum_inclusive) {
            if (parsed < *minimum) {
                return false;
            }
        } else if (parsed <= *minimum) {
            return false;
        }
    }

    if (maximum.has_value()) {
        if (maximum_inclusive) {
            if (parsed > *maximum) {
                return false;
            }
        } else if (parsed >= *maximum) {
            return false;
        }
    }

    return true;
}

std::string DescribeFloatParameterBounds(
    const std::optional<double>& minimum,
    const std::optional<double>& maximum,
    bool minimum_inclusive,
    bool maximum_inclusive) {
    if (minimum.has_value() && maximum.has_value()) {
        return "between " + std::to_string(*minimum) + " and " +
               std::to_string(*maximum);
    }
    if (minimum.has_value()) {
        return std::string(minimum_inclusive ? "greater than or equal to "
                                             : "greater than ") +
               std::to_string(*minimum);
    }
    if (maximum.has_value()) {
        return std::string(maximum_inclusive ? "less than or equal to "
                                             : "less than ") +
               std::to_string(*maximum);
    }
    return "finite";
}

bool ValidateFloatParameterBounds(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    const std::optional<double>& minimum,
    const std::optional<double>& maximum,
    bool minimum_inclusive,
    bool maximum_inclusive,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    if (IsFloatInRange(it->second, minimum, maximum, minimum_inclusive,
                       maximum_inclusive)) {
        return true;
    }
    error = node_type + " " + parameter_name + " must be a number " +
            DescribeFloatParameterBounds(minimum, maximum, minimum_inclusive,
                                         maximum_inclusive);
    return false;
}

bool TryParseFiniteDouble(const std::string& value, double& out) {
    const std::string trimmed = TrimString(value);
    if (trimmed.empty()) {
        return false;
    }

    const char* begin = trimmed.data();
    const char* end = trimmed.data() + trimmed.size();
    auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc() && ptr == end && std::isfinite(out);
}

double FloatParameterOrDefault(
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name,
    double default_value) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return default_value;
    }
    double value = default_value;
    return TryParseFiniteDouble(it->second, value) ? value : default_value;
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

std::string NormalizeDataInputFileType(
    const std::map<std::string, std::string>& parameters) {
    auto type_it = parameters.find("type");
    if (type_it != parameters.end() && !type_it->second.empty()) {
        return NormalizeDataInputFileType(type_it->second);
    }

    auto file_type_it = parameters.find("file_type");
    if (file_type_it != parameters.end() && !file_type_it->second.empty()) {
        return NormalizeDataInputFileType(file_type_it->second);
    }

    return "auto";
}

std::string NormalizeDatasetOutputFormat(
    const std::map<std::string, std::string>& parameters) {
    auto format_it = parameters.find("format");
    if (format_it != parameters.end() && !format_it->second.empty()) {
        return ToLowerAscii(TrimString(format_it->second));
    }

    auto file_type_it = parameters.find("file_type");
    if (file_type_it != parameters.end() && !file_type_it->second.empty()) {
        return ToLowerAscii(TrimString(file_type_it->second));
    }

    return "csv";
}

std::string NormalizeDataOutputPath(
    const std::map<std::string, std::string>& parameters) {
    auto file_path_it = parameters.find("file_path");
    if (file_path_it != parameters.end() && !file_path_it->second.empty()) {
        return file_path_it->second;
    }

    auto path_it = parameters.find("path");
    if (path_it != parameters.end() && !path_it->second.empty()) {
        return path_it->second;
    }

    return {};
}

std::string ParameterOrDefault(
    const std::map<std::string, std::string>& parameters,
    const char* key,
    const std::string& fallback = {}) {
    auto it = parameters.find(key);
    return it != parameters.end() ? it->second : fallback;
}

bool OptionalBooleanParameterOrDefault(
    const std::map<std::string, std::string>& parameters,
    const char* key,
    bool fallback) {
    auto it = parameters.find(key);
    if (it == parameters.end() || it->second.empty()) {
        return fallback;
    }

    bool parsed = false;
    return TryParseBooleanParameterValue(it->second, parsed) ? parsed : fallback;
}

int64_t OptionalIntegerParameterOrDefault(
    const std::map<std::string, std::string>& parameters,
    const char* key,
    int64_t fallback) {
    auto it = parameters.find(key);
    if (it == parameters.end() || it->second.empty()) {
        return fallback;
    }

    int64_t parsed = fallback;
    return TryParseInteger(it->second, parsed) ? parsed : fallback;
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

    if (node_type == "ExportCSV" || node_type == "ExportJSON" ||
        node_type == "ExportParquet") {
        return (HasNonEmptyParameter(parameters, "file_path") ||
                HasNonEmptyParameter(parameters, "path"))
            ? nullptr
            : "file_path";
    }

    if (node_type == "DataOutput") {
        return NormalizeDataOutputPath(parameters).empty()
            ? "file_path"
            : nullptr;
    }

    if (node_type == "DataConvert") {
        if (!HasNonEmptyParameter(parameters, "input_path")) {
            return "input_path";
        }
        if (!HasNonEmptyParameter(parameters, "output_path")) {
            return "output_path";
        }
        return nullptr;
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
    const std::vector<PipelineFloatParameterRuntimeCapability>&
        float_parameters,
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
                    capability.minimum, capability.forbidden_values, error)) {
                return false;
            }
        } else if (!ValidateIntegerParameterAtLeast(
                       parameters, node_type, capability.parameter_name,
                       capability.minimum, capability.forbidden_values,
                       error)) {
            return false;
        }
    }

    for (const auto& capability : float_parameters) {
        if (!ValidateFloatParameterBounds(
                parameters, node_type, capability.parameter_name,
                capability.minimum, capability.maximum,
                capability.minimum_inclusive, capability.maximum_inclusive,
                error)) {
            return false;
        }
    }

    if (node_type == "TimeSeriesSplit") {
        const double train_ratio =
            FloatParameterOrDefault(parameters, "train_ratio", 0.8);
        const double val_ratio =
            FloatParameterOrDefault(parameters, "val_ratio", 0.1);
        const double test_ratio =
            FloatParameterOrDefault(parameters, "test_ratio", 0.1);
        if (train_ratio <= 0.0) {
            error = "TimeSeriesSplit train_ratio must be > 0";
            return false;
        }
        const double sum = train_ratio + val_ratio + test_ratio;
        if (std::fabs(sum - 1.0) > 0.01) {
            error = "TimeSeriesSplit ratios must sum to 1.0";
            return false;
        }
    }

    if (node_type == "RobustScaler") {
        const double quantile_min =
            FloatParameterOrDefault(parameters, "quantile_min", 25.0);
        const double quantile_max =
            FloatParameterOrDefault(parameters, "quantile_max", 75.0);
        if (quantile_max <= quantile_min) {
            error = "RobustScaler quantile_min must be less than quantile_max";
            return false;
        }
    }

    if (node_type == "HierarchicalCluster") {
        const auto linkage_it = parameters.find("linkage");
        const std::string linkage =
            (linkage_it != parameters.end() && !linkage_it->second.empty())
                ? ToLowerAscii(TrimString(linkage_it->second))
                : "ward";
        const auto metric_it = parameters.find("metric");
        const std::string metric =
            (metric_it != parameters.end() && !metric_it->second.empty())
                ? ToLowerAscii(TrimString(metric_it->second))
                : "euclidean";
        if (linkage == "ward" && metric != "euclidean") {
            error = "HierarchicalCluster linkage='ward' requires metric='euclidean'";
            return false;
        }
    }

    if (node_type == "Convolution1D") {
        auto kernel_it = parameters.find("kernel");
        if (kernel_it != parameters.end()) {
            bool saw_value = false;
            std::stringstream tokens(kernel_it->second);
            std::string token;
            while (std::getline(tokens, token, ',')) {
                double value = 0.0;
                if (!TryParseFiniteDouble(token, value)) {
                    error = "Convolution1D kernel must be a comma-separated list of finite numbers";
                    return false;
                }
                saw_value = true;
            }
            if (!saw_value) {
                error = "Convolution1D kernel must be a comma-separated list of finite numbers";
                return false;
            }
        }
    }

    if (node_type == "FilterDesigner") {
        const auto filter_type_it = parameters.find("filter_type");
        const std::string filter_type =
            (filter_type_it != parameters.end() &&
             !filter_type_it->second.empty())
                ? ToLowerAscii(TrimString(filter_type_it->second))
                : "lowpass";
        if (filter_type == "bandpass" || filter_type == "bandstop") {
            if (!HasNonEmptyParameter(parameters, "cutoff_high")) {
                error = "FilterDesigner " + filter_type +
                        " requires cutoff_high";
                return false;
            }
            const double cutoff =
                FloatParameterOrDefault(parameters, "cutoff", 0.5);
            const double cutoff_high =
                FloatParameterOrDefault(parameters, "cutoff_high", 0.0);
            if (cutoff_high <= cutoff) {
                error = "FilterDesigner " + filter_type +
                        " requires cutoff_high > cutoff";
                return false;
            }
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

    if (node_type == "TableCropper") {
        const auto start_row_it = parameters.find("start_row");
        const auto end_row_it = parameters.find("end_row");
        const int64_t start_row =
            (start_row_it != parameters.end() && !start_row_it->second.empty())
                ? std::stoll(start_row_it->second)
                : 0;
        const int64_t end_row =
            (end_row_it != parameters.end() && !end_row_it->second.empty())
                ? std::stoll(end_row_it->second)
                : -1;
        if (end_row >= 0 && end_row < start_row) {
            error = "TableCropper end_row must be >= start_row";
            return false;
        }
    }

    if (node_type == "TextCleanNode" || node_type == "TextClean") {
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
            error = node_type + " remove_stopwords is not supported by PipelineExecutor";
            return false;
        }
    }

    if (node_type == "TextVectorize" &&
        HasNonEmptyParameter(parameters, "max_features")) {
        error = "TextVectorize max_features is not supported by the legacy PipelineExecutor path; use CountVectorizer or TFIDFVectorizer";
        return false;
    }

    if (node_type == "TextTokenizer") {
        if (!ValidateOptionalBooleanParameter(
                parameters, node_type, "lowercase", error) ||
            !ValidateOptionalBooleanParameter(
                parameters, node_type, "vocab_build_if_missing", error)) {
            return false;
        }
    }

    if (node_type == "TSWindow") {
        const auto stride_it = parameters.find("stride");
        if (stride_it != parameters.end() && !stride_it->second.empty() &&
            std::stoi(stride_it->second) != 1) {
            error = "TSWindow stride values other than 1 are not supported by PipelineExecutor";
            return false;
        }
    }

    if (node_type == "BinningNode" || node_type == "Binning") {
        const auto columns_it = parameters.find("columns");
        if (columns_it != parameters.end() &&
            columns_it->second.find(',') != std::string::npos) {
            error = node_type + " columns supports exactly one column";
            return false;
        }
    }

    if (node_type == "PolynomialFeaturesNode" ||
        node_type == "PolynomialFeatures") {
        const auto columns_it = parameters.find("columns");
        if (columns_it != parameters.end() &&
            columns_it->second.find(',') != std::string::npos) {
            error = node_type + " columns supports exactly one column";
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

        auto type_it = parameters.find("type");
        auto file_type_it = parameters.find("file_type");
        if (type_it != parameters.end() && !type_it->second.empty() &&
            file_type_it != parameters.end() &&
            !file_type_it->second.empty()) {
            const std::string type =
                NormalizeDataInputFileType(type_it->second);
            const std::string file_type =
                NormalizeDataInputFileType(file_type_it->second);
            if (type != file_type) {
                error = "DataInput type and file_type disagree";
                return false;
            }
        }

        const std::string file_type =
            NormalizeDataInputFileType(parameters);
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
        if (source_type == "folder") {
            const auto category_it = parameters.find("file_category");
            const std::string file_category =
                (category_it != parameters.end() && !category_it->second.empty())
                    ? ToLowerAscii(TrimString(category_it->second))
                    : "image";
            if (file_category != "image") {
                error = "DataInput folder source only supports image category in PipelineExecutor";
                return false;
            }
        }
    }

    if (node_type == "DataOutput" || node_type == "SaveDataset") {
        auto format_it = parameters.find("format");
        auto file_type_it = parameters.find("file_type");
        if (format_it != parameters.end() && !format_it->second.empty() &&
            file_type_it != parameters.end() && !file_type_it->second.empty()) {
            const std::string format =
                ToLowerAscii(TrimString(format_it->second));
            const std::string file_type =
                ToLowerAscii(TrimString(file_type_it->second));
            if (format != file_type) {
                error = node_type + " format and file_type disagree";
                return false;
            }
        }
    }

    if (node_type == "DataConvert") {
        const auto skip_rows_it = parameters.find("skip_rows");
        if (skip_rows_it != parameters.end() && !skip_rows_it->second.empty() &&
            !IsIntegerAtLeast(skip_rows_it->second, 0)) {
            error = "DataConvert skip_rows must be a non-negative integer";
            return false;
        }
        const auto row_group_it = parameters.find("row_group_size");
        if (row_group_it != parameters.end() && !row_group_it->second.empty() &&
            !IsIntegerAtLeast(row_group_it->second, 1)) {
            error = "DataConvert row_group_size must be an integer >= 1";
            return false;
        }
        for (const char* bool_param : {"header", "has_header",
                                       "allow_newlines_in_values",
                                       "overwrite",
                                       "create_parent_dirs",
                                       "write_manifest"}) {
            if (!ValidateOptionalBooleanParameter(parameters, node_type,
                                                  bool_param, error)) {
                return false;
            }
        }
    }

    return true;
}

bool ValidateOptionalRoleColumnKind(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name,
    const std::string& role,
    const std::string& kind,
    bool (*predicate)(const std::shared_ptr<arrow::DataType>&),
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    return RequireRoleColumnKind(table, node_type, it->second, role, kind,
                                 predicate, error);
}

bool ValidateOptionalRoleColumnExists(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name,
    const std::string& role,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    if (!table || !table->schema()) {
        error = node_type + ": input table schema is unavailable";
        return false;
    }
    if (table->schema()->GetFieldIndex(it->second) < 0) {
        error = node_type + ": " + role + " column '" + it->second +
                "' not found";
        return false;
    }
    return true;
}

bool ValidateTextOperatorInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    const auto text_it = parameters.find("text_col");
    if (text_it == parameters.end() || text_it->second.empty()) {
        return true;
    }
    if (!RequireRoleColumnKind(table, node_type, text_it->second, "text",
                               "string/large_string", IsStringArrowType,
                               error)) {
        return false;
    }
    return ValidateOptionalRoleColumnKind(
        table, node_type, parameters, "label_col", "label",
        "string or numeric label", IsTextLabelArrowType, error);
}

bool ValidateRequiredRoleColumnKind(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name,
    const std::string& role,
    const std::string& kind,
    bool (*predicate)(const std::shared_ptr<arrow::DataType>&),
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    return RequireRoleColumnKind(table, node_type, it->second, role, kind,
                                 predicate, error);
}

bool ValidateOptionalRoleColumnListKind(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    const char* parameter_name,
    const std::string& role,
    const std::string& kind,
    bool (*predicate)(const std::shared_ptr<arrow::DataType>&),
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }

    const std::vector<std::string> columns = ParseCommaSeparatedNames(it->second);
    if (columns.empty()) {
        const std::string column_role =
            (role == "column") ? "columns" : role + " columns";
        error = node_type + ": no " + column_role + " were provided";
        return false;
    }
    for (const auto& column : columns) {
        if (!RequireRoleColumnKind(table, node_type, column, role, kind,
                                   predicate, error)) {
            return false;
        }
    }
    return true;
}

bool ValidateSignalColumnInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    return ValidateRequiredRoleColumnKind(
        table, node_type, parameters, "signal_col", "signal", "numeric",
        IsNumericArrowType, error);
}

bool ValidateValueColumnInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    return ValidateRequiredRoleColumnKind(
        table, node_type, parameters, "value_col", "value", "numeric",
        IsNumericArrowType, error);
}

bool ValidateTimeSeriesWindowInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateValueColumnInputSchema(table, node_type, parameters,
                                        error)) {
        return false;
    }
    if (!ValidateOptionalRoleColumnListKind(
            table, node_type, parameters, "feature_cols", "feature",
            "numeric", IsNumericArrowType, error)) {
        return false;
    }
    return ValidateOptionalRoleColumnKind(
        table, node_type, parameters, "time_col", "time", "numeric",
        IsNumericArrowType, error);
}

bool ValidateFeatureColumnsInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    return ValidateOptionalRoleColumnListKind(
        table, node_type, parameters, "feature_cols", "feature", "numeric",
        IsNumericArrowType, error);
}

bool ValidateFeatureColumnsWithLabelInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateFeatureColumnsInputSchema(table, node_type, parameters,
                                           error)) {
        return false;
    }
    return ValidateOptionalRoleColumnExists(
        table, node_type, parameters, "label_col", "label", error);
}

bool ValidateNumericColumnsInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    return ValidateOptionalRoleColumnListKind(
        table, node_type, parameters, "columns", "column", "numeric",
        IsNumericArrowType, error);
}

bool ValidateNumericColumnsWithLabelInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateNumericColumnsInputSchema(table, node_type, parameters,
                                           error)) {
        return false;
    }
    return ValidateOptionalRoleColumnExists(
        table, node_type, parameters, "label_col", "label", error);
}

bool ValidateOutlierDetectorInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    const auto columns_it = parameters.find("columns");
    if (columns_it != parameters.end() &&
        ToLowerAscii(TrimString(columns_it->second)) == "all") {
        return ValidateOptionalRoleColumnExists(
            table, node_type, parameters, "label_col", "label", error);
    }
    return ValidateNumericColumnsWithLabelInputSchema(table, node_type,
                                                     parameters, error);
}

bool ValidateCategoricalColumnsInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    return ValidateOptionalRoleColumnListKind(
        table, node_type, parameters, "columns", "categorical",
        "string/large_string", IsStringArrowType, error);
}

bool ValidatePCAInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateFeatureColumnsInputSchema(table, node_type, parameters,
                                           error)) {
        return false;
    }
    return ValidateOptionalRoleColumnKind(
        table, node_type, parameters, "label_col", "label",
        "string or numeric label", IsTextLabelArrowType, error);
}

bool ValidateLinearRegressionInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateFeatureColumnsInputSchema(table, node_type, parameters,
                                           error)) {
        return false;
    }
    return ValidateRequiredRoleColumnKind(
        table, node_type, parameters, "target_col", "target", "numeric",
        IsNumericArrowType, error);
}

bool ValidatePolynomialRegressionInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateRequiredRoleColumnKind(
            table, node_type, parameters, "feature_col", "feature",
            "numeric", IsNumericArrowType, error)) {
        return false;
    }
    return ValidateRequiredRoleColumnKind(
        table, node_type, parameters, "target_col", "target", "numeric",
        IsNumericArrowType, error);
}

bool ValidateTargetEncoderInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    if (!ValidateCategoricalColumnsInputSchema(table, node_type, parameters,
                                               error)) {
        return false;
    }
    return ValidateRequiredRoleColumnKind(
        table, node_type, parameters, "target_col", "target", "numeric",
        IsNumericArrowType, error);
}

bool ValidatePipelineOperatorInputSchema(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    gui::NodeType type,
    std::string& error) {
    switch (type) {
        case gui::NodeType::TextTokenizer:
        case gui::NodeType::CountVectorizer:
        case gui::NodeType::TFIDFVectorizer:
        case gui::NodeType::SentimentAnalyzer:
            return ValidateTextOperatorInputSchema(table, node_type,
                                                   parameters, error);
        case gui::NodeType::FFTNode:
        case gui::NodeType::Convolution1D:
        case gui::NodeType::FilterDesigner:
        case gui::NodeType::TimeSeriesDecomposition:
        case gui::NodeType::ACFNode:
        case gui::NodeType::PACFNode:
        case gui::NodeType::StationarityTest:
        case gui::NodeType::SeasonalityDetector:
        case gui::NodeType::ARIMAForecaster:
        case gui::NodeType::ExponentialSmoothing:
            return ValidateSignalColumnInputSchema(table, node_type,
                                                   parameters, error);
        case gui::NodeType::TimeSeriesWindow:
            return ValidateTimeSeriesWindowInputSchema(table, node_type,
                                                       parameters, error);
        case gui::NodeType::TimeSeriesFeatures:
        case gui::NodeType::LogTransform:
        case gui::NodeType::Differencing:
            return ValidateValueColumnInputSchema(table, node_type,
                                                  parameters, error);
        case gui::NodeType::StandardScaler:
        case gui::NodeType::MinMaxScaler:
        case gui::NodeType::RobustScaler:
            return ValidateNumericColumnsWithLabelInputSchema(
                table, node_type, parameters, error);
        case gui::NodeType::OutlierDetector:
            return ValidateOutlierDetectorInputSchema(table, node_type,
                                                      parameters, error);
        case gui::NodeType::PCANode:
            return ValidatePCAInputSchema(table, node_type, parameters,
                                          error);
        case gui::NodeType::KMeansCluster:
        case gui::NodeType::DBSCANCluster:
        case gui::NodeType::HierarchicalCluster:
        case gui::NodeType::GMMCluster:
            return ValidateFeatureColumnsWithLabelInputSchema(
                table, node_type, parameters, error);
        case gui::NodeType::LinearRegressionNode:
            return ValidateLinearRegressionInputSchema(table, node_type,
                                                       parameters, error);
        case gui::NodeType::PolynomialRegressionNode:
            return ValidatePolynomialRegressionInputSchema(
                table, node_type, parameters, error);
        case gui::NodeType::LabelEncoder:
            return ValidateRequiredRoleColumnKind(
                table, node_type, parameters, "column", "column",
                "string/large_string", IsStringArrowType, error);
        case gui::NodeType::OrdinalEncoder:
            return ValidateCategoricalColumnsInputSchema(table, node_type,
                                                         parameters, error);
        case gui::NodeType::TargetEncoder:
            return ValidateTargetEncoderInputSchema(table, node_type,
                                                    parameters, error);
        default:
            return true;
    }
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
                runtime_support.float_parameters,
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

    const auto support = ResolveNodeRuntimeSupport(node);
    if (support.mode == PipelineRuntimeSupportMode::FailClosed &&
        support.fail_closed_reason != nullptr) {
        return FailUnsupportedNode(node, support.fail_closed_reason);
    }

    bool typed_legacy_handled = false;
    if (ExecuteTypedLegacyNode(node, ctx, typed_legacy_handled)) {
        return true;
    }
    if (typed_legacy_handled) {
        return false;
    }

    if (support.mode == PipelineRuntimeSupportMode::OperatorBacked &&
        support.operator_type.has_value()) {
        return ExecutePipelineOperatorNode(node, ctx, *support.operator_type);
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
    case gui::NodeType::DataInput:
        return ExecuteDataInput(node, ctx);
    case gui::NodeType::DataOutput:
        if (node.type == "SaveDataset") {
            return ExecuteSaveDataset(node, ctx);
        }
        return ExecuteDataOutput(node, ctx);
    case gui::NodeType::DeployToNodeEditorNode:
        return ExecuteDeployToNodeEditor(node, ctx);
    case gui::NodeType::DataConvert:
        return ExecuteDataConvert(node, ctx);
    case gui::NodeType::FilterRows:
        return ExecuteFilterRows(node, ctx);
    case gui::NodeType::SelectColumns:
        return ExecuteSelectColumns(node, ctx);
    case gui::NodeType::RemoveDuplicateRows:
        return ExecuteRemoveDuplicates(node, ctx);
    case gui::NodeType::BinningNode:
        return ExecuteBinning(node, ctx);
    case gui::NodeType::PolynomialFeaturesNode:
        return ExecutePolynomialFeatures(node, ctx);
    case gui::NodeType::TimeSeriesLag:
        return ExecuteTSLag(node, ctx);
    case gui::NodeType::TimeSeriesWindow:
        return ExecuteTSWindow(node, ctx);
    case gui::NodeType::TimeSeriesFeatures:
        return ExecuteTSFeatures(node, ctx);
    case gui::NodeType::Differencing:
        return ExecuteTSDiff(node, ctx);
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
    case gui::NodeType::ExportJSON:
        return ExecuteExportJSON(node, ctx);
    case gui::NodeType::ExportParquet:
        return ExecuteExportParquet(node, ctx);
    case gui::NodeType::RuleEngine:
        return ExecuteRuleEngine(node, ctx);
    case gui::NodeType::UnitConverter:
        return ExecuteUnitConverter(node, ctx);
    case gui::NodeType::CalculatorNode:
        return ExecuteCalculatorNode(node, ctx);
    case gui::NodeType::JSONPathExtractor:
        return ExecuteJSONPathExtractor(node, ctx);
    case gui::NodeType::RegexTester:
        return ExecuteRegexTester(node, ctx);
    case gui::NodeType::DataProfiler:
        return ExecuteDataProfiler(node, ctx);
    case gui::NodeType::RegressionMetricsNode:
        return ExecuteRegressionMetrics(node, ctx);
    case gui::NodeType::ConfusionMatrixNode:
        return ExecuteConfusionMatrix(node, ctx);
    case gui::NodeType::ROCCurveNode:
        return ExecuteROCCurve(node, ctx);
    case gui::NodeType::PRCurveNode:
        return ExecutePRCurve(node, ctx);
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
    case gui::NodeType::CellExtractor:
        return ExecuteCellExtractor(node, ctx);
    case gui::NodeType::CellUpdater:
        return ExecuteCellUpdater(node, ctx);
    case gui::NodeType::RowAppender:
        return ExecuteRowAppender(node, ctx);
    case gui::NodeType::ColumnAppender:
        return ExecuteColumnAppender(node, ctx);
    case gui::NodeType::Unpivot:
        return ExecuteUnpivot(node, ctx);
    case gui::NodeType::CountVectorizer:
        return ExecuteTextVectorize(node, ctx);
    case gui::NodeType::TextTokenizer:
        return ExecuteTextTokenize(node, ctx);
    case gui::NodeType::TextCleanNode:
        return ExecuteTextClean(node, ctx);
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
    auto format_it = node.parameters.find("format");
    const std::string format =
        (format_it != node.parameters.end() && !format_it->second.empty())
            ? NormalizeDataInputFileType(format_it->second)
            : "auto";
    std::string dataset_name = "ds_input_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Loading file: {} as dataset '{}' (format: {})",
                 file_path, dataset_name, format);

    try {
        // Use DataRegistry's Arrow support to load the file
        auto& registry = DataRegistry::Instance();
        std::shared_ptr<ArrowDataset> arrow_dataset;
        if (format == "csv") {
            arrow_dataset = registry.LoadCSVToArrow(file_path, dataset_name);
        } else if (format == "parquet") {
            arrow_dataset = registry.LoadParquetToArrow(file_path, dataset_name);
        } else {
            arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);
        }

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
            std::string file_type =
                NormalizeDataInputFileType(node.parameters);
            if (node.type == "ParquetInput" && file_type == "auto") {
                file_type = "parquet";
            }

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

    const std::string output_path = NormalizeDataOutputPath(node.parameters);
    if (output_path.empty()) {
        ReportError(GetImprovedErrorMessage("DataOutput", "missing_parameter", "file_path"));
        return false;
    }

    std::string format = NormalizeDatasetOutputFormat(node.parameters);

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

bool PipelineExecutor::ExecuteDataConvert(const Node& node, ExecutionContext& ctx) {
    DataConvertOptions options;
    options.input_path = ParameterOrDefault(node.parameters, "input_path");
    options.output_path = ParameterOrDefault(node.parameters, "output_path");
    options.input_format = ParameterOrDefault(node.parameters, "input_format", "auto");
    options.output_format = ParameterOrDefault(node.parameters, "output_format", "auto");

    const std::string delimiter =
        ToLowerAscii(TrimString(ParameterOrDefault(node.parameters,
                                                  "delimiter", "auto")));
    if (delimiter == "auto" || delimiter.empty()) {
        options.auto_detect_delimiter = true;
        options.delimiter = ',';
    } else {
        options.auto_detect_delimiter = false;
        options.delimiter = delimiter.front();
    }

    options.has_header = OptionalBooleanParameterOrDefault(
        node.parameters, "has_header",
        OptionalBooleanParameterOrDefault(node.parameters, "header", true));
    options.allow_newlines_in_values = OptionalBooleanParameterOrDefault(
        node.parameters, "allow_newlines_in_values", true);
    options.skip_rows = static_cast<int>(OptionalIntegerParameterOrDefault(
        node.parameters, "skip_rows", 0));
    options.parquet_compression = ParameterOrDefault(
        node.parameters, "compression", "snappy");
    options.row_group_size = OptionalIntegerParameterOrDefault(
        node.parameters, "row_group_size", 1024 * 1024);
    options.overwrite = OptionalBooleanParameterOrDefault(
        node.parameters, "overwrite", false);
    options.create_parent_dirs = OptionalBooleanParameterOrDefault(
        node.parameters, "create_parent_dirs", true);
    options.write_manifest = OptionalBooleanParameterOrDefault(
        node.parameters, "write_manifest", true);

    spdlog::info("[Pipeline] DataConvert converting '{}' -> '{}'",
                 options.input_path, options.output_path);

    try {
        auto result = DataConvertService::Convert(options);
        if (!result.ok) {
            ReportError("DataConvert: " + result.error);
            return false;
        }

        const std::string dataset_name = "ds_dataconvert_" + std::to_string(node.id);
        auto output_dataset = LoadDataConvertOutputDataset(
            result.output_path, dataset_name, options.output_format);
        if (!output_dataset || !output_dataset->GetArrowTable()) {
            ReportError("DataConvert: conversion succeeded, but generated output could not be loaded: " +
                        result.output_path);
            return false;
        }

        auto registered = DataRegistry::Instance().RegisterArrowTable(
            output_dataset->GetArrowTable(), dataset_name);
        if (!registered) {
            ReportError("DataConvert: conversion succeeded, but output dataset could not be registered");
            return false;
        }

        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) {
            ctx.input_dataset = dataset_name;
        }
        ctx.output_dataset = result.output_path;

        spdlog::info("[Pipeline] DataConvert wrote {} rows, {} columns to {}",
                     result.rows_written, result.columns, result.output_path);
        return true;
    } catch (const std::exception& e) {
        ReportError("DataConvert error: " + std::string(e.what()));
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
    const std::string diagnostic_name =
        node.type == "RemoveDuplicateRows" ? "RemoveDuplicateRows" :
                                              "RemoveDuplicates";
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
            ReportError(diagnostic_name + ": Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError(diagnostic_name + ": Input table schema is unavailable");
            return false;
        }

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError(diagnostic_name +
                        ": Failed to register table with DuckDB");
            return false;
        }

        // Execute DISTINCT query
        std::string sql = "SELECT DISTINCT * FROM " + temp_table;
        auto columns_it = node.parameters.find("columns");
        if (columns_it != node.parameters.end() &&
            !TrimString(columns_it->second).empty()) {
            std::vector<std::string> dedupe_columns;
            std::string column_error;
            if (!ResolveExistingColumns(input_table, diagnostic_name,
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
            ReportError(diagnostic_name + ": Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] {}: {} -> {} rows",
                    diagnostic_name, input_table->num_rows(),
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError(diagnostic_name + " error: " + std::string(e.what()));
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
    const std::string format = NormalizeDatasetOutputFormat(node.parameters);

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
        ctx.node_results[node.id] = output_name;
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
                if (!IsNumericArrowType(field->type())) {
                    duckdb_->UnregisterTable(temp_table);
                    ReportError("FillMissing: strategy 'mean' requires numeric column '" +
                                field->name() + "' (found " +
                                field->type()->ToString() + ")");
                    return false;
                }
                expression = "COALESCE(" + quoted_column + ", (SELECT AVG(" +
                             quoted_column + ") FROM " + temp_table + "))";
            } else if (strategy == "median") {
                if (!IsNumericArrowType(field->type())) {
                    duckdb_->UnregisterTable(temp_table);
                    ReportError("FillMissing: strategy 'median' requires numeric column '" +
                                field->name() + "' (found " +
                                field->type()->ToString() + ")");
                    return false;
                }
                expression = "COALESCE(" + quoted_column + ", (SELECT MEDIAN(" +
                             quoted_column + ") FROM " + temp_table + "))";
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
        ctx.node_results[node.id] = deployment_name;
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

    std::string schema_error;
    if (!ValidatePipelineOperatorInputSchema(input_table, node.type,
                                             node.parameters, type,
                                             schema_error)) {
        ReportError(schema_error);
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
    const std::string diagnostic_name =
        node.type == "TextCleanNode" ? "TextCleanNode" : "TextClean";
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(diagnostic_name +
                    ": No input connection or dataset not found");
        return false;
    }

    // Get parameters
    bool lowercase = OptionalBooleanParameterIsTrue(node.parameters, "lowercase");
    bool remove_html = OptionalBooleanParameterIsTrue(node.parameters, "remove_html");
    bool remove_special_chars =
        OptionalBooleanParameterIsTrue(node.parameters,
                                       "remove_special_chars");
    // remove_stopwords=true is rejected by ValidateNodeRuntimeParameters
    // until dictionary-backed stop-word removal is implemented.

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    std::string output_dataset_name = "ds_textclean_" + std::to_string(node.id);

    spdlog::info("[Data Studio] {} on column '{}' from '{}'",
                 diagnostic_name, text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError(diagnostic_name + ": Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, diagnostic_name, text_column,
                               "string", IsStringArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_text_column = QuoteSqlIdentifier(text_column);
        const std::string quoted_output_column =
            QuoteSqlIdentifier(text_column + "_cleaned");
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError(diagnostic_name + ": Failed to register table");
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
            ReportError(diagnostic_name + ": Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] {} completed: {} rows",
                     diagnostic_name, result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError(diagnostic_name + " error: " + std::string(e.what()));
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
    const std::string diagnostic_name =
        node.type == "TimeSeriesLag" ? "TimeSeriesLag" : "TSLag";
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(diagnostic_name +
                    ": No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto lags_it = node.parameters.find("lag_periods");
    std::string lag_periods = (lags_it != node.parameters.end()) ? lags_it->second : "1,7,30";

    std::string output_dataset_name = "ds_tslag_" + std::to_string(node.id);

    spdlog::info("[Data Studio] {} (periods={}) on '{}' from '{}'",
                 diagnostic_name, lag_periods, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError(diagnostic_name + ": Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, diagnostic_name, columns,
                               "numeric", IsNumericArrowType,
                               schema_error)) {
            ReportError(schema_error);
            return false;
        }
        const std::string quoted_column = QuoteSqlIdentifier(columns);
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError(diagnostic_name + ": Failed to register table");
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
            ReportError(diagnostic_name + ": Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] {} completed: {} rows with lag features",
                     diagnostic_name, result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError(diagnostic_name + " error: " + std::string(e.what()));
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
    const std::string diagnostic_name =
        node.type == "PolynomialFeaturesNode" ? "PolynomialFeaturesNode"
                                               : "PolynomialFeatures";
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(diagnostic_name +
                    ": No input connection or dataset not found");
        return false;
    }

    auto degree_it = node.parameters.find("degree");
    int degree = (degree_it != node.parameters.end()) ? std::stoi(degree_it->second) : 2;

    auto columns_it = node.parameters.find("columns");
    std::string column = (columns_it != node.parameters.end()) ? columns_it->second : "";
    if (column.empty()) {
        ReportError(diagnostic_name + ": Column name required");
        return false;
    }

    std::string output_dataset_name = "ds_poly_" + std::to_string(node.id);

    spdlog::info("[Data Studio] {} (degree={}) on '{}' from '{}'",
                 diagnostic_name, degree, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError(diagnostic_name + ": Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, diagnostic_name, column,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError(diagnostic_name + ": Failed to register table");
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
            ReportError(diagnostic_name + ": Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] {} completed: {} rows",
                     diagnostic_name, result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError(diagnostic_name + " error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteBinning(const Node& node, ExecutionContext& ctx) {
    const std::string diagnostic_name =
        node.type == "BinningNode" ? "BinningNode" : "Binning";
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(diagnostic_name + ": No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string column = (column_it != node.parameters.end()) ? column_it->second : "";
    if (column.empty()) {
        ReportError(diagnostic_name + ": Column name required");
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

    spdlog::info("[Data Studio] {} (method={}, bins={}) on '{}' from '{}'",
                diagnostic_name, method, n_bins, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError(diagnostic_name + ": Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string schema_error;
        if (!RequireColumnKind(input_table, diagnostic_name, column,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError(diagnostic_name + ": Failed to register table");
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
            ReportError(diagnostic_name + ": Unsupported method '" + method + "'");
            return false;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError(diagnostic_name + ": Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] {} completed: {} rows with {} bins",
                    diagnostic_name, result_table->num_rows(), n_bins);
        return true;

    } catch (const std::exception& e) {
        ReportError(diagnostic_name + " error: " + std::string(e.what()));
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

bool PipelineExecutor::ExecuteExportJSON(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportJSON: No input dataset");
        return false;
    }

    const std::string output_path = NormalizeDataOutputPath(node.parameters);
    if (output_path.empty()) {
        ReportError("ExportJSON: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to JSON: {}", output_path);
    auto& registry = DataRegistry::Instance();
    auto input_dataset = registry.GetArrowDataset(input_dataset_name);
    if (!input_dataset) {
        ReportError("ExportJSON: Input dataset not found in registry");
        return false;
    }

    auto table = input_dataset->GetArrowTable();
    if (!table || !table->schema()) {
        ReportError("ExportJSON: Input table is null");
        return false;
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        ReportError("ExportJSON: Failed to open output file");
        return false;
    }

    output << "[\n";
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        output << "  {";
        for (int column = 0; column < table->num_columns(); ++column) {
            if (column > 0) {
                output << ", ";
            }
            output << QuoteJsonString(table->schema()->field(column)->name())
                   << ": ";
            auto scalar_result = table->column(column)->GetScalar(row);
            if (!scalar_result.ok()) {
                ReportError("ExportJSON: Failed to read scalar value");
                return false;
            }
            output << ScalarToJsonValue(*scalar_result);
        }
        output << "}";
        if (row + 1 < table->num_rows()) {
            output << ",";
        }
        output << "\n";
    }
    output << "]\n";
    if (!output) {
        ReportError("ExportJSON: Failed to write output file");
        return false;
    }

    ctx.node_results[node.id] = input_dataset_name;
    ctx.output_dataset = output_path;
    return true;
}

bool PipelineExecutor::ExecuteExportParquet(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportParquet: No input dataset");
        return false;
    }

    const std::string output_path = NormalizeDataOutputPath(node.parameters);
    if (output_path.empty()) {
        ReportError("ExportParquet: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to Parquet: {}", output_path);
    auto& registry = DataRegistry::Instance();
    if (!registry.GetArrowDataset(input_dataset_name)) {
        ReportError("ExportParquet: Input dataset not found in registry");
        return false;
    }
    if (!registry.ExportArrowToParquet(input_dataset_name, output_path)) {
        ReportError("ExportParquet: Export failed");
        return false;
    }

    ctx.node_results[node.id] = input_dataset_name;
    ctx.output_dataset = output_path;
    return true;
}

bool PipelineExecutor::ExecuteRuleEngine(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RuleEngine: No input dataset");
        return false;
    }

    const auto rules_it = node.parameters.find("rules");
    const std::string rules =
        rules_it != node.parameters.end() ? rules_it->second : "";
    const auto output_column_it = node.parameters.find("output_column");
    const std::string output_column =
        output_column_it != node.parameters.end() &&
                !output_column_it->second.empty()
            ? output_column_it->second
            : "result";
    const auto default_value_it = node.parameters.find("default_value");
    const std::string default_value =
        default_value_it != node.parameters.end() ? default_value_it->second
                                                  : "NULL";
    if (rules.empty()) {
        ReportError("RuleEngine: rules are required");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RuleEngine: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string case_expression;
        std::string rule_error;
        if (!BuildRuleEngineCaseExpression(input_table, rules, default_value,
                                           case_expression, rule_error)) {
            ReportError(rule_error);
            return false;
        }

        const std::string temp_table = "temp_rule_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("RuleEngine: Failed to register table");
            return false;
        }

        const std::string sql = "SELECT *, " + case_expression + " AS " +
                                QuoteSqlIdentifier(output_column) + " FROM " +
                                temp_table;
        auto output_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);
        if (!output_table) {
            ReportError("RuleEngine: Query failed");
            return false;
        }

        const std::string output_dataset_name =
            "ds_ruleengine_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RuleEngine error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteUnitConverter(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("UnitConverter: No input dataset");
        return false;
    }

    const std::string category =
        ParameterOrDefault(node.parameters, "category", "length");
    const std::string from_unit =
        ParameterOrDefault(node.parameters, "from_unit", "m");
    const std::string to_unit =
        ParameterOrDefault(node.parameters, "to_unit", "ft");
    const std::string output_dataset_name =
        "ds_unitconverter_" + std::to_string(node.id);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("UnitConverter: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("UnitConverter: Input table is null");
            return false;
        }

        std::vector<std::string> select_expressions;
        select_expressions.reserve(input_table->num_columns());
        bool converted_column = false;
        for (int i = 0; i < input_table->num_columns(); ++i) {
            const auto field = input_table->schema()->field(i);
            const std::string quoted_column = QuoteSqlIdentifier(field->name());
            if (field && IsNumericArrowType(field->type())) {
                std::string converted_expression;
                std::string conversion_error;
                if (!BuildUnitConverterExpression(
                        category, from_unit, to_unit, quoted_column,
                        converted_expression, conversion_error)) {
                    ReportError(conversion_error);
                    return false;
                }
                select_expressions.push_back(converted_expression + " AS " +
                                             quoted_column);
                converted_column = true;
            } else {
                select_expressions.push_back(quoted_column);
            }
        }

        if (!converted_column) {
            ReportError("UnitConverter: input table has no numeric columns");
            return false;
        }

        const std::string temp_table = "temp_unit_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("UnitConverter: Failed to register table");
            return false;
        }

        std::string sql = "SELECT ";
        for (size_t i = 0; i < select_expressions.size(); ++i) {
            if (i > 0) {
                sql += ", ";
            }
            sql += select_expressions[i];
        }
        sql += " FROM " + temp_table;

        auto output_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);
        if (!output_table) {
            ReportError("UnitConverter: Query failed");
            return false;
        }

        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("UnitConverter error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteCalculatorNode(const Node& node, ExecutionContext& ctx) {
    const std::string expression =
        ParameterOrDefault(node.parameters, "expression", "2 + 2");
    const int64_t precision =
        OptionalIntegerParameterOrDefault(node.parameters, "precision", 6);
    std::string calculator_expression;
    std::string calculator_error;
    if (!BuildCalculatorExpression(expression, calculator_expression,
                                   calculator_error)) {
        ReportError(calculator_error);
        return false;
    }

    const std::string sql = "SELECT ROUND((" + calculator_expression + "), " +
                            std::to_string(precision) + ") AS result";
    try {
        auto output_table = duckdb_->Query(sql);
        if (!output_table) {
            ReportError("CalculatorNode: Query failed");
            return false;
        }

        const std::string output_dataset_name =
            "ds_calculator_" + std::to_string(node.id);
        DataRegistry::Instance().RegisterArrowTable(output_table,
                                                    output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("CalculatorNode error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteJSONPathExtractor(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("JSONPathExtractor: No input dataset");
        return false;
    }

    const std::string path = ParameterOrDefault(node.parameters, "path", "$");
    std::vector<std::string> path_segments;
    std::string path_error;
    if (!ParseSimpleJsonPath(path, path_segments, path_error)) {
        ReportError(path_error);
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("JSONPathExtractor: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("JSONPathExtractor: Input table is null");
            return false;
        }

        std::string json_column =
            ParameterOrDefault(node.parameters, "json_column", "");
        if (json_column.empty()) {
            json_column = ParameterOrDefault(node.parameters, "column", "");
        }
        int column_index = json_column.empty()
                               ? -1
                               : input_table->schema()->GetFieldIndex(json_column);
        if (column_index < 0 && !json_column.empty()) {
            ReportError("JSONPathExtractor: json column '" + json_column +
                        "' not found");
            return false;
        }
        if (column_index < 0) {
            for (int i = 0; i < input_table->num_columns(); ++i) {
                const auto field = input_table->schema()->field(i);
                if (field && IsStringArrowType(field->type())) {
                    column_index = i;
                    json_column = field->name();
                    break;
                }
            }
        }
        if (column_index < 0) {
            ReportError("JSONPathExtractor: input table has no JSON string column");
            return false;
        }

        arrow::StringBuilder result_builder;
        const auto column = input_table->column(column_index);
        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto scalar_result = column->GetScalar(row);
            if (!scalar_result.ok()) {
                ReportError("JSONPathExtractor: Failed to read JSON scalar");
                return false;
            }
            const auto scalar = *scalar_result;
            if (!scalar || !scalar->is_valid) {
                auto status = result_builder.AppendNull();
                if (!status.ok()) {
                    ReportError("JSONPathExtractor: Failed to append null result");
                    return false;
                }
                continue;
            }

            std::string json_text;
            if (scalar->type->id() == arrow::Type::STRING) {
                json_text =
                    std::static_pointer_cast<arrow::StringScalar>(scalar)->value;
            } else if (scalar->type->id() == arrow::Type::LARGE_STRING) {
                json_text =
                    std::static_pointer_cast<arrow::LargeStringScalar>(scalar)->value;
            } else {
                ReportError("JSONPathExtractor: selected column must be string");
                return false;
            }

            nlohmann::json document;
            try {
                document = nlohmann::json::parse(json_text);
            } catch (const std::exception& e) {
                ReportError("JSONPathExtractor: invalid JSON at row " +
                            std::to_string(row) + ": " + e.what());
                return false;
            }

            nlohmann::json value;
            if (!ExtractJsonPathValue(document, path_segments, value) ||
                value.is_null()) {
                auto status = result_builder.AppendNull();
                if (!status.ok()) {
                    ReportError("JSONPathExtractor: Failed to append null result");
                    return false;
                }
                continue;
            }

            auto status = result_builder.Append(JsonValueToDatasetString(value));
            if (!status.ok()) {
                ReportError("JSONPathExtractor: Failed to append extracted value");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> result_array;
        auto finish_status = result_builder.Finish(&result_array);
        if (!finish_status.ok()) {
            ReportError("JSONPathExtractor: Failed to build result array");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("value", arrow::utf8())}),
            {result_array});
        const std::string output_dataset_name =
            "ds_jsonpath_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("JSONPathExtractor error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRegexTester(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RegexTester: No input dataset");
        return false;
    }

    const std::string pattern = ParameterOrDefault(node.parameters, "pattern", ".*");
    if (pattern.empty()) {
        ReportError("RegexTester: pattern is required");
        return false;
    }

    std::regex::flag_type regex_options;
    std::string flags_error;
    if (!BuildRegexOptions(ParameterOrDefault(node.parameters, "flags", ""),
                           regex_options, flags_error)) {
        ReportError(flags_error);
        return false;
    }

    try {
        const std::regex regex_pattern(pattern, regex_options);
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RegexTester: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("RegexTester: Input table is null");
            return false;
        }

        std::string text_column =
            ParameterOrDefault(node.parameters, "text_column", "");
        if (text_column.empty()) {
            text_column = ParameterOrDefault(node.parameters, "column", "");
        }
        int column_index = text_column.empty()
                               ? -1
                               : input_table->schema()->GetFieldIndex(text_column);
        if (column_index < 0 && !text_column.empty()) {
            ReportError("RegexTester: text column '" + text_column + "' not found");
            return false;
        }
        if (column_index < 0) {
            for (int i = 0; i < input_table->num_columns(); ++i) {
                const auto field = input_table->schema()->field(i);
                if (field && IsStringArrowType(field->type())) {
                    column_index = i;
                    text_column = field->name();
                    break;
                }
            }
        }
        if (column_index < 0) {
            ReportError("RegexTester: input table has no string column");
            return false;
        }

        const auto field = input_table->schema()->field(column_index);
        if (!field || !IsStringArrowType(field->type())) {
            ReportError("RegexTester: selected column must be string");
            return false;
        }

        arrow::StringBuilder text_builder;
        arrow::BooleanBuilder matched_builder;
        arrow::StringBuilder match_builder;
        arrow::StringBuilder groups_builder;
        const auto column = input_table->column(column_index);
        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto scalar_result = column->GetScalar(row);
            if (!scalar_result.ok()) {
                ReportError("RegexTester: Failed to read text scalar");
                return false;
            }
            const auto scalar = *scalar_result;
            if (!scalar || !scalar->is_valid) {
                if (!text_builder.AppendNull().ok() ||
                    !matched_builder.Append(false).ok() ||
                    !match_builder.AppendNull().ok() ||
                    !groups_builder.Append("[]").ok()) {
                    ReportError("RegexTester: Failed to append null result");
                    return false;
                }
                continue;
            }

            std::string text;
            if (scalar->type->id() == arrow::Type::STRING) {
                text = std::static_pointer_cast<arrow::StringScalar>(scalar)->value;
            } else if (scalar->type->id() == arrow::Type::LARGE_STRING) {
                text = std::static_pointer_cast<arrow::LargeStringScalar>(scalar)->value;
            } else {
                ReportError("RegexTester: selected column must be string");
                return false;
            }

            std::smatch match;
            const bool matched = std::regex_search(text, match, regex_pattern);
            nlohmann::json groups = nlohmann::json::array();
            if (matched) {
                for (size_t i = 1; i < match.size(); ++i) {
                    groups.push_back(match[i].matched ? match[i].str() : "");
                }
            }

            if (!text_builder.Append(text).ok() ||
                !matched_builder.Append(matched).ok()) {
                ReportError("RegexTester: Failed to append match result");
                return false;
            }
            if (matched) {
                if (!match_builder.Append(match.str()).ok()) {
                    ReportError("RegexTester: Failed to append match text");
                    return false;
                }
            } else if (!match_builder.AppendNull().ok()) {
                ReportError("RegexTester: Failed to append missing match");
                return false;
            }
            if (!groups_builder.Append(groups.dump()).ok()) {
                ReportError("RegexTester: Failed to append capture groups");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> text_array;
        std::shared_ptr<arrow::Array> matched_array;
        std::shared_ptr<arrow::Array> match_array;
        std::shared_ptr<arrow::Array> groups_array;
        if (!text_builder.Finish(&text_array).ok() ||
            !matched_builder.Finish(&matched_array).ok() ||
            !match_builder.Finish(&match_array).ok() ||
            !groups_builder.Finish(&groups_array).ok()) {
            ReportError("RegexTester: Failed to build output arrays");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("text", arrow::utf8()),
                           arrow::field("matched", arrow::boolean()),
                           arrow::field("match", arrow::utf8()),
                           arrow::field("groups", arrow::utf8())}),
            {text_array, matched_array, match_array, groups_array});
        const std::string output_dataset_name =
            "ds_regextester_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::regex_error& e) {
        ReportError("RegexTester: invalid regex pattern: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        ReportError("RegexTester error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteDataProfiler(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("DataProfiler: No input dataset");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("DataProfiler: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("DataProfiler: Input table is null");
            return false;
        }

        arrow::StringBuilder column_builder;
        arrow::StringBuilder type_builder;
        arrow::BooleanBuilder nullable_builder;
        arrow::Int64Builder row_count_builder;
        arrow::Int64Builder null_count_builder;
        arrow::Int64Builder non_null_count_builder;
        const int64_t row_count = input_table->num_rows();

        for (int i = 0; i < input_table->num_columns(); ++i) {
            const auto field = input_table->schema()->field(i);
            const auto column = input_table->column(i);
            const int64_t null_count = column ? column->null_count() : row_count;
            const std::string type_name =
                field && field->type() ? field->type()->ToString() : "unknown";

            if (!column_builder.Append(field ? field->name() : "").ok() ||
                !type_builder.Append(type_name).ok() ||
                !nullable_builder.Append(field ? field->nullable() : true).ok() ||
                !row_count_builder.Append(row_count).ok() ||
                !null_count_builder.Append(null_count).ok() ||
                !non_null_count_builder.Append(row_count - null_count).ok()) {
                ReportError("DataProfiler: Failed to append profile row");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> column_array;
        std::shared_ptr<arrow::Array> type_array;
        std::shared_ptr<arrow::Array> nullable_array;
        std::shared_ptr<arrow::Array> row_count_array;
        std::shared_ptr<arrow::Array> null_count_array;
        std::shared_ptr<arrow::Array> non_null_count_array;
        if (!column_builder.Finish(&column_array).ok() ||
            !type_builder.Finish(&type_array).ok() ||
            !nullable_builder.Finish(&nullable_array).ok() ||
            !row_count_builder.Finish(&row_count_array).ok() ||
            !null_count_builder.Finish(&null_count_array).ok() ||
            !non_null_count_builder.Finish(&non_null_count_array).ok()) {
            ReportError("DataProfiler: Failed to build profile table");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("column", arrow::utf8()),
                           arrow::field("type", arrow::utf8()),
                           arrow::field("nullable", arrow::boolean()),
                           arrow::field("row_count", arrow::int64()),
                           arrow::field("null_count", arrow::int64()),
                           arrow::field("non_null_count", arrow::int64())}),
            {column_array, type_array, nullable_array, row_count_array,
             null_count_array, non_null_count_array});
        const std::string output_dataset_name =
            "ds_dataprofiler_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("DataProfiler error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRegressionMetrics(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RegressionMetricsNode: No input dataset");
        return false;
    }

    const auto get_parameter = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            auto it = node.parameters.find(name);
            if (it != node.parameters.end() && !TrimString(it->second).empty()) {
                return TrimString(it->second);
            }
        }
        return std::string{};
    };

    const std::string actual_col = get_parameter(
        {"actual_col", "y_true_col", "truth_col", "target_col", "ground_truth_col"});
    const std::string predicted_col = get_parameter(
        {"predicted_col", "y_pred_col", "prediction_col"});
    if (actual_col.empty()) {
        ReportError("RegressionMetricsNode: actual_col is required");
        return false;
    }
    if (predicted_col.empty()) {
        ReportError("RegressionMetricsNode: predicted_col is required");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RegressionMetricsNode: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("RegressionMetricsNode: Input table is null");
            return false;
        }

        std::string schema_error;
        if (!RequireColumnKind(input_table, "RegressionMetricsNode", actual_col,
                               "numeric", IsNumericArrowType, schema_error) ||
            !RequireColumnKind(input_table, "RegressionMetricsNode", predicted_col,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }

        const int actual_index = input_table->schema()->GetFieldIndex(actual_col);
        const int predicted_index = input_table->schema()->GetFieldIndex(predicted_col);
        auto actual_column = input_table->column(actual_index);
        auto predicted_column = input_table->column(predicted_index);
        if (!actual_column || !predicted_column) {
            ReportError("RegressionMetricsNode: metric columns are missing");
            return false;
        }

        const auto scalar_to_double = [](const std::shared_ptr<arrow::Scalar>& scalar,
                                         double& value) {
            if (!scalar || !scalar->is_valid || !scalar->type) {
                return false;
            }
            switch (scalar->type->id()) {
                case arrow::Type::INT8:
                    value = std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT16:
                    value = std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT32:
                    value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::UINT8:
                    value = std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT16:
                    value = std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT32:
                    value = std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::FLOAT:
                    value = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
                    return true;
                case arrow::Type::DOUBLE:
                    value = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
                    return true;
                default:
                    return false;
            }
        };

        int64_t count = 0;
        double sum_abs_error = 0.0;
        double sum_squared_error = 0.0;
        double sum_actual = 0.0;
        double sum_actual_squared = 0.0;
        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto actual_scalar_result = actual_column->GetScalar(row);
            auto predicted_scalar_result = predicted_column->GetScalar(row);
            if (!actual_scalar_result.ok() || !predicted_scalar_result.ok()) {
                ReportError("RegressionMetricsNode: Failed to read metric column value");
                return false;
            }

            double actual = 0.0;
            double predicted = 0.0;
            if (!scalar_to_double(*actual_scalar_result, actual) ||
                !scalar_to_double(*predicted_scalar_result, predicted)) {
                continue;
            }

            const double error = actual - predicted;
            sum_abs_error += std::fabs(error);
            sum_squared_error += error * error;
            sum_actual += actual;
            sum_actual_squared += actual * actual;
            ++count;
        }

        if (count == 0) {
            ReportError("RegressionMetricsNode: no non-null actual/predicted pairs");
            return false;
        }

        const double mse = sum_squared_error / static_cast<double>(count);
        const double rmse = std::sqrt(mse);
        const double mae = sum_abs_error / static_cast<double>(count);
        const double actual_mean = sum_actual / static_cast<double>(count);
        const double ss_total =
            sum_actual_squared - static_cast<double>(count) * actual_mean * actual_mean;
        const double r2 =
            (std::fabs(ss_total) <= 1e-12)
                ? (std::fabs(sum_squared_error) <= 1e-12 ? 1.0 : 0.0)
                : (1.0 - (sum_squared_error / ss_total));

        std::vector<std::string> metrics;
        auto metrics_it = node.parameters.find("metrics");
        if (metrics_it != node.parameters.end() &&
            !TrimString(metrics_it->second).empty()) {
            ParseCommaList(metrics_it->second, metrics);
        }
        if (metrics.empty()) {
            metrics = {"mse", "rmse", "mae", "r2"};
        }

        arrow::StringBuilder metric_builder;
        arrow::DoubleBuilder value_builder;
        for (std::string metric : metrics) {
            metric = ToLowerAscii(TrimString(metric));
            double value = 0.0;
            if (metric == "mse") {
                value = mse;
            } else if (metric == "rmse") {
                value = rmse;
            } else if (metric == "mae") {
                value = mae;
            } else if (metric == "r2" || metric == "r_squared") {
                metric = "r2";
                value = r2;
            } else if (metric == "count" || metric == "n") {
                metric = "count";
                value = static_cast<double>(count);
            } else {
                ReportError("RegressionMetricsNode: unsupported metric '" + metric + "'");
                return false;
            }

            if (!metric_builder.Append(metric).ok() ||
                !value_builder.Append(value).ok()) {
                ReportError("RegressionMetricsNode: Failed to append metric row");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> metric_array;
        std::shared_ptr<arrow::Array> value_array;
        if (!metric_builder.Finish(&metric_array).ok() ||
            !value_builder.Finish(&value_array).ok()) {
            ReportError("RegressionMetricsNode: Failed to build metrics table");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("metric", arrow::utf8()),
                           arrow::field("value", arrow::float64())}),
            {metric_array, value_array});
        const std::string output_dataset_name =
            "ds_regression_metrics_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RegressionMetricsNode error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteConfusionMatrix(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ConfusionMatrixNode: No input dataset");
        return false;
    }

    const auto get_parameter = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            auto it = node.parameters.find(name);
            if (it != node.parameters.end() && !TrimString(it->second).empty()) {
                return TrimString(it->second);
            }
        }
        return std::string{};
    };

    const std::string actual_col = get_parameter(
        {"actual_col", "y_true_col", "truth_col", "target_col", "label_col"});
    const std::string predicted_col = get_parameter(
        {"predicted_col", "y_pred_col", "prediction_col"});
    if (actual_col.empty()) {
        ReportError("ConfusionMatrixNode: actual_col is required");
        return false;
    }
    if (predicted_col.empty()) {
        ReportError("ConfusionMatrixNode: predicted_col is required");
        return false;
    }

    std::string normalize = "none";
    auto normalize_it = node.parameters.find("normalize");
    if (normalize_it != node.parameters.end() &&
        !TrimString(normalize_it->second).empty()) {
        normalize = ToLowerAscii(TrimString(normalize_it->second));
    }
    if (normalize == "false") {
        normalize = "none";
    }
    if (normalize != "none" && normalize != "true" &&
        normalize != "pred" && normalize != "all") {
        ReportError("ConfusionMatrixNode: normalize must be one of none, true, pred, all");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("ConfusionMatrixNode: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("ConfusionMatrixNode: Input table is null");
            return false;
        }

        const int actual_index = input_table->schema()->GetFieldIndex(actual_col);
        if (actual_index < 0) {
            ReportError("ConfusionMatrixNode: column '" + actual_col + "' not found");
            return false;
        }
        const int predicted_index = input_table->schema()->GetFieldIndex(predicted_col);
        if (predicted_index < 0) {
            ReportError("ConfusionMatrixNode: column '" + predicted_col + "' not found");
            return false;
        }

        auto actual_column = input_table->column(actual_index);
        auto predicted_column = input_table->column(predicted_index);
        if (!actual_column || !predicted_column) {
            ReportError("ConfusionMatrixNode: label columns are missing");
            return false;
        }

        std::map<std::pair<std::string, std::string>, int64_t> counts;
        std::map<std::string, int64_t> actual_totals;
        std::map<std::string, int64_t> predicted_totals;
        int64_t valid_count = 0;

        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto actual_scalar_result = actual_column->GetScalar(row);
            auto predicted_scalar_result = predicted_column->GetScalar(row);
            if (!actual_scalar_result.ok() || !predicted_scalar_result.ok()) {
                ReportError("ConfusionMatrixNode: Failed to read label column value");
                return false;
            }

            auto actual_scalar = *actual_scalar_result;
            auto predicted_scalar = *predicted_scalar_result;
            if (!actual_scalar || !predicted_scalar ||
                !actual_scalar->is_valid || !predicted_scalar->is_valid) {
                continue;
            }

            const std::string actual_label = actual_scalar->ToString();
            const std::string predicted_label = predicted_scalar->ToString();
            ++counts[{actual_label, predicted_label}];
            ++actual_totals[actual_label];
            ++predicted_totals[predicted_label];
            ++valid_count;
        }

        if (valid_count == 0) {
            ReportError("ConfusionMatrixNode: no non-null actual/predicted pairs");
            return false;
        }

        arrow::StringBuilder actual_builder;
        arrow::StringBuilder predicted_builder;
        arrow::Int64Builder count_builder;
        arrow::DoubleBuilder value_builder;

        for (const auto& entry : counts) {
            const std::string& actual_label = entry.first.first;
            const std::string& predicted_label = entry.first.second;
            const int64_t count = entry.second;
            double value = static_cast<double>(count);
            if (normalize == "true") {
                value = static_cast<double>(count) /
                        static_cast<double>(actual_totals[actual_label]);
            } else if (normalize == "pred") {
                value = static_cast<double>(count) /
                        static_cast<double>(predicted_totals[predicted_label]);
            } else if (normalize == "all") {
                value = static_cast<double>(count) /
                        static_cast<double>(valid_count);
            }

            if (!actual_builder.Append(actual_label).ok() ||
                !predicted_builder.Append(predicted_label).ok() ||
                !count_builder.Append(count).ok() ||
                !value_builder.Append(value).ok()) {
                ReportError("ConfusionMatrixNode: Failed to append matrix row");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> actual_array;
        std::shared_ptr<arrow::Array> predicted_array;
        std::shared_ptr<arrow::Array> count_array;
        std::shared_ptr<arrow::Array> value_array;
        if (!actual_builder.Finish(&actual_array).ok() ||
            !predicted_builder.Finish(&predicted_array).ok() ||
            !count_builder.Finish(&count_array).ok() ||
            !value_builder.Finish(&value_array).ok()) {
            ReportError("ConfusionMatrixNode: Failed to build matrix table");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("actual_label", arrow::utf8()),
                           arrow::field("predicted_label", arrow::utf8()),
                           arrow::field("count", arrow::int64()),
                           arrow::field("value", arrow::float64())}),
            {actual_array, predicted_array, count_array, value_array});
        const std::string output_dataset_name =
            "ds_confusion_matrix_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("ConfusionMatrixNode error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteROCCurve(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ROCCurveNode: No input dataset");
        return false;
    }

    const auto get_parameter = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            auto it = node.parameters.find(name);
            if (it != node.parameters.end() && !TrimString(it->second).empty()) {
                return TrimString(it->second);
            }
        }
        return std::string{};
    };

    const std::string actual_col = get_parameter(
        {"actual_col", "y_true_col", "truth_col", "target_col", "label_col"});
    const std::string score_col = get_parameter(
        {"score_col", "y_score_col", "probability_col", "prediction_score_col"});
    const std::string positive_label =
        get_parameter({"positive_label", "positive_class"}).empty()
            ? "1"
            : get_parameter({"positive_label", "positive_class"});
    if (actual_col.empty()) {
        ReportError("ROCCurveNode: actual_col is required");
        return false;
    }
    if (score_col.empty()) {
        ReportError("ROCCurveNode: score_col is required");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("ROCCurveNode: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("ROCCurveNode: Input table is null");
            return false;
        }

        const int actual_index = input_table->schema()->GetFieldIndex(actual_col);
        if (actual_index < 0) {
            ReportError("ROCCurveNode: column '" + actual_col + "' not found");
            return false;
        }
        std::string schema_error;
        if (!RequireColumnKind(input_table, "ROCCurveNode", score_col,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }

        auto actual_column = input_table->column(actual_index);
        auto score_column = input_table->column(
            input_table->schema()->GetFieldIndex(score_col));
        if (!actual_column || !score_column) {
            ReportError("ROCCurveNode: ROC columns are missing");
            return false;
        }

        const auto scalar_to_double = [](const std::shared_ptr<arrow::Scalar>& scalar,
                                         double& value) {
            if (!scalar || !scalar->is_valid || !scalar->type) {
                return false;
            }
            switch (scalar->type->id()) {
                case arrow::Type::INT8:
                    value = std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT16:
                    value = std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT32:
                    value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::UINT8:
                    value = std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT16:
                    value = std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT32:
                    value = std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::FLOAT:
                    value = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
                    return true;
                case arrow::Type::DOUBLE:
                    value = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
                    return true;
                default:
                    return false;
            }
        };

        struct RocPointInput {
            std::string label;
            double score = 0.0;
            bool positive = false;
        };
        std::vector<RocPointInput> samples;
        int64_t positive_count = 0;
        int64_t negative_count = 0;
        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto actual_scalar_result = actual_column->GetScalar(row);
            auto score_scalar_result = score_column->GetScalar(row);
            if (!actual_scalar_result.ok() || !score_scalar_result.ok()) {
                ReportError("ROCCurveNode: Failed to read ROC column value");
                return false;
            }

            auto actual_scalar = *actual_scalar_result;
            if (!actual_scalar || !actual_scalar->is_valid) {
                continue;
            }
            double score = 0.0;
            if (!scalar_to_double(*score_scalar_result, score)) {
                continue;
            }

            RocPointInput sample;
            sample.label = actual_scalar->ToString();
            sample.score = score;
            sample.positive = sample.label == positive_label;
            if (sample.positive) {
                ++positive_count;
            } else {
                ++negative_count;
            }
            samples.push_back(sample);
        }

        if (samples.empty()) {
            ReportError("ROCCurveNode: no non-null label/score pairs");
            return false;
        }
        if (positive_count == 0 || negative_count == 0) {
            ReportError("ROCCurveNode: ROC requires at least one positive and one negative sample");
            return false;
        }

        std::sort(samples.begin(), samples.end(),
                  [](const RocPointInput& a, const RocPointInput& b) {
                      return a.score > b.score;
                  });

        struct RocPoint {
            double threshold = 0.0;
            double fpr = 0.0;
            double tpr = 0.0;
        };
        std::vector<RocPoint> points;
        int64_t tp = 0;
        int64_t fp = 0;
        size_t index = 0;
        while (index < samples.size()) {
            const double threshold = samples[index].score;
            while (index < samples.size() && samples[index].score == threshold) {
                if (samples[index].positive) {
                    ++tp;
                } else {
                    ++fp;
                }
                ++index;
            }
            points.push_back(
                {threshold,
                 static_cast<double>(fp) / static_cast<double>(negative_count),
                 static_cast<double>(tp) / static_cast<double>(positive_count)});
        }

        double auc = 0.0;
        double prev_fpr = 0.0;
        double prev_tpr = 0.0;
        for (const auto& point : points) {
            auc += (point.fpr - prev_fpr) * (point.tpr + prev_tpr) / 2.0;
            prev_fpr = point.fpr;
            prev_tpr = point.tpr;
        }
        if (prev_fpr < 1.0 || prev_tpr < 1.0) {
            auc += (1.0 - prev_fpr) * (1.0 + prev_tpr) / 2.0;
        }

        arrow::DoubleBuilder threshold_builder;
        arrow::DoubleBuilder fpr_builder;
        arrow::DoubleBuilder tpr_builder;
        arrow::DoubleBuilder auc_builder;
        for (const auto& point : points) {
            if (!threshold_builder.Append(point.threshold).ok() ||
                !fpr_builder.Append(point.fpr).ok() ||
                !tpr_builder.Append(point.tpr).ok() ||
                !auc_builder.Append(auc).ok()) {
                ReportError("ROCCurveNode: Failed to append ROC point");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> threshold_array;
        std::shared_ptr<arrow::Array> fpr_array;
        std::shared_ptr<arrow::Array> tpr_array;
        std::shared_ptr<arrow::Array> auc_array;
        if (!threshold_builder.Finish(&threshold_array).ok() ||
            !fpr_builder.Finish(&fpr_array).ok() ||
            !tpr_builder.Finish(&tpr_array).ok() ||
            !auc_builder.Finish(&auc_array).ok()) {
            ReportError("ROCCurveNode: Failed to build ROC table");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("threshold", arrow::float64()),
                           arrow::field("fpr", arrow::float64()),
                           arrow::field("tpr", arrow::float64()),
                           arrow::field("auc", arrow::float64())}),
            {threshold_array, fpr_array, tpr_array, auc_array});
        const std::string output_dataset_name =
            "ds_roc_curve_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("ROCCurveNode error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecutePRCurve(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("PRCurveNode: No input dataset");
        return false;
    }

    const auto get_parameter = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            auto it = node.parameters.find(name);
            if (it != node.parameters.end() && !TrimString(it->second).empty()) {
                return TrimString(it->second);
            }
        }
        return std::string{};
    };

    const std::string actual_col = get_parameter(
        {"actual_col", "y_true_col", "truth_col", "target_col", "label_col"});
    const std::string score_col = get_parameter(
        {"score_col", "y_score_col", "probability_col", "prediction_score_col"});
    const std::string positive_label =
        get_parameter({"positive_label", "positive_class"}).empty()
            ? "1"
            : get_parameter({"positive_label", "positive_class"});
    if (actual_col.empty()) {
        ReportError("PRCurveNode: actual_col is required");
        return false;
    }
    if (score_col.empty()) {
        ReportError("PRCurveNode: score_col is required");
        return false;
    }

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("PRCurveNode: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("PRCurveNode: Input table is null");
            return false;
        }

        const int actual_index = input_table->schema()->GetFieldIndex(actual_col);
        if (actual_index < 0) {
            ReportError("PRCurveNode: column '" + actual_col + "' not found");
            return false;
        }
        std::string schema_error;
        if (!RequireColumnKind(input_table, "PRCurveNode", score_col,
                               "numeric", IsNumericArrowType, schema_error)) {
            ReportError(schema_error);
            return false;
        }

        auto actual_column = input_table->column(actual_index);
        auto score_column = input_table->column(
            input_table->schema()->GetFieldIndex(score_col));
        if (!actual_column || !score_column) {
            ReportError("PRCurveNode: PR columns are missing");
            return false;
        }

        const auto scalar_to_double = [](const std::shared_ptr<arrow::Scalar>& scalar,
                                         double& value) {
            if (!scalar || !scalar->is_valid || !scalar->type) {
                return false;
            }
            switch (scalar->type->id()) {
                case arrow::Type::INT8:
                    value = std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT16:
                    value = std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT32:
                    value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::INT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::UINT8:
                    value = std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT16:
                    value = std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT32:
                    value = std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value;
                    return true;
                case arrow::Type::UINT64:
                    value = static_cast<double>(
                        std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
                    return true;
                case arrow::Type::FLOAT:
                    value = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
                    return true;
                case arrow::Type::DOUBLE:
                    value = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
                    return true;
                default:
                    return false;
            }
        };

        struct PRPointInput {
            std::string label;
            double score = 0.0;
            bool positive = false;
        };
        std::vector<PRPointInput> samples;
        int64_t positive_count = 0;
        int64_t negative_count = 0;
        for (int64_t row = 0; row < input_table->num_rows(); ++row) {
            auto actual_scalar_result = actual_column->GetScalar(row);
            auto score_scalar_result = score_column->GetScalar(row);
            if (!actual_scalar_result.ok() || !score_scalar_result.ok()) {
                ReportError("PRCurveNode: Failed to read PR column value");
                return false;
            }

            auto actual_scalar = *actual_scalar_result;
            if (!actual_scalar || !actual_scalar->is_valid) {
                continue;
            }
            double score = 0.0;
            if (!scalar_to_double(*score_scalar_result, score)) {
                continue;
            }

            PRPointInput sample;
            sample.label = actual_scalar->ToString();
            sample.score = score;
            sample.positive = sample.label == positive_label;
            if (sample.positive) {
                ++positive_count;
            } else {
                ++negative_count;
            }
            samples.push_back(sample);
        }

        if (samples.empty()) {
            ReportError("PRCurveNode: no non-null label/score pairs");
            return false;
        }
        if (positive_count == 0 || negative_count == 0) {
            ReportError("PRCurveNode: precision-recall requires at least one positive and one negative sample");
            return false;
        }

        std::sort(samples.begin(), samples.end(),
                  [](const PRPointInput& a, const PRPointInput& b) {
                      return a.score > b.score;
                  });

        struct PRPoint {
            double threshold = 0.0;
            double precision = 0.0;
            double recall = 0.0;
        };
        std::vector<PRPoint> points;
        int64_t tp = 0;
        int64_t fp = 0;
        size_t index = 0;
        while (index < samples.size()) {
            const double threshold = samples[index].score;
            while (index < samples.size() && samples[index].score == threshold) {
                if (samples[index].positive) {
                    ++tp;
                } else {
                    ++fp;
                }
                ++index;
            }
            const double predicted_positive = static_cast<double>(tp + fp);
            points.push_back(
                {threshold,
                 predicted_positive > 0.0
                     ? static_cast<double>(tp) / predicted_positive
                     : 1.0,
                 static_cast<double>(tp) / static_cast<double>(positive_count)});
        }

        double average_precision = 0.0;
        double previous_recall = 0.0;
        for (const auto& point : points) {
            if (point.recall > previous_recall) {
                average_precision +=
                    (point.recall - previous_recall) * point.precision;
                previous_recall = point.recall;
            }
        }

        arrow::DoubleBuilder threshold_builder;
        arrow::DoubleBuilder precision_builder;
        arrow::DoubleBuilder recall_builder;
        arrow::DoubleBuilder ap_builder;
        for (const auto& point : points) {
            if (!threshold_builder.Append(point.threshold).ok() ||
                !precision_builder.Append(point.precision).ok() ||
                !recall_builder.Append(point.recall).ok() ||
                !ap_builder.Append(average_precision).ok()) {
                ReportError("PRCurveNode: Failed to append PR point");
                return false;
            }
        }

        std::shared_ptr<arrow::Array> threshold_array;
        std::shared_ptr<arrow::Array> precision_array;
        std::shared_ptr<arrow::Array> recall_array;
        std::shared_ptr<arrow::Array> ap_array;
        if (!threshold_builder.Finish(&threshold_array).ok() ||
            !precision_builder.Finish(&precision_array).ok() ||
            !recall_builder.Finish(&recall_array).ok() ||
            !ap_builder.Finish(&ap_array).ok()) {
            ReportError("PRCurveNode: Failed to build PR table");
            return false;
        }

        auto output_table = arrow::Table::Make(
            arrow::schema({arrow::field("threshold", arrow::float64()),
                           arrow::field("precision", arrow::float64()),
                           arrow::field("recall", arrow::float64()),
                           arrow::field("average_precision", arrow::float64())}),
            {threshold_array, precision_array, recall_array, ap_array});
        const std::string output_dataset_name =
            "ds_pr_curve_" + std::to_string(node.id);
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("PRCurveNode error: " + std::string(e.what()));
        return false;
    }
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

bool PipelineExecutor::ExecuteCellExtractor(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("CellExtractor: No input dataset");
        return false;
    }

    const auto column_it = node.parameters.find("column");
    const std::string column =
        (column_it != node.parameters.end()) ? column_it->second : "";
    if (column.empty()) {
        ReportError("CellExtractor: column is required");
        return false;
    }

    const auto row_it = node.parameters.find("row");
    const int row =
        (row_it != node.parameters.end() && !row_it->second.empty())
            ? std::stoi(row_it->second)
            : 0;

    std::string output_dataset_name = "ds_cell_" + std::to_string(node.id);
    spdlog::info("[Data Studio] CellExtractor row={} column='{}' from '{}'",
                 row, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("CellExtractor: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("CellExtractor: Input table is null");
            return false;
        }
        if (input_table->schema()->GetFieldIndex(column) < 0) {
            ReportError("CellExtractor: column '" + column + "' not found");
            return false;
        }
        if (row >= input_table->num_rows()) {
            ReportError("CellExtractor: row index out of range");
            return false;
        }

        const std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("CellExtractor: Failed to register table");
            return false;
        }

        const std::string sql =
            "SELECT " + QuoteSqlIdentifier(column) + " AS value FROM " +
            temp_table + " LIMIT 1 OFFSET " + std::to_string(row);
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);
        if (!result_table || result_table->num_rows() != 1) {
            ReportError("CellExtractor: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("CellExtractor error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteCellUpdater(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("CellUpdater: No input dataset");
        return false;
    }

    const auto column_it = node.parameters.find("column");
    const std::string column =
        (column_it != node.parameters.end()) ? column_it->second : "";
    if (column.empty()) {
        ReportError("CellUpdater: column is required");
        return false;
    }

    const auto value_it = node.parameters.find("value");
    const std::string value =
        (value_it != node.parameters.end()) ? value_it->second : "";
    if (value.empty()) {
        ReportError("CellUpdater: value is required");
        return false;
    }

    const auto row_it = node.parameters.find("row");
    const int row =
        (row_it != node.parameters.end() && !row_it->second.empty())
            ? std::stoi(row_it->second)
            : 0;
    if (row < 0) {
        ReportError("CellUpdater: row must be >= 0");
        return false;
    }

    std::string output_dataset_name = "ds_cell_update_" + std::to_string(node.id);
    spdlog::info("[Data Studio] CellUpdater row={} column='{}' from '{}'",
                 row, column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("CellUpdater: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("CellUpdater: Input table is null");
            return false;
        }
        const int column_index = input_table->schema()->GetFieldIndex(column);
        if (column_index < 0) {
            ReportError("CellUpdater: column '" + column + "' not found");
            return false;
        }
        if (row >= input_table->num_rows()) {
            ReportError("CellUpdater: row index out of range");
            return false;
        }

        std::string update_value_expression;
        std::string value_error;
        if (!BuildFillMissingConstantExpression(
                input_table->schema()->field(column_index), value,
                update_value_expression, value_error)) {
            ReportError("CellUpdater: unsupported value for column '" + column + "'");
            return false;
        }

        const std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("CellUpdater: Failed to register table");
            return false;
        }

        std::vector<std::string> select_exprs;
        select_exprs.reserve(input_table->num_columns());
        for (int i = 0; i < input_table->num_columns(); ++i) {
            const auto field = input_table->schema()->field(i);
            const std::string quoted_column = QuoteSqlIdentifier(field->name());
            if (field->name() == column) {
                select_exprs.push_back(
                    "CASE WHEN row_number() OVER () - 1 = " +
                    std::to_string(row) + " THEN " + update_value_expression +
                    " ELSE " + quoted_column + " END AS " + quoted_column);
            } else {
                select_exprs.push_back(quoted_column);
            }
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
        duckdb_->UnregisterTable(temp_table);
        if (!result_table) {
            ReportError("CellUpdater: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("CellUpdater error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRowAppender(const Node& node, ExecutionContext& ctx) {
    const auto input_dataset_names = GetInputDatasetNames(node, ctx, 2);
    if (input_dataset_names.empty()) {
        return false;
    }

    const std::string& top_dataset_name = input_dataset_names[0];
    const std::string& bottom_dataset_name = input_dataset_names[1];
    std::string output_dataset_name = "ds_row_append_" + std::to_string(node.id);

    spdlog::info("[Data Studio] RowAppender appending '{}' and '{}'",
                 top_dataset_name, bottom_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto top_dataset = registry.GetArrowDataset(top_dataset_name);
        auto bottom_dataset = registry.GetArrowDataset(bottom_dataset_name);
        if (!top_dataset || !bottom_dataset) {
            ReportError("RowAppender: Input datasets not found");
            return false;
        }

        auto top_table = top_dataset->GetArrowTable();
        auto bottom_table = bottom_dataset->GetArrowTable();
        if (!top_table || !bottom_table || !top_table->schema() ||
            !bottom_table->schema()) {
            ReportError("RowAppender: Input table is null");
            return false;
        }
        if (top_table->num_columns() != bottom_table->num_columns()) {
            ReportError("RowAppender: input schemas must have the same column count");
            return false;
        }
        if (top_table->num_columns() == 0) {
            ReportError("RowAppender: input tables must have at least one column");
            return false;
        }

        std::vector<std::string> output_columns;
        output_columns.reserve(top_table->num_columns());
        for (int i = 0; i < top_table->num_columns(); ++i) {
            const auto top_field = top_table->schema()->field(i);
            const auto bottom_field = bottom_table->schema()->field(i);
            if (top_field->name() != bottom_field->name()) {
                ReportError("RowAppender: input column names differ at index " +
                            std::to_string(i));
                return false;
            }
            if (!top_field->type()->Equals(*bottom_field->type())) {
                ReportError("RowAppender: input column type differs for '" +
                            top_field->name() + "'");
                return false;
            }
            output_columns.push_back(top_field->name());
        }

        const std::string top_table_name = "temp_top_" + std::to_string(node.id);
        const std::string bottom_table_name =
            "temp_bottom_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(top_table_name, top_table)) {
            ReportError("RowAppender: Failed to register top table");
            return false;
        }
        if (!duckdb_->RegisterTable(bottom_table_name, bottom_table)) {
            duckdb_->UnregisterTable(top_table_name);
            ReportError("RowAppender: Failed to register bottom table");
            return false;
        }

        const std::string selected_columns = JoinQuotedColumns(output_columns);
        const std::string sql = "SELECT " + selected_columns + " FROM " +
                                top_table_name + " UNION ALL SELECT " +
                                selected_columns + " FROM " +
                                bottom_table_name;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(top_table_name);
        duckdb_->UnregisterTable(bottom_table_name);
        if (!result_table) {
            ReportError("RowAppender: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RowAppender error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteColumnAppender(const Node& node, ExecutionContext& ctx) {
    const auto input_dataset_names = GetInputDatasetNames(node, ctx, 2);
    if (input_dataset_names.empty()) {
        return false;
    }

    const std::string& left_dataset_name = input_dataset_names[0];
    const std::string& right_dataset_name = input_dataset_names[1];
    const auto suffix_it = node.parameters.find("suffix");
    const std::string suffix =
        (suffix_it != node.parameters.end() && !suffix_it->second.empty())
            ? suffix_it->second
            : "_right";
    std::string output_dataset_name =
        "ds_column_append_" + std::to_string(node.id);

    spdlog::info("[Data Studio] ColumnAppender appending '{}' and '{}'",
                 left_dataset_name, right_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto left_dataset = registry.GetArrowDataset(left_dataset_name);
        auto right_dataset = registry.GetArrowDataset(right_dataset_name);
        if (!left_dataset || !right_dataset) {
            ReportError("ColumnAppender: Input datasets not found");
            return false;
        }

        auto left_table = left_dataset->GetArrowTable();
        auto right_table = right_dataset->GetArrowTable();
        if (!left_table || !right_table || !left_table->schema() ||
            !right_table->schema()) {
            ReportError("ColumnAppender: Input table is null");
            return false;
        }
        if (left_table->num_rows() != right_table->num_rows()) {
            ReportError("ColumnAppender: input row counts must match");
            return false;
        }

        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
        fields.reserve(left_table->num_columns() + right_table->num_columns());
        columns.reserve(left_table->num_columns() + right_table->num_columns());

        std::set<std::string> output_names;
        for (int i = 0; i < left_table->num_columns(); ++i) {
            const auto field = left_table->schema()->field(i);
            if (!output_names.insert(field->name()).second) {
                ReportError("ColumnAppender: duplicate left column name '" +
                            field->name() + "'");
                return false;
            }
            fields.push_back(field);
            columns.push_back(left_table->column(i));
        }

        for (int i = 0; i < right_table->num_columns(); ++i) {
            const auto field = right_table->schema()->field(i);
            std::string output_name = field->name();
            if (output_names.find(output_name) != output_names.end()) {
                output_name += suffix;
            }
            if (!output_names.insert(output_name).second) {
                ReportError("ColumnAppender: duplicate output column name '" +
                            output_name + "'");
                return false;
            }
            fields.push_back(output_name == field->name()
                                 ? field
                                 : field->WithName(output_name));
            columns.push_back(right_table->column(i));
        }

        auto output_table = arrow::Table::Make(
            arrow::schema(fields), columns, left_table->num_rows());
        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("ColumnAppender error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteUnpivot(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("Unpivot: No input dataset");
        return false;
    }

    const auto value_name_it = node.parameters.find("value_name");
    const std::string value_name =
        (value_name_it != node.parameters.end() && !value_name_it->second.empty())
            ? TrimString(value_name_it->second)
            : "value";
    const auto variable_name_it = node.parameters.find("variable_name");
    const std::string variable_name =
        (variable_name_it != node.parameters.end() &&
         !variable_name_it->second.empty())
            ? TrimString(variable_name_it->second)
            : "variable";
    if (value_name.empty() || variable_name.empty()) {
        ReportError("Unpivot: value_name and variable_name are required");
        return false;
    }
    if (value_name == variable_name) {
        ReportError("Unpivot: value_name and variable_name must differ");
        return false;
    }

    const auto id_columns_it = node.parameters.find("id_columns");
    const std::vector<std::string> requested_id_columns =
        id_columns_it != node.parameters.end()
            ? ParseCommaSeparatedNames(id_columns_it->second)
            : std::vector<std::string>{};
    const std::string output_dataset_name =
        "ds_unpivot_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Unpivot '{}' with {} id columns",
                 input_dataset_name, requested_id_columns.size());

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("Unpivot: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        if (!input_table || !input_table->schema()) {
            ReportError("Unpivot: Input table is null");
            return false;
        }

        std::set<std::string> id_column_names;
        std::vector<std::string> id_columns;
        id_columns.reserve(requested_id_columns.size());
        for (const auto& column : requested_id_columns) {
            if (input_table->schema()->GetFieldIndex(column) < 0) {
                ReportError("Unpivot: id column '" + column + "' not found");
                return false;
            }
            if (!id_column_names.insert(column).second) {
                ReportError("Unpivot: duplicate id column '" + column + "'");
                return false;
            }
            if (column == value_name || column == variable_name) {
                ReportError("Unpivot: id column '" + column +
                            "' conflicts with output column names");
                return false;
            }
            id_columns.push_back(column);
        }

        std::vector<std::string> value_columns;
        for (int i = 0; i < input_table->num_columns(); ++i) {
            const std::string field_name = input_table->schema()->field(i)->name();
            if (id_column_names.find(field_name) == id_column_names.end()) {
                value_columns.push_back(field_name);
            }
        }
        if (value_columns.empty()) {
            ReportError("Unpivot: no value columns remain after id_columns");
            return false;
        }

        const std::string temp_table = "temp_unpivot_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("Unpivot: Failed to register table");
            return false;
        }

        std::string sql;
        for (size_t value_index = 0; value_index < value_columns.size();
             ++value_index) {
            if (value_index > 0) {
                sql += " UNION ALL ";
            }
            sql += "SELECT ";
            for (const auto& id_column : id_columns) {
                sql += QuoteSqlIdentifier(id_column) + ", ";
            }
            sql += QuoteSqlStringLiteral(value_columns[value_index]) + " AS " +
                   QuoteSqlIdentifier(variable_name) + ", CAST(" +
                   QuoteSqlIdentifier(value_columns[value_index]) +
                   " AS VARCHAR) AS " + QuoteSqlIdentifier(value_name) +
                   " FROM " + temp_table;
        }

        auto output_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);
        if (!output_table) {
            ReportError("Unpivot: Query failed");
            return false;
        }

        registry.RegisterArrowTable(output_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("Unpivot error: " + std::string(e.what()));
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
