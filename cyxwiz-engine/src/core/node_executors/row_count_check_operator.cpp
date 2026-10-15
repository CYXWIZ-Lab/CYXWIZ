#include "row_count_check_operator.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <spdlog/spdlog.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <unordered_set>

namespace cyxwiz {
namespace {

bool ParseCount(const std::map<std::string, std::string>& params, const char* key,
                std::optional<int64_t>& out, std::string& error) {
    out.reset();
    const auto it = params.find(key);
    if (it == params.end()) return true;
    std::string text = it->second;
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    text = text.substr(start);
    if (text.empty()) return true;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            error = std::string("RowCountCheck: ") + key + " must be a whole number >= 0 (got '" +
                    it->second + "')";
            return false;
        }
    }
    try {
        out = std::stoll(text);
    } catch (const std::exception&) {
        error = std::string("RowCountCheck: ") + key + " is out of range";
        return false;
    }
    return true;
}

std::string Param(const std::map<std::string, std::string>& params, const char* key) {
    const auto it = params.find(key);
    return it == params.end() ? std::string() : it->second;
}

std::string Trim(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    return text.substr(start);
}

std::vector<std::string> CommaList(const std::string& text) {
    std::vector<std::string> items;
    std::string item;
    for (const char c : text + ",") {
        if (c != ',') {
            item += c;
            continue;
        }
        item = Trim(item);
        if (!item.empty()) items.push_back(item);
        item.clear();
    }
    return items;
}

std::string JoinList(const std::vector<std::string>& items) {
    std::string out;
    for (const auto& item : items) out += (out.empty() ? "" : ", ") + item;
    return out;
}

bool ParseNumber(const std::map<std::string, std::string>& params, const char* key, std::optional<double>& out,
                 std::string& error) {
    out.reset();
    const std::string text = Trim(Param(params, key));
    if (text.empty()) return true;
    try {
        size_t used = 0;
        out = std::stod(text, &used);
        if (used != text.size() || !std::isfinite(*out)) throw std::invalid_argument(text);
    } catch (const std::exception&) {
        error = std::string("RowCountCheck: ") + key + " must be a number (got '" + text + "')";
        return false;
    }
    return true;
}

// A column's values as text (the guided checks compare CAST(x AS VARCHAR)).
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> AsText(const std::shared_ptr<arrow::ChunkedArray>& column) {
    if (column->type()->id() == arrow::Type::STRING) return column;
    ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(arrow::Datum(column), arrow::utf8()));
    return cast.chunked_array();
}

std::string Shown(const std::string& value) {
    return value.size() > 60 ? "'" + value.substr(0, 57) + "...'" : "'" + value + "'";
}

}  // namespace

bool RowCountCheckOperator::Configure(const std::map<std::string, std::string>& params,
                                      std::string& error) {
    check_ = Trim(Param(params, "check"));
    if (check_.empty()) check_ = "row_count";
    check_name_ = Param(params, "check_name");
    const std::string policy = Param(params, "on_failure");
    if (!policy.empty() && policy != "stop") {
        error = "RowCountCheck: on_failure '" + policy + "' is not supported yet; checks stop the run";
        return false;
    }
    if (check_ != "row_count") {
        column_ = Trim(Param(params, "column"));
        columns_ = CommaList(Param(params, "columns"));
        allowed_ = CommaList(Param(params, "allowed"));
        if (!ParseNumber(params, "min_value", min_value_, error) || !ParseNumber(params, "max_value", max_value_, error)) return false;
        if (check_ == "unique") {
            if (columns_.empty() && !column_.empty()) columns_.push_back(column_);
            if (columns_.empty()) {
                error = "RowCountCheck: a unique check needs columns";
                return false;
            }
            return true;
        }
        if (check_ != "allowed_values" && check_ != "not_empty" && check_ != "range") {
            error = "RowCountCheck: check '" + check_ + "' is not one of row_count, allowed_values, unique, not_empty, range";
            return false;
        }
        if (column_.empty()) {
            error = "RowCountCheck: a " + check_ + " check needs a column";
            return false;
        }
        if (check_ == "allowed_values" && allowed_.empty()) {
            error = "RowCountCheck: list the allowed values";
            return false;
        }
        if (check_ == "range" && !min_value_ && !max_value_) {
            error = "RowCountCheck: a range check needs min_value and / or max_value";
            return false;
        }
        if (min_value_ && max_value_ && *min_value_ > *max_value_) {
            error = "RowCountCheck: min_value is greater than max_value";
            return false;
        }
        return true;
    }
    count_true_column_ = Param(params, "count_true_column");
    count_column_ = Param(params, "count_column");
    count_value_ = Param(params, "count_value");
    if (!count_true_column_.empty() && !count_column_.empty()) {
        error = "RowCountCheck: use count_true_column or count_column, not both";
        return false;
    }
    if (!count_column_.empty() && count_value_.empty()) {
        error = "RowCountCheck: count_column '" + count_column_ + "' needs a count_value";
        return false;
    }
    if (!ParseCount(params, "expected_rows", expected_rows_, error) ||
        !ParseCount(params, "min_rows", min_rows_, error) ||
        !ParseCount(params, "max_rows", max_rows_, error)) {
        return false;
    }
    if (!expected_rows_ && !min_rows_ && !max_rows_) {
        error = "RowCountCheck: set expected_rows, or min_rows and/or max_rows";
        return false;
    }
    if (min_rows_ && max_rows_ && *min_rows_ > *max_rows_) {
        error = "RowCountCheck: min_rows is greater than max_rows";
        return false;
    }
    return true;
}

std::string RowCountCheckOperator::Label() const {
    if (!check_name_.empty()) return "'" + check_name_ + "'";
    return check_ == "row_count" ? std::string("Row count check") : "The " + check_ + " check";
}

arrow::Result<std::shared_ptr<arrow::Schema>> RowCountCheckOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {
    if (check_ != "row_count") {
        const std::vector<std::string> needed = check_ == "unique" ? columns_ : std::vector<std::string>{column_};
        for (const auto& name : needed) {
            if (!input_schema->GetFieldByName(name)) return arrow::Status::Invalid("RowCountCheck: column '", name, "' not found");
        }
        if (check_ == "range") {
            const auto& type = *input_schema->GetFieldByName(column_)->type();
            if (!arrow::is_integer(type.id()) && !arrow::is_floating(type.id()) && !arrow::is_decimal(type.id())) {
                return arrow::Status::Invalid("RowCountCheck: a range check needs a numeric column; '", column_, "' is ",
                                              type.ToString());
            }
        }
        return input_schema;
    }
    if (!count_true_column_.empty()) {
        const auto field = input_schema->GetFieldByName(count_true_column_);
        if (!field) {
            return arrow::Status::Invalid("RowCountCheck: column '", count_true_column_,
                                          "' not found");
        }
        if (field->type()->id() != arrow::Type::BOOL) {
            return arrow::Status::Invalid("RowCountCheck: column '", count_true_column_,
                                          "' must be boolean, not ", field->type()->ToString());
        }
    }
    if (!count_column_.empty() && !input_schema->GetFieldByName(count_column_)) {
        return arrow::Status::Invalid("RowCountCheck: column '", count_column_, "' not found");
    }
    return input_schema;
}

arrow::Result<std::shared_ptr<arrow::Table>> RowCountCheckOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid("RowCountCheck: no input table");
    ARROW_RETURN_NOT_OK(InferOutputSchema(input->schema()).status());
    return check_ == "row_count" ? ApplyRowCount(input) : ApplyValueCheck(input);
}

arrow::Result<std::shared_ptr<arrow::Table>> RowCountCheckOperator::ApplyValueCheck(
    const std::shared_ptr<arrow::Table>& input) {
    const int64_t rows = input->num_rows();
    int64_t failing = 0;
    std::string first;  // the first failing value, shown in the message
    std::string rule, what;
    if (check_ == "unique") {
        // One key per row from the columns as text (an empty table has no chunks to join).
        std::vector<std::shared_ptr<arrow::StringArray>> keys;
        for (const auto& name : columns_) {
            if (rows == 0) break;
            ARROW_ASSIGN_OR_RAISE(auto text, AsText(input->GetColumnByName(name)));
            ARROW_ASSIGN_OR_RAISE(auto flat, arrow::Concatenate(text->chunks()));
            keys.push_back(std::static_pointer_cast<arrow::StringArray>(flat));
        }
        std::unordered_set<std::string> seen;
        seen.reserve(static_cast<size_t>(rows));
        for (int64_t r = 0; r < rows; ++r) {
            std::string key;
            for (const auto& values : keys) {
                key += values->IsNull(r) ? std::string("\x01") : std::string(values->GetView(r));
                key += '\x1f';
            }
            if (!seen.insert(key).second) {
                if (failing++ == 0) first = key.substr(0, key.size() - 1);
            }
        }
        for (char& c : first) {
            if (c == '\x1f') c = '|';
        }
        what = JoinList(columns_);
        rule = "unique";
        if (failing > 0) {
            return arrow::Status::Invalid(Label(), " failed: ", what, " repeats in ", failing, " row(s) (first: ", Shown(first),
                                          "). Nothing downstream ran; fix the data or the check.");
        }
        last_outcome_ = "passed: " + what + " unique in " + std::to_string(rows) + " rows";
    } else if (check_ == "range") {
        const auto number = [](double v) {
            char text[32];
            std::snprintf(text, sizeof(text), "%g", v);
            return std::string(text);
        };
        ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(arrow::Datum(input->GetColumnByName(column_)), arrow::float64()));
        for (const auto& chunk : cast.chunked_array()->chunks()) {
            const auto& values = static_cast<const arrow::DoubleArray&>(*chunk);
            for (int64_t i = 0; i < values.length(); ++i) {
                const bool bad = values.IsNull(i) || (min_value_ && values.Value(i) < *min_value_) ||
                                 (max_value_ && values.Value(i) > *max_value_);
                if (bad && failing++ == 0) first = values.IsNull(i) ? std::string("(empty)") : number(values.Value(i));
            }
        }
        rule = min_value_ && max_value_ ? "within " + number(*min_value_) + " to " + number(*max_value_)
               : min_value_             ? "at least " + number(*min_value_)
                                        : "at most " + number(*max_value_);
        if (failing > 0) {
            return arrow::Status::Invalid(Label(), " failed: ", column_, " is not ", rule, " in ", failing, " row(s) (first: ", first,
                                          "). Nothing downstream ran; fix the data or the check.");
        }
        last_outcome_ = "passed: " + column_ + " " + rule + " in " + std::to_string(rows) + " rows";
    } else {
        ARROW_ASSIGN_OR_RAISE(auto text, AsText(input->GetColumnByName(column_)));
        const std::unordered_set<std::string> allowed(allowed_.begin(), allowed_.end());
        for (const auto& chunk : text->chunks()) {
            const auto& values = static_cast<const arrow::StringArray&>(*chunk);
            for (int64_t i = 0; i < values.length(); ++i) {
                const bool empty = values.IsNull(i) || values.GetView(i).empty();
                const bool bad = check_ == "not_empty" ? empty : values.IsNull(i) || !allowed.count(std::string(values.GetView(i)));
                if (bad && failing++ == 0) first = values.IsNull(i) ? std::string("(empty)") : Shown(std::string(values.GetView(i)));
            }
        }
        if (check_ == "not_empty") {
            if (failing > 0) {
                return arrow::Status::Invalid(Label(), " failed: ", column_, " is empty in ", failing,
                                              " row(s). Nothing downstream ran; fix the data or the check.");
            }
            last_outcome_ = "passed: " + column_ + " never empty in " + std::to_string(rows) + " rows";
        } else {
            if (failing > 0) {
                return arrow::Status::Invalid(Label(), " failed: ", column_, " has ", failing, " value(s) outside ", JoinList(allowed_),
                                              " (first: ", first, "). Nothing downstream ran; fix the data or the check.");
            }
            last_outcome_ = "passed: every " + column_ + " is one of " + JoinList(allowed_) + " (" + std::to_string(rows) + " rows)";
        }
    }
    spdlog::info("{} {}", Label(), last_outcome_);
    return input;
}

arrow::Result<std::shared_ptr<arrow::Table>> RowCountCheckOperator::ApplyRowCount(
    const std::shared_ptr<arrow::Table>& input) {
    int64_t counted = input->num_rows();
    std::string what = "rows";
    if (!count_true_column_.empty()) {
        counted = 0;
        for (const auto& chunk : input->GetColumnByName(count_true_column_)->chunks()) {
            const auto& values = static_cast<const arrow::BooleanArray&>(*chunk);
            for (int64_t i = 0; i < values.length(); ++i) {
                if (values.IsValid(i) && values.Value(i)) ++counted;
            }
        }
        what = "rows where " + count_true_column_ + " is true";
    } else if (!count_column_.empty()) {
        counted = 0;
        for (const auto& chunk : input->GetColumnByName(count_column_)->chunks()) {
            for (int64_t i = 0; i < chunk->length(); ++i) {
                if (!chunk->IsValid(i)) continue;
                ARROW_ASSIGN_OR_RAISE(auto scalar, chunk->GetScalar(i));
                if (scalar->ToString() == count_value_) ++counted;
            }
        }
        what = "rows where " + count_column_ + " = '" + count_value_ + "'";
    }

    std::string expectation;
    bool ok = true;
    if (expected_rows_) {
        ok = ok && counted == *expected_rows_;
        expectation = "exactly " + std::to_string(*expected_rows_);
    }
    if (min_rows_) {
        ok = ok && counted >= *min_rows_;
        expectation += (expectation.empty() ? "" : ", ") + std::string("at least ") +
                       std::to_string(*min_rows_);
    }
    if (max_rows_) {
        ok = ok && counted <= *max_rows_;
        expectation += (expectation.empty() ? "" : ", ") + std::string("at most ") +
                       std::to_string(*max_rows_);
    }
    const std::string label = Label();
    if (!ok) {
        return arrow::Status::Invalid(label, " failed: expected ", expectation, " ", what,
                                      ", found ", counted,
                                      ". Nothing downstream ran; fix the data or the expectation.");
    }
    spdlog::info("[Data Studio] {} passed: {} {} (expected {})", label, counted, what, expectation);
    last_outcome_ = "passed: " + std::to_string(counted) + " " + what + " (expected " + expectation + ")";
    return input;
}

} // namespace cyxwiz
