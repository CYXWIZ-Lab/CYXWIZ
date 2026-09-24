#include "row_count_check_operator.h"

#include <arrow/api.h>

#include <spdlog/spdlog.h>

#include <cctype>

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

}  // namespace

bool RowCountCheckOperator::Configure(const std::map<std::string, std::string>& params,
                                      std::string& error) {
    check_name_ = Param(params, "check_name");
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
    const std::string policy = Param(params, "on_failure");
    if (!policy.empty() && policy != "stop") {
        error = "RowCountCheck: on_failure '" + policy +
                "' is not supported yet; count checks stop the run";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> RowCountCheckOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {
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
    const std::string label = check_name_.empty() ? std::string("Row count check") : "'" + check_name_ + "'";
    if (!ok) {
        return arrow::Status::Invalid(label, " failed: expected ", expectation, " ", what,
                                      ", found ", counted,
                                      ". Nothing downstream ran; fix the data or the expectation.");
    }
    spdlog::info("[Data Studio] {} passed: {} {} (expected {})", label, counted, what, expectation);
    return input;
}

} // namespace cyxwiz
