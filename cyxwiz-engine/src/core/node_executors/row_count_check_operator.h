#pragma once

#include "pipeline_operator.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

/**
 * RowCountCheckOperator - the Check step (TOFIX101 package E,
 * data_studio_design.md section 5.3; TOFIX148 D4: one check model).
 *
 * Passes the input table through unchanged when the check holds, and fails
 * the run otherwise ("stop" policy), so wrong data can never be published.
 * `check` picks the kind (default row_count):
 *
 * - row_count: the counted rows are all rows; or, when count_true_column is
 *   set, the rows where that boolean column is true (for example empty_text
 *   to count empty verses); or, when count_column and count_value are set,
 *   the rows whose value in that column equals count_value (for example
 *   role_candidate = chapter_candidate). Values compare as text.
 *   expected_rows (exact) and/or min_rows / max_rows (inclusive); at least
 *   one is required. "Expect none" is expected_rows 0 with a condition.
 * - allowed_values: every value of `column` is one of `allowed` (a comma
 *   list, compared as text); an empty value fails.
 * - unique: no two rows share their values in `columns` (a comma list).
 * - not_empty: `column` has no empty value (missing, or empty text).
 * - range: every value of the numeric `column` lies within min_value and / or
 *   max_value (inclusive); an empty value fails.
 *
 * The guided step's checks (core/guided_sql add_checks) follow the same
 * rules. check_name labels the failure message.
 */
class RowCountCheckOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "RowCountCheck"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) override;

    std::string OutcomeSummary() const override { return last_outcome_; }

    // Streams (TOFIX154): every batch is counted or checked, the rule is
    // judged at EndFit, and the input passes through uncopied. `unique` keeps
    // every key and so reads its input whole (not streamed).
    bool SupportsStreaming() const override { return check_ != "unique"; }
    bool PassesInputThrough() const override { return true; }
    arrow::Status BeginStream(const std::shared_ptr<arrow::Schema>& schema) override;
    arrow::Status FitBatch(const std::shared_ptr<arrow::Table>& batch) override;
    arrow::Status EndFit() override;

private:
    std::string Label() const;

    // The stream so far: rows seen, counted rows (row_count), failing rows
    // and the first failing value (value checks), the keys seen (unique).
    int64_t rows_ = 0;
    int64_t counted_ = 0;
    int64_t failing_ = 0;
    std::string first_;
    std::unordered_set<std::string> seen_;

    std::string last_outcome_;  // "passed: <counted> <what> (expected <rule>)"
    std::string check_ = "row_count";
    std::string check_name_;
    std::string count_true_column_;
    std::string count_column_;
    std::string count_value_;
    std::optional<int64_t> expected_rows_;
    std::optional<int64_t> min_rows_;
    std::optional<int64_t> max_rows_;
    std::string column_;                // allowed_values, not_empty, range
    std::vector<std::string> columns_;  // unique
    std::vector<std::string> allowed_;
    std::optional<double> min_value_;
    std::optional<double> max_value_;
};

} // namespace cyxwiz
