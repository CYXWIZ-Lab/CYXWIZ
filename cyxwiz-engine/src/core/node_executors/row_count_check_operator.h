#pragma once

#include "pipeline_operator.h"

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

/**
 * RowCountCheckOperator - the first Check step (TOFIX101 package E,
 * data_studio_design.md section 5.3, owner decision 5: count checks only).
 *
 * Passes the input table through unchanged when the counted rows satisfy the
 * expectation, and fails the run otherwise ("stop" policy), so a wrong count
 * can never be published. The counted rows are all rows; or, when
 * count_true_column is set, the rows where that boolean column is true
 * (for example empty_text to count empty verses); or, when count_column and
 * count_value are set, the rows whose value in that column equals count_value
 * (for example role_candidate = chapter_candidate). Values compare as text.
 *
 * Parameters: expected_rows (exact) and/or min_rows / max_rows (inclusive);
 * at least one is required. check_name labels the failure message.
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

private:
    std::string check_name_;
    std::string count_true_column_;
    std::string count_column_;
    std::string count_value_;
    std::optional<int64_t> expected_rows_;
    std::optional<int64_t> min_rows_;
    std::optional<int64_t> max_rows_;
};

} // namespace cyxwiz
