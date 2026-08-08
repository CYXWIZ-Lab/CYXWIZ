#pragma once

#include "pipeline_operator.h"

#include <string>
#include <utility>

namespace cyxwiz {

/**
 * TimeSeriesSegmentOperator - deterministic timestamp-integrity primitive.
 *
 * Appends two metadata columns to an ordered Arrow table:
 *   - segment_col (int64): zero-based continuous-segment identity.
 *   - delta_col (float64): seconds since the previous row; null on row zero.
 *
 * A new segment begins when the positive timestamp delta is greater than or
 * equal to gap_threshold_seconds. Null, duplicate, backward, or unparseable
 * timestamps fail closed. Timestamp input may be Arrow timestamp/date or an
 * ISO-8601-compatible string accepted by Arrow's safe timestamp cast.
 */
class TimeSeriesSegmentOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TimeSeriesSegment"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string timestamp_col_;
    double gap_threshold_seconds_ = 0.0;
    std::string segment_col_ = "__segment_id";
    std::string delta_col_ = "__time_delta_seconds";
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
