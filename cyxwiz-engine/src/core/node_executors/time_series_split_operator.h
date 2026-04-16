#pragma once

#include "pipeline_operator.h"

namespace cyxwiz {

/**
 * TimeSeriesSplitOperator — Cat-1 Band 2 pipeline operator.
 *
 * Chronologically partitions an already-windowed Arrow table into train,
 * val, and test sets by appending an int8 `__partition__` column:
 *
 *   0 = train    (first `train_ratio` fraction of rows)
 *   1 = val      (next `val_ratio` fraction of rows)
 *   2 = test     (remaining rows)
 *
 * Row order is preserved — assignment is purely based on row index, NOT a
 * random shuffle. This is the hard guarantee against temporal leakage:
 * windows emitted early (older data) go to train, windows emitted late
 * (newer data) go to val/test. Chronology of the source table must be
 * preserved by upstream nodes for this guarantee to hold, which is true
 * of TimeSeriesWindow as implemented.
 *
 * The `__partition__` column is read by ArrowDatasetBatcher when its
 * `partition_column` constructor param is set; otherwise the column is
 * inert and the batcher falls back to its legacy first-N% slicing. Names
 * with double-underscore prefix mark internal metadata columns so the
 * feature-column auto-detection skips them.
 *
 * Band 2 classification: deterministic given ratios, cacheable. No phase
 * flag read (that happens later in DataLoader / Band 3).
 *
 * v1 constraints:
 *   - Ratios must sum to approximately 1.0 (±0.01).
 *   - No stratification (not applicable to time series anyway).
 *   - No gap / purging between splits. Walk-forward CV is deferred.
 */
class TimeSeriesSplitOperator : public IPipelineOperator {
public:
    static constexpr const char* kPartitionColumnName = "__partition__";

    std::string GetName() const override { return "TimeSeriesSplit"; }
    PipelineBand GetBand() const override { return PipelineBand::Partition; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) override;

private:
    float train_ratio_ = 0.8f;
    float val_ratio_   = 0.1f;
    float test_ratio_  = 0.1f;
};

} // namespace cyxwiz
