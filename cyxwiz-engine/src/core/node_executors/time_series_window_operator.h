#pragma once

#include "pipeline_operator.h"

namespace cyxwiz {

/**
 * TimeSeriesWindowOperator — Cat-1 Band 1 pipeline operator.
 *
 * Transforms a univariate tabular Arrow table into a windowed supervised-
 * learning table. Reads a single `value_col` column and emits one row per
 * sliding window:
 *
 *     x_0, x_1, ..., x_{input_width-1}, y
 *
 * where `x_0..x_{input_width-1}` are `input_width` consecutive past values
 * and `y` is the first value at offset `input_width + shift - 1` past the
 * window start. This matches the classic timeseries_dataset_from_array
 * shape that feedforward / Conv1D / LSTM models all consume.
 *
 * v1 constraints (Phase 4 Session B):
 *   - label_width = 1 only (single-step forecasting). Multi-step output
 *     (y_0, y_1, ...) is deferred to Phase 4.x and will require a
 *     TimeSeriesBatcher that understands multi-output regression.
 *   - Univariate only. Multivariate (multiple value_cols) deferred.
 *   - Produces float32 columns regardless of input numeric type.
 *   - Preserves row order. Chronology of the source table is inherited
 *     by the windowed table — windows are emitted earliest-first.
 *
 * The output column name `y` is chosen deliberately so ArrowDatasetBatcher's
 * common-label auto-detection picks it up without the user having to type a
 * label column name. See dataset_batcher.cpp::InitializeColumns.
 *
 * Band 1 classification: stateless, deterministic given params. Identical
 * behavior on train/val/test. Cacheable (same output every run). Does not
 * read the training phase flag.
 */
class TimeSeriesWindowOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TimeSeriesWindow"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) override;

private:
    std::string value_col_;
    int input_width_ = 12;
    int label_width_ = 1;
    int shift_ = 1;
};

} // namespace cyxwiz
