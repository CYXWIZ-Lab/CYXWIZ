#pragma once

#include "pipeline_operator.h"
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * TimeSeriesWindowOperator — Cat-1 Band 1 pipeline operator.
 *
 * Transforms a univariate tabular Arrow table into a windowed supervised-
 * learning table. Reads a single `value_col` column and emits one row per
 * sliding window:
 *
 *     x_0, x_1, ..., x_{input_width-1}, y, y_1, ..., y_{label_width-1}
 *
 * where `x_0..x_{input_width-1}` are `input_width` consecutive past values
 * `y` is the first value at offset `input_width + shift - 1` past the
 * window start, and later targets are consecutive future values. A
 * `label_width` of one preserves the historical single `y` column.
 *
 * Multi-step extension:
 *   - label_width >= 1. Targets remain scalar float32 Arrow columns so
 *     preview, Parquet, and tabular materialization retain ordinary schemas.
 *   - Produces float32 columns regardless of input numeric type.
 *   - Preserves row order. Chronology of the source table is inherited
 *     by the windowed table — windows are emitted earliest-first.
 *
 * Multivariate extension (Phase 4 Session D): feature_cols is an optional
 * comma-separated list of additional column names to include alongside
 * value_col. Each extra feature contributes `input_width` more columns
 * to the output, named `<feat>_x_0 .. <feat>_x_{input_width-1}`. The
 * target `y` is always taken from value_col. Typical upstream source is
 * TimeSeriesFeatures producing lag / rolling columns.
 *
 * Forecast-plotting extension (Phase 4.x, 2026-04-16): `time_col` is an
 * optional numeric column (unix epoch / day index / frame number) that,
 * when set, produces a `__window_start_time` float64 metadata column in
 * the output. It holds the time value at each window's first input step,
 * so downstream plots can place forecast vs actual on a shared time
 * axis. The __-prefix marks it as internal metadata, excluded from
 * ArrowDatasetBatcher's feature auto-detection so training doesn't
 * inhale it as an extra input. String / arrow::TimestampType support
 * deferred — numeric only in v1.
 *
 * The first output column name `y` is retained deliberately so existing
 * label resolution remains compatible. Multi-step-aware batchers consume
 * `y`, then `y_1..y_{label_width-1}` in order.
 *
 * Every output row also carries hidden int64 `__target_start_index` and
 * `__target_end_index` metadata. TimeSeriesSplit uses these source-row
 * bounds to purge windows whose targets cross chronological partitions.
 * Double-underscore metadata is excluded from feature auto-detection.
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

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string value_col_;
    std::vector<std::string> feature_cols_;  // Optional multivariate extras
    std::string time_col_;                   // Optional numeric time column
    std::string segment_col_;                // Optional int64 segment identity
    int input_width_ = 12;
    int label_width_ = 1;
    int shift_ = 1;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
