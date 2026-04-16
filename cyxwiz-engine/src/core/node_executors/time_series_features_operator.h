#pragma once

#include "pipeline_operator.h"
#include <vector>

namespace cyxwiz {

/**
 * TimeSeriesFeaturesOperator — Cat-1 Band 1 pipeline operator.
 *
 * Engineers lag and rolling-aggregate feature columns from a source
 * `value_col`. Adds new float32 columns to the Arrow table; does NOT
 * modify the source column. Drops rows at the start of the series
 * where the features can't be computed (max of the largest lag and
 * the largest rolling window minus one). All existing columns are
 * sliced to the same length.
 *
 * Typical use: feeding richer features into a downstream multivariate
 * TimeSeriesWindow. For a Passengers series with lag_values="1,12",
 * rolling_windows="7", rolling_aggregations="mean,std":
 *   - Passengers_lag_1       = Passengers[t-1]
 *   - Passengers_lag_12      = Passengers[t-12]
 *   - Passengers_roll_7_mean = mean(Passengers[t-6..t])
 *   - Passengers_roll_7_std  = std(Passengers[t-6..t])
 *
 * Params:
 *   value_col            (required)        — source column.
 *   lag_values           (optional, csv)   — e.g. "1,12".
 *   rolling_windows      (optional, csv)   — e.g. "7,30".
 *   rolling_aggregations (default "mean")  — csv of {mean, std, min, max, median}.
 *                                             Applied to every window.
 */
class TimeSeriesFeaturesOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TimeSeriesFeatures"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string value_col_;
    std::vector<int> lags_;
    std::vector<int> rolling_windows_;
    std::vector<std::string> rolling_aggregations_;  // defaults to {"mean"}
};

} // namespace cyxwiz
