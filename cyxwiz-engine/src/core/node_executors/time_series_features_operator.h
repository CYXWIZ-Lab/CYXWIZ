#pragma once

#include "pipeline_operator.h"
#include <vector>

namespace cyxwiz {

/**
 * TimeSeriesFeaturesOperator — Cat-1 Band 1 pipeline operator.
 *
 * Engineers lag and rolling-mean feature columns from a source `value_col`.
 * Adds new float32 columns to the Arrow table; does NOT modify the source
 * column. Drops rows at the start of the series where the features can't
 * be computed (max of the largest lag and the largest rolling window minus
 * one). All existing columns are sliced to the same length.
 *
 * Typical use: feeding richer features into a downstream multivariate
 * TimeSeriesWindow. For a Passengers series with lag_values="1,12" and
 * rolling_windows="7":
 *   - Passengers_lag_1       = Passengers[t-1]
 *   - Passengers_lag_12      = Passengers[t-12]
 *   - Passengers_roll_7_mean = mean(Passengers[t-6..t])
 *
 * Params:
 *   value_col         (required)  — source column to engineer features from
 *   lag_values        (optional)  — comma-sep ints, e.g. "1,12" (empty = no lag features)
 *   rolling_windows   (optional)  — comma-sep ints, e.g. "7,30" (empty = no rolling features)
 *
 * v1 note: rolling agg is mean only. std/min/max deferred to Phase 4.x.
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
};

} // namespace cyxwiz
