#pragma once

#include "pipeline_operator.h"

namespace cyxwiz {

/**
 * DifferencingOperator — Cat-1 Band 1 pipeline operator.
 *
 * Computes `value[i] - value[i-lag]` on a single `value_col` column,
 * removing trend and/or seasonality. The first `lag * order` rows are
 * dropped (they have no lag-N predecessor). All other columns are
 * sliced to the same shorter length.
 *
 * Params:
 *   value_col  (required)  — which column to difference
 *   lag        (default 1) — seasonal lag (12 for monthly, 7 for daily)
 *   order      (default 1) — how many times to apply differencing
 *
 * Typically placed after LogTransform and before TimeSeriesWindow:
 *   DataInput → LogTransform → Differencing(lag=12) → TimeSeriesWindow → ...
 */
class DifferencingOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "Differencing"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string value_col_;
    int lag_ = 1;
    int order_ = 1;
};

} // namespace cyxwiz
