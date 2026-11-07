#pragma once

// Portfolio Weights (TOFIX129 D10): the least risky mix of assets from a
// table of their returns, one column per asset and one row per period
// (Markowitz). The means and the covariance come from the Train rows (all
// rows without a split), on the device; the weights from the convex
// quadratic program Optimization::PortfolioProgram poses (HiGHS). Output:
// one row per asset (asset, weight, mean_return, sd, risk_share).

#include "pipeline_operator.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class PortfolioWeightsOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "PortfolioWeights"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    std::vector<std::string> asset_cols_;   // empty: every number column
    double target_return_ = NAN;            // NaN: the minimum-variance mix
    bool long_only_ = true;
};

}  // namespace cyxwiz
