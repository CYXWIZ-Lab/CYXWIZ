#pragma once

#include "pipeline_operator.h"
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * LinearRegressionOperator — Cat-1 Band 1 pipeline operator.
 *
 * Fits a multiple linear regression on the input table via
 * `DataAnalyzer::MultipleLinearRegression` and appends two columns
 * to the output: `prediction` (float32) and `residual` (float32 =
 * target - prediction).
 *
 * Fits on ALL rows — there is no train/test split at this level.
 * Users who want proper train/test evaluation should chain
 * DataSplit → LinearRegression → per-partition evaluation. For
 * classical stats-style "fit on observations, look at residuals"
 * workflows, this operator is sufficient.
 *
 * Closes the LinearRegressionNode dead NodeType from the
 * "Tool-to-Node Migration" ML regression block.
 *
 * Params:
 *   feature_cols (required, comma-sep) — predictor columns.
 *   target_col   (required)            — response column.
 *   fit_intercept (default true)       — include b0 in the model.
 */
class LinearRegressionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LinearRegression"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::vector<std::string> feature_cols_;
    std::string target_col_;
    bool fit_intercept_ = true;
};

/**
 * PolynomialRegressionOperator — Cat-1 Band 1 pipeline operator.
 *
 * Fits a univariate polynomial regression y = b0 + b1*x + b2*x^2 +
 * ... + bk*x^k and appends `prediction` + `residual` columns. Single
 * predictor only (the backend's PolynomialRegression takes a single
 * x vector); multi-predictor polynomial requires manual feature
 * expansion upstream + LinearRegression.
 *
 * Closes the PolynomialRegressionNode dead NodeType.
 *
 * Params:
 *   feature_col (required)  — single predictor column.
 *   target_col  (required)  — response column.
 *   degree      (default 2) — polynomial degree.
 */
class PolynomialRegressionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "PolynomialRegression"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string feature_col_;
    std::string target_col_;
    int degree_ = 2;
};

} // namespace cyxwiz
