#pragma once

#include "pipeline_operator.h"
#include <utility>

namespace cyxwiz {

/**
 * LogTransformOperator — Cat-1 Band 1 pipeline operator.
 *
 * Applies log1p(x) to a single `value_col` column, stabilizing
 * exponential growth patterns (airline passengers, stock prices, etc.).
 * Schema is unchanged, row count is unchanged.
 *
 * Typically placed upstream of Differencing and TimeSeriesWindow:
 *   DataInput → LogTransform → Differencing(lag=12) → TimeSeriesWindow → ...
 */
class LogTransformOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LogTransform"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string value_col_;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
