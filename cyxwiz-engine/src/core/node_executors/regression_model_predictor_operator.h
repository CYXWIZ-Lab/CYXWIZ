#pragma once

#include "pipeline_operator.h"

#include <string>
#include <utility>

namespace cyxwiz {

class RegressionModelPredictorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "RegressionModelPredictor"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;
    bool CollectCacheDependencies(
        std::vector<PipelineOperatorCacheDependency>& dependencies,
        std::string& error) const override;
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string model_path_;
    std::string prediction_col_ = "prediction";
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
