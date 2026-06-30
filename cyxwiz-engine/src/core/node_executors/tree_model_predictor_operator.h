#pragma once

#include "pipeline_operator.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class TreeModelPredictorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TreeModelPredictor"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> feature_cols_;
    std::string model_path_;
    std::string prediction_col_ = "prediction";
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
