#pragma once

#include "pipeline_operator.h"
#include "random_forest_trainer.h"

#include <string>
#include <vector>

namespace cyxwiz {

class RandomForestClassifierOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "RandomForestClassifier"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::vector<std::string> feature_cols_;
    std::string target_col_;
    std::string prediction_col_ = "prediction";
    std::string model_path_;
    RandomForestTrainingOptions options_;
};

} // namespace cyxwiz
