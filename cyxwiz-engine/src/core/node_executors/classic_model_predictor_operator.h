#pragma once

// Classic Model Predictor node (TOFIX145): applies a fitted classic ML model
// (Naive Bayes, KNN, Logistic Regression, SVM Classifier or SVM Regressor) to a
// table. The model comes from the Model input (this run) or from a saved
// cyxwiz_classic_model file (an earlier run); the executor puts its path in
// `model_path`. Features are read by the model's saved column names.

#include "pipeline_operator.h"

#include <string>

namespace cyxwiz {

class ClassicModelPredictorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ClassicModelPredictor"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    bool CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies,
                                  std::string& error) const override;

private:
    std::string model_path_;
    std::string prediction_col_ = "prediction";
    bool probability_columns_ = false;
};

}  // namespace cyxwiz
