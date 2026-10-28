#pragma once

// Naive Bayes node (TOFIX145 Group C): Gaussian naive Bayes fitted on the
// device (ClassicML::FitGaussianNB, scikit-learn's GaussianNB); appends the
// predicted class and saves the fitted model.

#include "classic_model_common.h"

namespace cyxwiz {

class NaiveBayesClassifierOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "NaiveBayesClassifier"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool LearnsFromEveryRow() const override { return true; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    ClassicColumns columns_;
    double var_smoothing_ = 1e-9;
};

}  // namespace cyxwiz
