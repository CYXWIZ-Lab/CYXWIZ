#pragma once

// Logistic Regression node (TOFIX145 Group C): L2-regularised logistic
// regression fitted on the device (ClassicML::FitLogistic, scikit-learn's
// LogisticRegression with lbfgs); appends the predicted class and saves the
// coefficients.

#include "classic_model_common.h"

namespace cyxwiz {

class LogisticRegressionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LogisticRegressionNode"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    ClassicColumns columns_;
    double c_ = 1.0;
    int max_iter_ = 100;
    double tol_ = 1e-4;
};

}  // namespace cyxwiz
