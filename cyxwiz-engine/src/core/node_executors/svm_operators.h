#pragma once

// SVM nodes (TOFIX145 Group C): support vector machines fitted on the device
// (ClassicML::FitSVC / FitSVR, scikit-learn's SVC / SVR on libsvm's SMO). The
// regressor appends a numeric prediction column; its target must be numeric.
// The classifier's probability columns are scikit-learn's SVC(probability=True)
// (ClassicML::FitSVCProbability).

#include "classic_model_common.h"

namespace cyxwiz {

// Kernel settings shared by the SVM nodes, scikit-learn's names.
struct SVMKernelSettings {
    double C = 1.0;
    std::string kernel = "rbf";
    std::string gamma = "scale";  // scale, auto or a positive number
    int degree = 3;
    double coef0 = 0.0;
    double tol = 1e-3;
};

bool ParseSVMKernelSettings(const std::map<std::string, std::string>& params, const std::string& op_name,
                            SVMKernelSettings& out, std::string& error);

// The fitted model's learned values for the cyxwiz_classic_model artifact.
nlohmann::json SVMModelParameters(const SVMModel& model, const SVMKernelSettings& settings);

class SVMRegressorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SVMRegressor"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    ClassicColumns columns_;
    SVMKernelSettings settings_;
    double epsilon_ = 0.1;
};

class SVMClassifierOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SVMClassifier"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    ClassicColumns columns_;
    SVMKernelSettings settings_;
    // Probability columns: Platt's 5-fold calibration, with scikit-learn's
    // random_state for the folds.
    int random_state_ = 0;
};

}  // namespace cyxwiz
