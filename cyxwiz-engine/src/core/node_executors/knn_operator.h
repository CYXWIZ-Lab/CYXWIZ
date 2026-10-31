#pragma once

// KNN Classifier node (TOFIX145 Group C): k-nearest neighbours on the device
// (ClassicML::PredictKNN, scikit-learn's brute-force KNeighborsClassifier);
// appends the predicted class and saves the model (its training rows).

#include "classic_model_common.h"

namespace cyxwiz {

class KNNClassifierOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "KNNClassifier"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool LearnsFromEveryRow() const override { return true; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    ClassicColumns columns_;
    int n_neighbors_ = 5;
    std::string weights_ = "uniform";
    std::string metric_ = "euclidean";
};

}  // namespace cyxwiz
