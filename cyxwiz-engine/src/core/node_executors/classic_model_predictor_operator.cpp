#include "classic_model_predictor_operator.h"

#include "classic_model_artifact.h"
#include "classic_model_common.h"
#include "feature_matrix_utils.h"
#include "tree_classification_utils.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

bool ClassicModelPredictorOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    model_path_.clear();
    prediction_col_ = "prediction";
    if (const auto it = params.find("model_path"); it != params.end()) model_path_ = TrimAscii(it->second);
    if (model_path_.empty()) {
        error = GetName() + ": connect a classic model's Model output or choose a saved model file";
        return false;
    }
    if (const auto it = params.find("prediction_col"); it != params.end() && !TrimAscii(it->second).empty())
        prediction_col_ = TrimAscii(it->second);
    return true;
}

bool ClassicModelPredictorOperator::CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies,
                                                             std::string& error) const {
    ClassicModelArtifact artifact;
    if (!LoadClassicModelArtifact(model_path_, artifact, &error)) return false;
    dependencies.push_back({"classic_model", model_path_});
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> ClassicModelPredictorOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");
    ClassicModelArtifact artifact;
    std::string error;
    if (!LoadClassicModelArtifact(model_path_, artifact, &error))
        return arrow::Status::IOError(GetName() + ": cannot use the model '" + model_path_ + "': " + error);

    std::string missing;
    for (const auto& feature : artifact.feature_names) {
        if (!input->GetColumnByName(feature)) missing += (missing.empty() ? "" : ", ") + feature;
    }
    if (!missing.empty())
        return arrow::Status::KeyError(GetName() + ": the input has no column " + missing + " (the model was fitted on " +
                                       std::to_string(artifact.feature_names.size()) + " columns)");
    ClassicMatrix x;
    int64_t n = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, artifact.feature_names, GetName(), x, n, GetCancellationQuery()));
    if (n <= 0) return arrow::Status::Invalid(GetName() + ": input table has no rows");
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    if (artifact.IsRegressor()) {
        const std::vector<double> predicted = ClassicML::PredictSVR(artifact.svm, x, &error);
        if (predicted.size() != x.size()) return arrow::Status::Invalid(GetName() + ": " + error);
        spdlog::info("ClassicModelPredictor: {} on {} rows", artifact.model_type, x.size());
        return AppendNumericPredictions(input, prediction_col_, predicted, GetName());
    }

    ClassicPrediction prediction;
    if (artifact.model_type == "NaiveBayesClassifier") {
        prediction = ClassicML::PredictGaussianNB(artifact.naive_bayes, x);
    } else if (artifact.model_type == "KNNClassifier") {
        prediction = ClassicML::PredictKNN(artifact.knn, x);
    } else if (artifact.model_type == "LogisticRegressionNode") {
        prediction = ClassicML::PredictLogistic(artifact.logistic, x);
    } else {
        prediction = ClassicML::PredictSVC(artifact.svm, x);
    }
    if (!prediction.success) return arrow::Status::Invalid(GetName() + ": " + prediction.error_message);
    spdlog::info("ClassicModelPredictor: {} on {} rows, {} classes", artifact.model_type, x.size(),
                 artifact.class_labels.size());
    return AppendClassificationPredictions(input, prediction_col_, artifact.class_labels, artifact.numeric_labels,
                                           prediction.labels, GetName());
}

}  // namespace cyxwiz
