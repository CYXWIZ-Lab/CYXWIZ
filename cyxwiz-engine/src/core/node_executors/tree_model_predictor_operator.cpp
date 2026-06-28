#include "tree_model_predictor_operator.h"
#include "decision_tree_model.h"
#include "feature_matrix_utils.h"
#include "gradient_boosting_model.h"
#include "random_forest_model.h"
#include "tree_classification_utils.h"
#include "tree_model_artifact.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace cyxwiz {

namespace {

template <typename ModelT>
arrow::Result<std::shared_ptr<arrow::Table>> PredictWithLoadedModel(
    const std::shared_ptr<arrow::Table>& input,
    const ModelT& model,
    const std::vector<std::string>& configured_features,
    const std::string& prediction_col,
    const std::string& op_name) {
    std::vector<std::string> requested_features =
        configured_features.empty() ? model.FeatureNames() : configured_features;
    if (requested_features.empty()) {
        return arrow::Status::Invalid(
            op_name + ": model artifact has no feature names; set feature_cols");
    }

    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, requested_features, "", op_name, resolved_features));

    std::vector<std::vector<double>> features;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(
        input, resolved_features, op_name, features, n_samples));
    if (n_samples <= 0) {
        return arrow::Status::Invalid(op_name + ": input table has no rows");
    }

    const std::vector<int> predictions = model.PredictClasses(features);
    return AppendClassificationPredictions(
        input,
        prediction_col,
        model.ClassLabels(),
        model.HasNumericLabels(),
        predictions,
        op_name);
}

} // namespace

bool TreeModelPredictorOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    feature_cols_.clear();
    model_path_.clear();
    prediction_col_ = "prediction";

    auto model_path = params.find("model_path");
    if (model_path == params.end() || TrimAscii(model_path->second).empty()) {
        error = "TreeModelPredictor: 'model_path' parameter is required";
        return false;
    }
    model_path_ = TrimAscii(model_path->second);

    auto fc = params.find("feature_cols");
    if (fc != params.end()) {
        ParseCommaList(fc->second, feature_cols_);
    }

    auto prediction = params.find("prediction_col");
    if (prediction != params.end() && !TrimAscii(prediction->second).empty()) {
        prediction_col_ = TrimAscii(prediction->second);
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TreeModelPredictorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("TreeModelPredictor: input table is null");
    }

    std::string error;
    const std::string model_type =
        ReadTreeModelArtifactType(model_path_, &error);
    if (model_type.empty()) {
        return arrow::Status::IOError(
            "TreeModelPredictor: failed to read model artifact: " + error);
    }

    if (model_type == "DecisionTreeClassifier") {
        DecisionTreeModel model;
        if (!LoadDecisionTreeModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load DecisionTree artifact: " +
                error);
        }
        spdlog::info("TreeModelPredictor: loaded DecisionTree artifact '{}'",
                     model_path_);
        return PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName());
    }
    if (model_type == "RandomForestClassifier") {
        RandomForestModel model;
        if (!LoadRandomForestModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load RandomForest artifact: " +
                error);
        }
        spdlog::info("TreeModelPredictor: loaded RandomForest artifact '{}'",
                     model_path_);
        return PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName());
    }
    if (model_type == "GradientBoostingClassifier") {
        GradientBoostingModel model;
        if (!LoadGradientBoostingModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load GradientBoosting artifact: " +
                error);
        }
        spdlog::info(
            "TreeModelPredictor: loaded GradientBoosting artifact '{}'",
            model_path_);
        return PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName());
    }

    return arrow::Status::Invalid(
        "TreeModelPredictor: unsupported tree model artifact type '" +
        model_type + "'");
}

} // namespace cyxwiz
