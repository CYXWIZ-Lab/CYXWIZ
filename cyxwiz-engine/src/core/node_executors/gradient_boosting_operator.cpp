#include "gradient_boosting_operator.h"
#include "feature_matrix_utils.h"
#include "tree_classification_utils.h"
#include "tree_model_artifact.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace cyxwiz {

bool GradientBoostingClassifierOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    feature_cols_.clear();
    target_col_.clear();
    prediction_col_ = "prediction";
    model_path_.clear();
    options_ = GradientBoostingTrainingOptions{};

    auto fc = params.find("feature_cols");
    if (fc != params.end()) {
        ParseCommaList(fc->second, feature_cols_);
    }

    auto target = params.find("target_col");
    if (target == params.end() || TrimAscii(target->second).empty()) {
        error = "GradientBoostingClassifier: 'target_col' parameter is required";
        return false;
    }
    target_col_ = TrimAscii(target->second);

    auto prediction = params.find("prediction_col");
    if (prediction != params.end() && !TrimAscii(prediction->second).empty()) {
        prediction_col_ = TrimAscii(prediction->second);
    }
    auto model_path = params.find("model_path");
    if (model_path != params.end() && !TrimAscii(model_path->second).empty()) {
        model_path_ = TrimAscii(model_path->second);
    }

    if (!ParseIntParam(params, "n_estimators", options_.n_estimators,
                       GetName(), error) ||
        !ParseDoubleParam(params, "learning_rate", options_.learning_rate,
                          GetName(), error) ||
        !ParseIntParam(params, "max_depth", options_.max_depth,
                       GetName(), error) ||
        !ParseIntParam(params, "min_samples_split",
                       options_.min_samples_split, GetName(), error) ||
        !ParseIntParam(params, "min_samples_leaf",
                       options_.min_samples_leaf, GetName(), error)) {
        return false;
    }

    if (options_.n_estimators < 1) {
        error = "GradientBoostingClassifier: n_estimators must be >= 1";
        return false;
    }
    if (options_.learning_rate <= 0.0) {
        error = "GradientBoostingClassifier: learning_rate must be > 0";
        return false;
    }
    if (options_.max_depth < 1) {
        error = "GradientBoostingClassifier: max_depth must be >= 1";
        return false;
    }
    if (options_.min_samples_split < 2) {
        error = "GradientBoostingClassifier: min_samples_split must be >= 2";
        return false;
    }
    if (options_.min_samples_leaf < 1) {
        error = "GradientBoostingClassifier: min_samples_leaf must be >= 1";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
GradientBoostingClassifierOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid(
            "GradientBoostingClassifier: input table is null");
    }

    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, target_col_, GetName(), resolved_features));

    std::vector<std::vector<double>> features;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(
        input, resolved_features, GetName(), features, n_samples));
    if (n_samples <= 0) {
        return arrow::Status::Invalid(
            "GradientBoostingClassifier: input table has no rows");
    }

    std::vector<int> labels;
    std::vector<std::string> class_labels;
    bool numeric_labels = false;
    ARROW_RETURN_NOT_OK(ReadClassificationLabels(
        input, target_col_, GetName(), labels, class_labels, numeric_labels));
    if (labels.size() != static_cast<size_t>(n_samples)) {
        return arrow::Status::Invalid(
            "GradientBoostingClassifier: feature/label row mismatch");
    }

    GradientBoostingTrainer trainer(options_);
    GradientBoostingModel model;
    try {
        model = trainer.Fit(features, labels, class_labels.size(),
                            resolved_features, class_labels, numeric_labels);
    } catch (const std::exception& ex) {
        return arrow::Status::Invalid(ex.what());
    }

    const std::vector<int> predictions = model.PredictClasses(features);
    if (!model_path_.empty()) {
        std::string error;
        if (!SaveGradientBoostingModelArtifact(model, model_path_, &error)) {
            return arrow::Status::IOError(
                "GradientBoostingClassifier: failed to save model artifact: " +
                error);
        }
    }
    spdlog::info(
        "GradientBoostingClassifier: fit {} rows x {} features, classes={}, "
        "estimators={}, max_depth={}, learning_rate={}",
        n_samples,
        resolved_features.size(),
        class_labels.size(),
        model.Trees().size(),
        model.MaxDepth(),
        options_.learning_rate);

    return AppendClassificationPredictions(
        input,
        prediction_col_,
        model.ClassLabels(),
        model.HasNumericLabels(),
        predictions,
        GetName());
}

} // namespace cyxwiz
