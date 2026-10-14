#include "naive_bayes_operator.h"

#include "tree_classification_utils.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

bool NaiveBayesClassifierOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    var_smoothing_ = 1e-9;
    if (!ParseClassicColumns(params, GetName(), columns_, error)) return false;
    if (!ParseDoubleParam(params, "var_smoothing", var_smoothing_, GetName(), error)) return false;
    if (!(var_smoothing_ >= 0.0)) {
        error = GetName() + ": var_smoothing must be >= 0";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> NaiveBayesClassifierOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    ClassicTrainingData data;
    ARROW_RETURN_NOT_OK(ReadClassicTrainingData(input, columns_, GetName(), GetCancellationQuery(), data));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    const auto model =
        ClassicML::FitGaussianNB(data.x, data.labels, static_cast<int>(data.classes.size()), var_smoothing_);
    if (!model.success) return arrow::Status::Invalid(GetName() + ": " + model.error_message);
    const auto prediction = ClassicML::PredictGaussianNB(model, data.x);
    if (!prediction.success) return arrow::Status::Invalid(GetName() + ": " + prediction.error_message);

    if (!columns_.model_path.empty()) {
        const nlohmann::json parameters = {
            {"theta", model.theta},         {"var", model.var},
            {"class_prior", model.class_prior}, {"class_count", model.class_count},
            {"epsilon", model.epsilon},     {"var_smoothing", var_smoothing_},
        };
        std::string error;
        if (!SaveClassicModelArtifact(columns_.model_path, GetName(), data, parameters, &error))
            return arrow::Status::IOError(GetName() + ": failed to save the model: " + error);
    }
    spdlog::info("NaiveBayesClassifier: fit {} rows x {} features, {} classes", data.x.size(), data.features.size(),
                 data.classes.size());
    return AppendClassificationPredictions(input, columns_.prediction_col, data.classes, data.numeric_labels,
                                           prediction.labels, GetName());
}

}  // namespace cyxwiz
