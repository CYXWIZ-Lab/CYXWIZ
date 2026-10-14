#include "logistic_regression_operator.h"

#include "tree_classification_utils.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

bool LogisticRegressionOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    c_ = 1.0;
    max_iter_ = 100;
    tol_ = 1e-4;
    if (!ParseClassicColumns(params, GetName(), columns_, error)) return false;
    if (!ParseDoubleParam(params, "C", c_, GetName(), error)) return false;
    if (!ParseIntParam(params, "max_iter", max_iter_, GetName(), error)) return false;
    if (!ParseDoubleParam(params, "tol", tol_, GetName(), error)) return false;
    if (!(c_ > 0.0)) {
        error = GetName() + ": C must be > 0";
        return false;
    }
    if (max_iter_ < 1) {
        error = GetName() + ": max_iter must be >= 1";
        return false;
    }
    if (!(tol_ > 0.0)) {
        error = GetName() + ": tol must be > 0";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> LogisticRegressionOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    ClassicTrainingData data;
    ARROW_RETURN_NOT_OK(ReadClassicTrainingData(input, columns_, GetName(), GetCancellationQuery(), data));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    const auto model =
        ClassicML::FitLogistic(data.x, data.labels, static_cast<int>(data.classes.size()), c_, max_iter_, tol_);
    if (!model.success) return arrow::Status::Invalid(GetName() + ": " + model.error_message);
    if (!model.converged) {
        // As scikit-learn's ConvergenceWarning: the fit stands, the log says so.
        spdlog::warn("LogisticRegressionNode: stopped after max_iter={} without reaching tol={}; raise max_iter "
                     "or scale the features",
                     max_iter_, tol_);
    }
    const auto prediction = ClassicML::PredictLogistic(model, data.x);
    if (!prediction.success) return arrow::Status::Invalid(GetName() + ": " + prediction.error_message);

    if (!columns_.model_path.empty()) {
        const nlohmann::json parameters = {
            {"coef", model.coef},   {"intercept", model.intercept}, {"C", c_},
            {"max_iter", max_iter_}, {"tol", tol_},                 {"n_iter", model.n_iter},
            {"converged", model.converged},
        };
        std::string error;
        if (!SaveClassicModelArtifact(columns_.model_path, GetName(), data, parameters, &error))
            return arrow::Status::IOError(GetName() + ": failed to save the model: " + error);
    }
    spdlog::info("LogisticRegressionNode: fit {} rows x {} features, {} classes in {} iterations", data.x.size(),
                 data.features.size(), data.classes.size(), model.n_iter);
    return AppendClassificationPredictions(input, columns_.prediction_col, data.classes, data.numeric_labels,
                                           prediction.labels, GetName());
}

}  // namespace cyxwiz
