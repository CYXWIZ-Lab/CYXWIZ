#include "knn_operator.h"

#include "tree_classification_utils.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

bool KNNClassifierOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    n_neighbors_ = 5;
    weights_ = "uniform";
    metric_ = "euclidean";
    if (!ParseClassicColumns(params, GetName(), columns_, error)) return false;
    if (!ParseIntParam(params, "n_neighbors", n_neighbors_, GetName(), error)) return false;
    if (n_neighbors_ < 1) {
        error = GetName() + ": n_neighbors must be >= 1";
        return false;
    }
    if (const auto it = params.find("weights"); it != params.end() && !TrimAscii(it->second).empty())
        weights_ = TrimAscii(it->second);
    if (weights_ != "uniform" && weights_ != "distance") {
        error = GetName() + ": weights must be uniform or distance";
        return false;
    }
    if (const auto it = params.find("metric"); it != params.end() && !TrimAscii(it->second).empty())
        metric_ = TrimAscii(it->second);
    if (metric_ != "euclidean" && metric_ != "manhattan" && metric_ != "cosine") {
        error = GetName() + ": metric must be euclidean, manhattan or cosine";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> KNNClassifierOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    ClassicTrainingData data;
    ARROW_RETURN_NOT_OK(ReadClassicTrainingData(input, columns_, GetName(), GetCancellationQuery(), data));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    const auto model = ClassicML::FitKNN(data.x, data.labels, static_cast<int>(data.classes.size()), n_neighbors_,
                                         weights_, metric_);
    if (!model.success) return arrow::Status::Invalid(GetName() + ": " + model.error_message);
    const auto prediction = ClassicML::PredictKNN(model, data.x);
    if (!prediction.success) return arrow::Status::Invalid(GetName() + ": " + prediction.error_message);

    if (!columns_.model_path.empty()) {
        const nlohmann::json parameters = {
            {"n_neighbors", n_neighbors_}, {"weights", weights_}, {"metric", metric_},
            {"x", model.x},                {"labels", model.labels},
        };
        std::string error;
        if (!SaveClassicModelArtifact(columns_.model_path, GetName(), data, parameters, &error))
            return arrow::Status::IOError(GetName() + ": failed to save the model: " + error);
    }
    spdlog::info("KNNClassifier: {} rows x {} features, {} classes, k={}", data.x.size(), data.features.size(),
                 data.classes.size(), n_neighbors_);
    return AppendClassicPredictions(input, columns_.prediction_col, columns_.probability_columns, data.classes,
                                    data.numeric_labels, prediction, GetName());
}

}  // namespace cyxwiz
