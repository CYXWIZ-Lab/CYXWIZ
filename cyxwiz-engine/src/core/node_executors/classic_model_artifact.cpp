#include "classic_model_artifact.h"

#include "model_artifact_json_io.h"

#include <exception>

namespace cyxwiz {

namespace {

constexpr const char* kFormat = "cyxwiz_classic_model";

void ReadSVM(const artifact_json::Json& parameters, int n_classes, SVMModel& model) {
    model.kernel = parameters.at("kernel").get<std::string>();
    model.gamma = parameters.at("gamma").get<double>();
    model.degree = parameters.at("degree").get<int>();
    model.coef0 = parameters.at("coef0").get<double>();
    model.support_vectors = parameters.at("support_vectors").get<ClassicMatrix>();
    model.pair_first = parameters.at("pair_first").get<std::vector<int>>();
    model.pair_second = parameters.at("pair_second").get<std::vector<int>>();
    model.pair_support = parameters.at("pair_support").get<std::vector<std::vector<int>>>();
    model.pair_coef = parameters.at("pair_coef").get<std::vector<std::vector<double>>>();
    model.rho = parameters.at("rho").get<std::vector<double>>();
    model.prob_a = parameters.value("prob_a", std::vector<double>{});
    model.prob_b = parameters.value("prob_b", std::vector<double>{});
    model.iterations = parameters.value("iterations", 0);
    model.converged = parameters.value("converged", true);
    model.n_classes = n_classes;
    model.success = true;
}

}  // namespace

bool LoadClassicModelArtifact(const std::string& path, ClassicModelArtifact& artifact, std::string* error) {
    try {
        artifact_json::Json document;
        if (!artifact_json::ReadJsonFile(path, document, error)) return false;
        const std::string model_type = document.value("model_type", "");
        if (!artifact_json::ValidateEnvelope(document, kFormat, model_type, error)) return false;

        const auto& payload = document.at("model");
        ClassicModelArtifact loaded;
        loaded.model_type = model_type;
        loaded.feature_names = payload.at("feature_names").get<std::vector<std::string>>();
        if (loaded.feature_names.empty()) {
            artifact_json::SetError(error, "the saved model has no feature columns");
            return false;
        }
        const auto& parameters = payload.at("parameters");

        if (model_type == "SVMRegressor") {
            loaded.target = payload.value("target", "");
            ReadSVM(parameters, 0, loaded.svm);
            artifact = std::move(loaded);
            return true;
        }

        loaded.class_labels = payload.at("class_labels").get<std::vector<std::string>>();
        loaded.numeric_labels = payload.value("numeric_labels", false);
        const int n_classes = static_cast<int>(loaded.class_labels.size());
        if (n_classes < 2) {
            artifact_json::SetError(error, "the saved model has fewer than two classes");
            return false;
        }
        if (model_type == "NaiveBayesClassifier") {
            auto& model = loaded.naive_bayes;
            model.theta = parameters.at("theta").get<ClassicMatrix>();
            model.var = parameters.at("var").get<ClassicMatrix>();
            model.class_prior = parameters.at("class_prior").get<std::vector<double>>();
            model.class_count = parameters.value("class_count", std::vector<double>{});
            model.epsilon = parameters.value("epsilon", 0.0);
            model.success = true;
        } else if (model_type == "KNNClassifier") {
            auto& model = loaded.knn;
            model.x = parameters.at("x").get<ClassicMatrix>();
            model.labels = parameters.at("labels").get<std::vector<int>>();
            model.n_classes = n_classes;
            model.n_neighbors = parameters.at("n_neighbors").get<int>();
            model.weights = parameters.at("weights").get<std::string>();
            model.metric = parameters.at("metric").get<std::string>();
            model.success = true;
        } else if (model_type == "LogisticRegressionNode") {
            auto& model = loaded.logistic;
            model.coef = parameters.at("coef").get<ClassicMatrix>();
            model.intercept = parameters.at("intercept").get<std::vector<double>>();
            model.n_classes = n_classes;
            model.n_iter = parameters.value("n_iter", 0);
            model.converged = parameters.value("converged", true);
            model.success = true;
        } else if (model_type == "SVMClassifier") {
            ReadSVM(parameters, n_classes, loaded.svm);
        } else {
            artifact_json::SetError(error, "unknown classic model type '" + model_type + "'");
            return false;
        }
        artifact = std::move(loaded);
        return true;
    } catch (const std::exception& e) {
        artifact_json::SetError(error, std::string("the saved model is incomplete: ") + e.what());
        return false;
    }
}

}  // namespace cyxwiz
