#include "svm_operators.h"

#include "feature_matrix_utils.h"
#include "model_artifact_json_io.h"
#include "tree_classification_utils.h"
#include "ts_column_utils.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>

namespace cyxwiz {

bool ParseSVMKernelSettings(const std::map<std::string, std::string>& params, const std::string& op_name,
                            SVMKernelSettings& out, std::string& error) {
    out = SVMKernelSettings{};
    if (!ParseDoubleParam(params, "C", out.C, op_name, error)) return false;
    if (!ParseIntParam(params, "degree", out.degree, op_name, error)) return false;
    if (!ParseDoubleParam(params, "coef0", out.coef0, op_name, error)) return false;
    if (!ParseDoubleParam(params, "tol", out.tol, op_name, error)) return false;
    if (const auto it = params.find("kernel"); it != params.end() && !TrimAscii(it->second).empty())
        out.kernel = TrimAscii(it->second);
    if (const auto it = params.find("gamma"); it != params.end() && !TrimAscii(it->second).empty())
        out.gamma = TrimAscii(it->second);
    if (!(out.C > 0.0)) {
        error = op_name + ": C must be > 0";
        return false;
    }
    if (!(out.tol > 0.0)) {
        error = op_name + ": tol must be > 0";
        return false;
    }
    if (out.degree < 0) {
        error = op_name + ": degree must be >= 0";
        return false;
    }
    if (out.kernel != "linear" && out.kernel != "poly" && out.kernel != "rbf" && out.kernel != "sigmoid") {
        error = op_name + ": kernel must be linear, poly, rbf or sigmoid";
        return false;
    }
    if (out.gamma != "scale" && out.gamma != "auto") {
        char* end = nullptr;
        const double value = std::strtod(out.gamma.c_str(), &end);
        if (end == out.gamma.c_str() || *end != '\0' || !(value > 0.0) || !std::isfinite(value)) {
            error = op_name + ": gamma must be scale, auto or a positive number";
            return false;
        }
    }
    return true;
}

nlohmann::json SVMModelParameters(const SVMModel& model, const SVMKernelSettings& settings) {
    return {
        {"C", settings.C},
        {"kernel", model.kernel},
        {"gamma", model.gamma},
        {"gamma_setting", settings.gamma},
        {"degree", model.degree},
        {"coef0", model.coef0},
        {"tol", settings.tol},
        {"support_vectors", model.support_vectors},
        {"pair_first", model.pair_first},
        {"pair_second", model.pair_second},
        {"pair_support", model.pair_support},
        {"pair_coef", model.pair_coef},
        {"rho", model.rho},
        {"iterations", model.iterations},
        {"converged", model.converged},
    };
}

bool SVMClassifierOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    if (!ParseClassicColumns(params, GetName(), columns_, error)) return false;
    return ParseSVMKernelSettings(params, GetName(), settings_, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> SVMClassifierOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    ClassicTrainingData data;
    ARROW_RETURN_NOT_OK(ReadClassicTrainingData(input, columns_, GetName(), GetCancellationQuery(), data));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    std::string gamma_error;
    const double gamma = ClassicML::ResolveSVMGamma(data.x, settings_.gamma, &gamma_error);
    if (!(gamma > 0.0)) return arrow::Status::Invalid(GetName() + ": gamma: " + gamma_error);
    const auto model = ClassicML::FitSVC(data.x, data.labels, static_cast<int>(data.classes.size()), settings_.C,
                                         settings_.kernel, gamma, settings_.degree, settings_.coef0, settings_.tol);
    if (!model.success) return arrow::Status::Invalid(GetName() + ": " + model.error_message);
    if (!model.converged)
        spdlog::warn("SVMClassifier: the solver stopped at its iteration limit before reaching tol={}", settings_.tol);
    const auto prediction = ClassicML::PredictSVC(model, data.x);
    if (!prediction.success) return arrow::Status::Invalid(GetName() + ": " + prediction.error_message);

    if (!columns_.model_path.empty()) {
        std::string error;
        if (!SaveClassicModelArtifact(columns_.model_path, GetName(), data, SVMModelParameters(model, settings_),
                                      &error))
            return arrow::Status::IOError(GetName() + ": failed to save the model: " + error);
    }
    spdlog::info("SVMClassifier: fit {} rows x {} features, {} classes, {} support vectors, {} kernel", data.x.size(),
                 data.features.size(), data.classes.size(), model.support_vectors.size(), settings_.kernel);
    return AppendClassificationPredictions(input, columns_.prediction_col, data.classes, data.numeric_labels,
                                           prediction.labels, GetName());
}

bool SVMRegressorOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    epsilon_ = 0.1;
    if (!ParseClassicColumns(params, GetName(), columns_, error)) return false;
    if (!ParseSVMKernelSettings(params, GetName(), settings_, error)) return false;
    if (!ParseDoubleParam(params, "epsilon", epsilon_, GetName(), error)) return false;
    if (!(epsilon_ >= 0.0)) {
        error = GetName() + ": epsilon must be >= 0";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> SVMRegressorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");
    std::vector<std::string> features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(input, columns_.feature_cols, columns_.target_col, GetName(), features));
    ClassicMatrix x;
    int64_t n = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, features, GetName(), x, n, GetCancellationQuery()));
    if (n <= 0) return arrow::Status::Invalid(GetName() + ": input table has no rows");
    const auto target_column = input->GetColumnByName(columns_.target_col);
    if (!target_column) return arrow::Status::KeyError(GetName() + ": column '" + columns_.target_col + "' not found");
    std::vector<float> target_values;
    std::string bad_type;
    bool cancelled = false;
    if (!ReadColumnAsFloat(target_column, target_values, bad_type, GetCancellationQuery(), &cancelled)) {
        if (cancelled) return arrow::Status::Cancelled(GetName() + ": cancelled while reading the target");
        return arrow::Status::TypeError(GetName() + ": target column '" + columns_.target_col +
                                        "' must be numeric (got '" + bad_type + "')");
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
    const std::vector<double> target(target_values.begin(), target_values.end());

    std::string gamma_error;
    const double gamma = ClassicML::ResolveSVMGamma(x, settings_.gamma, &gamma_error);
    if (!(gamma > 0.0)) return arrow::Status::Invalid(GetName() + ": gamma: " + gamma_error);
    const auto model = ClassicML::FitSVR(x, target, settings_.C, epsilon_, settings_.kernel, gamma, settings_.degree,
                                         settings_.coef0, settings_.tol);
    if (!model.success) return arrow::Status::Invalid(GetName() + ": " + model.error_message);
    if (!model.converged)
        spdlog::warn("SVMRegressor: the solver stopped at its iteration limit before reaching tol={}", settings_.tol);
    std::string error;
    const std::vector<double> predicted = ClassicML::PredictSVR(model, x, &error);
    if (predicted.size() != x.size()) return arrow::Status::Invalid(GetName() + ": " + error);

    if (!columns_.model_path.empty()) {
        auto parameters = SVMModelParameters(model, settings_);
        parameters["epsilon"] = epsilon_;
        const nlohmann::json model_json = {
            {"feature_names", features}, {"target", columns_.target_col}, {"parameters", parameters}};
        std::string write_error;
        if (!artifact_json::WriteJsonFile(columns_.model_path,
                                          artifact_json::MakeEnvelope("cyxwiz_classic_model", GetName(), model_json),
                                          &write_error))
            return arrow::Status::IOError(GetName() + ": failed to save the model: " + write_error);
    }
    spdlog::info("SVMRegressor: fit {} rows x {} features, {} support vectors, {} kernel", x.size(), features.size(),
                 model.support_vectors.size(), settings_.kernel);

    if (input->schema()->GetFieldIndex(columns_.prediction_col) >= 0)
        return arrow::Status::Invalid(GetName() + ": prediction column '" + columns_.prediction_col +
                                      "' already exists");
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.AppendValues(predicted));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return input->AddColumn(input->num_columns(), arrow::field(columns_.prediction_col, arrow::float64()),
                            std::make_shared<arrow::ChunkedArray>(array));
}

}  // namespace cyxwiz
