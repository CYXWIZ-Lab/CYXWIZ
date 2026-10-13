#include "svm_operators.h"

#include "tree_classification_utils.h"

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

}  // namespace cyxwiz
