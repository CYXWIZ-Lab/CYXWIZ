#include "regression_model_artifact.h"
#include "model_artifact_json_io.h"

#include <cmath>
#include <exception>

namespace cyxwiz {
namespace {

constexpr const char* kFormat = "cyxwiz_regression_model";

const char* ModelTypeName(RegressionModelType type) {
    return type == RegressionModelType::Polynomial
        ? "PolynomialRegression"
        : "LinearRegression";
}

bool Validate(const RegressionModelArtifact& model, std::string* error) {
    if (model.feature_names.empty()) {
        artifact_json::SetError(error, "regression artifact has no feature names");
        return false;
    }
    if (model.target_name.empty()) {
        artifact_json::SetError(error, "regression artifact has no target name");
        return false;
    }
    const size_t expected = model.type == RegressionModelType::Polynomial
        ? static_cast<size_t>(model.degree + 1)
        : model.feature_names.size() + (model.fit_intercept ? 1U : 0U);
    if (model.type == RegressionModelType::Polynomial &&
        (model.feature_names.size() != 1 || model.degree < 1 ||
         !model.fit_intercept)) {
        artifact_json::SetError(error, "invalid polynomial regression contract");
        return false;
    }
    if (model.coefficients.size() != expected) {
        artifact_json::SetError(
            error, "regression coefficient count does not match feature contract");
        return false;
    }
    for (double coefficient : model.coefficients) {
        if (!std::isfinite(coefficient)) {
            artifact_json::SetError(error, "regression artifact has non-finite coefficients");
            return false;
        }
    }
    if (model.sample_count <= model.coefficients.size()) {
        artifact_json::SetError(
            error, "regression artifact sample count is not sufficient");
        return false;
    }
    for (double metric : {
             model.r_squared,
             model.adjusted_r_squared,
             model.mse,
             model.rmse,
             model.mae,
             model.residual_variance,
             model.residual_standard_error}) {
        if (!std::isfinite(metric)) {
            artifact_json::SetError(
                error, "regression artifact has non-finite metrics");
            return false;
        }
    }
    if (model.mse < 0.0 || model.rmse < 0.0 || model.mae < 0.0 ||
        model.residual_variance < 0.0 ||
        model.residual_standard_error < 0.0) {
        artifact_json::SetError(
            error, "regression artifact has negative error metrics");
        return false;
    }
    return true;
}

} // namespace

bool SaveRegressionModelArtifact(const RegressionModelArtifact& model,
                                 const std::string& path,
                                 std::string* error) {
    if (!Validate(model, error)) {
        return false;
    }
    const artifact_json::Json payload = {
        {"feature_names", model.feature_names},
        {"target_name", model.target_name},
        {"fit_intercept", model.fit_intercept},
        {"degree", model.degree},
        {"coefficients", model.coefficients},
        {"sample_count", model.sample_count},
        {"metrics", {
            {"r_squared", model.r_squared},
            {"adjusted_r_squared", model.adjusted_r_squared},
            {"mse", model.mse},
            {"rmse", model.rmse},
            {"mae", model.mae},
            {"residual_variance", model.residual_variance},
            {"residual_standard_error", model.residual_standard_error},
        }},
    };
    return artifact_json::WriteJsonFile(
        path,
        artifact_json::MakeEnvelope(kFormat, ModelTypeName(model.type), payload),
        error);
}

bool LoadRegressionModelArtifact(const std::string& path,
                                 RegressionModelArtifact& model,
                                 std::string* error) {
    try {
        artifact_json::Json document;
        if (!artifact_json::ReadJsonFile(path, document, error)) {
            return false;
        }
        const std::string model_type = document.value("model_type", "");
        if (model_type != "LinearRegression" &&
            model_type != "PolynomialRegression") {
            artifact_json::SetError(error, "unsupported regression model type");
            return false;
        }
        if (!artifact_json::ValidateEnvelope(
                document, kFormat, model_type, error)) {
            return false;
        }
        const auto& payload = document.at("model");
        RegressionModelArtifact loaded;
        loaded.type = model_type == "PolynomialRegression"
            ? RegressionModelType::Polynomial
            : RegressionModelType::Linear;
        loaded.feature_names =
            payload.at("feature_names").get<std::vector<std::string>>();
        loaded.target_name = payload.at("target_name").get<std::string>();
        loaded.fit_intercept = payload.at("fit_intercept").get<bool>();
        loaded.degree = payload.value("degree", 1);
        loaded.coefficients =
            payload.at("coefficients").get<std::vector<double>>();
        loaded.sample_count = payload.value("sample_count", size_t{0});
        const auto& metrics = payload.at("metrics");
        loaded.r_squared = metrics.value("r_squared", 0.0);
        loaded.adjusted_r_squared =
            metrics.value("adjusted_r_squared", 0.0);
        loaded.mse = metrics.value("mse", 0.0);
        loaded.rmse = metrics.value("rmse", 0.0);
        loaded.mae = metrics.value("mae", 0.0);
        if (metrics.contains("residual_variance") &&
            metrics.contains("residual_standard_error")) {
            loaded.residual_variance =
                metrics.value("residual_variance", 0.0);
            loaded.residual_standard_error =
                metrics.value("residual_standard_error", 0.0);
        } else {
            // Version-1 artifacts written before the metrics-truth fix used
            // mse/rmse for SSE/df_resid. Preserve that statistical value and
            // migrate prediction MSE to its canonical SSE/n definition.
            loaded.residual_variance = loaded.mse;
            loaded.residual_standard_error = loaded.rmse;
            const size_t parameter_count = loaded.coefficients.size();
            if (loaded.sample_count > parameter_count) {
                const double degrees_of_freedom = static_cast<double>(
                    loaded.sample_count - parameter_count);
                loaded.mse = loaded.residual_variance * degrees_of_freedom /
                    static_cast<double>(loaded.sample_count);
                loaded.rmse = std::sqrt(loaded.mse);
            }
        }
        if (!Validate(loaded, error)) {
            return false;
        }
        model = std::move(loaded);
        return true;
    } catch (const std::exception& ex) {
        artifact_json::SetError(error, ex.what());
        return false;
    }
}

} // namespace cyxwiz
