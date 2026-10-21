#include "regression_model_predictor_operator.h"
#include "feature_matrix_utils.h"
#include "regression_model_artifact.h"

#include <arrow/api.h>
#include <arrow/builder.h>

#include <cmath>
#include <vector>

namespace cyxwiz {
namespace {

std::string Trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

} // namespace

bool RegressionModelPredictorOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    model_path_.clear();
    prediction_col_ = "prediction";
    const auto path = params.find("model_path");
    if (path == params.end() || Trim(path->second).empty()) {
        error = "RegressionModelPredictor: fitted Model input is missing";
        return false;
    }
    model_path_ = Trim(path->second);
    const auto prediction = params.find("prediction_col");
    if (prediction != params.end() && !Trim(prediction->second).empty()) {
        prediction_col_ = Trim(prediction->second);
    }
    return true;
}

bool RegressionModelPredictorOperator::CollectCacheDependencies(
    std::vector<PipelineOperatorCacheDependency>& dependencies,
    std::string& error) const {
    RegressionModelArtifact artifact;
    if (!LoadRegressionModelArtifact(model_path_, artifact, &error)) {
        return false;
    }
    dependencies.push_back({"regression_model", model_path_});
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
RegressionModelPredictorOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid(
            "RegressionModelPredictor: input table is null");
    }
    if (input->GetColumnByName(prediction_col_)) {
        return arrow::Status::Invalid(
            "RegressionModelPredictor: output column already exists: " +
            prediction_col_);
    }

    RegressionModelArtifact artifact;
    std::string error;
    if (!LoadRegressionModelArtifact(model_path_, artifact, &error)) {
        return arrow::Status::IOError(
            "RegressionModelPredictor: failed to load artifact: " + error);
    }

    std::vector<std::vector<float>> features;
    features.reserve(artifact.feature_names.size());
    for (const auto& feature_name : artifact.feature_names) {
        const auto column = input->GetColumnByName(feature_name);
        if (!column) {
            return arrow::Status::KeyError(
                "RegressionModelPredictor: artifact feature '" +
                feature_name + "' is missing from input data");
        }
        std::vector<float> values;
        std::string bad_type;
        bool cancelled = false;
        if (!ReadColumnAsFloat(column, values, bad_type,
                               GetCancellationQuery(), &cancelled)) {
            if (cancelled) {
                return arrow::Status::Cancelled(
                    "RegressionModelPredictor: prediction cancelled");
            }
            return arrow::Status::TypeError(
                "RegressionModelPredictor: feature '" + feature_name +
                "' must be numeric (got '" + bad_type + "')");
        }
        features.push_back(std::move(values));
    }

    const int64_t row_count = input->num_rows();
    arrow::FloatBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(row_count));
    for (int64_t row = 0; row < row_count; ++row) {
        if ((row & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        double prediction = 0.0;
        if (artifact.type == RegressionModelType::Polynomial) {
            const double x = features.front()[static_cast<size_t>(row)];
            double power = 1.0;
            for (double coefficient : artifact.coefficients) {
                prediction += coefficient * power;
                power *= x;
            }
        } else {
            size_t coefficient_index = 0;
            if (artifact.fit_intercept) {
                prediction = artifact.coefficients[coefficient_index++];
            }
            for (size_t feature = 0; feature < features.size(); ++feature) {
                prediction += artifact.coefficients[coefficient_index++] *
                    features[feature][static_cast<size_t>(row)];
            }
        }
        if (!std::isfinite(prediction)) {
            return arrow::Status::Invalid(
                "RegressionModelPredictor: prediction became non-finite");
        }
        ARROW_RETURN_NOT_OK(builder.Append(static_cast<float>(prediction)));
    }

    std::shared_ptr<arrow::Array> predictions;
    ARROW_RETURN_NOT_OK(builder.Finish(&predictions));
    return input->AddColumn(
        input->num_columns(),
        arrow::field(prediction_col_, arrow::float32()),
        std::make_shared<arrow::ChunkedArray>(predictions));
}

} // namespace cyxwiz
