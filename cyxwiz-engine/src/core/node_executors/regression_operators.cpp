#include "regression_operators.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include "../data_analyzer.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Read a column as std::vector<double>, rejecting non-numeric.
arrow::Status ReadColumnDouble(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<double>& out) {

    auto col = input->GetColumnByName(col_name);
    if (!col) {
        return arrow::Status::KeyError(
            op_name + ": column '" + col_name + "' not found");
    }
    std::vector<float> floats;
    std::string bad;
    if (!ReadColumnAsFloat(col, floats, bad)) {
        return arrow::Status::TypeError(
            op_name + ": column '" + col_name +
            "' must be numeric (got '" + bad + "')");
    }
    out.assign(floats.begin(), floats.end());
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendPredictionResidual(
    const std::shared_ptr<arrow::Table>& input,
    const std::vector<double>& predicted,
    const std::vector<double>& residuals) {

    const int64_t n = input->num_rows();
    if (static_cast<int64_t>(predicted.size()) != n ||
        static_cast<int64_t>(residuals.size()) != n) {
        return arrow::Status::Invalid(
            "AppendPredictionResidual: predicted/residuals size mismatch "
            "(n=" + std::to_string(n) +
            ", pred=" + std::to_string(predicted.size()) +
            ", resid=" + std::to_string(residuals.size()) + ")");
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::FloatBuilder pred_builder(pool);
    arrow::FloatBuilder resid_builder(pool);
    ARROW_RETURN_NOT_OK(pred_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(resid_builder.Reserve(n));
    for (int64_t i = 0; i < n; ++i) {
        ARROW_RETURN_NOT_OK(pred_builder.Append(static_cast<float>(predicted[i])));
        ARROW_RETURN_NOT_OK(resid_builder.Append(static_cast<float>(residuals[i])));
    }
    std::shared_ptr<arrow::Array> pred_arr, resid_arr;
    ARROW_RETURN_NOT_OK(pred_builder.Finish(&pred_arr));
    ARROW_RETURN_NOT_OK(resid_builder.Finish(&resid_arr));

    auto pred_field = arrow::field("prediction", arrow::float32());
    auto resid_field = arrow::field("residual", arrow::float32());

    ARROW_ASSIGN_OR_RAISE(
        auto with_pred,
        input->AddColumn(input->num_columns(), pred_field,
                          std::make_shared<arrow::ChunkedArray>(pred_arr)));
    return with_pred->AddColumn(
        with_pred->num_columns(), resid_field,
        std::make_shared<arrow::ChunkedArray>(resid_arr));
}

} // namespace

// ============================================================================
// LinearRegressionOperator
// ============================================================================

bool LinearRegressionOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    feature_cols_.clear();
    target_col_.clear();
    fit_intercept_ = true;

    auto fc = params.find("feature_cols");
    if (fc == params.end() || fc->second.empty()) {
        error = "LinearRegression: 'feature_cols' parameter is required "
                "(comma-sep predictor column names)";
        return false;
    }
    ParseCommaList(fc->second, feature_cols_);
    if (feature_cols_.empty()) {
        error = "LinearRegression: 'feature_cols' parsed to empty list";
        return false;
    }

    auto tc = params.find("target_col");
    if (tc == params.end() || tc->second.empty()) {
        error = "LinearRegression: 'target_col' parameter is required";
        return false;
    }
    target_col_ = tc->second;

    auto fi = params.find("fit_intercept");
    if (fi != params.end() && !fi->second.empty()) {
        if (fi->second == "true") {
            fit_intercept_ = true;
        } else if (fi->second == "false") {
            fit_intercept_ = false;
        } else {
            error = "LinearRegression: 'fit_intercept' must be 'true' or "
                    "'false' (got '" + fi->second + "')";
            return false;
        }
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
LinearRegressionOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("LinearRegression: input table is null");
    }

    // Read target.
    std::vector<double> y;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(input, target_col_, GetName(), y));
    const size_t n = y.size();
    if (n == 0) {
        return arrow::Status::Invalid("LinearRegression: target column is empty");
    }

    // Read features.
    const size_t p = feature_cols_.size();
    std::vector<std::vector<double>> feature_cols(p);
    for (size_t f = 0; f < p; ++f) {
        ARROW_RETURN_NOT_OK(
            ReadColumnDouble(input, feature_cols_[f], GetName(), feature_cols[f]));
        if (feature_cols[f].size() != n) {
            return arrow::Status::Invalid(
                "LinearRegression: feature '" + feature_cols_[f] +
                "' has " + std::to_string(feature_cols[f].size()) +
                " rows, target has " + std::to_string(n));
        }
    }

    // Build X matrix [n x (p + intercept)], row-major.
    const size_t n_cols = fit_intercept_ ? p + 1 : p;
    std::vector<std::vector<double>> X(n, std::vector<double>(n_cols, 0.0));
    std::vector<std::string> names;
    names.reserve(n_cols);
    if (fit_intercept_) names.push_back("intercept");
    for (const auto& name : feature_cols_) names.push_back(name);

    for (size_t i = 0; i < n; ++i) {
        size_t col = 0;
        if (fit_intercept_) X[i][col++] = 1.0;
        for (size_t f = 0; f < p; ++f) {
            X[i][col++] = feature_cols[f][i];
        }
    }

    if (n <= n_cols) {
        return arrow::Status::Invalid(
            "LinearRegression: n_samples (" + std::to_string(n) +
            ") <= n_predictors+intercept (" + std::to_string(n_cols) +
            ") — system is underdetermined");
    }

    auto result = DataAnalyzer::MultipleLinearRegression(X, y, names);
    if (result.predicted.size() != n || result.residuals.size() != n) {
        return arrow::Status::ExecutionError(
            "LinearRegression: backend returned empty predictions "
            "(n_samples=" + std::to_string(n) +
            ", predicted=" + std::to_string(result.predicted.size()) +
            ") — likely singular X'X");
    }

    spdlog::info("LinearRegression: n={} predictors={} intercept={} "
                 "R^2={:.4f}, adj_R^2={:.4f}, RMSE={:.4f}",
                 n, p, fit_intercept_,
                 result.r_squared, result.adjusted_r_squared, result.rmse);

    return AppendPredictionResidual(input, result.predicted, result.residuals);
}

// ============================================================================
// PolynomialRegressionOperator
// ============================================================================

bool PolynomialRegressionOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    feature_col_.clear();
    target_col_.clear();
    degree_ = 2;

    auto fc = params.find("feature_col");
    if (fc == params.end() || fc->second.empty()) {
        error = "PolynomialRegression: 'feature_col' parameter is required";
        return false;
    }
    feature_col_ = fc->second;

    auto tc = params.find("target_col");
    if (tc == params.end() || tc->second.empty()) {
        error = "PolynomialRegression: 'target_col' parameter is required";
        return false;
    }
    target_col_ = tc->second;

    auto d = params.find("degree");
    if (d != params.end() && !d->second.empty()) {
        try { degree_ = std::stoi(d->second); }
        catch (...) {
            error = "PolynomialRegression: 'degree' is not a valid integer: " + d->second;
            return false;
        }
    }
    if (degree_ < 1) {
        error = "PolynomialRegression: degree must be >= 1 (got " +
                std::to_string(degree_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
PolynomialRegressionOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("PolynomialRegression: input table is null");
    }

    std::vector<double> x;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(input, feature_col_, GetName(), x));

    std::vector<double> y;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(input, target_col_, GetName(), y));

    if (x.size() != y.size()) {
        return arrow::Status::Invalid(
            "PolynomialRegression: feature_col has " + std::to_string(x.size()) +
            " rows but target_col has " + std::to_string(y.size()));
    }
    if (x.size() < static_cast<size_t>(degree_ + 2)) {
        return arrow::Status::Invalid(
            "PolynomialRegression: need at least degree+2 samples (got " +
            std::to_string(x.size()) + " for degree " + std::to_string(degree_) + ")");
    }

    auto result = DataAnalyzer::PolynomialRegression(x, y, degree_);
    if (result.predicted.size() != x.size() || result.residuals.size() != x.size()) {
        return arrow::Status::ExecutionError(
            "PolynomialRegression: backend returned empty predictions "
            "(n=" + std::to_string(x.size()) +
            ", predicted=" + std::to_string(result.predicted.size()) + ")");
    }

    spdlog::info("PolynomialRegression: n={} degree={} R^2={:.4f}, "
                 "adj_R^2={:.4f}, RMSE={:.4f}",
                 x.size(), degree_,
                 result.r_squared, result.adjusted_r_squared, result.rmse);

    return AppendPredictionResidual(input, result.predicted, result.residuals);
}

} // namespace cyxwiz
