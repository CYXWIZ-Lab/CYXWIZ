#include "regression_operators.h"
#include "../profiler_trace.h"
#include "dense_feature_memory_preflight.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include "../data_analyzer.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    double progress,
                    uint64_t rows_processed = 0,
                    uint64_t total_rows = 0,
                    uint64_t memory_bytes = 0) {
    if (!callback) return;

    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.progress = static_cast<float>(progress);
    event.processed_items = rows_processed;
    event.total_items = total_rows;
    event.estimated_memory_bytes = memory_bytes;
    callback(event);
}

// Read a column as std::vector<double>, rejecting non-numeric.
arrow::Status ReadColumnDouble(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<double>& out,
    const PipelineOperatorCancellationQuery& cancellation_requested) {

    auto col = input->GetColumnByName(col_name);
    if (!col) {
        return arrow::Status::KeyError(
            op_name + ": column '" + col_name + "' not found");
    }
    std::vector<float> floats;
    std::string bad;
    bool cancelled = false;
    if (!ReadColumnAsFloat(
            col, floats, bad, cancellation_requested, &cancelled)) {
        if (cancelled) {
            return arrow::Status::Cancelled(
                op_name + ": materialization cancelled while reading '" +
                col_name + "'");
        }
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
    const std::vector<double>& residuals,
    const PipelineOperatorCancellationQuery& cancellation_requested) {

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
        if ((i & 1023) == 0 && cancellation_requested &&
            cancellation_requested()) {
            return arrow::Status::Cancelled(
                "Regression: materialization cancelled while writing output");
        }
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
    CYXWIZ_PROFILE_ZONE("CyxWiz LinearRegression Materializer");
    if (!input) {
        return arrow::Status::Invalid("LinearRegression: input table is null");
    }

    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, target_col_, GetName(), resolved_features));
    auto target_column = input->GetColumnByName(target_col_);
    if (!target_column) {
        return arrow::Status::KeyError(
            GetName() + ": target column '" + target_col_ + "' not found");
    }
    if (!IsNumericChunked(target_column)) {
        return arrow::Status::TypeError(
            GetName() + ": target column '" + target_col_ +
            "' must be numeric");
    }

    const uint64_t regression_workspace_columns =
        4ULL + (fit_intercept_ ? 1ULL : 0ULL);
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitDenseFeatureMemoryPreflight(
            input,
            resolved_features,
            regression_workspace_columns,
            GetName(),
            "reduce training rows or predictor columns, fit on a sample "
            "first, or use a future chunked regression path",
            GetMaterializationMemoryContext(),
            progress_callback_));
    const uint64_t estimated_memory_bytes =
        preflight_estimate.estimated_peak_bytes;
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    // Read target.
    std::vector<double> y;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(
        input, target_col_, GetName(), y, GetCancellationQuery()));
    const size_t n = y.size();
    if (n == 0) {
        return arrow::Status::Invalid("LinearRegression: target column is empty");
    }

    // Read features.
    const size_t p = feature_cols_.size();
    std::vector<std::vector<double>> feature_cols(p);
    for (size_t f = 0; f < p; ++f) {
        ARROW_RETURN_NOT_OK(
            ReadColumnDouble(
                input, feature_cols_[f], GetName(), feature_cols[f],
                GetCancellationQuery()));
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
        if ((i & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
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

    ReportProgress(progress_callback_, "fit", "Fitting linear regression model", 0.65, static_cast<uint64_t>(n), static_cast<uint64_t>(n), estimated_memory_bytes);
    auto result = DataAnalyzer::MultipleLinearRegression(X, y, names);
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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

    return AppendPredictionResidual(
        input, result.predicted, result.residuals, GetCancellationQuery());
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
    CYXWIZ_PROFILE_ZONE("CyxWiz PolynomialRegression Materializer");
    if (!input) {
        return arrow::Status::Invalid("PolynomialRegression: input table is null");
    }

    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, {feature_col_}, target_col_, GetName(), resolved_features));
    auto target_column = input->GetColumnByName(target_col_);
    if (!target_column) {
        return arrow::Status::KeyError(
            GetName() + ": target column '" + target_col_ + "' not found");
    }
    if (!IsNumericChunked(target_column)) {
        return arrow::Status::TypeError(
            GetName() + ": target column '" + target_col_ +
            "' must be numeric");
    }

    const uint64_t regression_workspace_columns =
        static_cast<uint64_t>(degree_) + 4ULL;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitDenseFeatureMemoryPreflight(
            input,
            resolved_features,
            regression_workspace_columns,
            GetName(),
            "reduce polynomial degree or training rows, fit on a sample "
            "first, or use a future chunked regression path",
            GetMaterializationMemoryContext(),
            progress_callback_));
    const uint64_t estimated_memory_bytes =
        preflight_estimate.estimated_peak_bytes;
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    std::vector<double> x;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(
        input, feature_col_, GetName(), x, GetCancellationQuery()));

    std::vector<double> y;
    ARROW_RETURN_NOT_OK(ReadColumnDouble(
        input, target_col_, GetName(), y, GetCancellationQuery()));

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

    ReportProgress(progress_callback_, "fit", "Fitting polynomial regression model", 0.60, static_cast<uint64_t>(x.size()), static_cast<uint64_t>(x.size()), estimated_memory_bytes);
    auto result = DataAnalyzer::PolynomialRegression(x, y, degree_);
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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

    return AppendPredictionResidual(
        input, result.predicted, result.residuals, GetCancellationQuery());
}

} // namespace cyxwiz
