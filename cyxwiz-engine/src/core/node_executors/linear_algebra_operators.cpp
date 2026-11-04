#include "linear_algebra_operators.h"

#include <cyxwiz/linear_algebra.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace cyxwiz {

namespace {

bool ReadInt(const std::map<std::string, std::string>& params, const char* key, int fallback, int min, int& out,
             const std::string& op_name, std::string& error) {
    const auto it = params.find(key);
    out = fallback;
    if (it == params.end() || it->second.empty()) return true;
    char* end = nullptr;
    const long v = std::strtol(it->second.c_str(), &end, 10);
    if (!end || *end != '\0') return error = op_name + ": '" + key + "' is not a whole number: " + it->second, false;
    if (v < min) return error = op_name + ": " + key + " must be >= " + std::to_string(min) + " (got " + it->second + ")", false;
    out = static_cast<int>(v);
    return true;
}

// The rows of Vt [k × d] signed so each one's largest-magnitude entry is positive.
void SignComponents(LinearMatrix& vt) {
    for (auto& row : vt) {
        size_t peak = 0;
        for (size_t j = 1; j < row.size(); ++j)
            if (std::fabs(row[j]) > std::fabs(row[peak])) peak = j;
        if (!row.empty() && row[peak] < 0)
            for (double& v : row) v = -v;
    }
}

}  // namespace

// ---- the shared step

bool FittedLinearOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    const auto fc = params.find("feature_cols");
    feature_cols_ = ParseColumnList(fc != params.end() ? fc->second : "");
    const auto lc = params.find("label_col");
    label_col_ = lc != params.end() ? lc->second : "";
    if (!ReadBoolParameter(params, "keep_columns", false, keep_columns_, GetName(), error)) return false;
    if (!ParseFittedPreprocessingOptions(params, GetName(), options_, error)) return false;
    return ConfigureOwn(params, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> FittedLinearOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    const std::string name = GetName();
    if (!input) return arrow::Status::Invalid(name + ": input table is null");
    const auto progress = [&](const char* stage, const std::string& message, float value) {
        if (!progress_callback_) return;
        PipelineOperatorProgress event;
        event.stage = stage;
        event.message = message;
        event.status = "running";
        event.progress = value;
        progress_callback_(event);
    };
    std::vector<std::string> features;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, feature_cols_, label_col_, name, features));
    progress("Reading feature columns", "Reading " + std::to_string(features.size()) + " number columns", 0.1f);
    FeatureRows rows;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(input, features, name, false, rows, GetCancellationQuery()));
    ARROW_RETURN_NOT_OK(CheckCancellation(name));

    LinearMap map;
    int64_t fit_rows = 0;
    if (options_.IsTransformOnly()) {
        FittedPreprocessingState state;
        std::string error;
        if (!LoadFittedPreprocessingState(options_.state_path, name, state, error) ||
            !ValidateFittedPreprocessingConfiguration(state, Settings(), error) || !LinearMapFromState(state, map, error))
            return arrow::Status::Invalid(name + ": " + error);
        if (map.inputs != features)
            return arrow::Status::Invalid(name + ": the feature columns differ from the saved values' columns. Keep the columns used when fitting.");
        fit_rows = state.fit_rows;
    } else {
        // Fitted on the Train rows (all rows when the run marks none).
        const LinearMatrix train = TrainSubset(input, rows);
        if (train.size() < 2) return arrow::Status::Invalid(name + ": fewer than 2 Train rows have every feature");
        progress("Fitting", "Fitting on " + std::to_string(train.size()) + " Train rows", 0.4f);
        ARROW_RETURN_NOT_OK(Fit(train, features, map));
        fit_rows = static_cast<int64_t>(train.size());
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(name));
    progress("Mapping rows", "Mapping " + std::to_string(rows.rows.size()) + " rows on the device", 0.7f);
    LinearMatrix mapped;
    std::string error;
    if (!ApplyLinearMap(map, rows.data, mapped, error)) return arrow::Status::ExecutionError(name + ": " + error);
    if (!options_.IsTransformOnly() && options_.save_state) {
        if (!SaveFittedPreprocessingState(options_.state_path, LinearMapState(map, name, fit_rows, input->schema(), Settings()),
                                          options_.state_overwrite, error))
            return arrow::Status::IOError(name + ": failed to save the fitted values: " + error);
    }
    spdlog::info("{}: {} rows x {} features -> {} columns ({} on {} Train rows)", name, rows.rows.size(), features.size(),
                 map.outputs.size(), options_.IsTransformOnly() ? "saved values fitted" : "fitted", fit_rows);
    progress("Complete", name + " complete", 1.0f);
    return LinearMapTable(input, map, rows, mapped, keep_columns_, label_col_, name);
}

// ---- Truncated SVD

bool TruncatedSVDOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    return ReadInt(params, "n_components", 2, 1, n_components_, GetName(), error);
}

std::map<std::string, std::string> TruncatedSVDOperator::Settings() const { return {{"n_components", std::to_string(n_components_)}}; }

arrow::Status TruncatedSVDOperator::Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) {
    const size_t d = features.size(), k = static_cast<size_t>(n_components_);
    if (k > std::min(train.size(), d))
        return arrow::Status::Invalid("TruncatedSVD: n_components (" + std::to_string(k) + ") is above min(Train rows, features) = " +
                                      std::to_string(std::min(train.size(), d)));
    const SVDResult svd = LinearAlgebra::SVD(train, false);
    if (!svd.success) return arrow::Status::ExecutionError("TruncatedSVD: " + svd.error_message);
    LinearMatrix vt(svd.Vt.begin(), svd.Vt.begin() + static_cast<std::ptrdiff_t>(k));
    SignComponents(vt);
    map.inputs = features;
    map.shift.assign(d, 0.0);
    map.divisor.assign(d, 1.0);
    map.w.assign(d, std::vector<double>(k));
    for (size_t c = 0; c < k; ++c) {
        map.outputs.push_back("svd_" + std::to_string(c));
        for (size_t j = 0; j < d; ++j) map.w[j][c] = vt[c][j];
    }
    return arrow::Status::OK();
}

// ---- Whitening

bool WhiteningOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    const auto m = params.find("method");
    method_ = m == params.end() || m->second.empty() ? "pca" : m->second;
    if (method_ != "pca" && method_ != "zca") return error = "Whitening: method must be 'pca' or 'zca' (got '" + method_ + "')", false;
    if (!ReadInt(params, "n_components", 0, 0, n_components_, GetName(), error)) return false;
    epsilon_ = 0.0;
    if (const auto e = params.find("epsilon"); e != params.end() && !e->second.empty()) {
        char* end = nullptr;
        epsilon_ = std::strtod(e->second.c_str(), &end);
        if (!end || *end != '\0' || !(epsilon_ >= 0.0) || !std::isfinite(epsilon_))
            return error = "Whitening: epsilon must be a number of 0 or more (got '" + e->second + "')", false;
    }
    return true;
}

std::map<std::string, std::string> WhiteningOperator::Settings() const {
    char eps[32];
    std::snprintf(eps, sizeof(eps), "%.17g", epsilon_);
    return {{"method", method_}, {"n_components", std::to_string(n_components_)}, {"epsilon", eps}};
}

arrow::Status WhiteningOperator::Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) {
    const size_t d = features.size(), n = train.size();
    std::vector<double> mean, sd;
    LinearMatrix centred;
    std::string error;
    if (!ColumnMoments(train, mean, sd, error) || !Centred(train, mean, centred, error))
        return arrow::Status::ExecutionError("Whitening: " + error);
    const SVDResult svd = LinearAlgebra::SVD(centred, false);
    if (!svd.success) return arrow::Status::ExecutionError("Whitening: " + svd.error_message);
    LinearMatrix vt = svd.Vt;
    SignComponents(vt);
    // Each direction's variance (divides by n - 1, as PCA's explained variance).
    std::vector<double> scale(svd.S.size());
    const double top = svd.S.empty() ? 0.0 : svd.S[0] * svd.S[0] / static_cast<double>(n - 1);
    const size_t k = method_ == "zca" ? svd.S.size() : (n_components_ == 0 ? svd.S.size() : static_cast<size_t>(n_components_));
    if (k > svd.S.size())
        return arrow::Status::Invalid("Whitening: n_components (" + std::to_string(k) + ") is above min(Train rows, features) = " +
                                      std::to_string(svd.S.size()));
    for (size_t c = 0; c < svd.S.size(); ++c) {
        const double variance = svd.S[c] * svd.S[c] / static_cast<double>(n - 1);
        if (c < k && epsilon_ == 0.0 && !(variance > 1e-12 * std::max(1.0, top)))
            return arrow::Status::Invalid("Whitening: direction " + std::to_string(c + 1) +
                                          " of the Train rows has no variance, so it cannot be scaled to 1: " +
                                          (method_ == "pca" ? "keep fewer components, or " : "") + "set epsilon above 0.");
        scale[c] = 1.0 / std::sqrt(variance + epsilon_);
    }
    if (method_ == "zca" && svd.S.size() < d)
        return arrow::Status::Invalid("Whitening: ZCA needs more Train rows (" + std::to_string(n) + ") than features (" + std::to_string(d) + ")");
    map.inputs = features;
    map.shift = mean;
    map.divisor.assign(d, 1.0);
    if (method_ == "pca") {
        map.w.assign(d, std::vector<double>(k));
        for (size_t c = 0; c < k; ++c) {
            map.outputs.push_back("white_" + std::to_string(c));
            for (size_t j = 0; j < d; ++j) map.w[j][c] = vt[c][j] * scale[c];
        }
        return arrow::Status::OK();
    }
    // ZCA: V diag(scale) V^T, on the device.
    LinearMatrix scaled = vt;  // diag(scale) V^T
    for (size_t c = 0; c < scaled.size(); ++c)
        for (double& v : scaled[c]) v *= scale[c];
    const MatrixResult w = LinearAlgebra::Multiply(LinearAlgebra::Transpose(vt).matrix, scaled);
    if (!w.success) return arrow::Status::ExecutionError("Whitening: " + w.error_message);
    map.w = w.matrix;
    for (const auto& f : features) map.outputs.push_back(f + "_white");
    return arrow::Status::OK();
}

}  // namespace cyxwiz
