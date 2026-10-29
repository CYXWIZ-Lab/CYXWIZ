#include "projection_operators.h"

#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "feature_rows.h"

#include <cyxwiz/dimensionality_reduction.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

namespace {

bool ParseBool(const std::map<std::string, std::string>& params, const char* key, bool& out, const std::string& op, std::string& error) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    if (it->second != "true" && it->second != "false") {
        error = op + ": '" + key + "' must be 'true' or 'false' (got '" + it->second + "')";
        return false;
    }
    out = it->second == "true";
    return true;
}

template <typename T>
bool ParseNumber(const std::map<std::string, std::string>& params, const char* key, T& out, const std::string& op, std::string& error) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try {
        size_t used = 0;
        const double v = std::stod(it->second, &used);
        if (used != it->second.size()) throw std::invalid_argument(key);
        out = static_cast<T>(v);
    } catch (...) {
        error = op + ": '" + key + "' is not a number: " + it->second;
        return false;
    }
    return true;
}

}  // namespace

bool ProjectionOperatorBase::ConfigureBase(const std::map<std::string, std::string>& params, std::string& error) {
    feature_cols_.clear();
    label_col_.clear();
    standardise_ = true;
    keep_columns_ = true;
    n_components_ = 2;
    if (const auto it = params.find("feature_cols"); it != params.end()) ParseCommaList(it->second, feature_cols_);
    if (const auto it = params.find("label_col"); it != params.end()) label_col_ = it->second;
    if (!ParseBool(params, "standardise", standardise_, GetName(), error)) return false;
    if (!ParseBool(params, "keep_columns", keep_columns_, GetName(), error)) return false;
    if (!ParseNumber(params, "n_components", n_components_, GetName(), error)) return false;
    if (n_components_ < 2 || n_components_ > 3) {
        error = GetName() + ": n_components must be 2 or 3 (got " + std::to_string(n_components_) + ")";
        return false;
    }
    return true;
}

void ProjectionOperatorBase::Report(const std::string& message, double progress) const {
    if (!progress_callback_) return;
    PipelineOperatorProgress event;
    event.stage = GetName();
    event.message = message;
    event.status = "running";
    event.progress = static_cast<float>(progress);
    progress_callback_(event);
}

arrow::Result<std::shared_ptr<arrow::Table>> ProjectionOperatorBase::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz Projection Materializer");
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");
    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(input, feature_cols_, label_col_, GetName(), resolved));
    if (resolved.size() < 2) return arrow::Status::Invalid(GetName() + ": needs two numeric feature columns or more");

    // Both methods compare every row with every other: refuse what will not fit.
    const uint64_t rows = static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    const auto estimate = EstimateDenseMaterializationMemory(rows, rows, static_cast<uint64_t>(sizeof(double)));
    const auto decision = EvaluateMaterializationMemory(estimate, GetMaterializationMemoryContext());
    if (decision.blocked)
        return arrow::Status::CapacityError(GetName() + ": " + std::to_string(rows) + " rows need about " +
                                            FormatMaterializationBytes(estimate.estimated_peak_bytes) +
                                            " for the distances between rows; " + decision.reason +
                                            ". Sample the rows first (a Row Sampling step) or use PCA.");

    Report("Reading the feature columns", 0.05);
    FeatureRows fit;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(input, resolved, GetName(), standardise_, fit, GetCancellationQuery()));
    if (fit.rows.size() < 3) return arrow::Status::Invalid(GetName() + ": fewer than 3 rows have every feature");
    ARROW_ASSIGN_OR_RAISE(auto points, Embed(fit.data));
    if (points.size() != fit.rows.size()) return arrow::Status::ExecutionError(GetName() + ": the embedding has a row per input row");

    std::vector<std::string> names;
    for (int k = 0; k < n_components_; ++k) names.push_back(Prefix() + std::to_string(k));
    spdlog::info("{}: {} of {} rows x {} features -> {} dimensions{}", GetName(), fit.rows.size(), fit.total, resolved.size(),
                 n_components_, keep_columns_ ? " beside the input" : "");
    Report("Done", 1.0);
    if (keep_columns_) return AppendFittedColumns(input, names, fit.rows, points);
    // The embedding alone: a row per fitted row.
    auto empty = arrow::Table::Make(arrow::schema({}), std::vector<std::shared_ptr<arrow::ChunkedArray>>{}, static_cast<int64_t>(points.size()));
    std::vector<int64_t> all(points.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int64_t>(i);
    return AppendFittedColumns(empty, names, all, points);
}

bool TSNEOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    perplexity_ = 30;
    learning_rate_ = 200.0;
    iterations_ = 1000;
    if (!ConfigureBase(params, error)) return false;
    if (!ParseNumber(params, "perplexity", perplexity_, GetName(), error)) return false;
    if (!ParseNumber(params, "learning_rate", learning_rate_, GetName(), error)) return false;
    if (!ParseNumber(params, "iterations", iterations_, GetName(), error)) return false;
    if (perplexity_ < 1) {
        error = GetName() + ": perplexity must be at least 1";
        return false;
    }
    if (!(learning_rate_ > 0.0)) {
        error = GetName() + ": learning_rate must be above 0";
        return false;
    }
    if (iterations_ < 250) {
        error = GetName() + ": iterations must be at least 250 (the first 250 are the early exaggeration)";
        return false;
    }
    return true;
}

arrow::Result<std::vector<std::vector<double>>> TSNEOperator::Embed(const std::vector<std::vector<double>>& data) {
    if (perplexity_ >= static_cast<int>(data.size()))
        return arrow::Status::Invalid(GetName() + ": perplexity (" + std::to_string(perplexity_) + ") must be below the rows used (" +
                                      std::to_string(data.size()) + ")");
    // The progress callback is the only way back into the loop: it throws to stop it.
    struct Stopped {};
    const auto& cancelled = GetCancellationQuery();
    tSNEResult r;
    try {
        r = DimensionalityReduction::ComputetSNE(data, n_components_, perplexity_, learning_rate_, iterations_, [&](int iteration, double kl) {
            char text[64];
            std::snprintf(text, sizeof(text), "iteration %d, KL %.3f", iteration, kl);
            Report(text, 0.1 + 0.85 * static_cast<double>(iteration) / static_cast<double>(iterations_));
            if (cancelled && cancelled()) throw Stopped{};
        });
    } catch (const Stopped&) {
        return arrow::Status::Cancelled(GetName() + ": cancelled");
    }
    if (!r.success) return arrow::Status::ExecutionError(GetName() + ": " + r.error_message);
    return r.embeddings;
}

bool UMAPOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    n_neighbors_ = 15;
    min_dist_ = 0.1;
    if (!ConfigureBase(params, error)) return false;
    if (!ParseNumber(params, "n_neighbors", n_neighbors_, GetName(), error)) return false;
    if (!ParseNumber(params, "min_dist", min_dist_, GetName(), error)) return false;
    if (n_neighbors_ < 2) {
        error = GetName() + ": n_neighbors must be at least 2";
        return false;
    }
    if (min_dist_ < 0.0) {
        error = GetName() + ": min_dist must be 0 or more";
        return false;
    }
    return true;
}

arrow::Result<std::vector<std::vector<double>>> UMAPOperator::Embed(const std::vector<std::vector<double>>& data) {
    if (n_neighbors_ > static_cast<int>(data.size()))
        return arrow::Status::Invalid(GetName() + ": n_neighbors (" + std::to_string(n_neighbors_) + ") must be at most the rows used (" +
                                      std::to_string(data.size()) + ")");
    Report("Building the neighbour graph and the layout", 0.2);
    const auto r = DimensionalityReduction::ComputeUMAP(data, n_components_, n_neighbors_, min_dist_, 0);
    if (!r.success) return arrow::Status::ExecutionError(GetName() + ": " + r.error_message);
    return r.embeddings;
}

}  // namespace cyxwiz
