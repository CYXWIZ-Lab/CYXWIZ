#include "pca_operator.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "linear_map.h"


#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

std::string BuildPcaMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision,
    uint64_t source_features,
    uint64_t output_components) {
    std::ostringstream ss;
    ss << "PCA memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", rows=" << estimate.rows
       << ", source_features=" << source_features
       << ", output_components=" << output_components
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: reduce feature_cols, source rows, n_components, "
          "or use a future chunked PCA/materialization path.";
    return ss.str();
}

} // namespace

bool PCAOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    feature_cols_.clear();
    label_col_.clear();
    n_components_ = 2;
    center_ = true;
    scale_ = false;
    keep_columns_ = false;

    const auto fc = params.find("feature_cols");
    feature_cols_ = ParseColumnList(fc != params.end() ? fc->second : "");

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto p = params.find("n_components");
    if (p != params.end() && !p->second.empty()) {
        try { n_components_ = std::stoi(p->second); }
        catch (...) {
            error = std::string("PCA: 'n_components' is not a valid integer: ") + p->second;
            return false;
        }
    }
    if (n_components_ < 1) {
        error = "PCA: n_components must be >= 1 (got " +
                std::to_string(n_components_) + ")";
        return false;
    }

    if (!ReadBoolParameter(params, "center", true, center_, GetName(), error)) return false;
    if (!ReadBoolParameter(params, "scale", false, scale_, GetName(), error)) return false;
    if (!ReadBoolParameter(params, "keep_columns", false, keep_columns_, GetName(), error)) return false;
    if (!ParseFittedPreprocessingOptions(params, GetName(), options_, error)) return false;

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
PCAOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz PCA Materializer");

    if (!input) {
        return arrow::Status::Invalid("PCA: input table is null");
    }

    auto report_progress = [&](std::string stage,
                               std::string message,
                               double progress,
                               uint64_t processed = 0,
                               uint64_t total = 0,
                               uint64_t memory = 0) {
        if (!progress_callback_) {
            return;
        }
        PipelineOperatorProgress event;
        event.stage = std::move(stage);
        event.message = std::move(message);
        event.status = "running";
        event.progress = static_cast<float>(progress);
        event.processed_items = processed;
        event.total_items = total;
        event.estimated_memory_bytes = memory;
        progress_callback_(event);
    };

    report_progress("Resolving features",
                    "Resolving numeric feature columns for PCA",
                    0.05);

    // Resolve which columns to use as features. Auto-detect = all
    // numeric columns except label_col and any internal `__*` metadata
    // columns (matches ArrowDatasetBatcher's feature detection).
    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, feature_cols_, label_col_, "PCA", resolved_features));
    report_progress("Features resolved",
                    "Resolved " + std::to_string(resolved_features.size()) +
                    " numeric PCA feature columns",
                    0.15,
                    static_cast<uint64_t>(resolved_features.size()),
                    static_cast<uint64_t>(resolved_features.size()));

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    const uint64_t planned_features =
        static_cast<uint64_t>(resolved_features.size());
    const uint64_t planned_components =
        static_cast<uint64_t>(std::max(1, n_components_));
    const uint64_t planned_columns =
        planned_features + planned_components + (label_col_.empty() ? 0ULL : 1ULL);
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_rows, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message = BuildPcaMemoryPreflightMessage(
        preflight_estimate, preflight_decision,
        planned_features, planned_components);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "PCA memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.17f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    // The rows with every feature, as doubles (rows missing one are left out).
    report_progress("Reading feature columns", "Reading the PCA feature columns", 0.20, 0,
                    static_cast<uint64_t>(resolved_features.size()), preflight_estimate.estimated_peak_bytes);
    const size_t n_features = resolved_features.size();
    NumberColumns fit;
    ARROW_RETURN_NOT_OK(ReadNumberColumns(input, resolved_features, "PCA", fit, GetCancellationQuery()));
    const int64_t n_samples = static_cast<int64_t>(fit.rows.size());
    const uint64_t estimated_input_matrix_bytes = preflight_estimate.estimated_peak_bytes;
    const uint64_t estimated_output_matrix_bytes =
        EstimateDenseMaterializationMemory(static_cast<uint64_t>(n_samples),
                                           static_cast<uint64_t>(n_components_ + (label_col_.empty() ? 0 : 1)),
                                           static_cast<uint64_t>(sizeof(float)))
            .estimated_peak_bytes;
    const std::map<std::string, std::string> settings = {
        {"n_components", std::to_string(n_components_)}, {"center", center_ ? "true" : "false"}, {"scale", scale_ ? "true" : "false"}};
    LinearMap map;
    int64_t fit_rows = 0;
    if (options_.IsTransformOnly()) {
        FittedPreprocessingState state;
        std::string state_error;
        if (!LoadFittedPreprocessingState(options_.state_path, GetName(), state, state_error) ||
            !ValidateFittedPreprocessingConfiguration(state, settings, state_error) ||
            !LinearMapFromState(state, map, state_error))
            return arrow::Status::Invalid("PCA: " + state_error);
        if (map.inputs != resolved_features)
            return arrow::Status::Invalid("PCA: the feature columns differ from the saved values' columns. Keep the columns used when fitting.");
        fit_rows = state.fit_rows;
    } else {
        // Fitted on the Train rows (all rows when the run marks none), as the scalers are.
        const NumberColumns train = TrainColumns(input, fit);
        if (static_cast<int64_t>(train.rows.size()) < std::max(2, n_components_))
            return arrow::Status::Invalid("PCA: Train rows with every feature (" + std::to_string(train.rows.size()) +
                                          ") < n_components (" + std::to_string(n_components_) + ")");
        if (static_cast<int>(n_features) < n_components_)
            return arrow::Status::Invalid("PCA: n_features (" + std::to_string(n_features) + ") < n_components (" +
                                          std::to_string(n_components_) + ")");
        report_progress("Computing PCA", "Computing PCA on the Train rows", 0.65, 0, static_cast<uint64_t>(n_components_),
                        estimated_input_matrix_bytes + estimated_output_matrix_bytes);
        // ComputePCA's model from one summary of the Train rows: the eigenvectors of
        // the n - 1 covariance of the (centred, scaled) columns, signed as its.
        const ColumnSummaryResult summary = LinearAlgebra::ColumnSummary(train.View());
        if (!summary.success) return arrow::Status::ExecutionError("PCA: " + summary.error_message);
        map.inputs = resolved_features;
        map.shift = center_ ? summary.mean : std::vector<double>(n_features, 0.0);
        map.divisor.assign(n_features, 1.0);
        if (scale_)  // ComputePCA's deviation: divides by n, 1 when below 1e-10
            for (size_t j = 0; j < n_features; ++j) map.divisor[j] = summary.sd[j] < 1e-10 ? 1.0 : summary.sd[j];
        const LinearMatrix covariance = SecondMoments(summary, !center_, map.divisor, static_cast<double>(train.rows.size() - 1));
        LinearMatrix vt;
        std::vector<double> variances;
        std::string error;
        if (!TopDirections(covariance, static_cast<size_t>(n_components_), vt, variances, error))
            return arrow::Status::ExecutionError("PCA: " + error);
        double total = 0, explained = 0;
        for (size_t j = 0; j < n_features; ++j) total += covariance[j][j];
        for (const double v : variances) explained += v;
        if (!(total > 0)) return arrow::Status::ExecutionError("PCA: the Train rows have zero variance");
        map.w.assign(n_features, std::vector<double>(static_cast<size_t>(n_components_)));
        for (int c = 0; c < n_components_; ++c) {
            map.outputs.push_back("pc_" + std::to_string(c));
            for (size_t j = 0; j < n_features; ++j) map.w[j][static_cast<size_t>(c)] = vt[static_cast<size_t>(c)][j];
        }
        fit_rows = static_cast<int64_t>(train.rows.size());
        spdlog::info("PCA: fitted on {} Train rows x {} features -> {} components, explained variance {:.2f}%, center={}, scale={}",
                     train.rows.size(), n_features, n_components_, 100.0 * explained / total, center_, scale_);
    }

    // Every row with all its features, through the fitted map on the device.
    std::vector<double> projected;
    std::string error;
    if (!ApplyLinearMap(map, fit, projected, error)) return arrow::Status::ExecutionError("PCA: " + error);
    if (!options_.IsTransformOnly() && options_.save_state) {
        std::string state_error;
        if (!SaveFittedPreprocessingState(options_.state_path, LinearMapState(map, GetName(), fit_rows, input->schema(), settings),
                                          options_.state_overwrite, state_error))
            return arrow::Status::IOError("PCA: failed to save the fitted values: " + state_error);
    }
    report_progress("Building Arrow columns", "Writing the PCA columns", 0.85, 0, static_cast<uint64_t>(n_samples),
                    estimated_output_matrix_bytes);
    ARROW_ASSIGN_OR_RAISE(auto out_table, LinearMapTable(input, map, fit, projected, keep_columns_, label_col_, "PCA"));
    report_progress("Complete", "PCA materialization complete", 1.0, static_cast<uint64_t>(n_samples),
                    static_cast<uint64_t>(n_samples), estimated_output_matrix_bytes);
    return out_table;
}

} // namespace cyxwiz
