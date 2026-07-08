#include "tree_model_predictor_operator.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "decision_tree_model.h"
#include "feature_matrix_utils.h"
#include "gradient_boosting_model.h"
#include "random_forest_model.h"
#include "tree_classification_utils.h"
#include "tree_model_artifact.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

const char* MaterializationMemoryProgressStatus(
    MaterializationMemoryRisk risk) {
    switch (risk) {
    case MaterializationMemoryRisk::Safe:
        return "running";
    case MaterializationMemoryRisk::Warning:
        return "warning";
    case MaterializationMemoryRisk::Risky:
        return "risky";
    case MaterializationMemoryRisk::Blocked:
        return "blocked";
    }
    return "running";
}

std::string BuildPredictorMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision) {
    std::ostringstream ss;
    ss << "TreeModelPredictor memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", samples=" << estimate.rows
       << ", planned_columns=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: reduce prediction rows or feature columns, predict on a sample first, "
          "or use a future chunked prediction materialization path.";
    return ss.str();
}

arrow::Result<MaterializationMemoryEstimate> EmitPredictorMemoryPreflight(
    const std::shared_ptr<arrow::Table>& input,
    const std::vector<std::string>& resolved_features,
    const PipelineOperatorProgressCallback& callback) {
    const uint64_t planned_samples =
        static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    if (planned_samples == 0) {
        return arrow::Status::Invalid(
            "TreeModelPredictor: input table has no rows");
    }
    if (resolved_features.empty()) {
        return arrow::Status::Invalid(
            "TreeModelPredictor: no numeric feature columns resolved");
    }

    const uint64_t planned_columns =
        static_cast<uint64_t>(resolved_features.size()) + 1ULL;
    const auto estimate = EstimateDenseMaterializationMemory(
        planned_samples, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto decision = EvaluateMaterializationMemory(
        estimate, DetectMaterializationMemorySnapshot());
    const std::string preflight_message =
        BuildPredictorMemoryPreflightMessage(estimate, decision);

    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_samples, planned_columns, planned_cells)) {
        planned_cells = (std::numeric_limits<uint64_t>::max)();
    }

    if (callback) {
        PipelineOperatorProgress event;
        event.stage = "TreeModelPredictor memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryProgressStatus(decision.risk);
        event.progress = 0.40f;
        event.processed_items = 0;
        event.total_items = planned_cells;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        callback(event);
    }
    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }
    return estimate;
}

template <typename ModelT>
arrow::Result<std::shared_ptr<arrow::Table>> PredictWithLoadedModel(
    const std::shared_ptr<arrow::Table>& input,
    const ModelT& model,
    const std::vector<std::string>& configured_features,
    const std::string& prediction_col,
    const std::string& op_name,
    const PipelineOperatorProgressCallback& progress_callback) {
    auto report_progress = [&](std::string stage,
                               std::string message,
                               double progress,
                               uint64_t processed = 0,
                               uint64_t total = 0,
                               uint64_t memory = 0) {
        if (!progress_callback) {
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
        progress_callback(event);
    };

    std::vector<std::string> requested_features =
        configured_features.empty() ? model.FeatureNames() : configured_features;
    if (requested_features.empty()) {
        return arrow::Status::Invalid(
            op_name + ": model artifact has no feature names; set feature_cols");
    }

    report_progress("Resolving features",
                    "Resolving tree model prediction feature columns",
                    0.35,
                    0,
                    static_cast<uint64_t>(requested_features.size()));
    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, requested_features, "", op_name, resolved_features));

    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPredictorMemoryPreflight(
            input, resolved_features, progress_callback));

    report_progress("Reading feature matrix",
                    "Reading tree model prediction feature matrix",
                    0.50,
                    0,
                    static_cast<uint64_t>(resolved_features.size()),
                    preflight_estimate.estimated_peak_bytes);
    std::vector<std::vector<double>> features;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(
        input, resolved_features, op_name, features, n_samples));
    if (n_samples <= 0) {
        return arrow::Status::Invalid(op_name + ": input table has no rows");
    }

    const uint64_t estimated_matrix_bytes =
        preflight_estimate.estimated_peak_bytes;
    report_progress("Predicting rows",
                    "Generating tree model predictions",
                    0.75,
                    0,
                    static_cast<uint64_t>(n_samples),
                    estimated_matrix_bytes);
    const std::vector<int> predictions = model.PredictClasses(features);
    report_progress("Appending predictions",
                    "Appending tree model predictions",
                    0.90,
                    static_cast<uint64_t>(n_samples),
                    static_cast<uint64_t>(n_samples),
                    estimated_matrix_bytes);
    return AppendClassificationPredictions(
        input,
        prediction_col,
        model.ClassLabels(),
        model.HasNumericLabels(),
        predictions,
        op_name);
}

} // namespace

bool TreeModelPredictorOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    feature_cols_.clear();
    model_path_.clear();
    prediction_col_ = "prediction";

    auto model_path = params.find("model_path");
    if (model_path == params.end() || TrimAscii(model_path->second).empty()) {
        error = "TreeModelPredictor: 'model_path' parameter is required";
        return false;
    }
    model_path_ = TrimAscii(model_path->second);

    auto fc = params.find("feature_cols");
    if (fc != params.end()) {
        ParseCommaList(fc->second, feature_cols_);
    }

    auto prediction = params.find("prediction_col");
    if (prediction != params.end() && !TrimAscii(prediction->second).empty()) {
        prediction_col_ = TrimAscii(prediction->second);
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TreeModelPredictorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TreeModelPredictor Materializer");

    if (!input) {
        return arrow::Status::Invalid("TreeModelPredictor: input table is null");
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

    report_progress("Reading artifact",
                    "Reading tree model artifact type",
                    0.05);
    std::string error;
    const std::string model_type =
        ReadTreeModelArtifactType(model_path_, &error);
    if (model_type.empty()) {
        return arrow::Status::IOError(
            "TreeModelPredictor: failed to read model artifact: " + error);
    }

    if (model_type == "DecisionTreeClassifier") {
        report_progress("Loading artifact",
                        "Loading DecisionTree model artifact",
                        0.20);
        DecisionTreeModel model;
        if (!LoadDecisionTreeModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load DecisionTree artifact: " +
                error);
        }
        spdlog::info("TreeModelPredictor: loaded DecisionTree artifact '{}'",
                     model_path_);
        ARROW_ASSIGN_OR_RAISE(auto out, PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName(),
            progress_callback_));
        report_progress("Complete",
                        "TreeModelPredictor materialization complete",
                        1.0,
                        static_cast<uint64_t>(input->num_rows()),
                        static_cast<uint64_t>(input->num_rows()));
        return out;
    }
    if (model_type == "RandomForestClassifier") {
        report_progress("Loading artifact",
                        "Loading RandomForest model artifact",
                        0.20);
        RandomForestModel model;
        if (!LoadRandomForestModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load RandomForest artifact: " +
                error);
        }
        spdlog::info("TreeModelPredictor: loaded RandomForest artifact '{}'",
                     model_path_);
        ARROW_ASSIGN_OR_RAISE(auto out, PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName(),
            progress_callback_));
        report_progress("Complete",
                        "TreeModelPredictor materialization complete",
                        1.0,
                        static_cast<uint64_t>(input->num_rows()),
                        static_cast<uint64_t>(input->num_rows()));
        return out;
    }
    if (model_type == "GradientBoostingClassifier") {
        report_progress("Loading artifact",
                        "Loading GradientBoosting model artifact",
                        0.20);
        GradientBoostingModel model;
        if (!LoadGradientBoostingModelArtifact(model_path_, model, &error)) {
            return arrow::Status::IOError(
                "TreeModelPredictor: failed to load GradientBoosting artifact: " +
                error);
        }
        spdlog::info(
            "TreeModelPredictor: loaded GradientBoosting artifact '{}'",
            model_path_);
        ARROW_ASSIGN_OR_RAISE(auto out, PredictWithLoadedModel(
            input, model, feature_cols_, prediction_col_, GetName(),
            progress_callback_));
        report_progress("Complete",
                        "TreeModelPredictor materialization complete",
                        1.0,
                        static_cast<uint64_t>(input->num_rows()),
                        static_cast<uint64_t>(input->num_rows()));
        return out;
    }

    return arrow::Status::Invalid(
        "TreeModelPredictor: unsupported tree model artifact type '" +
        model_type + "'");
}

} // namespace cyxwiz
