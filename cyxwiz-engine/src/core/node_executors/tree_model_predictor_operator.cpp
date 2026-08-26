#include "tree_model_predictor_operator.h"
#include "../profiler_trace.h"
#include "decision_tree_model.h"
#include "dense_feature_memory_preflight.h"
#include "feature_matrix_utils.h"
#include "gradient_boosting_model.h"
#include "random_forest_model.h"
#include "tree_classification_utils.h"
#include "tree_model_artifact.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

template <typename ModelT>
arrow::Result<std::shared_ptr<arrow::Table>> PredictWithLoadedModel(
    const std::shared_ptr<arrow::Table>& input,
    const ModelT& model,
    const std::vector<std::string>& configured_features,
    const std::string& prediction_col,
    const std::string& op_name,
    const MaterializationMemoryContext& memory_context,
    const PipelineOperatorCancellationQuery& cancellation_requested,
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
        EmitDenseFeatureMemoryPreflight(
            input,
            resolved_features,
            1,
            "TreeModelPredictor",
            "Reduce prediction rows or feature columns, predict on a sample "
            "first, or use a future chunked prediction materialization path.",
            memory_context,
            progress_callback,
            0.40f));
    if (cancellation_requested && cancellation_requested()) {
        return arrow::Status::Cancelled(
            "TreeModelPredictor: materialization cancelled");
    }

    report_progress("Reading feature matrix",
                    "Reading tree model prediction feature matrix",
                    0.50,
                    0,
                    static_cast<uint64_t>(resolved_features.size()),
                    preflight_estimate.estimated_peak_bytes);
    std::vector<std::vector<double>> features;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(
        input, resolved_features, op_name, features, n_samples,
        cancellation_requested));
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
            GetMaterializationMemoryContext(),
            GetCancellationQuery(),
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
            GetMaterializationMemoryContext(),
            GetCancellationQuery(),
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
            GetMaterializationMemoryContext(),
            GetCancellationQuery(),
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
