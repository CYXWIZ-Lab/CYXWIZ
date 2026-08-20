#include "debug_loss_metric_explainer.h"

#include "classification_decision.h"

#include <algorithm>
#include <optional>
#include <string>

namespace cyxwiz {

namespace {

const char* ReductionName(Reduction reduction) {
    switch (reduction) {
        case Reduction::None: return "none";
        case Reduction::Mean: return "mean";
        case Reduction::Sum: return "sum";
    }
    return "mean";
}

bool IsLossNode(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::MSELoss:
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::BCELoss:
        case gui::NodeType::BCEWithLogits:
        case gui::NodeType::L1Loss:
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
        case gui::NodeType::NLLLoss:
        case gui::NodeType::FocalLoss:
        case gui::NodeType::SoftDiceLoss:
        case gui::NodeType::TverskyLoss:
        case gui::NodeType::JaccardLoss:
        case gui::NodeType::ContrastiveLoss:
        case gui::NodeType::CosineEmbeddingLoss:
        case gui::NodeType::TripletLoss:
            return true;
        default:
            return false;
    }
}

bool IsMetricNode(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::ClassificationMetricsNode:
        case gui::NodeType::RegressionMetricsNode:
        case gui::NodeType::ConfusionMatrixNode:
        case gui::NodeType::ROCCurveNode:
        case gui::NodeType::PRCurveNode:
        case gui::NodeType::TopK:
        case gui::NodeType::ThresholdFilter:
        case gui::NodeType::PairMetrics:
        case gui::NodeType::RetrievalMetrics:
            return true;
        default:
            return false;
    }
}

const char* NodeTypeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::MSELoss: return "MSE Loss";
        case gui::NodeType::CrossEntropyLoss: return "Cross Entropy Loss";
        case gui::NodeType::BCELoss: return "Binary Cross Entropy Loss";
        case gui::NodeType::BCEWithLogits: return "BCE With Logits";
        case gui::NodeType::L1Loss: return "L1 Loss";
        case gui::NodeType::SmoothL1Loss: return "Smooth L1 Loss";
        case gui::NodeType::HuberLoss: return "Huber Loss";
        case gui::NodeType::NLLLoss: return "NLL Loss";
        case gui::NodeType::FocalLoss: return "Focal Loss";
        case gui::NodeType::SoftDiceLoss: return "Soft Dice Loss";
        case gui::NodeType::TverskyLoss: return "Tversky Loss";
        case gui::NodeType::JaccardLoss: return "Jaccard Loss";
        case gui::NodeType::ContrastiveLoss: return "Contrastive Loss";
        case gui::NodeType::CosineEmbeddingLoss: return "Cosine Embedding Loss";
        case gui::NodeType::TripletLoss: return "Triplet Loss";
        case gui::NodeType::ClassificationMetricsNode:
            return "Classification Metrics";
        case gui::NodeType::RegressionMetricsNode:
            return "Regression Metrics";
        case gui::NodeType::ConfusionMatrixNode: return "Confusion Matrix";
        case gui::NodeType::ROCCurveNode: return "ROC Curve";
        case gui::NodeType::PRCurveNode: return "Precision-Recall Curve";
        case gui::NodeType::TopK: return "Top K";
        case gui::NodeType::ThresholdFilter: return "Threshold Filter";
        case gui::NodeType::PairMetrics: return "Pair Metrics";
        case gui::NodeType::RetrievalMetrics: return "Retrieval Metrics";
        default: return "Unknown";
    }
}

nlohmann::json BoundedParameters(
    const std::map<std::string, std::string>& parameters) {
    nlohmann::json out = nlohmann::json::object();
    size_t retained = 0;
    for (const auto& [key, value] : parameters) {
        if (retained >= DebugLossMetricExplainer::kMaxParametersPerRow) {
            break;
        }
        out[key] = value;
        ++retained;
    }
    return out;
}

const DebugTraceRecord* FindObservedLossTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces) {
    const auto it = std::find_if(
        traces.begin(), traces.end(),
        [&run_id](const DebugTraceRecord& trace) {
            return trace.run_id == run_id && trace.phase == "Loss" &&
                   trace.role == DebugTraceRole::Loss &&
                   trace.payload.contains("prediction_shape");
        });
    return it == traces.end() ? nullptr : &*it;
}

std::vector<size_t> ExpectedPredictionShape(
    const TrainingConfiguration& config) {
    std::vector<size_t> shape;
    if (!config.layers.empty()) {
        shape = config.layers.back().output_shape;
    }
    if (shape.empty()) {
        shape.push_back(std::max<size_t>(config.output_size, 1));
    }
    shape.insert(shape.begin(), 1);
    return shape;
}

std::vector<size_t> ExpectedTargetShape(
    const TrainingConfiguration& config,
    const std::vector<size_t>& prediction_shape) {
    switch (config.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::FocalLoss:
        case gui::NodeType::NLLLoss:
            return {1};
        default:
            return prediction_shape;
    }
}

size_t ClassCount(const TrainingConfiguration& config) {
    if (UsesScalarBinaryTargets(config.loss_type)) {
        return 2;
    }
    if (config.preprocessing.num_classes > 0) {
        return config.preprocessing.num_classes;
    }
    return config.output_size;
}

void AttachOptionalFloat(nlohmann::json& row,
                         const char* key,
                         const std::optional<float>& value) {
    row[std::string(key) + "_applicable"] = value.has_value();
    if (value) {
        row[key] = *value;
    }
}

nlohmann::json SelectedLossRow(
    const TrainingConfiguration& config,
    const ResolvedLossConfiguration& resolved,
    const gui::MLNode* node,
    const DebugTraceRecord* observed) {
    const auto expected_prediction = ExpectedPredictionShape(config);
    const auto expected_target = ExpectedTargetShape(
        config, expected_prediction);
    nlohmann::json row = {
        {"kind", "loss"},
        {"source", "compiled_training_configuration"},
        {"selected_for_training", true},
        {"node_id", config.loss_node_id},
        {"node_name", node ? node->name : resolved.loss_name},
        {"node_type", NodeTypeName(config.loss_type)},
        {"evidence_state", observed ? "observed_local_debug" : "configured_only"},
        {"expected_prediction_shape", expected_prediction},
        {"expected_target_shape", expected_target},
        {"expected_prediction_contract",
         UsesScalarBinaryTargets(config.loss_type)
            ? "one scalar probability/logit per sample"
            : (UsesContinuousTargetMetrics(config)
                ? "prediction shape must match the continuous target shape"
                : "one score per class; class-index targets")},
        {"class_count", ClassCount(config)},
        {"output_width", config.output_size},
        {"reduction", ReductionName(resolved.reduction)},
        {"ignore_index_applicable", resolved.ignore_index_applicable},
        {"class_weights_applicable", resolved.class_weights_applicable},
        {"class_weight_count", resolved.class_weights.size()},
        {"label_smoothing_applicable", resolved.label_smoothing_applicable},
        {"label_smoothing", resolved.label_smoothing},
        {"configured_parameters", node
            ? BoundedParameters(node->parameters)
            : BoundedParameters(config.loss_params)},
        {"why_misleading", nlohmann::json::array()},
    };
    if (resolved.ignore_index_applicable) {
        row["ignore_index"] = resolved.ignore_index;
    }
    nlohmann::json weights = nlohmann::json::array();
    for (size_t i = 0;
         i < resolved.class_weights.size() &&
         i < DebugLossMetricExplainer::kMaxClassWeights;
         ++i) {
        weights.push_back(resolved.class_weights[i]);
    }
    row["class_weights"] = std::move(weights);
    row["class_weights_truncated"] =
        resolved.class_weights.size() > DebugLossMetricExplainer::kMaxClassWeights;
    AttachOptionalFloat(row, "pos_weight", resolved.pos_weight);
    AttachOptionalFloat(row, "alpha", resolved.alpha);
    AttachOptionalFloat(row, "beta", resolved.beta);
    AttachOptionalFloat(row, "gamma", resolved.gamma);
    AttachOptionalFloat(row, "smooth", resolved.smooth);

    if (observed) {
        const auto& payload = observed->payload;
        if (payload.contains("prediction_shape")) {
            row["actual_prediction_shape"] = payload["prediction_shape"];
        }
        if (payload.contains("target_shape")) {
            row["actual_target_shape"] = payload["target_shape"];
        }
        const bool compatibility_available = payload.value(
            "prediction_target_shape_check_available", false);
        row["shape_compatibility_observed"] = compatibility_available;
        if (compatibility_available) {
            row["shapes_compatible"] = payload.value(
                "prediction_target_shapes_compatible", false);
            row["shape_reason"] = payload.value(
                "prediction_target_shape_reason", std::string{});
        }
        row["loss_finite_observed"] = payload.contains("loss_finite");
        if (payload.contains("loss_finite")) {
            row["loss_finite"] = payload.value("loss_finite", false);
        }
    } else {
        row["shape_compatibility_observed"] = false;
        row["loss_finite_observed"] = false;
        row["why_misleading"].push_back(
            "Actual shapes and loss numerics were not observed in this run.");
    }
    if (resolved.reduction == Reduction::Sum) {
        row["why_misleading"].push_back(
            "Sum reduction scales with batch or element count, so values are not directly comparable across differently sized batches.");
    } else if (resolved.reduction == Reduction::None) {
        row["why_misleading"].push_back(
            "No reduction returns elementwise losses; a single displayed scalar would not represent the full loss tensor.");
    }
    if (resolved.pos_weight && *resolved.pos_weight != 1.0f) {
        row["why_misleading"].push_back(
            "Positive-class weighting changes the loss scale and does not make accuracy class-balanced.");
    }
    if (!resolved.class_weights.empty()) {
        row["why_misleading"].push_back(
            "Class weighting changes optimization emphasis; compare the class-wise metrics, not loss magnitude alone.");
    }
    if (resolved.label_smoothing > 0.0f) {
        row["why_misleading"].push_back(
            "Label smoothing intentionally prevents hard one-hot targets and changes the minimum attainable loss.");
    }
    return row;
}

nlohmann::json RuntimeMetricRow(const TrainingConfiguration& config) {
    nlohmann::json row = {
        {"kind", "metric_policy"},
        {"source", "training_runtime_policy"},
        {"selected_for_training", true},
        {"node_id", -1},
        {"node_name", "Training Runtime Metrics"},
        {"node_type", "Implicit runtime metric policy"},
        {"evidence_state", "configured_policy"},
        {"class_count", ClassCount(config)},
        {"threshold_applicable", false},
        {"top_k_applicable", false},
        {"why_misleading", nlohmann::json::array({
            "A one-step Local Debug result is not a dataset-wide validation metric.",
            "Accuracy can hide minority-class failures; inspect class-wise metrics and the confusion matrix for imbalanced data."
        })},
    };
    if (UsesContinuousTargetMetrics(config)) {
        row["metrics"] = nlohmann::json::array({"mae", "rmse"});
        row["decision_rule"] = "continuous prediction/target comparison";
        row["why_misleading"][1] =
            "MAE and RMSE depend on target scale; compare them after confirming the target transform and inverse transform.";
        return row;
    }

    row["metrics"] = nlohmann::json::array({"accuracy"});
    const auto mode = ClassificationDecisionModeForLoss(config.loss_type);
    if (mode == ClassificationDecisionMode::BinaryLogit) {
        row["threshold_applicable"] = true;
        row["threshold"] = 0.0f;
        row["threshold_space"] = "logit";
        row["equivalent_probability_threshold"] = 0.5f;
        row["decision_rule"] = "logit >= 0 predicts the positive class";
    } else if (mode == ClassificationDecisionMode::BinaryProbability) {
        row["threshold_applicable"] = true;
        row["threshold"] = 0.5f;
        row["threshold_space"] = "probability";
        row["decision_rule"] = "probability >= 0.5 predicts the positive class";
    } else {
        row["top_k_applicable"] = true;
        row["top_k"] = 1;
        row["decision_rule"] = "argmax over class scores (top-1)";
    }
    return row;
}

nlohmann::json ConfiguredNodeRow(const gui::MLNode& node,
                                 const char* kind,
                                 bool selected) {
    nlohmann::json row = {
        {"kind", kind},
        {"source", "graph_snapshot"},
        {"selected_for_training", selected},
        {"node_id", node.id},
        {"node_name", node.name},
        {"node_type", NodeTypeName(node.type)},
        {"evidence_state", "configured_only"},
        {"configured_parameters", BoundedParameters(node.parameters)},
        {"parameter_count", node.parameters.size()},
        {"parameters_truncated",
         node.parameters.size() > DebugLossMetricExplainer::kMaxParametersPerRow},
        {"actual_shapes_observed", false},
        {"result_observed", false},
        {"why_misleading", nlohmann::json::array()},
    };
    if (const auto it = node.parameters.find("threshold");
        it != node.parameters.end()) {
        row["threshold_configured"] = it->second;
    }
    if (const auto it = node.parameters.find("k");
        it != node.parameters.end()) {
        row["top_k_configured"] = it->second;
    }
    if (kind == std::string("metric")) {
        row["why_misleading"].push_back(
            "This node is configured in the frozen graph, but this training debugger run did not execute its dataset pipeline result.");
    } else if (!selected) {
        row["why_misleading"].push_back(
            "This loss node was not selected by the compiled training path; its settings did not produce the observed Local Debug loss.");
    }
    return row;
}

} // namespace

DebugTraceRecord DebugLossMetricExplainer::BuildTrace(
    const std::string& run_id,
    const TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<DebugTraceRecord>& traces) const {
    const ResolvedLossConfiguration resolved =
        ResolveLossConfiguration(config);
    const auto selected_it = std::find_if(
        nodes.begin(), nodes.end(),
        [&config](const gui::MLNode& node) {
            return node.id == config.loss_node_id;
        });
    const gui::MLNode* selected = selected_it == nodes.end()
        ? nullptr
        : &*selected_it;
    const DebugTraceRecord* observed = FindObservedLossTrace(run_id, traces);

    nlohmann::json rows = nlohmann::json::array();
    rows.push_back(SelectedLossRow(config, resolved, selected, observed));
    rows.push_back(RuntimeMetricRow(config));
    size_t total_rows = 2;

    for (const auto& node : nodes) {
        if (IsLossNode(node.type) && node.id != config.loss_node_id) {
            ++total_rows;
            if (rows.size() < kMaxRows) {
                rows.push_back(ConfiguredNodeRow(node, "loss", false));
            }
        } else if (IsMetricNode(node.type)) {
            ++total_rows;
            if (rows.size() < kMaxRows) {
                rows.push_back(ConfiguredNodeRow(node, "metric", false));
            }
        }
    }

    bool needs_attention = false;
    if (observed) {
        const auto& payload = observed->payload;
        needs_attention =
            (payload.value("prediction_target_shape_check_available", false) &&
             !payload.value("prediction_target_shapes_compatible", false)) ||
            (payload.contains("loss_finite") &&
             !payload.value("loss_finite", false));
    }

    DebugTraceRecord result = DebugNodeTraceContract::Make(
        run_id,
        config.loss_node_id,
        selected ? selected->name : resolved.loss_name,
        NodeTypeName(config.loss_type),
        "LossMetricExplanation",
        DebugTraceRole::Loss,
        {}, {}, "loss_metric_metadata", "canonical_debug_evidence",
        needs_attention ? "needs_attention" : "captured");
    auto& payload = result.payload;
    payload["loss_metric_explanation_schema"] = kSchema;
    payload["trace_producer"] = "DebugLossMetricExplainer";
    payload["observation_scope"] =
        "compiled_loss_contract_plus_same_run_local_debug_evidence";
    payload["row_count"] = total_rows;
    payload["retained_row_count"] = rows.size();
    payload["row_limit"] = kMaxRows;
    payload["rows_truncated"] = total_rows > rows.size();
    payload["selected_loss_node_id"] = config.loss_node_id;
    payload["actual_loss_observed"] = observed != nullptr;
    payload["tensor_reads_added"] = false;
    payload["raw_tensor_values_included"] = false;
    payload["rows"] = std::move(rows);
    payload["scope_note"] =
        "Actual shapes come only from this run's Local Debug loss trace. "
        "Explicit evaluation nodes are configuration evidence unless their "
        "own pipeline execution is observed.";
    DebugNodeTraceContract::AttachDiagnosticContext(
        result,
        "loss_metric_explanation",
        "DebugLossMetricExplainer",
        "cyxwiz-engine/src/core/debug_loss_metric_explainer.cpp",
        "cyxwiz::DebugLossMetricExplainer::BuildTrace");
    return result;
}

} // namespace cyxwiz
