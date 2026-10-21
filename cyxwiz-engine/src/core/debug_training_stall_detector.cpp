#include "debug_training_stall_detector.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string_view>

namespace cyxwiz {

namespace {

constexpr size_t kMinimumLossObservations = 3;
constexpr double kFlatLossRelativeSpan = 1.0e-6;
constexpr double kLearningRateLow = 1.0e-8;
constexpr double kLearningRateHigh = 1.0;
constexpr double kZeroUpdateNorm = 1.0e-12;
constexpr size_t kMinimumClassBalanceRows = 8;
constexpr double kClassMajorityRatio = 0.90;
constexpr double kActivationSaturationRatio = 0.95;

double PayloadNumber(const nlohmann::json& payload,
                     const char* key,
                     double fallback = 0.0) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_number()
        ? it->get<double>()
        : fallback;
}

bool PayloadBool(const nlohmann::json& payload,
                 const char* key,
                 bool fallback = false) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_boolean()
        ? it->get<bool>()
        : fallback;
}

std::string PayloadString(const nlohmann::json& payload,
                          const char* key) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_string()
        ? it->get<std::string>()
        : std::string{};
}

bool Contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

void AddFinding(nlohmann::json& findings,
                const char* cause,
                const char* label,
                const char* status,
                const std::string& detail,
                const std::string& evidence_scope,
                nlohmann::json evidence = nlohmann::json::object()) {
    findings.push_back({
        {"cause", cause},
        {"label", label},
        {"status", status},
        {"detail", detail},
        {"evidence_scope", evidence_scope},
        {"evidence", std::move(evidence)},
    });
}

std::vector<double> CollectLosses(
    const std::vector<DebugTraceRecord>& traces,
    const TrainingTraceSummary& training_trace) {
    std::vector<double> losses;
    for (const auto& trace : traces) {
        if (trace.phase != "SmokeRun.Loss") {
            continue;
        }
        const auto it = trace.payload.find("loss");
        if (it != trace.payload.end() && it->is_number()) {
            const double value = it->get<double>();
            if (std::isfinite(value)) {
                losses.push_back(value);
            }
        }
    }
    if (!losses.empty()) {
        return losses;
    }
    for (const auto& event : training_trace.recent_events) {
        // ComputeLoss is the canonical per-batch observation. Do not also
        // count BatchCallback/EpochComplete copies of the same value as
        // independent evidence for a flat trend.
        if (event.stage == "ComputeLoss" &&
            event.status != "failed" && std::isfinite(event.loss)) {
            losses.push_back(event.loss);
        }
    }
    return losses;
}

void AddLossFinding(nlohmann::json& findings,
                    const std::vector<DebugTraceRecord>& traces,
                    const TrainingTraceSummary& training_trace) {
    const auto losses = CollectLosses(traces, training_trace);
    if (losses.size() < kMinimumLossObservations) {
        AddFinding(findings, "loss_not_changing", "Loss not changing",
                   "unobserved",
                   "At least three finite loss observations are required.",
                   "current_debug_and_linked_training_trace",
                   {{"observation_count", losses.size()},
                    {"minimum_observation_count", kMinimumLossObservations}});
        return;
    }
    const auto [minimum, maximum] = std::minmax_element(
        losses.begin(), losses.end());
    const double scale = std::max(
        1.0, std::max(std::abs(*minimum), std::abs(*maximum)));
    const double relative_span = (*maximum - *minimum) / scale;
    const bool flat = relative_span <= kFlatLossRelativeSpan;
    AddFinding(findings, "loss_not_changing", "Loss not changing",
               flat ? "suspected" : "not_detected",
               flat
                   ? "The bounded finite loss window is effectively flat."
                   : "The bounded finite loss window changed beyond the flat-loss threshold.",
               "bounded_loss_window",
               {{"observation_count", losses.size()},
                {"minimum_loss", *minimum},
                {"maximum_loss", *maximum},
                {"relative_span", relative_span},
                {"flat_relative_span_threshold", kFlatLossRelativeSpan}});
}

void AddGradientFinding(nlohmann::json& findings,
                        const std::vector<DebugTraceRecord>& traces) {
    size_t observed = 0;
    size_t zero = 0;
    for (const auto& trace : traces) {
        if (trace.phase == "SmokeRun.Backward") {
            const size_t count = static_cast<size_t>(std::max(
                0.0, PayloadNumber(trace.payload, "gradient_tensor_count")));
            observed += count;
            zero += static_cast<size_t>(std::max(
                0.0, PayloadNumber(trace.payload,
                                   "zero_gradient_tensor_count")));
        } else if (trace.phase == "Backward" &&
                   trace.role == DebugTraceRole::Gradient &&
                   PayloadBool(trace.payload, "has_gradient", false)) {
            ++observed;
            if (PayloadBool(trace.payload, "is_zero", false)) {
                ++zero;
            }
        }
    }
    if (observed == 0) {
        AddFinding(findings, "gradients_zero", "Gradients zero",
                   "unobserved",
                   "No parameter-gradient norm evidence was captured.",
                   "current_debug_session");
        return;
    }
    const bool all_zero = zero >= observed;
    AddFinding(findings, "gradients_zero", "Gradients zero",
               all_zero ? "suspected" : "not_detected",
               all_zero
                   ? "Every observed parameter gradient norm was zero."
                   : "At least one observed parameter gradient was non-zero.",
               "current_debug_gradient_traces",
               {{"gradient_tensor_count", observed},
                {"zero_gradient_tensor_count", zero}});
}

void AddLearningRateFindings(
    nlohmann::json& findings,
    const DebugTrainingStallConfigEvidence& config) {
    if (!config.learning_rate_available ||
        !std::isfinite(config.learning_rate)) {
        AddFinding(findings, "learning_rate_too_low",
                   "Learning rate too low", "unobserved",
                   "The compiled learning rate was not available.",
                   "compiled_training_configuration");
        AddFinding(findings, "learning_rate_too_high",
                   "Learning rate too high", "unobserved",
                   "The compiled learning rate was not available.",
                   "compiled_training_configuration");
        return;
    }
    AddFinding(findings, "learning_rate_too_low",
               "Learning rate too low",
               config.learning_rate < kLearningRateLow
                   ? "suspected" : "not_detected",
               config.learning_rate < kLearningRateLow
                   ? "The configured learning rate is below the conservative stall threshold."
                   : "The configured learning rate is not below the conservative stall threshold.",
               "compiled_training_configuration",
               {{"learning_rate", config.learning_rate},
                {"low_threshold", kLearningRateLow},
                {"threshold_is_heuristic", true}});
    AddFinding(findings, "learning_rate_too_high",
               "Learning rate too high",
               config.learning_rate > kLearningRateHigh
                   ? "suspected" : "not_detected",
               config.learning_rate > kLearningRateHigh
                   ? "The configured learning rate is above the conservative instability threshold."
                   : "The configured learning rate is not above the conservative instability threshold.",
               "compiled_training_configuration",
               {{"learning_rate", config.learning_rate},
                {"high_threshold", kLearningRateHigh},
                {"threshold_is_heuristic", true}});
}

void AddActivationFinding(nlohmann::json& findings,
                          const std::vector<DebugTraceRecord>& traces) {
    size_t observed = 0;
    size_t saturated = 0;
    double maximum_ratio = 0.0;
    for (const auto& trace : traces) {
        if (trace.role != DebugTraceRole::Activation ||
            !PayloadBool(trace.payload, "saturation_summary_available", false)) {
            continue;
        }
        ++observed;
        const double ratio = PayloadNumber(
            trace.payload, "saturation_candidate_ratio");
        maximum_ratio = std::max(maximum_ratio, ratio);
        if (ratio >= kActivationSaturationRatio) {
            ++saturated;
        }
    }
    if (observed == 0) {
        AddFinding(findings, "saturated_activations",
                   "Saturated activations", "unobserved",
                   "No activation trace carried bounded saturation evidence.",
                   "current_debug_activation_traces");
        return;
    }
    AddFinding(findings, "saturated_activations",
               "Saturated activations",
               saturated > 0 ? "suspected" : "not_detected",
               saturated > 0
                   ? "At least one activation trace exceeded the saturation-candidate threshold."
                   : "Observed activation traces stayed below the saturation-candidate threshold.",
               "local_debug_forward_outputs",
               {{"activation_trace_count", observed},
                {"suspected_activation_count", saturated},
                {"maximum_saturation_candidate_ratio", maximum_ratio},
                {"saturation_ratio_threshold", kActivationSaturationRatio}});
}

const DebugTraceRecord* FindBatchInspection(
    const std::vector<DebugTraceRecord>& traces) {
    const auto it = std::find_if(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.phase == "SmokeRun.BatchInput" &&
                PayloadBool(trace.payload, "batch_inspector", false);
        });
    return it == traces.end() ? nullptr : &*it;
}

void AddClassBalanceFinding(nlohmann::json& findings,
                            const std::vector<DebugTraceRecord>& traces) {
    const auto* trace = FindBatchInspection(traces);
    if (!trace ||
        !PayloadBool(trace->payload, "class_balance_available", false)) {
        AddFinding(findings, "class_imbalance", "Class imbalance",
                   "unobserved",
                   trace
                       ? PayloadString(trace->payload, "class_balance_reason")
                       : "No bounded first-batch class summary was captured.",
                   "first_smoke_batch");
        return;
    }
    const auto counts_it = trace->payload.find("class_counts");
    if (counts_it == trace->payload.end() || !counts_it->is_array()) {
        AddFinding(findings, "class_imbalance", "Class imbalance",
                   "unobserved", "The class summary has no count array.",
                   "first_smoke_batch");
        return;
    }
    size_t total = 0;
    size_t maximum = 0;
    size_t zero_classes = 0;
    for (const auto& row : *counts_it) {
        if (!row.is_object()) {
            continue;
        }
        const size_t count = static_cast<size_t>(std::max(
            0.0, PayloadNumber(row, "count")));
        total += count;
        maximum = std::max(maximum, count);
        if (count == 0) {
            ++zero_classes;
        }
    }
    const double majority_ratio = total > 0
        ? static_cast<double>(maximum) / static_cast<double>(total)
        : 0.0;
    if (total < kMinimumClassBalanceRows) {
        AddFinding(findings, "class_imbalance", "Class imbalance",
                   "unobserved",
                   "The bounded class sample is too small for the heuristic.",
                   "first_smoke_batch",
                   {{"observed_rows", total},
                    {"minimum_rows", kMinimumClassBalanceRows}});
        return;
    }
    const bool imbalanced = zero_classes > 0 ||
        majority_ratio >= kClassMajorityRatio || counts_it->size() < 2;
    AddFinding(findings, "class_imbalance", "Class imbalance",
               imbalanced ? "suspected" : "not_detected",
               imbalanced
                   ? "The bounded first smoke batch has highly uneven class coverage."
                   : "The bounded first smoke batch did not cross the imbalance heuristic.",
               "first_smoke_batch_only",
               {{"observed_rows", total},
                {"observed_class_count", counts_it->size()},
                {"zero_count_class_count", zero_classes},
                {"majority_ratio", majority_ratio},
                {"majority_ratio_threshold", kClassMajorityRatio},
                {"dataset_level_conclusion", false}});
}

void AddMalformedLabelFinding(
    nlohmann::json& findings,
    const std::vector<DebugTraceRecord>& traces) {
    const auto* trace = FindBatchInspection(traces);
    if (!trace) {
        AddFinding(findings, "malformed_labels", "Malformed labels",
                   "unobserved",
                   "No bounded first-batch label inspection was captured.",
                   "first_smoke_batch");
        return;
    }
    const double nulls = PayloadNumber(trace->payload, "label_null_count");
    const double columns = PayloadNumber(trace->payload, "label_column_count");
    const std::string reason =
        PayloadString(trace->payload, "class_balance_reason");
    const bool explicit_bad_reason =
        Contains(reason, "non-finite") ||
        Contains(reason, "not integer class ids") ||
        Contains(reason, "shape is not scalar ids or one-hot classes");
    const bool missing_column =
        PayloadBool(trace->payload, "source_metadata_available", false) &&
        columns <= 0.0;
    const bool malformed = nulls > 0.0 || missing_column || explicit_bad_reason;
    AddFinding(findings, "malformed_labels", "Malformed labels",
               malformed ? "suspected" : "not_detected",
               malformed
                   ? "The bounded label inspection found null, missing, non-finite, non-integer, or incompatible-shape evidence."
                   : "The bounded label inspection found no explicit malformed-label evidence.",
               "first_smoke_batch_only",
               {{"label_null_count", nulls},
                {"label_column_count", columns},
                {"class_balance_reason", reason},
                {"label_values_materialized", false}});
}

void AddBatcherFinding(nlohmann::json& findings,
                       const std::vector<DebugTraceRecord>& traces,
                       const TrainingTraceSummary& training_trace) {
    std::set<std::pair<int, int>> cursors;
    std::map<std::pair<int, int>, size_t> successful_fetch_counts;
    for (const auto& event : training_trace.recent_events) {
        if (event.stage != "GetNextBatch" || event.status == "failed") {
            continue;
        }
        const auto cursor = std::make_pair(event.epoch, event.batch);
        cursors.insert(cursor);
        ++successful_fetch_counts[cursor];
    }
    for (const auto& trace : traces) {
        if (trace.phase != "SmokeRun.BatchInput") {
            continue;
        }
        const int batch = static_cast<int>(
            PayloadNumber(trace.payload, "batch", -1.0));
        if (batch >= 0) {
            cursors.insert({0, batch});
        }
    }
    bool repeated_cursor = false;
    size_t maximum_repeat = 0;
    for (const auto& [cursor, count] : successful_fetch_counts) {
        (void)cursor;
        maximum_repeat = std::max(maximum_repeat, count);
        if (count >= 4) {
            repeated_cursor = true;
        }
    }
    if (repeated_cursor) {
        AddFinding(findings, "batcher_not_advancing",
                   "Batcher not advancing", "suspected",
                   "The training trace repeatedly completed fetch events at one batch cursor.",
                   "linked_training_recent_events",
                   {{"distinct_batch_cursor_count", cursors.size()},
                    {"maximum_same_cursor_fetch_count", maximum_repeat},
                    {"repeat_threshold", 4}});
    } else if (cursors.size() >= 2) {
        AddFinding(findings, "batcher_not_advancing",
                   "Batcher not advancing", "not_detected",
                   "At least two distinct batch cursors were observed.",
                   "current_debug_and_linked_training_trace",
                   {{"distinct_batch_cursor_count", cursors.size()},
                    {"maximum_same_cursor_fetch_count", maximum_repeat}});
    } else {
        AddFinding(findings, "batcher_not_advancing",
                   "Batcher not advancing", "unobserved",
                   "The bounded evidence does not contain enough distinct or repeated fetch events.",
                   "current_debug_and_linked_training_trace",
                   {{"distinct_batch_cursor_count", cursors.size()},
                    {"maximum_same_cursor_fetch_count", maximum_repeat}});
    }
}

void AddOptimizerFinding(nlohmann::json& findings,
                         const std::vector<DebugTraceRecord>& traces) {
    size_t observed = 0;
    size_t non_zero = 0;
    double maximum_norm = 0.0;
    for (const auto& trace : traces) {
        if (trace.phase != "Backward" ||
            trace.role != DebugTraceRole::Gradient ||
            !PayloadBool(trace.payload, "update_observed", false)) {
            continue;
        }
        ++observed;
        const double norm = PayloadNumber(trace.payload, "update_l2_norm");
        maximum_norm = std::max(maximum_norm, norm);
        if (std::isfinite(norm) && norm > kZeroUpdateNorm) {
            ++non_zero;
        }
    }
    if (observed == 0) {
        AddFinding(findings, "optimizer_not_updating_parameters",
                   "Optimizer not updating parameters", "unobserved",
                   "No before/after parameter-update norm evidence was captured.",
                   "local_debug_optimizer_step");
        return;
    }
    const bool no_update = non_zero == 0;
    AddFinding(findings, "optimizer_not_updating_parameters",
               "Optimizer not updating parameters",
               no_update ? "suspected" : "not_detected",
               no_update
                   ? "Every observed parameter update norm was effectively zero."
                   : "At least one observed parameter changed after the optimizer step.",
               "local_debug_optimizer_step",
               {{"observed_parameter_count", observed},
                {"non_zero_update_count", non_zero},
                {"maximum_update_l2_norm", maximum_norm},
                {"zero_update_norm_threshold", kZeroUpdateNorm}});
}

} // namespace

DebugTraceRecord DebugTrainingStallDetector::BuildTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces,
    const TrainingTraceSummary& training_trace,
    const DebugTrainingStallConfigEvidence& config) const {
    nlohmann::json findings = nlohmann::json::array();
    AddLossFinding(findings, traces, training_trace);
    AddGradientFinding(findings, traces);
    AddLearningRateFindings(findings, config);
    AddActivationFinding(findings, traces);
    AddClassBalanceFinding(findings, traces);
    AddMalformedLabelFinding(findings, traces);
    AddBatcherFinding(findings, traces, training_trace);
    AddOptimizerFinding(findings, traces);

    size_t suspected = 0;
    size_t not_detected = 0;
    size_t unobserved = 0;
    for (const auto& finding : findings) {
        const std::string status = PayloadString(finding, "status");
        if (status == "suspected") {
            ++suspected;
        } else if (status == "not_detected") {
            ++not_detected;
        } else {
            ++unobserved;
        }
    }

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id, -1, "Training Stall Analysis", "TrainingDiagnostics",
        "TrainingStallAnalysis",
        suspected > 0 ? DebugTraceRole::Warning
                      : DebugTraceRole::CompileArtifact,
        {}, {}, "diagnostic", "canonical_debug_evidence",
        suspected > 0 ? "warning" : "captured");
    auto& payload = trace.payload;
    payload["training_stall_schema"] = kSchema;
    payload["training_stall_detector"] = true;
    payload["trace_producer"] = "DebugTrainingStallDetector";
    payload["finding_count"] = findings.size();
    payload["suspected_count"] = suspected;
    payload["not_detected_count"] = not_detected;
    payload["unobserved_count"] = unobserved;
    payload["has_suspected_stall_cause"] = suspected > 0;
    payload["canonical_trace_count_scanned"] = traces.size();
    payload["training_event_count_scanned"] =
        training_trace.recent_events.size();
    payload["source_training_run_id"] = training_trace.run_id;
    payload["findings"] = std::move(findings);
    payload["scope_note"] =
        "Findings are bounded heuristics over captured canonical evidence, not proof of dataset-wide or long-run behavior.";
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace, "training_stall_detection", "DebugTrainingStallDetector",
        "cyxwiz-engine/src/core/debug_training_stall_detector.cpp",
        "cyxwiz::DebugTrainingStallDetector::BuildTrace");
    return trace;
}

} // namespace cyxwiz
