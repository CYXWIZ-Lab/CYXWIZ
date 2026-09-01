#include "debug_slow_path_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace cyxwiz {

namespace {

struct StageTimingAggregate {
    StageTimingAggregate(const char* category_value,
                         const char* label_value,
                         const char* unavailable_reason_value)
        : category(category_value),
          label(label_value),
          unavailable_reason(unavailable_reason_value) {}

    const char* category;
    const char* label;
    const char* unavailable_reason;
    size_t event_count = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    std::string source;
    std::string evidence_scope;

    bool Available() const { return event_count > 0; }

    void Add(double duration_ms,
             const std::string& observed_source,
             const std::string& observed_scope) {
        if (!std::isfinite(duration_ms) || duration_ms <= 0.0) {
            return;
        }
        ++event_count;
        total_ms += duration_ms;
        max_ms = std::max(max_ms, duration_ms);
        if (source.empty()) {
            source = observed_source;
            evidence_scope = observed_scope;
        } else if (source != observed_source) {
            source = "multiple_current_debug_sources";
            evidence_scope = "current_debug_session";
        }
    }
};

constexpr size_t kPreprocessing = 0;
constexpr size_t kBatchFetch = 1;
constexpr size_t kForward = 5;
constexpr size_t kBackward = 6;
constexpr size_t kOptimizer = 7;
constexpr size_t kExportImport = 8;

std::array<StageTimingAggregate, 9> MakeAggregates() {
    return {{
        {"preprocessing_operator", "Preprocessing operators",
         "No timed operator-backed preprocessing trace was captured."},
        {"batch_fetch", "Batch fetch",
         "No timed GetNextBatch or Smoke Run batch-input event was captured."},
        {"batch_creation", "Batch creation",
         "Current GetNextBatch timing combines batch construction and source wait."},
        {"data_wait", "Data wait",
         "Current GetNextBatch timing combines source wait and batch construction."},
        {"native_cpu_fallback", "Native CPU fallback",
         "Fallback events record occurrence and reason, not elapsed duration."},
        {"forward", "Model forward",
         "No timed forward event was captured."},
        {"backward", "Model backward",
         "No timed backward event was captured."},
        {"optimizer_step", "Optimizer step",
         "No timed optimizer-step event was captured."},
        {"export_import", "Export / import",
         "No timed export or import trace was captured."},
    }};
}

bool Contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

double PayloadNumber(const nlohmann::json& payload,
                     const char* key,
                     double fallback = -1.0) {
    const auto it = payload.find(key);
    if (it == payload.end() || !it->is_number()) {
        return fallback;
    }
    return it->get<double>();
}

std::string PayloadString(const nlohmann::json& payload,
                          const char* key) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_string()
        ? it->get<std::string>()
        : std::string{};
}

void AddCurrentTraceTimings(
    std::array<StageTimingAggregate, 9>& stages,
    const std::vector<DebugTraceRecord>& traces) {
    for (const auto& trace : traces) {
        const std::string producer =
            PayloadString(trace.payload, "trace_producer");
        if (producer == "DebugOperatorTraceProducer" &&
            trace.phase == "OperatorTransform") {
            stages[kPreprocessing].Add(
                trace.duration_ms,
                "operator_trace",
                "selected_sample_operator_apply");
        }

        if (trace.phase == "SmokeRun.BatchInput") {
            stages[kBatchFetch].Add(
                trace.duration_ms,
                "smoke_run_trace",
                "combined_batch_creation_and_data_wait");
        }
        if (trace.phase == "SmokeRun.Loss") {
            stages[kForward].Add(
                PayloadNumber(trace.payload, "forward_duration_ms"),
                "smoke_run_trace",
                "bounded_real_data_smoke_batch");
        }
        if (trace.phase == "SmokeRun.Backward") {
            stages[kBackward].Add(
                trace.duration_ms,
                "smoke_run_trace",
                "bounded_real_data_smoke_batch");
            stages[kOptimizer].Add(
                PayloadNumber(trace.payload, "optimizer_step_duration_ms"),
                "smoke_run_trace",
                "bounded_real_data_smoke_batch");
        }

        if (trace.phase != "RuntimeSlowPath" &&
            (Contains(trace.phase, "Export") ||
             Contains(trace.phase, "Import"))) {
            stages[kExportImport].Add(
                trace.duration_ms,
                "canonical_trace",
                "current_debug_session_export_import");
        }
    }
}

void AddLocalTimings(
    std::array<StageTimingAggregate, 9>& stages,
    const DebugSlowPathLocalTimings& timings) {
    if (timings.forward_available) {
        stages[kForward].Add(
            timings.forward_ms,
            "local_debug_result",
            "synthetic_local_debug_step");
    }
    if (timings.backward_available) {
        stages[kBackward].Add(
            timings.backward_ms,
            "local_debug_result",
            "synthetic_local_debug_step");
    }
    if (timings.optimizer_available) {
        stages[kOptimizer].Add(
            timings.optimizer_ms,
            "local_debug_result",
            "synthetic_local_debug_step");
    }
}

void AddTrainingTiming(
    StageTimingAggregate& stage,
    const TrainingTraceEvent& event,
    const TrainingTraceSummary& trace) {
    stage.Add(
        event.duration_ms,
        "training_trace",
        trace.status == "running"
            ? "active_training_recent_events"
            : "latest_training_recent_events");
}

void AddTrainingTimings(
    std::array<StageTimingAggregate, 9>& stages,
    const TrainingTraceSummary& trace) {
    if (!trace.available) {
        return;
    }
    const bool add_batch_fetch = !stages[kBatchFetch].Available();
    const bool add_forward = !stages[kForward].Available();
    const bool add_backward = !stages[kBackward].Available();
    const bool add_optimizer = !stages[kOptimizer].Available();
    for (const auto& event : trace.recent_events) {
        if (add_batch_fetch && event.stage == "GetNextBatch") {
            AddTrainingTiming(
                stages[kBatchFetch], event, trace);
        } else if (add_forward && event.stage == "Forward") {
            AddTrainingTiming(
                stages[kForward], event, trace);
        } else if (add_backward && event.stage == "Backward") {
            AddTrainingTiming(
                stages[kBackward], event, trace);
        } else if (add_optimizer && event.stage == "UpdateParameters") {
            AddTrainingTiming(
                stages[kOptimizer], event, trace);
        }
    }
}

nlohmann::json StageJson(const StageTimingAggregate& stage,
                         uint64_t fallback_count) {
    nlohmann::json value = {
        {"category", stage.category},
        {"label", stage.label},
        {"timing_available", stage.Available()},
        {"event_count", stage.event_count},
    };
    if (stage.Available()) {
        value["total_ms"] = stage.total_ms;
        value["mean_ms"] =
            stage.total_ms / static_cast<double>(stage.event_count);
        value["max_ms"] = stage.max_ms;
        value["source"] = stage.source;
        value["evidence_scope"] = stage.evidence_scope;
    } else {
        value["unavailable_reason"] = stage.unavailable_reason;
    }
    if (std::string_view(stage.category) == "native_cpu_fallback") {
        value["occurrence_count"] = fallback_count;
        value["occurrence_evidence_available"] = true;
    }
    return value;
}

} // namespace

DebugTraceRecord DebugSlowPathDetector::BuildTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces,
    const TrainingTraceSummary& training_trace,
    const DebugSlowPathLocalTimings& local_timings) const {
    auto stages = MakeAggregates();
    AddCurrentTraceTimings(stages, traces);
    AddLocalTimings(stages, local_timings);
    AddTrainingTimings(stages, training_trace);

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "Slow Path Summary",
        "RuntimeProfile",
        "RuntimeSlowPath",
        DebugTraceRole::CompileArtifact,
        {},
        {},
        "timing",
        training_trace.effective_backend.empty()
            ? "observed_host_wall_clock"
            : training_trace.effective_backend,
        "captured");

    auto& payload = trace.payload;
    payload["slow_path_schema"] = kSchema;
    payload["slow_path_detector"] = true;
    payload["trace_producer"] = "DebugSlowPathDetector";
    payload["success"] = true;
    payload["measurement_semantics"] =
        "host_wall_clock_no_forced_arrayfire_sync";
    payload["device_kernel_time_proven"] = false;
    payload["candidate_basis"] = "largest_observed_stage_max_ms";
    payload["candidate_is_regression_verdict"] = false;
    payload["canonical_trace_count_scanned"] = traces.size();
    payload["training_event_count_scanned"] =
        training_trace.recent_events.size();
    payload["source_training_run_id"] = training_trace.run_id;
    payload["native_cpu_fallback_count"] =
        training_trace.native_cpu_fallback_count;

    nlohmann::json stage_json = nlohmann::json::array();
    const StageTimingAggregate* candidate = nullptr;
    size_t available_count = 0;
    for (const auto& stage : stages) {
        stage_json.push_back(StageJson(
            stage, training_trace.native_cpu_fallback_count));
        if (!stage.Available()) {
            continue;
        }
        ++available_count;
        if (!candidate || stage.max_ms > candidate->max_ms) {
            candidate = &stage;
        }
    }
    payload["stage_timings"] = std::move(stage_json);
    payload["timing_category_count"] = stages.size();
    payload["available_timing_category_count"] = available_count;
    payload["unavailable_timing_category_count"] =
        stages.size() - available_count;
    payload["slow_path_candidate_available"] = candidate != nullptr;
    if (candidate) {
        payload["slow_path_candidate_category"] = candidate->category;
        payload["slow_path_candidate_label"] = candidate->label;
        payload["slow_path_candidate_max_ms"] = candidate->max_ms;
        payload["slow_path_candidate_source"] = candidate->source;
        payload["slow_path_candidate_scope"] = candidate->evidence_scope;
    } else {
        payload["slow_path_candidate_reason"] =
            "No measured stage duration was available.";
    }

    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "slow_path_detection",
        "DebugSlowPathDetector",
        "cyxwiz-engine/src/core/debug_slow_path_detector.cpp",
        "cyxwiz::DebugSlowPathDetector::BuildTrace");
    return trace;
}

} // namespace cyxwiz
