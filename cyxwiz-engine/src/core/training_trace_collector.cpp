#include "training_trace_collector.h"
#include "debug_run_paths.h"
#include "algorithms/arrayfire_backend_utils.h"
#include "execution_device_context.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/memory_manager.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <thread>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

std::filesystem::path TraceDir() {
    return GetDebugRunRoot();
}

std::filesystem::path CurrentTracePath() {
    return TraceDir() / "current_training_trace.json";
}

void PopulateMemorySnapshot(TrainingTraceEvent& event) {
    event.cpu_allocated_bytes = static_cast<uint64_t>(MemoryManager::GetAllocatedBytes());
    event.cpu_peak_bytes = static_cast<uint64_t>(MemoryManager::GetPeakBytes());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (cyxwiz::IsInitialized()) {
        try {
            size_t alloc_bytes = 0;
            size_t alloc_buffers = 0;
            size_t lock_bytes = 0;
            size_t lock_buffers = 0;
            af::deviceMemInfo(
                &alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
            event.af_allocated_bytes = static_cast<uint64_t>(alloc_bytes);
            event.af_alloc_buffers = static_cast<uint64_t>(alloc_buffers);
            event.af_locked_bytes = static_cast<uint64_t>(lock_bytes);
            event.af_lock_buffers = static_cast<uint64_t>(lock_buffers);
        } catch (...) {
            // Memory tracing must never affect training.
        }
    }
#endif
}

void PopulateStageExecutionContext(TrainingTraceEvent& event) {
    const auto* context = CurrentExecutionDeviceContext();
    if (!context) {
        return;
    }

    event.stage_backend = context->effective_backend;
    event.stage_device_id = context->effective_device_id;
    event.stage_device_name = context->device_name;
    event.execution_platform = context->platform;
    event.execution_context_id = context->stable_identity;
    event.capability_generation = context->capability_generation;
}

std::string TransferSummaryKey(const std::string& mode,
                               const std::string& reason) {
    const std::string safe_mode = mode.empty() ? "unknown" : mode;
    const std::string safe_reason = reason.empty() ? "unknown" : reason;
    return safe_mode + "/" + safe_reason;
}

std::string ReasonSummaryKey(const std::string& reason) {
    return reason.empty() ? "unknown" : reason;
}

std::string FormatReasonCounts(const std::map<std::string, uint64_t>& counts) {
    if (counts.empty()) {
        return "";
    }

    std::ostringstream out;
    bool first = true;
    for (const auto& [reason, count] : counts) {
        if (!first) {
            out << "; ";
        }
        out << reason << "=" << count;
        first = false;
    }
    return out.str();
}

std::string FormatShape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return "scalar";
    }
    std::ostringstream out;
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            out << 'x';
        }
        out << shape[index];
    }
    return out.str();
}

std::string HostSyncGroupKey(const TrainingTraceHostSyncGroup& group) {
    return group.category + "\n" + group.reason + "\n" +
           group.operation + "\n" +
           FormatShape(group.shape) + "\n" + group.dtype + "\n" +
           group.layout;
}

void AddHostSyncGroup(
    std::map<std::string, TrainingTraceHostSyncGroup>& groups,
    TrainingTraceHostSyncGroup group) {
    const std::string key = HostSyncGroupKey(group);
    auto [it, inserted] = groups.emplace(key, group);
    if (!inserted) {
        it->second.event_count += group.event_count;
        it->second.bytes += group.bytes;
    }
}

std::vector<TrainingTraceHostSyncGroup> HostSyncGroupValues(
    const std::map<std::string, TrainingTraceHostSyncGroup>& groups) {
    std::vector<TrainingTraceHostSyncGroup> values;
    values.reserve(groups.size());
    for (const auto& [key, group] : groups) {
        (void)key;
        values.push_back(group);
    }
    return values;
}

std::string FormatHostSyncGroups(
    const std::vector<TrainingTraceHostSyncGroup>& groups) {
    std::ostringstream out;
    bool first = true;
    for (const auto& group : groups) {
        if (!first) {
            out << "; ";
        }
        out << (group.category.empty() ? "unknown" : group.category)
            << '/' << (group.reason.empty() ? "unknown" : group.reason)
            << '/' << (group.operation.empty() ? "unknown" : group.operation)
            << '/' << FormatShape(group.shape)
            << '/' << (group.dtype.empty() ? "unknown" : group.dtype)
            << '/' << (group.layout.empty() ? "unknown" : group.layout)
            << "=" << group.event_count << " events," << group.bytes
            << " bytes";
        first = false;
    }
    return out.str();
}

nlohmann::json HostSyncGroupToJson(
    const TrainingTraceHostSyncGroup& group) {
    return {
        {"category", group.category},
        {"reason", group.reason},
        {"operation", group.operation},
        {"shape", group.shape},
        {"dtype", group.dtype},
        {"layout", group.layout},
        {"event_count", group.event_count},
        {"bytes", group.bytes}
    };
}

TrainingTraceHostSyncGroup HostSyncGroupFromJson(
    const nlohmann::json& j) {
    TrainingTraceHostSyncGroup group;
    group.category = j.value("category", "unknown");
    group.reason = j.value("reason", "");
    group.operation = j.value("operation", "");
    group.shape = j.value("shape", std::vector<size_t>{});
    group.dtype = j.value("dtype", "");
    group.layout = j.value("layout", "");
    group.event_count = j.value("event_count", uint64_t{0});
    group.bytes = j.value("bytes", uint64_t{0});
    return group;
}

nlohmann::json EventToJson(const TrainingTraceEvent& event) {
    return {
        {"timestamp", event.timestamp},
        {"run_id", event.run_id},
        {"stage", event.stage},
        {"thread_id", event.thread_id},
        {"epoch", event.epoch},
        {"batch", event.batch},
        {"total_batches", event.total_batches},
        {"loss", event.loss},
        {"accuracy", event.accuracy},
        {"validation_loss", event.validation_loss},
        {"validation_accuracy", event.validation_accuracy},
        {"duration_ms", event.duration_ms},
        {"cpu_allocated_bytes", event.cpu_allocated_bytes},
        {"cpu_peak_bytes", event.cpu_peak_bytes},
        {"af_allocated_bytes", event.af_allocated_bytes},
        {"af_locked_bytes", event.af_locked_bytes},
        {"af_alloc_buffers", event.af_alloc_buffers},
        {"af_lock_buffers", event.af_lock_buffers},
        {"status", event.status},
        {"message", event.message},
        {"metric_scope", event.metric_scope},
        {"checkpoint_path", event.checkpoint_path},
        {"is_best_checkpoint", event.is_best_checkpoint},
        {"terminal_reason", event.terminal_reason},
        {"task_id", event.task_id},
        {"task_name", event.task_name},
        {"task_stage", event.task_stage},
        {"task_progress", event.task_progress},
        {"node_id", event.node_id},
        {"node_name", event.node_name},
        {"estimated_memory_bytes", event.estimated_memory_bytes},
        {"memory_risk_level", event.memory_risk_level},
        {"processed_items", event.processed_items},
        {"total_items", event.total_items},
        {"pin_memory_requested", event.pin_memory_requested},
        {"transfer_mode", event.transfer_mode},
        {"transfer_reason", event.transfer_reason},
        {"transfer_backend", event.transfer_backend},
        {"transfer_batch_size", event.transfer_batch_size},
        {"compute_backend", event.compute_backend},
        {"stage_backend", event.stage_backend},
        {"stage_device_id", event.stage_device_id},
        {"stage_device_name", event.stage_device_name},
        {"requested_backend", event.requested_backend},
        {"requested_device_id", event.requested_device_id},
        {"effective_backend", event.effective_backend},
        {"effective_device_id", event.effective_device_id},
        {"effective_device_name", event.effective_device_name},
        {"execution_platform", event.execution_platform},
        {"execution_context_id", event.execution_context_id},
        {"capability_generation", event.capability_generation},
        {"fallback_target", event.fallback_target},
        {"fallback_operation", event.fallback_operation},
        {"fallback_reason", event.fallback_reason},
        {"fallback_policy", event.fallback_policy},
        {"native_cpu_fallback", event.native_cpu_fallback},
        {"arrayfire_host_sync_bytes", event.arrayfire_host_sync_bytes},
        {"arrayfire_host_sync_reason", event.arrayfire_host_sync_reason},
        {"arrayfire_host_sync_category", event.arrayfire_host_sync_category},
        {"arrayfire_host_sync_operation", event.arrayfire_host_sync_operation},
        {"arrayfire_host_sync_shape", event.arrayfire_host_sync_shape},
        {"arrayfire_host_sync_dtype", event.arrayfire_host_sync_dtype},
        {"arrayfire_host_sync_layout", event.arrayfire_host_sync_layout},
        {"placement_fingerprint", event.placement_fingerprint},
        {"placement_entry_count", event.placement_entry_count},
        {"placement_summary", event.placement_summary}
    };
}

TrainingTraceEvent EventFromJson(const nlohmann::json& j) {
    TrainingTraceEvent event;
    event.timestamp = j.value("timestamp", "");
    event.run_id = j.value("run_id", "");
    event.stage = j.value("stage", "");
    event.thread_id = j.value("thread_id", "");
    event.epoch = j.value("epoch", 0);
    event.batch = j.value("batch", 0);
    event.total_batches = j.value("total_batches", 0);
    event.loss = j.value("loss", 0.0f);
    event.accuracy = j.value("accuracy", 0.0f);
    event.validation_loss = j.value("validation_loss", 0.0f);
    event.validation_accuracy = j.value("validation_accuracy", 0.0f);
    event.duration_ms = j.value("duration_ms", 0.0f);
    event.cpu_allocated_bytes = j.value("cpu_allocated_bytes", uint64_t{0});
    event.cpu_peak_bytes = j.value("cpu_peak_bytes", uint64_t{0});
    event.af_allocated_bytes = j.value("af_allocated_bytes", uint64_t{0});
    event.af_locked_bytes = j.value("af_locked_bytes", uint64_t{0});
    event.af_alloc_buffers = j.value("af_alloc_buffers", uint64_t{0});
    event.af_lock_buffers = j.value("af_lock_buffers", uint64_t{0});
    event.status = j.value("status", "ok");
    event.message = j.value("message", "");
    event.metric_scope = j.value("metric_scope", "");
    event.checkpoint_path = j.value("checkpoint_path", "");
    event.is_best_checkpoint = j.value("is_best_checkpoint", false);
    event.terminal_reason = j.value("terminal_reason", "");
    event.task_id = j.value("task_id", uint64_t{0});
    event.task_name = j.value("task_name", "");
    event.task_stage = j.value("task_stage", "");
    event.task_progress = j.value("task_progress", 0.0f);
    event.node_id = j.value("node_id", -1);
    event.node_name = j.value("node_name", "");
    event.estimated_memory_bytes = j.value("estimated_memory_bytes", uint64_t{0});
    event.memory_risk_level = j.value("memory_risk_level", "");
    event.processed_items = j.value("processed_items", uint64_t{0});
    event.total_items = j.value("total_items", uint64_t{0});
    event.pin_memory_requested = j.value("pin_memory_requested", false);
    event.transfer_mode = j.value("transfer_mode", "");
    event.transfer_reason = j.value("transfer_reason", "");
    event.transfer_backend = j.value("transfer_backend", "");
    event.transfer_batch_size = j.value("transfer_batch_size", 0);
    event.compute_backend = j.value("compute_backend", "");
    event.stage_backend = j.value("stage_backend", "");
    event.stage_device_id = j.value("stage_device_id", 0);
    event.stage_device_name = j.value("stage_device_name", "");
    event.requested_backend = j.value("requested_backend", "");
    event.requested_device_id = j.value("requested_device_id", 0);
    event.effective_backend = j.value("effective_backend", "");
    event.effective_device_id = j.value("effective_device_id", 0);
    event.effective_device_name = j.value("effective_device_name", "");
    event.execution_platform = j.value("execution_platform", "");
    event.execution_context_id = j.value("execution_context_id", "");
    event.capability_generation =
        j.value("capability_generation", uint64_t{0});
    event.fallback_target = j.value("fallback_target", "");
    event.fallback_operation = j.value("fallback_operation", "");
    event.fallback_reason = j.value("fallback_reason", "");
    event.fallback_policy = j.value("fallback_policy", "");
    event.native_cpu_fallback = j.value("native_cpu_fallback", false);
    event.arrayfire_host_sync_bytes =
        j.value("arrayfire_host_sync_bytes", uint64_t{0});
    event.arrayfire_host_sync_reason =
        j.value("arrayfire_host_sync_reason", "");
    event.arrayfire_host_sync_category =
        j.value("arrayfire_host_sync_category", "");
    event.arrayfire_host_sync_operation =
        j.value("arrayfire_host_sync_operation", "");
    event.arrayfire_host_sync_shape =
        j.value("arrayfire_host_sync_shape", std::vector<size_t>{});
    event.arrayfire_host_sync_dtype =
        j.value("arrayfire_host_sync_dtype", "");
    event.arrayfire_host_sync_layout =
        j.value("arrayfire_host_sync_layout", "");
    event.placement_fingerprint = j.value("placement_fingerprint", "");
    event.placement_entry_count =
        j.value("placement_entry_count", uint64_t{0});
    event.placement_summary = j.value("placement_summary", "");
    return event;
}

bool IsTerminalStatus(const std::string& status) {
    return status == "completed" ||
           status == "early_stopped" ||
           status == "cancelled" ||
           status == "failed";
}

bool IsSuccessfulTerminalStatus(const std::string& status) {
    return status == "completed" || status == "early_stopped";
}

void PopulateRunLevelTraceSummary(TrainingTraceSummary& summary) {
    bool saw_context_bind = !summary.execution_context_id.empty();
    bool saw_placement_plan = !summary.placement_fingerprint.empty();
    const bool derive_fallback_count =
        summary.native_cpu_fallback_count == 0;
    const bool derive_transfer_counts =
        summary.transfer_event_count == 0 &&
        summary.transfer_known_bytes == 0;
    const bool derive_synchronization_counts =
        summary.synchronization_event_count == 0 &&
        summary.synchronization_known_bytes == 0;
    const bool derive_output_boundary_count =
        summary.declared_output_boundary_count == 0;
    const bool derive_host_sync =
        summary.arrayfire_host_sync_count == 0 &&
        summary.arrayfire_host_sync_bytes == 0;
    uint64_t derived_fallback_count = 0;
    uint64_t derived_transfer_event_count = 0;
    uint64_t derived_transfer_known_bytes = 0;
    uint64_t derived_synchronization_event_count = 0;
    uint64_t derived_synchronization_known_bytes = 0;
    uint64_t derived_output_boundary_count = 0;
    uint64_t derived_host_sync_count = 0;
    uint64_t derived_host_sync_bytes = 0;
    std::map<std::string, uint64_t> derived_transfer_reasons;
    std::map<std::string, uint64_t> derived_synchronization_reasons;
    std::map<std::string, TrainingTraceHostSyncGroup>
        derived_host_sync_groups;
    for (const auto& event : summary.recent_events) {
        if (event.native_cpu_fallback) {
            ++derived_fallback_count;
        }
        if (!event.transfer_mode.empty()) {
            ++derived_transfer_event_count;
            derived_transfer_known_bytes += event.arrayfire_host_sync_bytes;
            ++derived_transfer_reasons[
                TransferSummaryKey(event.transfer_mode,
                                   event.transfer_reason)];
        }
        if (event.stage == "ArrayFire.HostSync" ||
            event.arrayfire_host_sync_bytes > 0) {
            ++derived_synchronization_event_count;
            derived_synchronization_known_bytes +=
                event.arrayfire_host_sync_bytes;
            ++derived_synchronization_reasons[
                ReasonSummaryKey(event.arrayfire_host_sync_reason.empty()
                                     ? event.transfer_reason
                                     : event.arrayfire_host_sync_reason)];
        }
        if (event.stage == "TrainingExecutor.OutputBoundary") {
            ++derived_output_boundary_count;
        }
        if (event.stage == "ArrayFire.HostSync" ||
            event.arrayfire_host_sync_bytes > 0) {
            ++derived_host_sync_count;
            derived_host_sync_bytes += event.arrayfire_host_sync_bytes;
            TrainingTraceHostSyncGroup group;
            group.category = event.arrayfire_host_sync_category.empty()
                ? "unknown"
                : event.arrayfire_host_sync_category;
            group.reason = event.arrayfire_host_sync_reason;
            group.operation = event.arrayfire_host_sync_operation;
            group.shape = event.arrayfire_host_sync_shape;
            group.dtype = event.arrayfire_host_sync_dtype;
            group.layout = event.arrayfire_host_sync_layout;
            group.event_count = 1;
            group.bytes = event.arrayfire_host_sync_bytes;
            AddHostSyncGroup(derived_host_sync_groups, std::move(group));
        }
        if (!saw_context_bind &&
            event.stage == "ExecutionDeviceContext.Bind") {
            saw_context_bind = true;
            summary.execution_platform = event.execution_platform;
            summary.requested_backend = event.requested_backend;
            summary.requested_device_id = event.requested_device_id;
            summary.effective_backend = event.effective_backend;
            summary.effective_device_id = event.effective_device_id;
            summary.effective_device_name = event.effective_device_name;
            summary.execution_context_id = event.execution_context_id;
            summary.fallback_policy = event.fallback_policy;
        }
        if (!saw_placement_plan &&
            event.stage == "TrainingExecutor.PlacementPlan") {
            saw_placement_plan = true;
            summary.placement_fingerprint = event.placement_fingerprint;
            summary.placement_entry_count = event.placement_entry_count;
            summary.placement_summary = event.placement_summary;
        }
    }
    if (derive_fallback_count) {
        summary.native_cpu_fallback_count = derived_fallback_count;
    }
    if (derive_transfer_counts) {
        summary.transfer_event_count = derived_transfer_event_count;
        summary.transfer_known_bytes = derived_transfer_known_bytes;
    }
    if (summary.transfer_summary.empty()) {
        summary.transfer_summary =
            FormatReasonCounts(derived_transfer_reasons);
    }
    if (derive_synchronization_counts) {
        summary.synchronization_event_count =
            derived_synchronization_event_count;
        summary.synchronization_known_bytes =
            derived_synchronization_known_bytes;
    }
    if (summary.synchronization_summary.empty()) {
        summary.synchronization_summary =
            FormatReasonCounts(derived_synchronization_reasons);
    }
    if (derive_output_boundary_count) {
        summary.declared_output_boundary_count =
            derived_output_boundary_count;
    }
    if (derive_host_sync) {
        summary.arrayfire_host_sync_count = derived_host_sync_count;
        summary.arrayfire_host_sync_bytes = derived_host_sync_bytes;
    }
    if (summary.arrayfire_host_sync_groups.empty()) {
        summary.arrayfire_host_sync_groups =
            HostSyncGroupValues(derived_host_sync_groups);
    }
    if (summary.arrayfire_host_sync_summary.empty()) {
        summary.arrayfire_host_sync_summary =
            FormatHostSyncGroups(summary.arrayfire_host_sync_groups);
    }

    if (!summary.available) {
        summary.residency_verdict = "unavailable";
        summary.residency_reason = "No training trace is available.";
        return;
    }
    if (!saw_context_bind) {
        summary.residency_verdict = "missing_execution_context";
        summary.residency_reason =
            "No execution device context bind event was recorded.";
        return;
    }
    if (summary.native_cpu_fallback_count > 0) {
        summary.residency_verdict = "native_cpu_fallback_observed";
        summary.residency_reason =
            "Native CPU fallback events were recorded for this run.";
        return;
    }
    if (!IsTerminalStatus(summary.status)) {
        summary.residency_verdict = "in_progress";
        summary.residency_reason =
            "Run has not reached a terminal state yet.";
        return;
    }
    if (!IsSuccessfulTerminalStatus(summary.status)) {
        summary.residency_verdict = "terminal_without_residency_pass";
        summary.residency_reason =
            "Run ended without native CPU fallback, but did not complete "
            "successfully.";
        return;
    }
    if (summary.fallback_policy == "forbid_native_cpu_fallback" &&
        summary.execution_platform == "arrayfire") {
        summary.residency_verdict =
            "strict_arrayfire_declared_boundaries";
        summary.residency_reason =
            "Strict ArrayFire run completed with zero native CPU fallback "
            "events; declared host output boundaries are recorded separately.";
        return;
    }

    summary.residency_verdict = "compatibility_mode_no_fallback_observed";
    summary.residency_reason =
        "Run completed without recorded native CPU fallback, but native CPU "
        "fallback was allowed by policy.";
}

} // namespace

TrainingTraceCollector& TrainingTraceCollector::Instance() {
    static TrainingTraceCollector collector;
    return collector;
}

void TrainingTraceCollector::StartRun(const std::string& run_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    run_id_ = run_id;
    status_ = "running";
    events_.clear();
    materialization_events_.clear();
    warnings_.clear();
    execution_context_event_ = TrainingTraceEvent{};
    has_execution_context_event_ = false;
    placement_plan_event_ = TrainingTraceEvent{};
    has_placement_plan_event_ = false;
    native_cpu_fallback_count_ = 0;
    transfer_event_count_ = 0;
    transfer_known_bytes_ = 0;
    transfer_reason_counts_.clear();
    synchronization_event_count_ = 0;
    synchronization_known_bytes_ = 0;
    synchronization_reason_counts_.clear();
    arrayfire_host_sync_count_ = 0;
    arrayfire_host_sync_bytes_ = 0;
    arrayfire_host_sync_groups_.clear();
    declared_output_boundary_count_ = 0;
    events_since_write_ = 0;
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

void TrainingTraceCollector::RecordStage(TrainingTraceStage stage,
                                         int epoch,
                                         int batch,
                                         int total_batches,
                                         float loss,
                                         float accuracy,
                                         float duration_ms,
                                         const std::string& status,
                                         const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = CrashRunRecorder::StageName(stage);
    event.thread_id = ThreadIdString();
    event.epoch = epoch;
    event.batch = batch;
    event.total_batches = total_batches;
    event.loss = loss;
    event.accuracy = accuracy;
    event.duration_ms = duration_ms;
    event.status = status;
    event.message = message;
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    if (event.stage == "TrainingExecutor.OutputBoundary") {
        ++declared_output_boundary_count_;
    }
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }
    if (event.node_id >= 0) {
        materialization_events_.push_back(event);
        while (materialization_events_.size() > settings_.max_recent_events) {
            materialization_events_.pop_front();
        }
    }

    if (status != "ok" && !message.empty()) {
        warnings_.push_back(message);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    events_since_write_++;
    const int write_interval = std::max(1, settings_.persist_every_n_events);
    if (settings_.persist_enabled &&
        (events_since_write_ >= static_cast<size_t>(write_interval) ||
         status != "ok")) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordRuntimeWarning(const std::string& source,
                                                  const std::string& message) {
    RecordRuntimeEvent(source, message, "warning");
}

void TrainingTraceCollector::RecordRuntimeEvent(const std::string& stage,
                                                const std::string& message,
                                                const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    if (status != "ok") {
        std::string warning = stage;
        if (!warning.empty() && !message.empty()) {
            warning += ": ";
        }
        warning += message;
        warnings_.push_back(warning);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = stage.empty() ? "RuntimeEvent" : stage;
    event.thread_id = ThreadIdString();
    event.status = status.empty() ? "ok" : status;
    event.message = message;
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
    }
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordPinMemoryTransferStatus(
    const PinMemoryTransferStatus& status,
    const std::string& message,
    const std::string& severity) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    const std::string event_status = severity.empty() ? "ok" : severity;
    if (event_status != "ok") {
        std::string warning = "DataLoader.PinMemoryTransfer";
        if (!message.empty()) {
            warning += ": " + message;
        }
        warnings_.push_back(warning);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "DataLoader.PinMemoryTransfer";
    event.thread_id = ThreadIdString();
    event.status = event_status;
    event.message = message;
    event.node_id = status.node_id;
    event.node_name = status.node_name;
    event.pin_memory_requested = status.requested;
    event.transfer_mode = status.effective_mode;
    event.transfer_reason = status.reason_code;
    event.transfer_backend = status.backend;
    event.transfer_batch_size = status.batch_size;
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
    }
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    ++transfer_event_count_;
    ++transfer_reason_counts_[
        TransferSummaryKey(event.transfer_mode, event.transfer_reason)];
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordNativeCpuFallback(
    const ArrayFireNativeCpuFallbackEvent& fallback) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "ArrayFire.NativeCpuFallback";
    event.thread_id = ThreadIdString();
    event.status = fallback.fallback_forbidden ? "error" : "warning";
    event.compute_backend = fallback.selected_backend;
    event.fallback_target = "native_cpu";
    event.fallback_operation = fallback.operation_name;
    event.fallback_reason = fallback.reason_code;
    event.fallback_policy = fallback.fallback_forbidden
        ? "forbid_native_cpu_fallback"
        : "allow_native_cpu_fallback";
    event.native_cpu_fallback = true;
    event.message = "ArrayFire operation '" + fallback.operation_name +
        "' on backend '" + fallback.selected_backend +
        "' attempted native CPU fallback";
    if (!fallback.reason_code.empty()) {
        event.message += " (reason=" + fallback.reason_code + ")";
    }
    if (!fallback.context.empty()) {
        event.message += ". Context: " + fallback.context;
    }
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
        event.validation_loss = latest.validation_loss;
        event.validation_accuracy = latest.validation_accuracy;
    }
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    ++native_cpu_fallback_count_;
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    warnings_.push_back(event.message);
    if (warnings_.size() > 50) {
        warnings_.erase(warnings_.begin());
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordArrayFireHostSync(
    const ArrayFireHostSyncEvent& sync) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "ArrayFire.HostSync";
    event.thread_id = ThreadIdString();
    event.status = "ok";
    event.compute_backend = sync.selected_backend;
    event.transfer_mode = "arrayfire_to_host";
    event.transfer_reason = sync.reason_code;
    event.transfer_backend = sync.selected_backend;
    event.arrayfire_host_sync_bytes = sync.bytes;
    event.arrayfire_host_sync_reason = sync.reason_code;
    event.arrayfire_host_sync_category = sync.attribution_category.empty()
        ? "unknown"
        : sync.attribution_category;
    event.arrayfire_host_sync_operation = sync.attribution_operation;
    event.arrayfire_host_sync_shape = sync.tensor_shape;
    event.arrayfire_host_sync_dtype = sync.tensor_dtype;
    event.arrayfire_host_sync_layout = sync.tensor_layout;
    event.message = sync.operation_name.empty()
        ? "ArrayFire tensor data synchronized to host"
        : sync.operation_name + " synchronized ArrayFire tensor data to host";
    if (!sync.context.empty()) {
        event.message += ". Context: " + sync.context;
    }
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
        event.validation_loss = latest.validation_loss;
        event.validation_accuracy = latest.validation_accuracy;
    }
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    ++transfer_event_count_;
    transfer_known_bytes_ += sync.bytes;
    ++transfer_reason_counts_[
        TransferSummaryKey(event.transfer_mode, event.transfer_reason)];
    ++synchronization_event_count_;
    synchronization_known_bytes_ += sync.bytes;
    ++synchronization_reason_counts_[
        ReasonSummaryKey(event.arrayfire_host_sync_reason)];
    ++arrayfire_host_sync_count_;
    arrayfire_host_sync_bytes_ += sync.bytes;
    TrainingTraceHostSyncGroup group;
    group.category = event.arrayfire_host_sync_category;
    group.reason = event.arrayfire_host_sync_reason;
    group.operation = event.arrayfire_host_sync_operation;
    group.shape = event.arrayfire_host_sync_shape;
    group.dtype = event.arrayfire_host_sync_dtype;
    group.layout = event.arrayfire_host_sync_layout;
    group.event_count = 1;
    group.bytes = sync.bytes;
    AddHostSyncGroup(arrayfire_host_sync_groups_, std::move(group));
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordExecutionDeviceContext(
    const ExecutionDeviceContext& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "ExecutionDeviceContext.Bind";
    event.thread_id = ThreadIdString();
    event.status = context.valid ? "ok" : "warning";
    event.message = context.Describe();
    event.compute_backend = context.effective_backend;
    event.requested_backend = context.requested_backend;
    event.requested_device_id = context.requested_device_id;
    event.effective_backend = context.effective_backend;
    event.effective_device_id = context.effective_device_id;
    event.effective_device_name = context.device_name;
    event.execution_platform = context.platform;
    event.execution_context_id = context.stable_identity;
    event.capability_generation = context.capability_generation;
    event.stage_backend = context.effective_backend;
    event.stage_device_id = context.effective_device_id;
    event.stage_device_name = context.device_name;
    event.fallback_policy = context.FallbackPolicyName();
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    execution_context_event_ = event;
    has_execution_context_event_ = true;
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (!context.valid) {
        warnings_.push_back(event.message);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordPlacementPlan(
    const std::string& fingerprint,
    uint64_t entry_count,
    const std::string& placement_summary,
    const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "TrainingExecutor.PlacementPlan";
    event.thread_id = ThreadIdString();
    event.status = "ok";
    event.message = message;
    event.placement_fingerprint = fingerprint;
    event.placement_entry_count = entry_count;
    event.placement_summary = placement_summary;
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    placement_plan_event_ = event;
    has_placement_plan_event_ = true;
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordTaskProgress(
    uint64_t task_id,
    const std::string& task_name,
    const std::string& task_stage,
    float progress,
    const std::string& message,
    const std::string& status,
    int node_id,
    const std::string& node_name,
    uint64_t estimated_memory_bytes,
    uint64_t processed_items,
    uint64_t total_items,
    const std::string& memory_risk_level) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = task_stage.empty() ? "TaskProgress" : task_stage;
    event.thread_id = ThreadIdString();
    event.status = status.empty() ? "running" : status;
    event.message = message;
    event.metric_scope = "task";
    event.task_id = task_id;
    event.task_name = task_name;
    event.task_stage = task_stage;
    event.task_progress = std::clamp(progress, 0.0f, 1.0f);
    event.node_id = node_id;
    event.node_name = node_name;
    event.estimated_memory_bytes = estimated_memory_bytes;
    event.memory_risk_level = memory_risk_level;
    event.processed_items = processed_items;
    event.total_items = total_items;
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
        event.validation_loss = latest.validation_loss;
        event.validation_accuracy = latest.validation_accuracy;
    }
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (event.status != "running" && !event.message.empty()) {
        warnings_.push_back(task_name + ": " + event.message);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordValidationMetrics(
    int epoch,
    float train_loss,
    float train_accuracy,
    float validation_loss,
    float validation_accuracy,
    float duration_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = "ValidationCompleted";
    event.thread_id = ThreadIdString();
    event.epoch = epoch;
    event.loss = train_loss;
    event.accuracy = train_accuracy;
    event.validation_loss = validation_loss;
    event.validation_accuracy = validation_accuracy;
    event.duration_ms = duration_ms;
    event.metric_scope = "validation";
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordCheckpointSaved(
    int epoch,
    const std::string& checkpoint_path,
    float validation_loss,
    float validation_accuracy,
    bool is_best_checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = is_best_checkpoint ? "BestCheckpointUpdated" : "CheckpointSaved";
    event.thread_id = ThreadIdString();
    event.epoch = epoch;
    event.validation_loss = validation_loss;
    event.validation_accuracy = validation_accuracy;
    event.metric_scope = "checkpoint";
    event.checkpoint_path = checkpoint_path;
    event.is_best_checkpoint = is_best_checkpoint;
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordTerminalEvent(
    const std::string& terminal_status,
    const std::string& terminal_reason,
    int epoch,
    float loss,
    float accuracy) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = terminal_status == "early_stopped"
        ? "EarlyStopped"
        : "TrainingTerminal";
    event.thread_id = ThreadIdString();
    event.epoch = epoch;
    event.loss = loss;
    event.accuracy = accuracy;
    event.status = terminal_status.empty() ? "completed" : terminal_status;
    event.message = terminal_reason;
    event.terminal_reason = terminal_reason;
    PopulateStageExecutionContext(event);
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::FinishRun(const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

void TrainingTraceCollector::Configure(const TrainingTraceSettings& settings) {
    std::lock_guard<std::mutex> lock(mutex_);
    settings_ = settings;
    if (settings_.persist_every_n_events < 1) {
        settings_.persist_every_n_events = 1;
    }
    if (settings_.max_recent_events < 20) {
        settings_.max_recent_events = 20;
    }
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

TrainingTraceSettings TrainingTraceCollector::GetSettings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settings_;
}

TrainingTraceSummary TrainingTraceCollector::Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TrainingTraceSummary summary;
    summary.available = !run_id_.empty();
    summary.run_id = run_id_;
    summary.status = status_;
    summary.warnings = warnings_;
    summary.recent_events.assign(events_.begin(), events_.end());
    summary.materialization_events.assign(
        materialization_events_.begin(), materialization_events_.end());
    summary.native_cpu_fallback_count = native_cpu_fallback_count_;
    summary.transfer_event_count = transfer_event_count_;
    summary.transfer_known_bytes = transfer_known_bytes_;
    summary.transfer_summary = FormatReasonCounts(transfer_reason_counts_);
    summary.synchronization_event_count = synchronization_event_count_;
    summary.synchronization_known_bytes = synchronization_known_bytes_;
    summary.synchronization_summary =
        FormatReasonCounts(synchronization_reason_counts_);
    summary.arrayfire_host_sync_count = arrayfire_host_sync_count_;
    summary.arrayfire_host_sync_bytes = arrayfire_host_sync_bytes_;
    summary.arrayfire_host_sync_groups =
        HostSyncGroupValues(arrayfire_host_sync_groups_);
    summary.arrayfire_host_sync_summary =
        FormatHostSyncGroups(summary.arrayfire_host_sync_groups);
    if (has_placement_plan_event_) {
        summary.placement_fingerprint =
            placement_plan_event_.placement_fingerprint;
        summary.placement_entry_count =
            placement_plan_event_.placement_entry_count;
        summary.placement_summary = placement_plan_event_.placement_summary;
    }
    summary.declared_output_boundary_count =
        declared_output_boundary_count_;
    if (has_execution_context_event_) {
        summary.execution_platform =
            execution_context_event_.execution_platform;
        summary.requested_backend =
            execution_context_event_.requested_backend;
        summary.requested_device_id =
            execution_context_event_.requested_device_id;
        summary.effective_backend =
            execution_context_event_.effective_backend;
        summary.effective_device_id =
            execution_context_event_.effective_device_id;
        summary.effective_device_name =
            execution_context_event_.effective_device_name;
        summary.execution_context_id =
            execution_context_event_.execution_context_id;
        summary.fallback_policy = execution_context_event_.fallback_policy;
    }
    if (!events_.empty()) {
        const auto& latest = events_.back();
        summary.latest_stage = latest.stage;
        summary.latest_timestamp = latest.timestamp;
        summary.latest_epoch = latest.epoch;
        summary.latest_batch = latest.batch;
        summary.latest_total_batches = latest.total_batches;
        summary.latest_loss = latest.loss;
        summary.latest_accuracy = latest.accuracy;
    }
    PopulateRunLevelTraceSummary(summary);
    return summary;
}

TrainingTraceSummary TrainingTraceCollector::LatestTrace() {
    auto live = Instance().Snapshot();
    if (live.available && !live.run_id.empty()) {
        return live;
    }
    if (const auto persisted = LoadLastTrace()) {
        return *persisted;
    }
    return live;
}

std::optional<TrainingTraceSummary> TrainingTraceCollector::LoadLastTrace() {
    const auto path = CurrentTracePath();
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    try {
        std::ifstream file(path);
        nlohmann::json j;
        file >> j;

        TrainingTraceSummary summary;
        summary.available = true;
        summary.run_id = j.value("run_id", "");
        summary.status = j.value("status", "");
        summary.warnings = j.value("warnings", std::vector<std::string>{});
        summary.native_cpu_fallback_count =
            j.value("native_cpu_fallback_count", uint64_t{0});
        summary.transfer_event_count =
            j.value("transfer_event_count", uint64_t{0});
        summary.transfer_known_bytes =
            j.value("transfer_known_bytes", uint64_t{0});
        summary.transfer_summary = j.value("transfer_summary", "");
        summary.synchronization_event_count =
            j.value("synchronization_event_count", uint64_t{0});
        summary.synchronization_known_bytes =
            j.value("synchronization_known_bytes", uint64_t{0});
        summary.synchronization_summary =
            j.value("synchronization_summary", "");
        summary.arrayfire_host_sync_count =
            j.value("arrayfire_host_sync_count", uint64_t{0});
        summary.arrayfire_host_sync_bytes =
            j.value("arrayfire_host_sync_bytes", uint64_t{0});
        summary.arrayfire_host_sync_summary =
            j.value("arrayfire_host_sync_summary", "");
        if (j.contains("arrayfire_host_sync_groups") &&
            j["arrayfire_host_sync_groups"].is_array()) {
            for (const auto& item : j["arrayfire_host_sync_groups"]) {
                summary.arrayfire_host_sync_groups.push_back(
                    HostSyncGroupFromJson(item));
            }
        }
        summary.placement_fingerprint =
            j.value("placement_fingerprint", "");
        summary.placement_entry_count =
            j.value("placement_entry_count", uint64_t{0});
        summary.placement_summary = j.value("placement_summary", "");
        summary.execution_platform = j.value("execution_platform", "");
        summary.requested_backend = j.value("requested_backend", "");
        summary.requested_device_id = j.value("requested_device_id", 0);
        summary.effective_backend = j.value("effective_backend", "");
        summary.effective_device_id = j.value("effective_device_id", 0);
        summary.effective_device_name = j.value("effective_device_name", "");
        summary.execution_context_id = j.value("execution_context_id", "");
        summary.fallback_policy = j.value("fallback_policy", "");
        summary.declared_output_boundary_count =
            j.value("declared_output_boundary_count", uint64_t{0});
        if (j.contains("events") && j["events"].is_array()) {
            for (const auto& item : j["events"]) {
                summary.recent_events.push_back(EventFromJson(item));
            }
        }
        if (j.contains("materialization_events") &&
            j["materialization_events"].is_array()) {
            for (const auto& item : j["materialization_events"]) {
                summary.materialization_events.push_back(EventFromJson(item));
            }
        } else {
            for (const auto& event : summary.recent_events) {
                if (event.node_id >= 0 &&
                    (event.metric_scope == "task" || event.task_id != 0)) {
                    summary.materialization_events.push_back(event);
                }
            }
        }
        if (!summary.recent_events.empty()) {
            const auto& latest = summary.recent_events.back();
            summary.latest_stage = latest.stage;
            summary.latest_timestamp = latest.timestamp;
            summary.latest_epoch = latest.epoch;
            summary.latest_batch = latest.batch;
            summary.latest_total_batches = latest.total_batches;
            summary.latest_loss = latest.loss;
            summary.latest_accuracy = latest.accuracy;
        }
        PopulateRunLevelTraceSummary(summary);
        return summary;
    } catch (...) {
        return std::nullopt;
    }
}

void TrainingTraceCollector::WriteLocked() const {
    try {
        std::filesystem::create_directories(TraceDir());
        nlohmann::json events = nlohmann::json::array();
        for (const auto& event : events_) {
            events.push_back(EventToJson(event));
        }
        nlohmann::json materialization_events = nlohmann::json::array();
        for (const auto& event : materialization_events_) {
            materialization_events.push_back(EventToJson(event));
        }
        nlohmann::json host_sync_groups = nlohmann::json::array();
        for (const auto& group :
             HostSyncGroupValues(arrayfire_host_sync_groups_)) {
            host_sync_groups.push_back(HostSyncGroupToJson(group));
        }
        nlohmann::json j = {
            {"run_id", run_id_},
            {"status", status_},
            {"events", events},
            {"materialization_events", materialization_events},
            {"warnings", warnings_},
            {"native_cpu_fallback_count", native_cpu_fallback_count_},
            {"transfer_event_count", transfer_event_count_},
            {"transfer_known_bytes", transfer_known_bytes_},
            {"transfer_summary",
             FormatReasonCounts(transfer_reason_counts_)},
            {"synchronization_event_count", synchronization_event_count_},
            {"synchronization_known_bytes", synchronization_known_bytes_},
            {"synchronization_summary",
             FormatReasonCounts(synchronization_reason_counts_)},
            {"arrayfire_host_sync_count", arrayfire_host_sync_count_},
            {"arrayfire_host_sync_bytes", arrayfire_host_sync_bytes_},
            {"arrayfire_host_sync_groups", host_sync_groups},
            {"arrayfire_host_sync_summary",
             FormatHostSyncGroups(
                 HostSyncGroupValues(arrayfire_host_sync_groups_))}
        };
        TrainingTraceSummary summary;
        summary.available = !run_id_.empty();
        summary.run_id = run_id_;
        summary.status = status_;
        summary.recent_events.assign(events_.begin(), events_.end());
        summary.native_cpu_fallback_count = native_cpu_fallback_count_;
        summary.transfer_event_count = transfer_event_count_;
        summary.transfer_known_bytes = transfer_known_bytes_;
        summary.transfer_summary =
            FormatReasonCounts(transfer_reason_counts_);
        summary.synchronization_event_count =
            synchronization_event_count_;
        summary.synchronization_known_bytes =
            synchronization_known_bytes_;
        summary.synchronization_summary =
            FormatReasonCounts(synchronization_reason_counts_);
        summary.arrayfire_host_sync_count = arrayfire_host_sync_count_;
        summary.arrayfire_host_sync_bytes = arrayfire_host_sync_bytes_;
        summary.arrayfire_host_sync_groups =
            HostSyncGroupValues(arrayfire_host_sync_groups_);
        summary.arrayfire_host_sync_summary =
            FormatHostSyncGroups(summary.arrayfire_host_sync_groups);
        if (has_placement_plan_event_) {
            summary.placement_fingerprint =
                placement_plan_event_.placement_fingerprint;
            summary.placement_entry_count =
                placement_plan_event_.placement_entry_count;
            summary.placement_summary =
                placement_plan_event_.placement_summary;
        }
        summary.declared_output_boundary_count =
            declared_output_boundary_count_;
        if (has_execution_context_event_) {
            summary.execution_platform =
                execution_context_event_.execution_platform;
            summary.requested_backend =
                execution_context_event_.requested_backend;
            summary.requested_device_id =
                execution_context_event_.requested_device_id;
            summary.effective_backend =
                execution_context_event_.effective_backend;
            summary.effective_device_id =
                execution_context_event_.effective_device_id;
            summary.effective_device_name =
                execution_context_event_.effective_device_name;
            summary.execution_context_id =
                execution_context_event_.execution_context_id;
            summary.fallback_policy =
                execution_context_event_.fallback_policy;
        }
        PopulateRunLevelTraceSummary(summary);
        j["execution_platform"] = summary.execution_platform;
        j["requested_backend"] = summary.requested_backend;
        j["requested_device_id"] = summary.requested_device_id;
        j["effective_backend"] = summary.effective_backend;
        j["effective_device_id"] = summary.effective_device_id;
        j["effective_device_name"] = summary.effective_device_name;
        j["execution_context_id"] = summary.execution_context_id;
        j["fallback_policy"] = summary.fallback_policy;
        j["declared_output_boundary_count"] =
            summary.declared_output_boundary_count;
        j["transfer_event_count"] = summary.transfer_event_count;
        j["transfer_known_bytes"] = summary.transfer_known_bytes;
        j["transfer_summary"] = summary.transfer_summary;
        j["synchronization_event_count"] =
            summary.synchronization_event_count;
        j["synchronization_known_bytes"] =
            summary.synchronization_known_bytes;
        j["synchronization_summary"] =
            summary.synchronization_summary;
        j["arrayfire_host_sync_count"] =
            summary.arrayfire_host_sync_count;
        j["arrayfire_host_sync_bytes"] =
            summary.arrayfire_host_sync_bytes;
        j["arrayfire_host_sync_groups"] = host_sync_groups;
        j["arrayfire_host_sync_summary"] =
            summary.arrayfire_host_sync_summary;
        j["placement_fingerprint"] = summary.placement_fingerprint;
        j["placement_entry_count"] = summary.placement_entry_count;
        j["placement_summary"] = summary.placement_summary;
        j["residency_verdict"] = summary.residency_verdict;
        j["residency_reason"] = summary.residency_reason;
        std::ofstream file(CurrentTracePath(), std::ios::trunc);
        file << std::setw(2) << j << '\n';
    } catch (...) {
        // Debug tracing must never break training.
    }
}

std::string TrainingTraceCollector::NowLocal() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

std::string TrainingTraceCollector::ThreadIdString() {
    std::ostringstream out;
    out << std::this_thread::get_id();
    return out.str();
}

} // namespace cyxwiz
