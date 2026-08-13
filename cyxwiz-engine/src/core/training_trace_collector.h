#pragma once

#include "crash_run_recorder.h"
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct PinMemoryTransferStatus;
struct ArrayFireNativeCpuFallbackEvent;
struct ArrayFireHostSyncEvent;
struct ExecutionDeviceContext;

struct TrainingTraceEvent {
    std::string timestamp;
    std::string run_id;
    std::string stage;
    std::string thread_id;
    int epoch = 0;
    int batch = 0;
    int total_batches = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    float validation_loss = 0.0f;
    float validation_accuracy = 0.0f;
    float duration_ms = 0.0f;
    uint64_t cpu_allocated_bytes = 0;
    uint64_t cpu_peak_bytes = 0;
    uint64_t af_allocated_bytes = 0;
    uint64_t af_locked_bytes = 0;
    uint64_t af_alloc_buffers = 0;
    uint64_t af_lock_buffers = 0;
    std::string status = "ok";
    std::string message;
    std::string metric_scope;
    std::string checkpoint_path;
    bool is_best_checkpoint = false;
    std::string terminal_reason;
    uint64_t task_id = 0;
    std::string task_name;
    std::string task_stage;
    float task_progress = 0.0f;
    int node_id = -1;
    std::string node_name;
    uint64_t estimated_memory_bytes = 0;
    std::string memory_risk_level;
    uint64_t processed_items = 0;
    uint64_t total_items = 0;
    bool pin_memory_requested = false;
    std::string transfer_mode;
    std::string transfer_reason;
    std::string transfer_backend;
    int transfer_batch_size = 0;
    std::string compute_backend;
    std::string stage_backend;
    int stage_device_id = 0;
    std::string stage_device_name;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::string execution_platform;
    std::string execution_context_id;
    std::string physical_fingerprint;
    std::string identity_confidence;
    bool requested_qualification_evidence_available = false;
    bool requested_route_qualified = false;
    std::string requested_qualification_matrix_id;
    std::string requested_qualification_message;
    bool effective_qualification_evidence_available = false;
    bool effective_route_qualified = false;
    std::string effective_qualification_matrix_id;
    std::string effective_qualification_message;
    uint64_t capability_generation = 0;
    bool activation_succeeded = false;
    bool execution_validated = false;
    bool selection_fallback_applied = false;
    std::string preflight_stage;
    int preflight_error_code = 0;
    std::string fallback_target;
    std::string fallback_operation;
    std::string fallback_reason;
    std::string fallback_policy;
    bool native_cpu_fallback = false;
    uint64_t arrayfire_host_sync_bytes = 0;
    std::string arrayfire_host_sync_reason;
    std::string arrayfire_host_sync_category;
    std::string arrayfire_host_sync_operation;
    std::vector<size_t> arrayfire_host_sync_shape;
    std::string arrayfire_host_sync_dtype;
    std::string arrayfire_host_sync_layout;
    std::string placement_fingerprint;
    uint64_t placement_entry_count = 0;
    std::string placement_summary;
};

struct TrainingTraceHostSyncGroup {
    std::string category;
    std::string reason;
    std::string operation;
    std::vector<size_t> shape;
    std::string dtype;
    std::string layout;
    uint64_t event_count = 0;
    uint64_t bytes = 0;
};

struct TrainingTraceSummary {
    bool available = false;
    std::string run_id;
    std::string status;
    std::string latest_stage;
    std::string latest_timestamp;
    int latest_epoch = 0;
    int latest_batch = 0;
    int latest_total_batches = 0;
    float latest_loss = 0.0f;
    float latest_accuracy = 0.0f;
    std::vector<TrainingTraceEvent> recent_events;
    std::vector<TrainingTraceEvent> materialization_events;
    std::vector<std::string> warnings;
    uint64_t native_cpu_fallback_count = 0;
    uint64_t transfer_event_count = 0;
    uint64_t transfer_known_bytes = 0;
    std::string transfer_summary;
    uint64_t synchronization_event_count = 0;
    uint64_t synchronization_known_bytes = 0;
    std::string synchronization_summary;
    uint64_t arrayfire_host_sync_count = 0;
    uint64_t arrayfire_host_sync_bytes = 0;
    std::vector<TrainingTraceHostSyncGroup> arrayfire_host_sync_groups;
    std::string arrayfire_host_sync_summary;
    std::string placement_fingerprint;
    uint64_t placement_entry_count = 0;
    std::string placement_summary;
    std::string execution_platform;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::string execution_context_id;
    std::string physical_fingerprint;
    std::string identity_confidence;
    bool requested_qualification_evidence_available = false;
    bool requested_route_qualified = false;
    std::string requested_qualification_matrix_id;
    std::string requested_qualification_message;
    bool effective_qualification_evidence_available = false;
    bool effective_route_qualified = false;
    std::string effective_qualification_matrix_id;
    std::string effective_qualification_message;
    bool activation_succeeded = false;
    bool execution_validated = false;
    bool selection_fallback_applied = false;
    std::string preflight_stage;
    int preflight_error_code = 0;
    std::string fallback_policy;
    uint64_t declared_output_boundary_count = 0;
    std::string residency_verdict;
    std::string residency_reason;
};

struct TrainingTraceSettings {
    bool persist_enabled = true;
    int persist_every_n_events = 10;
    size_t max_recent_events = 200;
};

class TrainingTraceCollector {
public:
    static TrainingTraceCollector& Instance();

    void StartRun(const std::string& run_id);
    void RecordStage(TrainingTraceStage stage,
                     int epoch,
                     int batch,
                     int total_batches,
                     float loss = 0.0f,
                     float accuracy = 0.0f,
                     float duration_ms = 0.0f,
                     const std::string& status = "ok",
                     const std::string& message = "");
    void RecordRuntimeWarning(const std::string& source,
                              const std::string& message);
    void RecordRuntimeEvent(const std::string& stage,
                            const std::string& message,
                            const std::string& status = "ok");
    void RecordPinMemoryTransferStatus(
        const PinMemoryTransferStatus& status,
        const std::string& message,
        const std::string& severity);
    void RecordNativeCpuFallback(
        const ArrayFireNativeCpuFallbackEvent& fallback);
    void RecordArrayFireHostSync(const ArrayFireHostSyncEvent& sync);
    void RecordExecutionDeviceContext(
        const ExecutionDeviceContext& context);
    void RecordPlacementPlan(const std::string& fingerprint,
                             uint64_t entry_count,
                             const std::string& placement_summary,
                             const std::string& message);
    void RecordTaskProgress(uint64_t task_id,
                            const std::string& task_name,
                            const std::string& task_stage,
                            float progress,
                            const std::string& message,
                            const std::string& status = "running",
                            int node_id = -1,
                            const std::string& node_name = "",
                            uint64_t estimated_memory_bytes = 0,
                            uint64_t processed_items = 0,
                            uint64_t total_items = 0,
                            const std::string& memory_risk_level = "");
    void RecordValidationMetrics(int epoch,
                                 float train_loss,
                                 float train_accuracy,
                                 float validation_loss,
                                 float validation_accuracy,
                                 float duration_ms = 0.0f);
    void RecordCheckpointSaved(int epoch,
                               const std::string& checkpoint_path,
                               float validation_loss,
                               float validation_accuracy,
                               bool is_best_checkpoint);
    void RecordTerminalEvent(const std::string& terminal_status,
                             const std::string& terminal_reason,
                             int epoch,
                             float loss,
                             float accuracy);
    void FinishRun(const std::string& status);

    void Configure(const TrainingTraceSettings& settings);
    TrainingTraceSettings GetSettings() const;
    TrainingTraceSummary Snapshot() const;
    static TrainingTraceSummary LatestTrace();
    static std::optional<TrainingTraceSummary> LoadLastTrace();

private:
    TrainingTraceCollector() = default;

    void WriteLocked() const;
    static std::string NowLocal();
    static std::string ThreadIdString();

    mutable std::mutex mutex_;
    std::string run_id_;
    std::string status_ = "idle";
    std::deque<TrainingTraceEvent> events_;
    std::deque<TrainingTraceEvent> materialization_events_;
    std::vector<std::string> warnings_;
    TrainingTraceEvent execution_context_event_;
    bool has_execution_context_event_ = false;
    TrainingTraceEvent placement_plan_event_;
    bool has_placement_plan_event_ = false;
    uint64_t native_cpu_fallback_count_ = 0;
    uint64_t transfer_event_count_ = 0;
    uint64_t transfer_known_bytes_ = 0;
    std::map<std::string, uint64_t> transfer_reason_counts_;
    uint64_t synchronization_event_count_ = 0;
    uint64_t synchronization_known_bytes_ = 0;
    std::map<std::string, uint64_t> synchronization_reason_counts_;
    uint64_t arrayfire_host_sync_count_ = 0;
    uint64_t arrayfire_host_sync_bytes_ = 0;
    std::map<std::string, TrainingTraceHostSyncGroup>
        arrayfire_host_sync_groups_;
    uint64_t declared_output_boundary_count_ = 0;
    TrainingTraceSettings settings_;
    size_t events_since_write_ = 0;
};

} // namespace cyxwiz
