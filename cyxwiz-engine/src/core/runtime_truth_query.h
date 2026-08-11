#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

struct RuntimeTrainingEventTruth {
    std::string timestamp;
    std::string stage;
    std::string status;
    std::string message;
    int epoch = 0;
    int batch = 0;
    int total_batches = 0;
    bool native_cpu_fallback = false;
    std::string fallback_operation;
    std::string fallback_reason;
    std::string fallback_policy;
    uint64_t host_sync_bytes = 0;
    std::string host_sync_reason;
    std::string host_sync_category;
    std::string host_sync_operation;
    uint64_t task_id = 0;
    std::string task_name;
    std::string task_stage;
    float task_progress = 0.0f;
    int node_id = -1;
    std::string node_name;
};

struct RuntimeTrainingTruth {
    bool available = false;
    bool active = false;
    std::string source;
    std::string run_id;
    std::string status;
    std::string latest_stage;
    std::string latest_timestamp;
    int epoch = 0;
    int total_epochs = 0;
    int batch = 0;
    int total_batches = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::string execution_context_id;
    std::string fallback_policy;
    uint64_t native_cpu_fallback_count = 0;
    uint64_t transfer_event_count = 0;
    uint64_t transfer_known_bytes = 0;
    uint64_t synchronization_event_count = 0;
    uint64_t synchronization_known_bytes = 0;
    uint64_t host_sync_count = 0;
    uint64_t host_sync_bytes = 0;
    std::string host_sync_summary;
    std::string placement_fingerprint;
    uint64_t placement_entry_count = 0;
    std::string placement_summary;
    std::string residency_verdict;
    std::string residency_reason;
    std::vector<RuntimeTrainingEventTruth> recent_events;
    std::vector<RuntimeTrainingEventTruth> materialization_events;
};

struct RuntimeDeviceEntryTruth {
    std::string backend;
    int device_id = 0;
    std::string name;
    uint64_t memory_total = 0;
};

struct RuntimeDeviceTruth {
    bool active_available = false;
    std::string active_source;
    std::string active_run_id;
    std::string active_backend;
    int active_device_id = 0;
    std::string active_device_name;
    bool queued_available = false;
    std::string queued_backend;
    int queued_device_id = 0;
    std::string next_run_policy;
    std::string next_run_policy_source;
    std::vector<RuntimeDeviceEntryTruth> available_devices;
    std::string inventory_source;
    std::string inventory_status;
};

struct RuntimeRunIssueTruth {
    std::string source;
    std::string level;
    std::string code;
    int node_id = -1;
    std::string node_name;
    std::string message;
};

struct RuntimeRunEventTruth {
    std::string timestamp;
    std::string source;
    std::string stage;
    std::string status;
    int node_id = -1;
    std::string message;
};

struct RuntimeRunTruth {
    bool available = false;
    std::string source;
    std::string run_id;
    std::string debug_run_id;
    std::string training_run_id;
    std::string timestamp;
    std::string status;
    std::string summary;
    bool success = false;
    uint64_t graph_hash = 0;
    uint64_t issue_count = 0;
    uint64_t trace_count = 0;
    uint64_t event_count = 0;
    uint64_t recommendation_count = 0;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::string execution_context_id;
    std::string placement_fingerprint;
    std::string residency_verdict;
    uint64_t native_cpu_fallback_count = 0;
    uint64_t transfer_event_count = 0;
    uint64_t transfer_known_bytes = 0;
    uint64_t synchronization_event_count = 0;
    uint64_t synchronization_known_bytes = 0;
    bool training_evidence_available = false;
    RuntimeTrainingTruth training;
    std::vector<RuntimeRunIssueTruth> issues;
    std::vector<RuntimeRunEventTruth> events;
};

class RuntimeTruthQueryProvider {
public:
    virtual ~RuntimeTruthQueryProvider() = default;

    virtual RuntimeTrainingTruth GetCurrentTraining() const = 0;
    virtual RuntimeTrainingTruth GetLastTraining() const = 0;
    virtual RuntimeDeviceTruth GetDeviceTruth(bool include_inventory) = 0;
    virtual RuntimeRunTruth GetCurrentRun() const = 0;
    virtual RuntimeRunTruth GetRun(std::string_view run_id) const = 0;
};

std::unique_ptr<RuntimeTruthQueryProvider> CreateEngineRuntimeTruthProvider();

} // namespace cyxwiz
