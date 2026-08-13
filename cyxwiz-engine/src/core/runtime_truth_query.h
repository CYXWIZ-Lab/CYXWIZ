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
    bool backend_available = false;
    bool device_selectable = false;
    bool execution_validated = false;
    std::string device_kind;
    std::string identity_confidence;
    std::string provider;
    std::string driver_version;
    uint32_t hardware_vendor_id = 0;
    uint32_t hardware_device_id = 0;
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    int pci_function = 0;
    std::string hardware_uuid;
    std::string hardware_luid;
    std::string physical_fingerprint;
    bool provider_known = false;
    bool driver_version_known = false;
    bool hardware_vendor_id_known = false;
    bool hardware_device_id_known = false;
    bool pci_location_known = false;
    bool hardware_uuid_known = false;
    bool hardware_luid_known = false;
    bool physical_fingerprint_known = false;
    std::string metadata_status;
    int metadata_error_code = 0;
    std::string metadata_message;
    bool name_known = false;
    bool name_is_fallback = false;
    bool memory_total_known = false;
};

struct RuntimeDeviceTruth {
    bool active_available = false;
    std::string active_source;
    std::string active_run_id;
    std::string active_backend;
    int active_device_id = 0;
    std::string active_device_name;
    bool active_execution_validated = false;
    std::string active_preflight_stage;
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
    bool execution_validated = false;
    std::string preflight_stage;
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
