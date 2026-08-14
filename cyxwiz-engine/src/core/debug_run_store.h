#pragma once

#include "debug_recommendation_engine.h"
#include "debug_trace_record.h"
#include "graph_compiler.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct TrainingTraceSummary;
struct DebugSession;

struct DebugRunExecutionSummary {
    bool available = false;
    std::string training_run_id;
    std::string status;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::string execution_context_id;
    std::string placement_fingerprint;
    std::string residency_verdict;
    size_t native_cpu_fallback_count = 0;
    size_t transfer_event_count = 0;
    uint64_t transfer_known_bytes = 0;
    size_t synchronization_event_count = 0;
    uint64_t synchronization_known_bytes = 0;
};

DebugRunExecutionSummary MakeDebugRunExecutionSummary(
    const TrainingTraceSummary& trace);

struct DebugRunStoreSummary {
    std::string run_id;
    std::string timestamp;
    uint64_t graph_hash = 0;
    bool success = false;
    size_t issue_count = 0;
    size_t trace_count = 0;
    size_t event_count = 0;
    size_t recommendation_count = 0;
    std::string summary;
    std::string file_path;
    DebugRunExecutionSummary execution;
};

struct DebugReplayCompiledConfigSummary {
    bool available = false;
    bool valid = false;
    size_t layer_count = 0;
    std::vector<size_t> input_shape;
    size_t input_size = 0;
    size_t output_size = 0;
    int batch_size = 0;
    int epochs = 0;
    bool shuffle = false;
    bool drop_last = false;
    int num_workers = 0;
    int prefetch_factor = 0;
    int log_interval = 0;
    int validation_freq = 0;
    int grad_accum_steps = 0;
    float train_ratio = 0.0f;
    float val_ratio = 0.0f;
    float test_ratio = 0.0f;
    bool stratified = false;
    std::string loss;
    std::string optimizer;
    float learning_rate = 0.0f;
    float momentum = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float weight_decay = 0.0f;
    std::string compiler_placement_fingerprint;
    size_t backend_placement_count = 0;
    bool forbid_native_cpu_fallback = false;
};

struct DebugRunReplayCapsule {
    static constexpr const char* kSchema =
        "cyxwiz.debug.run_replay_capsule.v1";

    bool available = false;
    std::string mode;
    std::string replay_scope = "explain_and_recompile";
    uint64_t graph_hash = 0;
    bool graph_snapshot_trace_available = false;
    size_t graph_snapshot_trace_index = 0;
    std::string dataset_reference;
    size_t selected_sample_index = 0;
    size_t smoke_sample_limit = 0;
    size_t smoke_batch_size_limit = 0;
    DebugReplayCompiledConfigSummary compiled_config;
    int split_seed = 0;
    int dataloader_seed = 0;
    int balance_seed = 0;
    std::string backend_evidence_scope = "unobserved";
    std::string backend_source_run_id;
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string effective_device_name;
    std::map<std::string, std::string> environment;
    bool trace_records_embedded = true;
    bool issues_embedded = true;
    bool raw_dataset_values_included = false;
    bool exact_replay_claimed = false;
};

DebugRunReplayCapsule MakeDebugRunReplayCapsule(
    const DebugSession& session,
    const TrainingConfiguration* config,
    const DebugRunExecutionSummary& execution,
    size_t smoke_sample_limit = 0);

nlohmann::json DebugRunReplayCapsuleToJson(
    const DebugRunReplayCapsule& capsule);
DebugRunReplayCapsule DebugRunReplayCapsuleFromJson(
    const nlohmann::json& value);

struct DebugRunStoreRecord {
    DebugRunStoreSummary summary;
    DebugRunReplayCapsule replay_capsule;
    std::vector<ValidationIssue> issues;
    std::vector<DebugTraceRecord> traces;
    std::vector<StudioEventRecord> studio_events;
    std::vector<DebugRecommendation> recommendations;
};

class DebugRunStore {
public:
    static bool Save(const DebugRunStoreRecord& record);
    static std::optional<DebugRunStoreRecord> Load(const std::string& run_id);
    static std::vector<DebugRunStoreSummary> ListRecent(size_t max_runs = 10);
};

} // namespace cyxwiz
