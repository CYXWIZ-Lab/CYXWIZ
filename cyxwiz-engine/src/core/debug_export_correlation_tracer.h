#pragma once

#include "debug_trace_record.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugExportCorrelationInput {
    std::string artifact_kind;
    std::string artifact_path;
    std::string exporter_name;
    uint64_t graph_hash = 0;
    bool compile_success = false;
    std::string compile_status;
    std::vector<int> source_node_ids;
    std::string generated_content;
    std::string message;
};

struct DebugArtifactExpectations {
    bool graph_expected = false;
    bool manifest_expected = false;
    bool training_config_expected = false;
    bool weights_manifest_expected = false;
    bool tokenizer_config_required = false;
    bool tokenizer_vocabulary_required = false;
    bool optimizer_state_expected = false;
    bool training_history_expected = false;
    std::optional<size_t> parameter_count;
    std::optional<size_t> layer_count;
    std::string input_contract;
    std::string output_contract;
};

struct DebugArtifactObservation {
    bool available = false;
    bool manifest_valid = false;
    std::string evidence_scope = "unobserved";
    std::string format_version;
    std::string model_family;
    bool manifest_present = false;
    bool training_config_present = false;
    bool weights_manifest_present = false;
    bool graph_present = false;
    bool tokenizer_config_present = false;
    bool tokenizer_vocabulary_present = false;
    bool optimizer_state_present = false;
    bool training_history_present = false;
    bool sequence_assets_present = false;
    bool tree_model_artifact_present = false;
    std::optional<size_t> parameter_count;
    std::optional<size_t> layer_count;
    std::string input_contract;
    std::string output_contract;
    std::vector<std::string> package_warnings;
};

struct DebugArtifactConsistencyInput {
    std::string action;
    std::string artifact_kind;
    std::string artifact_path;
    std::string producer_name;
    uint64_t graph_hash = 0;
    bool operation_success = false;
    std::string operation_status;
    std::string source_graph_content;
    std::string artifact_graph_content;
    DebugArtifactExpectations expected;
    DebugArtifactObservation observed;
};

class DebugExportCorrelationTracer {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.export_correlation.v1";
    static constexpr const char* kConsistencySchema =
        "cyxwiz.debug.artifact_consistency.v1";

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const DebugExportCorrelationInput& input) const;

    DebugTraceRecord BuildConsistencyTrace(
        const std::string& run_id,
        const DebugArtifactConsistencyInput& input) const;

    static uint64_t Fingerprint(const std::string& content);
};

} // namespace cyxwiz
