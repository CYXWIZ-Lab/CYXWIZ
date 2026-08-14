#include "debug_export_correlation_tracer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

namespace {

std::string Hex64(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

} // namespace

DebugTraceRecord DebugExportCorrelationTracer::BuildTrace(
    const std::string& run_id,
    const DebugExportCorrelationInput& input) const {
    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        input.exporter_name.empty() ? "GeneratedCodeExport" : input.exporter_name,
        "ExportArtifact",
        "ExportCorrelation",
        DebugTraceRole::GeneratedCode,
        {},
        {},
        input.artifact_kind,
        "ExportCorrelation",
        input.compile_success ? "ok" : "failed");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "export_correlation",
        "DebugExportCorrelationTracer",
        "cyxwiz-engine/src/core/debug_export_correlation_tracer.cpp",
        "cyxwiz::DebugExportCorrelationTracer::BuildTrace");
    trace.payload["schema"] = kSchema;
    trace.payload["artifact_kind"] = input.artifact_kind;
    trace.payload["artifact_path"] = input.artifact_path;
    trace.payload["exporter_name"] = input.exporter_name;
    trace.payload["graph_hash"] = input.graph_hash;
    trace.payload["compile_success"] = input.compile_success;
    trace.payload["compile_status"] = input.compile_status;
    trace.payload["source_node_ids"] = input.source_node_ids;
    trace.payload["content_fingerprint"] =
        Fingerprint(input.generated_content);
    trace.payload["content_bytes"] = input.generated_content.size();
    trace.payload["message"] = input.message;

    if (input.artifact_path.empty()) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Generated-code/export trace has no artifact path.",
            errors::Serialization::ArtifactPathMissing);
    }
    if (!input.compile_success) {
        DebugNodeTraceContract::AddError(
            trace,
            input.compile_status.empty()
                ? "Generated-code/export compile correlation failed."
                : input.compile_status);
    }
    trace.payload["success"] = trace.status == "ok";

    return trace;
}

DebugTraceRecord DebugExportCorrelationTracer::BuildConsistencyTrace(
    const std::string& run_id,
    const DebugArtifactConsistencyInput& input) const {
    const bool is_import = input.action.rfind("import", 0) == 0;
    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        input.producer_name.empty()
            ? (is_import ? "ModelImport" : "ModelExport")
            : input.producer_name,
        "ModelArtifact",
        "ArtifactConsistency",
        DebugTraceRole::GeneratedCode,
        {},
        {},
        input.artifact_kind,
        "ArtifactConsistency",
        input.operation_success ? "ok" : "failed");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "artifact_consistency",
        "DebugExportCorrelationTracer",
        "cyxwiz-engine/src/core/debug_export_correlation_tracer.cpp",
        "cyxwiz::DebugExportCorrelationTracer::BuildConsistencyTrace");

    trace.payload["schema"] = kConsistencySchema;
    trace.payload["artifact_action"] = input.action;
    trace.payload["artifact_kind"] = input.artifact_kind;
    trace.payload["artifact_path"] = input.artifact_path;
    trace.payload["producer_name"] = input.producer_name;
    trace.payload["graph_hash"] = input.graph_hash;
    trace.payload["graph_hash_hex"] = Hex64(input.graph_hash);
    trace.payload["operation_success"] = input.operation_success;
    trace.payload["operation_status"] = input.operation_status;
    trace.payload["inspection_available"] = input.observed.available;
    trace.payload["inspection_scope"] = input.observed.evidence_scope;
    trace.payload["manifest_valid"] = input.observed.manifest_valid;
    trace.payload["format_version"] = input.observed.format_version;
    trace.payload["model_family"] = input.observed.model_family;

    const bool source_graph_available = !input.source_graph_content.empty();
    const bool artifact_graph_available = !input.artifact_graph_content.empty();
    const uint64_t source_graph_fingerprint = source_graph_available
        ? Fingerprint(input.source_graph_content)
        : 0;
    const uint64_t artifact_graph_fingerprint = artifact_graph_available
        ? Fingerprint(input.artifact_graph_content)
        : 0;
    const bool graph_fingerprints_comparable =
        source_graph_available && artifact_graph_available;
    const bool graph_fingerprints_match =
        graph_fingerprints_comparable &&
        source_graph_fingerprint == artifact_graph_fingerprint;
    trace.payload["source_graph_available"] = source_graph_available;
    trace.payload["artifact_graph_available"] = artifact_graph_available;
    trace.payload["source_graph_fingerprint"] = source_graph_fingerprint;
    trace.payload["source_graph_fingerprint_hex"] =
        Hex64(source_graph_fingerprint);
    trace.payload["artifact_graph_fingerprint"] =
        artifact_graph_fingerprint;
    trace.payload["artifact_graph_fingerprint_hex"] =
        Hex64(artifact_graph_fingerprint);
    trace.payload["graph_fingerprint_algorithm"] =
        "fnv1a64_noncryptographic";
    trace.payload["graph_fingerprints_comparable"] =
        graph_fingerprints_comparable;
    trace.payload["graph_fingerprints_match"] = graph_fingerprints_match;

    trace.payload["manifest_expected"] = input.expected.manifest_expected;
    trace.payload["manifest_present"] = input.observed.manifest_present;
    trace.payload["training_config_expected"] =
        input.expected.training_config_expected;
    trace.payload["training_config_present"] =
        input.observed.training_config_present;
    trace.payload["weights_manifest_expected"] =
        input.expected.weights_manifest_expected;
    trace.payload["weights_manifest_present"] =
        input.observed.weights_manifest_present;
    trace.payload["graph_expected"] = input.expected.graph_expected;
    trace.payload["graph_present"] = input.observed.graph_present;
    trace.payload["tokenizer_config_required"] =
        input.expected.tokenizer_config_required;
    trace.payload["tokenizer_config_present"] =
        input.observed.tokenizer_config_present;
    trace.payload["tokenizer_vocabulary_required"] =
        input.expected.tokenizer_vocabulary_required;
    trace.payload["tokenizer_vocabulary_present"] =
        input.observed.tokenizer_vocabulary_present;
    trace.payload["optimizer_state_expected"] =
        input.expected.optimizer_state_expected;
    trace.payload["optimizer_state_present"] =
        input.observed.optimizer_state_present;
    trace.payload["training_history_expected"] =
        input.expected.training_history_expected;
    trace.payload["training_history_present"] =
        input.observed.training_history_present;
    trace.payload["sequence_assets_present"] =
        input.observed.sequence_assets_present;
    trace.payload["tree_model_artifact_present"] =
        input.observed.tree_model_artifact_present;

    trace.payload["expected_parameter_count_available"] =
        input.expected.parameter_count.has_value();
    trace.payload["observed_parameter_count_available"] =
        input.observed.parameter_count.has_value();
    if (input.expected.parameter_count) {
        trace.payload["expected_parameter_count"] =
            *input.expected.parameter_count;
    }
    if (input.observed.parameter_count) {
        trace.payload["observed_parameter_count"] =
            *input.observed.parameter_count;
    }
    const bool parameter_counts_comparable =
        input.expected.parameter_count.has_value() &&
        input.observed.parameter_count.has_value();
    const bool parameter_counts_match =
        parameter_counts_comparable &&
        input.expected.parameter_count == input.observed.parameter_count;
    trace.payload["parameter_counts_comparable"] =
        parameter_counts_comparable;
    trace.payload["parameter_counts_match"] = parameter_counts_match;

    trace.payload["expected_layer_count_available"] =
        input.expected.layer_count.has_value();
    trace.payload["observed_layer_count_available"] =
        input.observed.layer_count.has_value();
    if (input.expected.layer_count) {
        trace.payload["expected_layer_count"] = *input.expected.layer_count;
    }
    if (input.observed.layer_count) {
        trace.payload["observed_layer_count"] = *input.observed.layer_count;
    }
    const bool layer_counts_comparable =
        input.expected.layer_count.has_value() &&
        input.observed.layer_count.has_value();
    const bool layer_counts_match =
        layer_counts_comparable &&
        input.expected.layer_count == input.observed.layer_count;
    trace.payload["layer_counts_comparable"] = layer_counts_comparable;
    trace.payload["layer_counts_match"] = layer_counts_match;

    trace.payload["expected_input_contract"] = input.expected.input_contract;
    trace.payload["observed_input_contract"] = input.observed.input_contract;
    trace.payload["expected_output_contract"] =
        input.expected.output_contract;
    trace.payload["observed_output_contract"] =
        input.observed.output_contract;
    const bool input_contracts_comparable =
        !input.expected.input_contract.empty() &&
        !input.observed.input_contract.empty();
    const bool output_contracts_comparable =
        !input.expected.output_contract.empty() &&
        !input.observed.output_contract.empty();
    trace.payload["input_contracts_comparable"] =
        input_contracts_comparable;
    trace.payload["input_contracts_match"] =
        input_contracts_comparable &&
        input.expected.input_contract == input.observed.input_contract;
    trace.payload["output_contracts_comparable"] =
        output_contracts_comparable;
    trace.payload["output_contracts_match"] =
        output_contracts_comparable &&
        input.expected.output_contract == input.observed.output_contract;
    trace.payload["package_warnings"] = input.observed.package_warnings;
    trace.payload["raw_artifact_content_included"] = false;

    size_t check_count = 0;
    const char* consistency_error_code = is_import
        ? errors::Serialization::ModelLoadFailed
        : errors::Serialization::ModelSaveFailed;
    const auto require_asset = [&](bool expected,
                                   bool present,
                                   const char* label,
                                   bool required) {
        if (!expected || !input.observed.available) {
            return;
        }
        ++check_count;
        if (present) {
            return;
        }
        const std::string message = std::string("Expected artifact asset is missing: ") +
            label + ".";
        if (required) {
            DebugNodeTraceContract::AddError(
                trace, message, consistency_error_code);
        } else {
            DebugNodeTraceContract::AddWarning(
                trace, message, consistency_error_code);
        }
    };

    if (!input.operation_success) {
        DebugNodeTraceContract::AddError(
            trace,
            input.operation_status.empty()
                ? (is_import ? "Model import/inspection failed."
                             : "Model export failed.")
                : input.operation_status,
            is_import ? errors::Serialization::ModelLoadFailed
                      : errors::Serialization::ModelSaveFailed);
    }
    if (input.observed.available) {
        ++check_count;
        if (input.operation_success && !input.observed.manifest_valid) {
            DebugNodeTraceContract::AddError(
                trace,
                "Artifact inspection did not validate its manifest/header.",
                consistency_error_code);
        }
    }

    require_asset(input.expected.manifest_expected,
                  input.observed.manifest_present,
                  "manifest.json", true);
    require_asset(input.expected.training_config_expected,
                  input.observed.training_config_present,
                  "config.json", true);
    require_asset(input.expected.weights_manifest_expected,
                  input.observed.weights_manifest_present,
                  "weights/manifest.json", true);
    require_asset(input.expected.graph_expected,
                  input.observed.graph_present,
                  "graph.cyxgraph", true);
    require_asset(input.expected.tokenizer_config_required,
                  input.observed.tokenizer_config_present,
                  "tokenizer/config.json", true);
    require_asset(input.expected.tokenizer_vocabulary_required,
                  input.observed.tokenizer_vocabulary_present,
                  "tokenizer/vocab.txt", true);
    require_asset(input.expected.optimizer_state_expected,
                  input.observed.optimizer_state_present,
                  "optimizer state", false);
    require_asset(input.expected.training_history_expected,
                  input.observed.training_history_present,
                  "history.json", false);

    if (graph_fingerprints_comparable) {
        ++check_count;
        if (!graph_fingerprints_match) {
            DebugNodeTraceContract::AddError(
                trace,
                "Packaged graph content does not match the exported source graph.",
                consistency_error_code);
        }
    }
    if (parameter_counts_comparable) {
        ++check_count;
        if (!parameter_counts_match) {
            DebugNodeTraceContract::AddError(
                trace,
                "Artifact parameter count does not match the source model.",
                consistency_error_code);
        }
    }
    if (layer_counts_comparable) {
        ++check_count;
        if (!layer_counts_match) {
            DebugNodeTraceContract::AddError(
                trace,
                "Artifact layer count does not match the source model.",
                consistency_error_code);
        }
    }
    if (input_contracts_comparable) {
        ++check_count;
        if (input.expected.input_contract != input.observed.input_contract) {
            DebugNodeTraceContract::AddError(
                trace,
                "Artifact input contract does not match the expected contract.",
                consistency_error_code);
        }
    }
    if (output_contracts_comparable) {
        ++check_count;
        if (input.expected.output_contract != input.observed.output_contract) {
            DebugNodeTraceContract::AddError(
                trace,
                "Artifact output contract does not match the expected contract.",
                consistency_error_code);
        }
    }

    const size_t error_count = trace.payload.value("error_count", size_t{0});
    const size_t warning_count =
        trace.payload.value("warning_count", size_t{0});
    std::string outcome;
    if (!input.operation_success) {
        outcome = "failed";
    } else if (error_count != 0) {
        outcome = "mismatch";
    } else if (warning_count != 0) {
        outcome = "warning";
        trace.status = "warning";
    } else if (!input.observed.available || check_count == 0) {
        outcome = "unobserved";
        trace.status = "unobserved";
    } else {
        outcome = "compatible";
        trace.status = "ok";
    }
    trace.payload["consistency_check_count"] = check_count;
    trace.payload["consistency_outcome"] = outcome;
    trace.payload["consistency_verified"] =
        outcome == "compatible" || outcome == "warning";
    trace.payload["success"] = input.operation_success && error_count == 0;
    return trace;
}

uint64_t DebugExportCorrelationTracer::Fingerprint(
    const std::string& content) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : content) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

} // namespace cyxwiz
