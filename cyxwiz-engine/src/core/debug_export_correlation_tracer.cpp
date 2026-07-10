#include "debug_export_correlation_tracer.h"

namespace cyxwiz {

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
