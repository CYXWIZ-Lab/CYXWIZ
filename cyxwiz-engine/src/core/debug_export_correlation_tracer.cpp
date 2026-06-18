#include "debug_export_correlation_tracer.h"

namespace cyxwiz {

DebugTraceRecord DebugExportCorrelationTracer::BuildTrace(
    const std::string& run_id,
    const DebugExportCorrelationInput& input) const {
    DebugTraceRecord trace;
    trace.run_id = run_id;
    trace.node_id = -1;
    trace.node_name = input.exporter_name.empty()
        ? "GeneratedCodeExport"
        : input.exporter_name;
    trace.node_type = "ExportArtifact";
    trace.phase = "ExportCorrelation";
    trace.role = DebugTraceRole::GeneratedCode;
    trace.dtype = input.artifact_kind;
    trace.status = input.compile_success ? "ok" : "failed";
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
            "Generated-code/export trace has no artifact path.");
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
