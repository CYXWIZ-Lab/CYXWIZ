#include "debug_graph_trace_executor.h"

namespace cyxwiz {

std::vector<DebugTraceRecord> DebugGraphTraceExecutor::TraceSteps(
    const std::string& run_id,
    const std::vector<DebugGraphTraceStep>& steps) const {
    std::vector<DebugTraceRecord> traces;
    traces.reserve(steps.size());

    for (const auto& step : steps) {
        DebugTraceRecord trace = DebugNodeTraceContract::Make(
            run_id,
            step.node_id,
            step.node_name,
            step.node_type,
            step.phase,
            step.role,
            step.input_shape,
            step.output_shape,
            step.dtype,
            step.backend,
            step.status);
        trace.duration_ms = step.duration_ms;
        DebugNodeTraceContract::AttachDiagnosticContext(
            trace,
            "graph_trace_step",
            "DebugGraphTraceExecutor",
            "cyxwiz-engine/src/core/debug_graph_trace_executor.cpp",
            "cyxwiz::DebugGraphTraceExecutor::TraceSteps");

        for (auto it = step.payload.begin(); it != step.payload.end(); ++it) {
            trace.payload[it.key()] = it.value();
        }

        for (const auto& warning : step.warnings) {
            DebugNodeTraceContract::AddWarning(trace, warning);
        }
        for (const auto& error : step.errors) {
            DebugNodeTraceContract::AddError(trace, error);
        }
        traces.push_back(std::move(trace));
    }

    return traces;
}

} // namespace cyxwiz
