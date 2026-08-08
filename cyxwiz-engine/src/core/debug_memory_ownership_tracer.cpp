#include "debug_memory_ownership_tracer.h"

#include <limits>

namespace cyxwiz {

namespace {

int64_t Delta(uint64_t after, uint64_t before) {
    if (after >= before) {
        const uint64_t diff = after - before;
        if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return std::numeric_limits<int64_t>::max();
        }
        return static_cast<int64_t>(diff);
    }

    const uint64_t diff = before - after;
    if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::min();
    }
    return -static_cast<int64_t>(diff);
}

bool NearBudget(uint64_t value, uint64_t budget) {
    return budget > 0 && value >= ((budget / 10) * 9);
}

} // namespace

DebugTraceRecord DebugMemoryOwnershipTracer::BuildTrace(
    const std::string& run_id,
    const DebugMemoryOwnershipInput& input) const {
    const uint64_t estimated_tensor_bytes =
        EstimateTensorBytes(input.output_shape, input.bytes_per_element);
    const bool cpu_peak_increased =
        input.after.cpu_peak_bytes > input.before.cpu_peak_bytes;
    const bool device_locked_increased =
        input.after.af_locked_bytes > input.before.af_locked_bytes;
    const bool host_oom_risk =
        NearBudget(input.after.cpu_peak_bytes, input.host_budget_bytes);
    const bool device_oom_risk =
        NearBudget(input.after.af_locked_bytes, input.device_budget_bytes);

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        input.node_id,
        input.node_name,
        input.node_type,
        input.phase,
        input.role,
        {},
        input.output_shape,
        input.dtype,
        input.backend,
        "ok");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "memory_ownership",
        "DebugMemoryOwnershipTracer",
        "cyxwiz-engine/src/core/debug_memory_ownership_tracer.cpp",
        "cyxwiz::DebugMemoryOwnershipTracer::BuildTrace");

    trace.payload["memory_schema"] = kSchema;
    trace.payload["memory_observation"] = "training_trace_delta";
    trace.payload["ownership_proven"] = false;
    trace.payload["ownership_note"] =
        "Per-node memory is inferred from before/after training snapshots; "
        "allocator-level tensor ownership is not proven in this slice.";
    trace.payload["estimated_tensor_bytes"] = estimated_tensor_bytes;
    trace.payload["bytes_per_element"] = input.bytes_per_element;
    trace.payload["cpu_allocated_before_bytes"] =
        input.before.cpu_allocated_bytes;
    trace.payload["cpu_allocated_after_bytes"] =
        input.after.cpu_allocated_bytes;
    trace.payload["cpu_allocated_delta_bytes"] = Delta(
        input.after.cpu_allocated_bytes,
        input.before.cpu_allocated_bytes);
    trace.payload["cpu_peak_before_bytes"] = input.before.cpu_peak_bytes;
    trace.payload["cpu_peak_after_bytes"] = input.after.cpu_peak_bytes;
    trace.payload["cpu_peak_increased"] = cpu_peak_increased;
    trace.payload["af_allocated_before_bytes"] =
        input.before.af_allocated_bytes;
    trace.payload["af_allocated_after_bytes"] =
        input.after.af_allocated_bytes;
    trace.payload["af_allocated_delta_bytes"] = Delta(
        input.after.af_allocated_bytes,
        input.before.af_allocated_bytes);
    trace.payload["af_locked_before_bytes"] = input.before.af_locked_bytes;
    trace.payload["af_locked_after_bytes"] = input.after.af_locked_bytes;
    trace.payload["af_locked_delta_bytes"] = Delta(
        input.after.af_locked_bytes,
        input.before.af_locked_bytes);
    trace.payload["af_alloc_buffers_before"] = input.before.af_alloc_buffers;
    trace.payload["af_alloc_buffers_after"] = input.after.af_alloc_buffers;
    trace.payload["af_lock_buffers_before"] = input.before.af_lock_buffers;
    trace.payload["af_lock_buffers_after"] = input.after.af_lock_buffers;
    trace.payload["device_locked_increased"] = device_locked_increased;
    trace.payload["host_budget_bytes"] = input.host_budget_bytes;
    trace.payload["device_budget_bytes"] = input.device_budget_bytes;
    trace.payload["host_oom_risk"] = host_oom_risk;
    trace.payload["device_oom_risk"] = device_oom_risk;

    if (host_oom_risk) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Host memory peak is close to the configured debug budget.");
    }
    if (device_oom_risk) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Device locked memory is close to the configured debug budget.");
    }
    trace.payload["success"] = trace.status == "ok";

    return trace;
}

uint64_t DebugMemoryOwnershipTracer::EstimateTensorBytes(
    const std::vector<size_t>& shape,
    uint64_t bytes_per_element) {
    if (shape.empty() || bytes_per_element == 0) {
        return 0;
    }

    uint64_t elements = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (elements > std::numeric_limits<uint64_t>::max() /
                           static_cast<uint64_t>(dim)) {
            return std::numeric_limits<uint64_t>::max();
        }
        elements *= static_cast<uint64_t>(dim);
    }

    if (elements > std::numeric_limits<uint64_t>::max() /
                       bytes_per_element) {
        return std::numeric_limits<uint64_t>::max();
    }
    return elements * bytes_per_element;
}

} // namespace cyxwiz
