#include "run_execution_device.h"

#include "error_codes.h"
#include "runtime_log_store.h"
#include "training_trace_collector.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace cyxwiz {

ExecutionPolicy ResolveRunExecutionPolicy(bool strict_by_default, const char* who) {
    const auto queued = GetNextRunExecutionPolicy();
    if (queued.has_value())
        spdlog::info("{}: applied next-run execution policy: {}", who, ExecutionPolicyDisplayName(*queued));
    const ExecutionPolicy policy =
        queued.has_value() ? *queued : (strict_by_default ? ExecutionPolicy::Strict : ExecutionPolicy::Compatibility);
    spdlog::info("{}: execution policy {}", who, ExecutionPolicyDisplayName(policy));
    return policy;
}

ExecutionDeviceContext PrepareRunExecutionDevice(ExecutionPolicy policy, const char* who,
                                                 const RouteUnsupportedReason& route_unsupported) {
    ExecutionDeviceContext context = PrepareExecutionDeviceForRun(policy, route_unsupported);
    if (context.selection_fallback_applied) spdlog::warn("{}: {}", who, context.error);
    spdlog::info("{}: ArrayFire device preflight completed: {}", who, context.Describe());
    return context;
}

void RecordRunExecutionDeviceBind(const ExecutionDeviceContext& context, const std::string& run_id,
                                  const char* who) {
    TrainingTraceCollector::Instance().RecordExecutionDeviceContext(context);
    RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::now();
    event.level = context.valid ? RuntimeLogLevel::Info : RuntimeLogLevel::Warning;
    event.category = "device";
    event.source = who;
    event.event_name = "ExecutionDeviceContext.Bind";
    event.run_id = run_id;
    event.backend = context.effective_backend;
    event.device_id = context.effective_device_id;
    event.device_name = context.device_name;
    event.message = context.Describe();
    if (!context.valid) event.primary_error_code = errors::Runtime::ExecutionFailed;
    event.details.emplace_back("requested_backend", context.requested_backend);
    event.details.emplace_back("requested_device_id", std::to_string(context.requested_device_id));
    event.details.emplace_back("execution_policy", context.ExecutionPolicyName());
    RuntimeLogStore::Instance().Append(std::move(event));
}

}  // namespace cyxwiz
