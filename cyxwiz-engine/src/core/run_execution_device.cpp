#include "run_execution_device.h"

#include "error_codes.h"
#include "runtime_log_store.h"
#include "training_trace_collector.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <algorithm>
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


// ---- One training run per GPU (TOFIX206 3.2 E)

struct GpuRunLease::Handle {
#ifdef _WIN32
    HANDLE mutex = nullptr;
    ~Handle() {
        if (mutex) {
            ReleaseMutex(mutex);
            CloseHandle(mutex);
        }
    }
#else
    int fd = -1;
    ~Handle() {
        if (fd >= 0) {
            flock(fd, LOCK_UN);
            close(fd);
        }
    }
#endif
};

GpuRunLease::GpuRunLease(std::unique_ptr<Handle> handle) : handle_(std::move(handle)) {}
GpuRunLease::~GpuRunLease() = default;

namespace {

// The device's name as a lock name: letters and digits, the rest '_'
std::string LeaseName(const ExecutionDeviceContext& context) {
    std::string name = context.device_name.empty() ? context.effective_backend : context.device_name;
    for (char& c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) c = '_';
    return "CyxWiz.GpuRun." + name;
}

}  // namespace

GpuRunLeaseResult AcquireGpuRunLease(const ExecutionDeviceContext& context, const std::function<bool()>& cancelled,
                                     const std::function<void(const std::string&)>& on_wait) {
    GpuRunLeaseResult result;
    if (!context.valid || context.effective_backend == "arrayfire_cpu") return result;
    const std::string name = LeaseName(context);
    // ArrayFire names devices with '_' for spaces: the user reads the name
    std::string device = context.device_name.empty() ? context.effective_backend : context.device_name;
    std::replace(device.begin(), device.end(), '_', ' ');
    auto handle = std::make_unique<GpuRunLease::Handle>();
    bool said = false;
    const auto wait_turn = [&]() {
        if (!said) {
            said = true;
            const std::string message = "Waiting for the GPU: another training run is using " + device;
            spdlog::info("{}", message);
            if (on_wait) on_wait(message);
        }
        return !(cancelled && cancelled());
    };
#ifdef _WIN32
    const std::wstring wide(name.begin(), name.end());
    handle->mutex = CreateMutexW(nullptr, FALSE, (L"Local\\" + wide).c_str());
    if (!handle->mutex) {
        result.error = "the GPU lease could not be created (error " + std::to_string(GetLastError()) + ")";
        return result;
    }
    for (;;) {
        const DWORD waited = WaitForSingleObject(handle->mutex, said ? 200 : 0);
        if (waited == WAIT_OBJECT_0 || waited == WAIT_ABANDONED) break;  // abandoned: its run's process died
        if (waited != WAIT_TIMEOUT) {
            result.error = "waiting for the GPU lease failed (error " + std::to_string(GetLastError()) + ")";
            return result;
        }
        if (!wait_turn()) {
            result.cancelled = true;
            return result;
        }
    }
#else
    const auto path = std::filesystem::temp_directory_path() / (name + ".lock");
    handle->fd = open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (handle->fd < 0) {
        result.error = "the GPU lease file could not be opened: " + path.string();
        return result;
    }
    while (flock(handle->fd, LOCK_EX | LOCK_NB) != 0) {
        if (!wait_turn()) {
            result.cancelled = true;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
#endif
    if (said) spdlog::info("The GPU is free: the run starts on {}", device);
    result.lease = std::unique_ptr<GpuRunLease>(new GpuRunLease(std::move(handle)));
    return result;
}
}  // namespace cyxwiz
