#include "run_execution_device.h"

#include "error_codes.h"
#include "runtime_log_store.h"
#include "training_trace_collector.h"

#include <spdlog/spdlog.h>

#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
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
    bool owned = false;  // the wait succeeded: only then is there a lock to release
#ifdef _WIN32
    HANDLE mutex = nullptr;
    ~Handle() {
        if (!mutex) return;
        if (owned) ReleaseMutex(mutex);
        CloseHandle(mutex);
    }
#else
    int fd = -1;
    ~Handle() {
        if (fd < 0) return;
        if (owned) flock(fd, LOCK_UN);
        close(fd);
    }
#endif
};

GpuRunLease::GpuRunLease(std::unique_ptr<Handle> handle) : handle_(std::move(handle)) {}
GpuRunLease::~GpuRunLease() = default;

namespace {

// The device as a lock name: its physical identity (the same card through
// CUDA or OpenCL, apart from another card of its model), else its name and
// index; letters and digits, the rest '_'
std::string LeaseName(const ExecutionDeviceContext& context) {
    std::string name = context.physical_fingerprint;
    if (name.empty())
        name = (context.device_name.empty() ? context.effective_backend : context.device_name) + " " +
               std::to_string(context.effective_device_id);
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
    std::string device = context.device_name.empty() ? context.effective_backend : context.device_name;
    // ArrayFire names devices with '_' for spaces: the user reads the name
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
    // Every session of the computer where the system allows it, else this one
    const std::wstring wide(name.begin(), name.end());
    handle->mutex = CreateMutexW(nullptr, FALSE, (L"Global\\" + wide).c_str());
    if (!handle->mutex) handle->mutex = CreateMutexW(nullptr, FALSE, (L"Local\\" + wide).c_str());
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
    // One file for every user of the computer (read-only is enough to lock
    // it); not inherited by the processes the run starts
    const std::string path = "/tmp/" + name + ".lock";
    handle->fd = open(path.c_str(), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
    // Another user's file: some systems refuse O_CREAT on it in a shared folder
    if (handle->fd < 0) handle->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (handle->fd < 0) {
        result.error = "the GPU lease file could not be opened: " + path;
        return result;
    }
    while (flock(handle->fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EINTR) continue;
        if (errno != EWOULDBLOCK) {  // locks not supported here: not "another run"
            result.error = "the GPU lease file could not be locked (error " + std::to_string(errno) + "): " + path;
            return result;
        }
        if (!wait_turn()) {
            result.cancelled = true;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
#endif
    handle->owned = true;
    if (said) spdlog::info("The GPU is free: the run starts on {}", device);
    result.lease = std::unique_ptr<GpuRunLease>(new GpuRunLease(std::move(handle)));
    return result;
}
}  // namespace cyxwiz
