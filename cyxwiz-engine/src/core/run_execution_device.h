#pragma once

// The device a training run uses, chosen the same way for every kind of run
// (supervised training and PPO; TOFIX206): the user's route from Preferences >
// Devices and the next-run execution policy (Strict refuses a route that fails
// its check, Compatibility falls back to the ArrayFire CPU route and says so),
// activated on the calling worker thread, and the bind recorded in the runtime
// log and the training trace with the run's id.

#include "execution_device_preferences.h"

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

// The run's policy: the queued next-run policy when there is one, else
// Strict when `strict_by_default`, else Compatibility.
ExecutionPolicy ResolveRunExecutionPolicy(bool strict_by_default, const char* who);

// Activates the run's route on this thread (PrepareExecutionDeviceForRun).
// Throws with the reason when no route can run; a Compatibility fallback is
// returned with selection_fallback_applied and its reason in `error`.
ExecutionDeviceContext PrepareRunExecutionDevice(ExecutionPolicy policy, const char* who,
                                                 const RouteUnsupportedReason& route_unsupported = {});

// "ExecutionDeviceContext.Bind" in the runtime log (category "device") and the
// training trace, under `run_id`.
void RecordRunExecutionDeviceBind(const ExecutionDeviceContext& context, const std::string& run_id,
                                  const char* who);

// One training run per GPU on this machine (TOFIX206 3.2 E, owner D9): a run on
// a GPU route holds its device's lease for its whole life, across Engines and
// tools, so runs never stack their traffic on one GPU; another run waits for
// it. Keyed by the device's physical identity (one GPU through CUDA or OpenCL
// is one lease, two cards of one model are two); without one, by its name
// and index. Released by the lease's end, or by the system when its process
// dies. Hold and release it on the run's own thread. Runs of another user
// account on the same computer are excluded only where the system lets them
// share the lock (Windows: the first user's lock is theirs alone).
class GpuRunLease {
public:
    ~GpuRunLease();
    GpuRunLease(const GpuRunLease&) = delete;
    GpuRunLease& operator=(const GpuRunLease&) = delete;
    struct Handle;  // the system's lock, defined with AcquireGpuRunLease (the one place that makes it)
    explicit GpuRunLease(std::unique_ptr<Handle> handle);

private:
    std::unique_ptr<Handle> handle_;
};

struct GpuRunLeaseResult {
    std::unique_ptr<GpuRunLease> lease;  // null for a CPU route, or when cancelled
    bool cancelled = false;              // `cancelled` said so while it waited
    std::string error;                   // the lease could not be taken (the run goes on without it)
};

// The lease of the context's GPU, or none for the ArrayFire CPU route. While
// another run holds it: on_wait once with what to tell the user ("Waiting for
// the GPU: another training run is using GeForce GTX 1050 Ti"), then a check
// of `cancelled` every 200 ms.
GpuRunLeaseResult AcquireGpuRunLease(const ExecutionDeviceContext& context, const std::function<bool()>& cancelled,
                                     const std::function<void(const std::string&)>& on_wait);

}  // namespace cyxwiz
