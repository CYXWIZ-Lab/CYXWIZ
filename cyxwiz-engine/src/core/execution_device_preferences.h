#pragma once

#include "algorithms/arrayfire_backend_utils.h"
#include "execution_device_context.h"
#include "route_qualification_snapshot.h"

#include <cyxwiz/cyxwiz.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>

namespace cyxwiz {

struct PendingExecutionDeviceSelection {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string physical_fingerprint;
};

enum class DeviceSelectionTransactionStage {
    NotStarted,
    Inventory,
    Qualification,
    Activation,
    Restore,
    Revalidation,
    Commit,
    Complete,
    ExecutionGuard
};

enum class DeviceSelectionTransactionStatus {
    Committed,
    InvalidBackend,
    RouteNotFound,
    IdentityMismatch,
    NotQualified,
    ActivationFailed,
    EffectiveRouteMismatch,
    RestoreFailed,
    RevalidationFailed,
    CommitFailed,
    ExecutionActive
};

struct DeviceSelectionTransactionResult {
    DeviceSelectionTransactionStage stage =
        DeviceSelectionTransactionStage::NotStarted;
    DeviceSelectionTransactionStatus status =
        DeviceSelectionTransactionStatus::RouteNotFound;
    bool committed = false;
    DeviceActivationResult activation;
    std::string message;
};

struct DeviceSelectionTransactionHooks {
    std::function<std::vector<DeviceInfo>()> inventory;
    std::function<bool(const DeviceInfo&, std::string&)> qualify;
    std::function<DeviceActivationResult(DeviceType, int)> activate;
    std::function<DeviceActivationResult()> restore;
    std::function<void(const PendingExecutionDeviceSelection&)> commit;
};

inline std::string ExecutionDeviceSelectionBackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "arrayfire_cpu";
        case DeviceType::CUDA: return "arrayfire_cuda";
        case DeviceType::OPENCL: return "arrayfire_opencl";
        case DeviceType::ONEAPI: return "arrayfire_oneapi";
        case DeviceType::METAL: return "unsupported_metal";
        case DeviceType::VULKAN: return "unsupported_vulkan";
        default: return "arrayfire_unknown";
    }
}

inline bool IsArrayFireExecutionDeviceSelection(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
        case DeviceType::CUDA:
        case DeviceType::OPENCL:
        case DeviceType::ONEAPI:
            return true;
        case DeviceType::METAL:
        case DeviceType::VULKAN:
        default:
            return false;
    }
}

inline std::mutex& PendingExecutionDeviceSelectionMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<PendingExecutionDeviceSelection>&
PendingExecutionDeviceSelectionSlot() {
    static std::optional<PendingExecutionDeviceSelection> selection;
    return selection;
}

inline std::optional<PendingExecutionDeviceSelection>&
SavedExecutionDeviceSelectionSlot() {
    static std::optional<PendingExecutionDeviceSelection> selection;
    return selection;
}

// The requested route's resolution for one selection, and the device
// inventory it was read from. Compile's provider target
// (RequestedRouteNeuralTarget) and the readiness preview
// (PreviewRequestedRouteTrainingReadiness) both read it, so two Compiles never
// disagree about the device and no Compile enumerates devices
// (Device::GetAvailableDevices probes every ArrayFire backend). It is filled
// whenever an inventory is read for the selection: the Engine's start, a
// committed selection, Preferences > Devices (its device list, read again on
// Refresh devices and after Verify), the run preflight, and, when none was
// read since the selection changed, the GUI Compile itself, once
// (EnsureExecutionRouteResolution). Keyed by the selection (pending, else
// saved; none: the process device), so a changed selection never reads
// another's answer; it is also dropped when the selection changes.
struct ExecutionRouteResolution {
    std::optional<PendingExecutionDeviceSelection> selection;  // what it was made for
    bool route_available = false;  // a route was found at all
    DeviceType type = DeviceType::CPU;  // the route a run would request
    int device_id = 0;
    // False when a saved physical identity matched no device of the
    // inventory (unplugged) or several; device_id is then the stored index
    bool resolved = true;
    std::string error;  // why it did not resolve
    std::vector<DeviceInfo> inventory;
    bool warned = false;  // Compile logged the unresolved route once
};

inline std::optional<ExecutionRouteResolution>& ExecutionRouteResolutionSlot() {
    static std::optional<ExecutionRouteResolution> resolution;
    return resolution;
}

inline bool SameExecutionDeviceSelection(const PendingExecutionDeviceSelection& a,
                                         const PendingExecutionDeviceSelection& b) {
    return a.type == b.type && a.device_id == b.device_id && a.physical_fingerprint == b.physical_fingerprint;
}

inline bool SameExecutionDeviceSelection(const std::optional<PendingExecutionDeviceSelection>& a,
                                         const std::optional<PendingExecutionDeviceSelection>& b) {
    if (!a || !b) return !a && !b;
    return SameExecutionDeviceSelection(*a, *b);
}

// The selection a run would request: pending, else saved. Caller holds
// PendingExecutionDeviceSelectionMutex.
inline std::optional<PendingExecutionDeviceSelection> RequestedExecutionDeviceSelectionLocked() {
    const auto& pending = PendingExecutionDeviceSelectionSlot();
    return pending ? pending : SavedExecutionDeviceSelectionSlot();
}

// Drops the resolution once the selection it was made for is no longer the
// requested one. Caller holds PendingExecutionDeviceSelectionMutex.
inline void DropStaleExecutionRouteResolutionLocked() {
    auto& resolution = ExecutionRouteResolutionSlot();
    if (resolution && !SameExecutionDeviceSelection(resolution->selection, RequestedExecutionDeviceSelectionLocked()))
        resolution.reset();
}

inline void RememberExecutionRouteResolution(ExecutionRouteResolution resolution) {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    auto& slot = ExecutionRouteResolutionSlot();
    // The same answer again keeps its "warned once"
    resolution.warned = slot && SameExecutionDeviceSelection(slot->selection, resolution.selection) &&
                        slot->resolved == resolution.resolved && slot->device_id == resolution.device_id &&
                        slot->warned;
    slot = std::move(resolution);
}

// The resolution of the selection a run would request now, when an
// inventory was read for it; nullopt when none was. Never enumerates.
inline std::optional<ExecutionRouteResolution> CachedExecutionRouteResolution() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    const auto& resolution = ExecutionRouteResolutionSlot();
    if (!resolution || !SameExecutionDeviceSelection(resolution->selection, RequestedExecutionDeviceSelectionLocked()))
        return std::nullopt;
    return resolution;
}

// True once per resolution: the first caller to report that `selection`'s
// cached resolution did not resolve.
inline bool TakeUnresolvedExecutionRouteWarning(const PendingExecutionDeviceSelection& selection) {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    auto& resolution = ExecutionRouteResolutionSlot();
    if (!resolution || resolution->warned || !resolution->selection ||
        !SameExecutionDeviceSelection(*resolution->selection, selection))
        return false;
    resolution->warned = true;
    return true;
}

inline void SetPendingExecutionDeviceSelection(DeviceType type, int device_id) {
    if (!IsArrayFireExecutionDeviceSelection(type)) {
        throw std::invalid_argument(
            "Requested device type is not a supported ArrayFire execution backend");
    }

    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot() =
        PendingExecutionDeviceSelection{type, device_id, {}};
    DropStaleExecutionRouteResolutionLocked();
}

inline std::optional<PendingExecutionDeviceSelection>
GetSavedExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    return SavedExecutionDeviceSelectionSlot();
}

inline void CommitExecutionDeviceSelectionState(
    const PendingExecutionDeviceSelection& selection) {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    SavedExecutionDeviceSelectionSlot() = selection;
    PendingExecutionDeviceSelectionSlot() = selection;
    DropStaleExecutionRouteResolutionLocked();
    Device::RecordProcessDevice(selection.type, selection.device_id);
}

inline std::optional<PendingExecutionDeviceSelection>
GetPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    return PendingExecutionDeviceSelectionSlot();
}

// A run's preflight clears the pending selection; the saved one it equals
// keeps its resolution.
inline void ClearPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot().reset();
    DropStaleExecutionRouteResolutionLocked();
}

inline void ClearSavedExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    SavedExecutionDeviceSelectionSlot().reset();
    DropStaleExecutionRouteResolutionLocked();
}

inline const char* DeviceSelectionTransactionStageName(
    DeviceSelectionTransactionStage stage) {
    switch (stage) {
        case DeviceSelectionTransactionStage::Inventory: return "inventory";
        case DeviceSelectionTransactionStage::Qualification: return "qualification";
        case DeviceSelectionTransactionStage::Activation: return "activation";
        case DeviceSelectionTransactionStage::Restore: return "restore";
        case DeviceSelectionTransactionStage::Revalidation: return "revalidation";
        case DeviceSelectionTransactionStage::Commit: return "commit";
        case DeviceSelectionTransactionStage::Complete: return "complete";
        case DeviceSelectionTransactionStage::ExecutionGuard:
            return "execution_guard";
        default: return "not_started";
    }
}

inline DeviceSelectionTransactionResult RunDeviceSelectionTransaction(
    const PendingExecutionDeviceSelection& candidate,
    const DeviceSelectionTransactionHooks& hooks) {
    DeviceSelectionTransactionResult result;
    if (!IsArrayFireExecutionDeviceSelection(candidate.type)) {
        result.status = DeviceSelectionTransactionStatus::InvalidBackend;
        result.message = "Candidate is not a supported ArrayFire backend";
        return result;
    }

    const auto fail = [&](DeviceSelectionTransactionStage stage,
                          DeviceSelectionTransactionStatus status,
                          std::string message) {
        result.stage = stage;
        result.status = status;
        result.message = std::move(message);
        return result;
    };
    const auto find_candidate = [&](const std::vector<DeviceInfo>& inventory)
        -> const DeviceInfo* {
        for (const auto& device : inventory) {
            if (device.type == candidate.type &&
                device.device_id == candidate.device_id) {
                return &device;
            }
        }
        return nullptr;
    };
    const auto identity_matches = [&](const DeviceInfo& device) {
        return candidate.physical_fingerprint.empty() ||
            (device.physical_fingerprint_known &&
             device.physical_fingerprint == candidate.physical_fingerprint);
    };

    std::vector<DeviceInfo> inventory;
    try {
        result.stage = DeviceSelectionTransactionStage::Inventory;
        inventory = hooks.inventory();
    } catch (const std::exception& error) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::RouteNotFound,
                    error.what());
    }
    const DeviceInfo* discovered = find_candidate(inventory);
    if (!discovered) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::RouteNotFound,
                    "Candidate route is not present in the current inventory");
    }
    if (!identity_matches(*discovered)) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::IdentityMismatch,
                    "Candidate physical identity no longer matches its backend ordinal");
    }

    result.stage = DeviceSelectionTransactionStage::Qualification;
    std::string qualification_message;
    bool qualified = false;
    try {
        qualified = hooks.qualify(*discovered, qualification_message);
    } catch (const std::exception& error) {
        qualification_message = error.what();
    }
    if (!qualified) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::NotQualified,
                    std::move(qualification_message));
    }

    result.stage = DeviceSelectionTransactionStage::Activation;
    try {
        result.activation = hooks.activate(candidate.type, candidate.device_id);
    } catch (const std::exception& error) {
        result.activation.message = error.what();
    }

    result.stage = DeviceSelectionTransactionStage::Restore;
    DeviceActivationResult restoration;
    try {
        restoration = hooks.restore();
    } catch (const std::exception& error) {
        restoration.message = error.what();
    }
    if (!restoration.success) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::RestoreFailed,
                    restoration.message.empty()
                        ? "Previous process route could not be restored"
                        : restoration.message);
    }
    if (!result.activation.success) {
        return fail(DeviceSelectionTransactionStage::Activation,
                    DeviceSelectionTransactionStatus::ActivationFailed,
                    result.activation.message);
    }
    if (result.activation.effective_type != candidate.type ||
        result.activation.effective_device_id != candidate.device_id ||
        !result.activation.execution_validated) {
        return fail(DeviceSelectionTransactionStage::Activation,
                    DeviceSelectionTransactionStatus::EffectiveRouteMismatch,
                    "Candidate did not validate on the exact requested route");
    }

    result.stage = DeviceSelectionTransactionStage::Revalidation;
    try {
        inventory = hooks.inventory();
    } catch (const std::exception& error) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::RevalidationFailed,
                    error.what());
    }
    discovered = find_candidate(inventory);
    if (!discovered || !identity_matches(*discovered)) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::RevalidationFailed,
                    "Candidate changed or disappeared before commit");
    }

    result.stage = DeviceSelectionTransactionStage::Commit;
    try {
        hooks.commit(candidate);
    } catch (const std::exception& error) {
        return fail(result.stage,
                    DeviceSelectionTransactionStatus::CommitFailed,
                    error.what());
    }
    result.stage = DeviceSelectionTransactionStage::Complete;
    result.status = DeviceSelectionTransactionStatus::Committed;
    result.committed = true;
    result.message = "Selection committed for the next training run";
    return result;
}

inline void RefreshExecutionRouteResolution(std::vector<DeviceInfo> inventory);  // below

inline DeviceSelectionTransactionResult CommitExecutionDeviceSelection(
    const PendingExecutionDeviceSelection& candidate,
    std::function<void(const PendingExecutionDeviceSelection&)>
        commit_override = {}) {
    runtime::RuntimeMutationLease runtime_mutation;
    if (!runtime_mutation.OwnsMutation() ||
        HasActiveExecutionDeviceContext()) {
        DeviceSelectionTransactionResult result;
        result.stage = DeviceSelectionTransactionStage::ExecutionGuard;
        result.status = DeviceSelectionTransactionStatus::ExecutionActive;
        result.message =
            "Device selection is blocked while an execution context is active";
        return result;
    }

    std::optional<PendingExecutionDeviceSelection> original;
    if (const auto* current = Device::GetCurrentDevice()) {
        const DeviceInfo info = current->GetInfo();
        original = PendingExecutionDeviceSelection{
            current->GetType(),
            current->GetDeviceId(),
            info.physical_fingerprint_known
                ? info.physical_fingerprint
                : std::string{}};
    }

    DeviceSelectionTransactionHooks hooks;
    // The inventory last read (the revalidation's, at a commit): the new
    // selection is resolved against it, without another enumeration
    auto last_inventory = std::make_shared<std::vector<DeviceInfo>>();
    hooks.inventory = [last_inventory] {
        *last_inventory = Device::GetAvailableDevices();
        return *last_inventory;
    };
    hooks.qualify = [](const DeviceInfo& info, std::string& message) {
        const auto qualification = EvaluateRouteQualification(info);
        const auto authorization =
            EvaluateRouteTrainingAuthorization(info, qualification);
        message = authorization.message;
        return authorization.authorized;
    };
    hooks.activate = [](DeviceType type, int device_id) {
        return Device(type, device_id).ActivateExact(true);
    };
    hooks.restore = [original] {
        if (!original.has_value()) {
            DeviceActivationResult failure;
            failure.message = "Previous process route was unavailable";
            return failure;
        }
        return Device(original->type, original->device_id).ActivateExact(false);
    };
    hooks.commit = commit_override
        ? std::move(commit_override)
        : std::function<void(const PendingExecutionDeviceSelection&)>(
              [](const PendingExecutionDeviceSelection& selection) {
                  CommitExecutionDeviceSelectionState(selection);
              });
    auto result = RunDeviceSelectionTransaction(candidate, hooks);
    if (result.committed) RefreshExecutionRouteResolution(*last_inventory);
    return result;
}

inline std::string FormatActivationFailure(
    const DeviceActivationResult& result) {
    return "stage=" +
        std::string(DeviceActivationStageName(result.stage)) +
        " error=" + std::to_string(result.error_code) + " (" +
        result.message + ")";
}

inline ExecutionRouteQualification MakeExecutionRouteQualification(
    const RouteQualificationDecision& decision) {
    ExecutionRouteQualification qualification;
    qualification.evidence_available = decision.evidence_available;
    qualification.qualified = decision.qualified;
    qualification.matrix_id = decision.matrix_id;
    qualification.message = decision.message;
    return qualification;
}

inline DeviceActivationResult MakeRouteQualificationFailure(
    DeviceType type,
    int device_id,
    const std::string& message) {
    DeviceActivationResult failure;
    failure.requested_type = type;
    failure.requested_device_id = device_id;
    failure.effective_type = type;
    failure.effective_device_id = device_id;
    failure.stage = DeviceActivationStage::ExecutionValidation;
    failure.message = message;
    return failure;
}

// Why a saved physical device identity did not resolve, and what to do.
inline std::string UnresolvedExecutionRouteMessage(const PendingExecutionDeviceSelection& selection,
                                                   DeviceRouteResolutionStatus status) {
    const std::string device =
        ExecutionDeviceSelectionBackendName(selection.type) + " device " + std::to_string(selection.device_id);
    return "The device chosen in Preferences > Devices (" + device + ") " +
           (status == DeviceRouteResolutionStatus::Ambiguous
                ? std::string("matches more than one of this machine's devices")
                : std::string("is not among this machine's devices (unplugged, or its driver changed)")) +
           ". Open Preferences > Devices and choose the device again.";
}

// The route a training run will request, judged against `inventory`: the
// pending selection, else the saved selection (a saved physical identity is
// resolved in the inventory), else the device the process selected (startup
// activation or SetActive, recorded process-wide), else the calling thread's
// ArrayFire device. Remembered as the selection's resolution, a failure
// included (no earlier answer outlives it). The process record matters
// because the preflight runs on the training worker thread, where
// Device::GetCurrentDevice() reports ArrayFire's thread default (e.g. oneAPI
// on a machine without CUDA) rather than the selected route.
inline ExecutionRouteResolution ResolveExecutionRoute(std::vector<DeviceInfo> inventory) {
    ExecutionRouteResolution resolution;
    resolution.selection = GetPendingExecutionDeviceSelection();
    if (!resolution.selection) resolution.selection = GetSavedExecutionDeviceSelection();
    if (const auto& selection = resolution.selection) {
        resolution.route_available = true;
        resolution.type = selection->type;
        resolution.device_id = selection->device_id;
        if (!selection->physical_fingerprint.empty()) {
            const auto found = ResolvePhysicalDeviceRoute(inventory, selection->type, selection->physical_fingerprint);
            if (found.status == DeviceRouteResolutionStatus::Resolved) {
                resolution.device_id = found.device_id;
            } else {
                resolution.resolved = false;
                resolution.error = UnresolvedExecutionRouteMessage(*selection, found.status);
            }
        }
    } else if (const auto process = Device::GetProcessDevice()) {
        resolution.route_available = true;
        resolution.type = process->type;
        resolution.device_id = process->device_id;
    } else if (const auto* current = Device::GetCurrentDevice()) {
        resolution.route_available = true;
        resolution.type = current->GetType();
        resolution.device_id = current->GetDeviceId();
    }
    resolution.inventory = std::move(inventory);
    RememberExecutionRouteResolution(resolution);
    return resolution;
}

// ResolveExecutionRoute's route, for the run preflight; false when no route
// exists at all. route_resolution_error: why a saved identity did not resolve.
inline bool ResolveRequestedExecutionRoute(
    const std::vector<DeviceInfo>& inventory,
    DeviceType& requested_type,
    int& requested_device_id,
    std::string& route_resolution_error) {
    const auto resolution = ResolveExecutionRoute(inventory);
    requested_type = resolution.type;
    requested_device_id = resolution.device_id;
    route_resolution_error = resolution.error;
    return resolution.route_available;
}

// Resolves the requested selection against an inventory a caller has just
// read (the Engine's start, a committed selection, Preferences > Devices).
inline void RefreshExecutionRouteResolution(std::vector<DeviceInfo> inventory) {
    (void)ResolveExecutionRoute(std::move(inventory));
}

// The requested selection's resolution: the cached one, else (no inventory
// read since the selection changed) the devices are enumerated once and
// resolved. The GUI Compile calls it before compiling, so the compiler's
// provider target and the readiness preview read the same answer.
inline ExecutionRouteResolution EnsureExecutionRouteResolution() {
    if (auto cached = CachedExecutionRouteResolution()) return std::move(*cached);
    return ResolveExecutionRoute(Device::GetAvailableDevices());
}

// The device a run started now would bind, as a neural provider target, for
// Compile, whose report asks the providers about the run's device. Never
// enumerates devices (Device::GetAvailableDevices probes every ArrayFire
// backend, and Compile runs on the GUI thread, in NAS, model import and the
// server node's graph jobs):
//  - inside a run (ScopedNeuralRunTarget), the device the run bound;
//  - else the route chosen in Preferences > Devices (pending, else saved):
//    the device its cached resolution names (CachedExecutionRouteResolution);
//    without one, or when its saved physical identity did not resolve, its
//    stored backend and index. An unresolved identity is logged as a warning
//    once per resolution, with what to do;
//  - else the process device, else the calling thread's ArrayFire device.
inline NeuralDeviceTarget RequestedRouteNeuralTarget() {
    NeuralDeviceTarget target;
    if (BoundNeuralRunTarget(target)) return target;
    auto selection = GetPendingExecutionDeviceSelection();
    if (!selection) selection = GetSavedExecutionDeviceSelection();
    if (selection) {
        target.platform = selection->type;
        target.device_id = selection->device_id;
        if (selection->physical_fingerprint.empty()) return target;
        const auto resolution = CachedExecutionRouteResolution();
        if (!resolution) {
            spdlog::debug("Compile: the saved device identity was not matched to this machine's devices yet; the "
                          "report asks the providers about its stored index ({} device {})",
                          ExecutionDeviceSelectionBackendName(target.platform), target.device_id);
        } else if (resolution->resolved) {
            target.device_id = resolution->device_id;
        } else if (TakeUnresolvedExecutionRouteWarning(*selection)) {
            spdlog::warn("Compile: {} Until then the report asks the providers about its stored index.",
                         resolution->error);
        }
        return target;
    }
    if (const auto process = Device::GetProcessDevice()) {
        target.platform = process->type;
        target.device_id = process->device_id;
        return target;
    }
    return CaptureCurrentNeuralDeviceTarget();
}

struct RequestedRouteTrainingReadiness {
    bool route_available = false;   // a route could be resolved at all
    bool authorized = false;        // the requested route may train
    bool cpu_recovery_qualified = false;
    // The saved physical identity matched no device, or several; message
    // says so and what to do (UnresolvedExecutionRouteMessage)
    bool route_unresolved = false;
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string route_name;
    std::string message;            // why the requested route may not train
};

// Non-throwing preview of the run preflight's route decision, for Compile:
// tells whether training would be refused (or fall back) before a run starts.
// Judged against the selection's cached inventory
// (EnsureExecutionRouteResolution): no enumeration on each Compile.
inline RequestedRouteTrainingReadiness PreviewRequestedRouteTrainingReadiness() {
    RequestedRouteTrainingReadiness readiness;
    const auto resolution = ResolveExecutionRoute(EnsureExecutionRouteResolution().inventory);
    const auto& inventory = resolution.inventory;
    readiness.type = resolution.type;
    readiness.device_id = resolution.device_id;
    const std::string& route_error = resolution.error;
    readiness.route_unresolved = !resolution.resolved;
    if (!resolution.route_available) {
        readiness.message = "ArrayFire process device is unavailable";
        return readiness;
    }
    readiness.route_available = true;
    const DeviceInfo* route = nullptr;
    const DeviceInfo* cpu_route = nullptr;
    for (const auto& candidate : inventory) {
        if (candidate.type == readiness.type && candidate.device_id == readiness.device_id) {
            route = &candidate;
        }
        if (candidate.type == DeviceType::CPU && cpu_route == nullptr) cpu_route = &candidate;
    }
    if (cpu_route != nullptr) {
        readiness.cpu_recovery_qualified = EvaluateRouteQualification(*cpu_route).qualified;
    }
    if (!route_error.empty()) {
        readiness.message = route_error;
        return readiness;
    }
    if (route == nullptr) {
        readiness.message = "Requested route is not present in the current inventory";
        return readiness;
    }
    readiness.route_name = route->name;
    const auto qualification = EvaluateRouteQualification(*route);
    const auto authorization = EvaluateRouteTrainingAuthorization(*route, qualification);
    readiness.authorized = authorization.authorized;
    if (!authorization.authorized) readiness.message = authorization.message;
    return readiness;
}

// `route_unsupported` (optional): why the run's model cannot execute on a
// route ("" when it can), asked of the requested route before it is
// activated. A reason counts as a failed preflight: Strict refuses with it,
// Compatibility recovers to the qualified ArrayFire CPU route (TOFIX178
// 1.0(d): Conv2D on CUDA without the NVIDIA provider).
using RouteUnsupportedReason = std::function<std::string(DeviceType type, int device_id)>;

inline ExecutionDeviceContext PrepareExecutionDeviceForRun(
    ExecutionPolicy execution_policy,
    const RouteUnsupportedReason& route_unsupported = {}) {
    const auto pending = GetPendingExecutionDeviceSelection();
    const auto inventory = Device::GetAvailableDevices();
    DeviceType requested_type = DeviceType::CPU;
    int requested_device_id = 0;
    std::string route_resolution_error;
    if (!ResolveRequestedExecutionRoute(
            inventory, requested_type, requested_device_id, route_resolution_error)) {
        throw std::runtime_error(
            "ArrayFire process device is unavailable before run preflight");
    }

    const auto find_route = [&](DeviceType type, int device_id)
        -> const DeviceInfo* {
        for (const auto& candidate : inventory) {
            if (candidate.type == type &&
                candidate.device_id == device_id) {
                return &candidate;
            }
        }
        return nullptr;
    };

    const Device requested(requested_type, requested_device_id);
    RouteQualificationDecision requested_qualification;
    DeviceActivationResult requested_activation;
    if (!route_resolution_error.empty()) {
        requested_activation = MakeRouteQualificationFailure(
            requested_type, requested_device_id, route_resolution_error);
        requested_activation.stage = DeviceActivationStage::DeviceSelection;
    } else if (const std::string unsupported =
                   route_unsupported ? route_unsupported(requested_type, requested_device_id) : std::string();
               !unsupported.empty()) {
        requested_activation = MakeRouteQualificationFailure(
            requested_type, requested_device_id, unsupported);
        requested_activation.stage = DeviceActivationStage::DeviceSelection;
    } else if (const DeviceInfo* requested_route =
                   find_route(requested_type, requested_device_id)) {
        requested_qualification =
            EvaluateRouteQualification(*requested_route);
        const auto authorization = EvaluateRouteTrainingAuthorization(
            *requested_route, requested_qualification);
        if (!authorization.authorized) {
            requested_activation = MakeRouteQualificationFailure(
                requested_type,
                requested_device_id,
                authorization.message);
        } else {
            requested_activation = requested.ActivateExact(true);
        }
    } else {
        requested_activation = MakeRouteQualificationFailure(
            requested_type,
            requested_device_id,
            "Requested route is not present in the current inventory");
        requested_activation.stage = DeviceActivationStage::DeviceSelection;
    }
    DeviceActivationResult effective_activation = requested_activation;
    RouteQualificationDecision effective_qualification =
        requested_qualification;
    bool selection_fallback_applied = false;

    if (!requested_activation.success) {
        if (execution_policy ==
            ExecutionPolicy::Strict) {
            throw std::runtime_error(
                "Requested ArrayFire device preflight failed: " +
                FormatActivationFailure(requested_activation));
        }

        const DeviceInfo* cpu_route = nullptr;
        for (const auto& candidate : inventory) {
            if (candidate.type == DeviceType::CPU) {
                cpu_route = &candidate;
                break;
            }
        }
        if (cpu_route != nullptr) {
            effective_qualification =
                EvaluateRouteQualification(*cpu_route);
        } else {
            effective_qualification.message =
                "ArrayFire CPU recovery route is not present in the current inventory";
        }
        if (!effective_qualification.qualified) {
            throw std::runtime_error(
                "Requested ArrayFire device preflight failed (" +
                FormatActivationFailure(requested_activation) +
                "); ArrayFire CPU recovery is not qualified (" +
                effective_qualification.message +
                "). Verify the device in Preferences > Devices > Verify Selected "
                "(it also verifies the CPU recovery route), then train again");
        }

        effective_activation =
            Device(DeviceType::CPU, cpu_route->device_id).ActivateExact(true);
        if (!effective_activation.success) {
            throw std::runtime_error(
                "Requested ArrayFire device preflight failed (" +
                FormatActivationFailure(requested_activation) +
                "); ArrayFire CPU recovery also failed (" +
                FormatActivationFailure(effective_activation) + ")");
        }
        selection_fallback_applied = true;
    }

    ExecutionDeviceContext context =
        CaptureCurrentExecutionDeviceContext(execution_policy);
    context.requested_backend =
        ExecutionDeviceSelectionBackendName(requested_type);
    context.requested_device_id = requested_device_id;
    context.requested_qualification =
        MakeExecutionRouteQualification(requested_qualification);
    context.effective_qualification =
        MakeExecutionRouteQualification(effective_qualification);
    context.activation_succeeded = effective_activation.success;
    context.execution_validated =
        effective_activation.execution_validated;
    context.selection_fallback_applied = selection_fallback_applied;
    context.preflight_stage =
        DeviceActivationStageName(effective_activation.stage);
    context.valid = context.valid && effective_activation.success &&
                    effective_activation.execution_validated;
    if (selection_fallback_applied) {
        context.preflight_error_code = requested_activation.error_code;
        context.error = "Requested selection failed at " +
            FormatActivationFailure(requested_activation) +
            "; ArrayFire CPU preflight succeeded";
    } else {
        context.preflight_error_code = effective_activation.error_code;
        if (!effective_activation.success) {
            context.error = FormatActivationFailure(effective_activation);
        }
    }

    if (!context.valid) {
        throw std::runtime_error(
            "Execution device context preflight is invalid: " +
            context.Describe());
    }

    if (pending.has_value()) {
        ClearPendingExecutionDeviceSelection();
    }
    return context;
}

// The session's execution policy: the policy every run started from now on
// takes (training, Run Test, RL training, policy playback, the stress test),
// until it is set again. It is a standing choice, not a one-off for the next
// run: Preferences > Devices sets it ("Next runs") with the machine default,
// the Engine and headless hosts set it at start from the saved machine
// preference (ApplyMachineComputePreference), and nothing consumes it.
// Unset, each runner takes its own default (ResolveRunExecutionPolicy).
// Clear returns to that; tests use it to start from a known state.
// (Named "next run" before; renamed because it was never cleared after use.)
inline std::mutex& SessionExecutionPolicyMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<ExecutionPolicy>&
SessionExecutionPolicySlot() {
    static std::optional<ExecutionPolicy> policy;
    return policy;
}

inline void SetSessionExecutionPolicy(ExecutionPolicy policy) {
    std::lock_guard<std::mutex> lock(SessionExecutionPolicyMutex());
    SessionExecutionPolicySlot() = policy;
}

inline std::optional<ExecutionPolicy> GetSessionExecutionPolicy() {
    std::lock_guard<std::mutex> lock(SessionExecutionPolicyMutex());
    return SessionExecutionPolicySlot();
}

inline void ClearSessionExecutionPolicy() {
    std::lock_guard<std::mutex> lock(SessionExecutionPolicyMutex());
    SessionExecutionPolicySlot().reset();
}

} // namespace cyxwiz
