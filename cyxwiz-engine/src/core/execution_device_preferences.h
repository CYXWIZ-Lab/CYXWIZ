#pragma once

#include "algorithms/arrayfire_backend_utils.h"
#include "execution_device_context.h"
#include "route_qualification_snapshot.h"

#include <cyxwiz/cyxwiz.h>

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
    Complete
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
    CommitFailed
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

inline void SetPendingExecutionDeviceSelection(DeviceType type, int device_id) {
    if (!IsArrayFireExecutionDeviceSelection(type)) {
        throw std::invalid_argument(
            "Requested device type is not a supported ArrayFire execution backend");
    }

    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot() =
        PendingExecutionDeviceSelection{type, device_id, {}};
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
}

inline std::optional<PendingExecutionDeviceSelection>
GetPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    return PendingExecutionDeviceSelectionSlot();
}

inline void ClearPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot().reset();
}

inline void ClearSavedExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    SavedExecutionDeviceSelectionSlot().reset();
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

inline DeviceSelectionTransactionResult CommitExecutionDeviceSelection(
    const PendingExecutionDeviceSelection& candidate,
    std::function<void(const PendingExecutionDeviceSelection&)>
        commit_override = {}) {
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
    hooks.inventory = [] { return Device::GetAvailableDevices(); };
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
    return RunDeviceSelectionTransaction(candidate, hooks);
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

inline ExecutionDeviceContext PrepareExecutionDeviceForRun(
    ArrayFireFallbackPolicy fallback_policy) {
    const auto pending = GetPendingExecutionDeviceSelection();
    const auto inventory = Device::GetAvailableDevices();

    DeviceType requested_type = DeviceType::CPU;
    int requested_device_id = 0;
    std::string route_resolution_error;
    if (pending.has_value()) {
        requested_type = pending->type;
        requested_device_id = pending->device_id;
        if (!pending->physical_fingerprint.empty()) {
            const auto resolution = ResolvePhysicalDeviceRoute(
                inventory,
                pending->type,
                pending->physical_fingerprint);
            if (resolution.status == DeviceRouteResolutionStatus::Resolved) {
                requested_device_id = resolution.device_id;
            } else {
                route_resolution_error =
                    "Saved physical device identity could not be resolved uniquely";
            }
        }
    } else if (const auto saved = GetSavedExecutionDeviceSelection()) {
        requested_type = saved->type;
        requested_device_id = saved->device_id;
        if (!saved->physical_fingerprint.empty()) {
            const auto resolution = ResolvePhysicalDeviceRoute(
                inventory,
                saved->type,
                saved->physical_fingerprint);
            if (resolution.status == DeviceRouteResolutionStatus::Resolved) {
                requested_device_id = resolution.device_id;
            } else {
                route_resolution_error =
                    "Saved physical device identity could not be resolved uniquely";
            }
        }
    } else if (const auto* current = Device::GetCurrentDevice()) {
        requested_type = current->GetType();
        requested_device_id = current->GetDeviceId();
    } else {
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
        if (fallback_policy ==
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback) {
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
                effective_qualification.message + ")");
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
        CaptureCurrentExecutionDeviceContext(fallback_policy);
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

inline std::mutex& NextRunExecutionPolicyMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<ArrayFireFallbackPolicy>&
NextRunExecutionPolicySlot() {
    static std::optional<ArrayFireFallbackPolicy> policy;
    return policy;
}

inline void SetNextRunExecutionPolicy(ArrayFireFallbackPolicy policy) {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    NextRunExecutionPolicySlot() = policy;
}

inline std::optional<ArrayFireFallbackPolicy> GetNextRunExecutionPolicy() {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    return NextRunExecutionPolicySlot();
}

inline void ClearNextRunExecutionPolicy() {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    NextRunExecutionPolicySlot().reset();
}

inline const char* ExecutionPolicyDisplayName(
    ArrayFireFallbackPolicy policy) {
    return policy == ArrayFireFallbackPolicy::ForbidNativeCpuFallback
        ? "Strict ArrayFire residency"
        : "Compatibility with recorded fallback";
}

} // namespace cyxwiz
