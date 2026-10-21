#include "backend_pack_remover.h"
#include "runtime_mutation_gate.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <system_error>
#include <utility>

namespace cyxwiz::runtime {
namespace {

bool IsIdentifier(const std::string& value) {
    if (value.empty() || value.size() > 128 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char value) {
        return std::islower(value) || std::isdigit(value) || value == '.' ||
               value == '_' || value == '-';
    });
}

bool IsOptionalBackend(const std::string& backend) {
    return backend == "cuda" || backend == "opencl" ||
           backend == "oneapi";
}

bool HasPack(
    const ActiveRuntimeState& state,
    const std::string& backend,
    const std::string& pack_id) {
    return std::any_of(
        state.packs.begin(), state.packs.end(),
        [&](const ActivePackState& pack) {
            return pack.backend == backend && pack.pack_id == pack_id;
        });
}

bool IsWithin(
    const std::filesystem::path& root,
    const std::filesystem::path& child) {
    const auto relative = child.lexically_relative(root);
    return !relative.empty() && !relative.is_absolute() &&
           *relative.begin() != "..";
}

bool ValidateRemovalTarget(
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& target,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        target, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Installed pack path is not a regular directory";
        return false;
    }
    const auto canonical_root = std::filesystem::canonical(
        runtime_root, filesystem_error);
    if (filesystem_error) {
        error = "Runtime root cannot be resolved safely";
        return false;
    }
    const auto canonical_target = std::filesystem::canonical(
        target, filesystem_error);
    if (filesystem_error || !IsWithin(canonical_root, canonical_target)) {
        error = "Installed pack resolves outside the runtime root";
        return false;
    }
    return true;
}

}  // namespace

BackendPackRemover::BackendPackRemover(
    std::filesystem::path runtime_root,
    BackendPackExecutionActiveCheck execution_active,
    BackendPackRemovalObserver observer,
    BackendPackRemovalCheckpointHook checkpoint)
    : runtime_root_(std::move(runtime_root)),
      execution_active_(std::move(execution_active)),
      observer_(std::move(observer)),
      checkpoint_(std::move(checkpoint)) {}

BackendPackRemovalResult BackendPackRemover::Remove(
    std::string backend,
    std::string pack_id) {
    std::unique_lock<std::mutex> removal_lock(
        removal_mutex_, std::try_to_lock);
    if (!removal_lock.owns_lock()) {
        return {BackendPackRemovalStatus::Busy,
                "A backend-pack removal is already running"};
    }
    cancel_requested_.store(false);
    BackendPackRemovalProgress progress;
    progress.stage = BackendPackRemovalStage::Validating;
    progress.backend = backend;
    progress.pack_id = pack_id;
    progress.message = "Validating the installed pack and active runtime";
    SetProgress(progress);

    if (!IsOptionalBackend(backend) || !IsIdentifier(pack_id) ||
        runtime_root_.empty() || !runtime_root_.is_absolute()) {
        return Finish(
            BackendPackRemovalStatus::InvalidRequest,
            "A safe optional backend, pack identity, and runtime root are required");
    }
    RuntimeMutationLease mutation;
    if (!mutation.OwnsMutation() ||
        (execution_active_ && execution_active_())) {
        return Finish(
            BackendPackRemovalStatus::ExecutionActive,
            "Backend-pack removal is blocked while an execution context is active");
    }

    ActiveRuntimeState active;
    std::string error;
    if (!LoadActiveRuntimeState(
            runtime_root_ / "active-runtime.json", active, error)) {
        return Finish(BackendPackRemovalStatus::InvalidRuntime, error);
    }
    const auto target = runtime_root_ / "packs" / backend / pack_id;
    std::error_code filesystem_error;
    const auto target_status = std::filesystem::symlink_status(
        target, filesystem_error);
    const bool target_missing =
        target_status.type() == std::filesystem::file_type::not_found ||
        filesystem_error ==
            std::make_error_code(std::errc::no_such_file_or_directory);
    if (target_missing) filesystem_error.clear();
    const bool target_exists = !target_missing && !filesystem_error;
    const bool target_active = HasPack(active, backend, pack_id);
    if (!target_exists) {
        if (target_active || filesystem_error) {
            return Finish(
                BackendPackRemovalStatus::InvalidRuntime,
                "Active runtime references a missing backend pack");
        }
    } else if (!ValidateRemovalTarget(runtime_root_, target, error)) {
        return Finish(BackendPackRemovalStatus::FilesystemFailure, error);
    }

    std::optional<BackendPackStateResult> deactivation;
    if (target_active) {
        progress.stage = BackendPackRemovalStage::Deactivating;
        progress.message = "Deactivating the backend pack before removal";
        SetProgress(progress);
        BackendPackStateService state_service(
            runtime_root_, execution_active_);
        deactivation = state_service.DeactivateOptionalPack(backend);
        if (deactivation->status != BackendPackStateStatus::Completed ||
            !deactivation->current) {
            return Finish(
                deactivation->status == BackendPackStateStatus::ExecutionActive
                    ? BackendPackRemovalStatus::ExecutionActive
                    : BackendPackRemovalStatus::InvalidRuntime,
                "Cannot deactivate the backend pack before removal: " +
                    deactivation->message,
                {}, deactivation);
        }
        active = *deactivation->current;
        if (checkpoint_ &&
            !checkpoint_(BackendPackRemovalCheckpoint::AfterDeactivation)) {
            return Finish(
                BackendPackRemovalStatus::Interrupted,
                "Backend-pack removal interrupted after safe deactivation",
                {}, deactivation);
        }
    } else {
        ActiveRuntime resolved;
        if (!ResolveRuntimeState(runtime_root_, active, resolved, error)) {
            return Finish(BackendPackRemovalStatus::InvalidRuntime, error);
        }
    }
    if (cancel_requested_.load()) {
        return Finish(
            BackendPackRemovalStatus::Interrupted,
            "Backend-pack removal cancelled before filesystem mutation",
            {}, deactivation);
    }

    progress.stage = BackendPackRemovalStage::UpdatingRollback;
    progress.message = "Removing stale rollback references to the pack";
    SetProgress(progress);
    const auto rollback_path = runtime_root_ / "rollback" /
        active.runtime_set_id / "previous-active-runtime.json";
    const bool rollback_exists =
        std::filesystem::exists(rollback_path, filesystem_error) &&
        !filesystem_error;
    if (filesystem_error) {
        return Finish(
            BackendPackRemovalStatus::FilesystemFailure,
            "Cannot inspect the runtime rollback state", {}, deactivation);
    }
    if (rollback_exists) {
        ActiveRuntimeState rollback;
        if (!LoadActiveRuntimeState(rollback_path, rollback, error)) {
            return Finish(
                BackendPackRemovalStatus::InvalidRuntime,
                "Cannot validate the runtime rollback state: " + error,
                {}, deactivation);
        }
        if (HasPack(rollback, backend, pack_id) &&
            !SaveActiveRuntimeStateAtomic(rollback_path, active, error)) {
            return Finish(
                BackendPackRemovalStatus::FilesystemFailure,
                "Cannot invalidate the removed pack rollback reference: " +
                    error,
                {}, deactivation);
        }
    }
    if (checkpoint_ &&
        !checkpoint_(BackendPackRemovalCheckpoint::AfterRollbackUpdate)) {
        return Finish(
            BackendPackRemovalStatus::Interrupted,
            "Backend-pack removal interrupted after rollback update",
            {}, deactivation);
    }

    const auto pending_root = runtime_root_ / "staging" / "removal" /
        backend / pack_id;
    if (!target_exists) {
        const bool pending_exists =
            std::filesystem::exists(pending_root, filesystem_error) &&
            !filesystem_error;
        if (filesystem_error) {
            return Finish(
                BackendPackRemovalStatus::FilesystemFailure,
                "Cannot inspect pending backend-pack cleanup", {},
                deactivation);
        }
        if (pending_exists) {
            if (!ValidateRemovalTarget(
                    runtime_root_, pending_root, error)) {
                return Finish(
                    BackendPackRemovalStatus::FilesystemFailure,
                    "Pending backend-pack cleanup is unsafe: " + error,
                    pending_root, deactivation);
            }
            std::filesystem::remove_all(pending_root, filesystem_error);
            if (filesystem_error ||
                std::filesystem::exists(pending_root, filesystem_error)) {
                return Finish(
                    BackendPackRemovalStatus::CleanupPending,
                    "Backend pack is absent, but quarantined cleanup is pending",
                    pending_root, deactivation);
            }
        }
        return Finish(
            BackendPackRemovalStatus::AlreadyAbsent,
            "Backend pack is absent and pending cleanup is complete", {},
            deactivation);
    }

    progress.stage = BackendPackRemovalStage::Quarantining;
    progress.message = "Quarantining the inactive pack directory";
    SetProgress(progress);
    std::filesystem::create_directories(
        pending_root, filesystem_error);
    if (filesystem_error ||
        !ValidateRemovalTarget(runtime_root_, pending_root, error)) {
        return Finish(
            BackendPackRemovalStatus::FilesystemFailure,
            "Cannot create a safe removal quarantine directory" +
                (error.empty() ? std::string{} : ": " + error), {},
            deactivation);
    }
    const auto quarantine = pending_root /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    if (std::filesystem::exists(quarantine, filesystem_error) ||
        filesystem_error) {
        return Finish(
            BackendPackRemovalStatus::FilesystemFailure,
            "Removal quarantine path is unavailable", {}, deactivation);
    }
    std::filesystem::rename(target, quarantine, filesystem_error);
    if (filesystem_error) {
        return Finish(
            BackendPackRemovalStatus::FilesystemFailure,
            "Cannot atomically quarantine the backend pack: " +
                filesystem_error.message(),
            {}, deactivation);
    }
    progress.quarantined_directory = quarantine;
    SetProgress(progress);
    if (checkpoint_ &&
        !checkpoint_(BackendPackRemovalCheckpoint::AfterQuarantine)) {
        return Finish(
            BackendPackRemovalStatus::Interrupted,
            "Backend-pack removal interrupted after quarantine",
            quarantine, deactivation);
    }

    progress.stage = BackendPackRemovalStage::Deleting;
    progress.message = "Deleting the quarantined backend pack";
    SetProgress(progress);
    std::filesystem::remove_all(quarantine, filesystem_error);
    if (filesystem_error ||
        std::filesystem::exists(quarantine, filesystem_error)) {
        return Finish(
            BackendPackRemovalStatus::CleanupPending,
            "The pack is inactive and quarantined, but cleanup is pending",
            quarantine, deactivation);
    }
    return Finish(
        BackendPackRemovalStatus::Removed,
        "Backend pack removed", {}, deactivation);
}

void BackendPackRemover::Cancel() {
    cancel_requested_.store(true);
}

BackendPackRemovalProgress BackendPackRemover::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

BackendPackRemovalResult BackendPackRemover::Finish(
    BackendPackRemovalStatus status,
    std::string message,
    std::filesystem::path quarantined_directory,
    std::optional<BackendPackStateResult> deactivation) {
    auto progress = GetProgress();
    progress.stage =
        status == BackendPackRemovalStatus::Removed ||
                status == BackendPackRemovalStatus::AlreadyAbsent ||
                status == BackendPackRemovalStatus::CleanupPending
            ? BackendPackRemovalStage::Complete
            : BackendPackRemovalStage::Failed;
    progress.message = message;
    progress.quarantined_directory = quarantined_directory;
    SetProgress(progress);
    return {status, std::move(message),
            std::move(quarantined_directory), std::move(deactivation)};
}

void BackendPackRemover::SetProgress(
    BackendPackRemovalProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

const char* BackendPackRemovalStatusName(BackendPackRemovalStatus status) {
    switch (status) {
        case BackendPackRemovalStatus::Removed: return "removed";
        case BackendPackRemovalStatus::AlreadyAbsent:
            return "already_absent";
        case BackendPackRemovalStatus::Busy: return "busy";
        case BackendPackRemovalStatus::ExecutionActive:
            return "execution_active";
        case BackendPackRemovalStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackRemovalStatus::InvalidRuntime:
            return "invalid_runtime";
        case BackendPackRemovalStatus::FilesystemFailure:
            return "filesystem_failure";
        case BackendPackRemovalStatus::Interrupted: return "interrupted";
        case BackendPackRemovalStatus::CleanupPending:
            return "cleanup_pending";
        default: return "unknown";
    }
}

const char* BackendPackRemovalStageName(BackendPackRemovalStage stage) {
    switch (stage) {
        case BackendPackRemovalStage::Idle: return "idle";
        case BackendPackRemovalStage::Validating: return "validating";
        case BackendPackRemovalStage::Deactivating: return "deactivating";
        case BackendPackRemovalStage::UpdatingRollback:
            return "updating_rollback";
        case BackendPackRemovalStage::Quarantining: return "quarantining";
        case BackendPackRemovalStage::Deleting: return "deleting";
        case BackendPackRemovalStage::Complete: return "complete";
        case BackendPackRemovalStage::Failed: return "failed";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
