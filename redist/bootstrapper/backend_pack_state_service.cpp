#include "backend_pack_state_service.h"
#include "runtime_mutation_gate.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace cyxwiz::runtime {
namespace {

bool IsOptionalBackend(const std::string& backend) {
    return backend == "cuda" || backend == "opencl" ||
           backend == "oneapi";
}

}  // namespace

BackendPackStateService::BackendPackStateService(
    std::filesystem::path runtime_root,
    BackendPackExecutionActiveCheck execution_active,
    BackendPackStateObserver observer)
    : runtime_root_(std::move(runtime_root)),
      execution_active_(std::move(execution_active)),
      observer_(std::move(observer)) {}

BackendPackStateResult BackendPackStateService::InitializeBase(
    std::string runtime_set_id,
    std::string base_pack_id) {
    std::unique_lock<std::mutex> mutation_lock(
        mutation_mutex_, std::try_to_lock);
    if (!mutation_lock.owns_lock()) {
        return {BackendPackStateStatus::Busy,
                "A runtime initialization is already running"};
    }
    RuntimeMutationLease runtime_mutation;
    if (!runtime_mutation.OwnsMutation() ||
        (execution_active_ && execution_active_())) {
        return {BackendPackStateStatus::ExecutionActive,
                "Runtime initialization is blocked while an execution context is active"};
    }
    if (runtime_set_id.empty() || base_pack_id.empty()) {
        return {BackendPackStateStatus::InvalidRequest,
                "Runtime-set and base-pack identities are required"};
    }

    const auto active_path = runtime_root_ / "active-runtime.json";
    std::error_code filesystem_error;
    if (std::filesystem::exists(active_path, filesystem_error) ||
        filesystem_error) {
        return {BackendPackStateStatus::InvalidRuntime,
                filesystem_error
                    ? "Cannot inspect the active runtime state: " +
                          filesystem_error.message()
                    : "An active runtime already exists"};
    }

    BackendPackStateProgress progress;
    progress.stage = BackendPackStateStage::Validating;
    progress.operation = "initialize_base";
    progress.pack_id = base_pack_id;
    progress.generation = 1;
    progress.message = "Validating the complete CPU-base runtime";
    SetProgress(progress);

    ActiveRuntimeState candidate;
    candidate.runtime_set_id = std::move(runtime_set_id);
    candidate.generation = 1;
    candidate.base_pack_id = std::move(base_pack_id);
    ActiveRuntime resolved;
    std::string error;
    if (!ResolveRuntimeState(runtime_root_, candidate, resolved, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = error;
        SetProgress(progress);
        return {BackendPackStateStatus::InvalidRuntime, error};
    }

    progress.stage = BackendPackStateStage::Publishing;
    progress.message = "Publishing the initial runtime state";
    SetProgress(progress);
    if (std::filesystem::exists(active_path, filesystem_error) ||
        filesystem_error ||
        !SaveActiveRuntimeStateAtomic(active_path, candidate, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = filesystem_error
            ? filesystem_error.message()
            : (error.empty() ? "An active runtime appeared during initialization"
                             : error);
        SetProgress(progress);
        return {BackendPackStateStatus::PublishFailed, progress.message};
    }

    progress.stage = BackendPackStateStage::Complete;
    progress.message = "Initial CPU-base runtime published atomically";
    SetProgress(progress);
    return {BackendPackStateStatus::Completed,
            progress.message, std::nullopt, candidate};
}

BackendPackStateResult BackendPackStateService::ActivateOptionalPack(
    std::string backend,
    std::string pack_id) {
    if (!IsOptionalBackend(backend) || pack_id.empty()) {
        return {BackendPackStateStatus::InvalidRequest,
                "Optional backend and pack identity are required"};
    }
    return Mutate(
        "activate", std::move(backend), std::move(pack_id),
        [](const ActiveRuntimeState& previous,
           ActiveRuntimeState& candidate,
           std::string&) {
            (void)previous;
            (void)candidate;
            return true;
        });
}

BackendPackStateResult BackendPackStateService::DeactivateOptionalPack(
    std::string backend) {
    if (!IsOptionalBackend(backend)) {
        return {BackendPackStateStatus::InvalidRequest,
                "An optional backend identity is required"};
    }
    return Mutate(
        "deactivate", std::move(backend), {},
        [](const ActiveRuntimeState& previous,
           ActiveRuntimeState& candidate,
           std::string& error) {
            if (candidate.packs.size() == previous.packs.size()) {
                error = "The backend is not active";
                return false;
            }
            return true;
        });
}

BackendPackStateResult BackendPackStateService::Rollback() {
    return Mutate(
        "rollback", {}, {},
        [&](const ActiveRuntimeState& previous,
            ActiveRuntimeState& candidate,
            std::string& error) {
            ActiveRuntimeState rollback;
            if (!LoadActiveRuntimeState(
                    RollbackPath(previous), rollback, error)) {
                error = "No valid previous runtime state is available: " +
                        error;
                return false;
            }
            candidate = std::move(rollback);
            return true;
        });
}

BackendPackStateResult BackendPackStateService::Mutate(
    std::string operation,
    std::string backend,
    std::string pack_id,
    const CandidateBuilder& build_candidate) {
    std::unique_lock<std::mutex> mutation_lock(
        mutation_mutex_, std::try_to_lock);
    if (!mutation_lock.owns_lock()) {
        return {BackendPackStateStatus::Busy,
                "A backend-pack state mutation is already running"};
    }
    RuntimeMutationLease runtime_mutation;
    if (!runtime_mutation.OwnsMutation()) {
        return {BackendPackStateStatus::ExecutionActive,
                "Backend-pack mutation is blocked while an execution context is active"};
    }
    if (execution_active_ && execution_active_()) {
        return {BackendPackStateStatus::ExecutionActive,
                "Backend-pack mutation is blocked while an execution context is active"};
    }

    BackendPackStateProgress progress;
    progress.stage = BackendPackStateStage::Reading;
    progress.operation = operation;
    progress.backend = backend;
    progress.pack_id = pack_id;
    progress.message = "Reading active runtime state";
    SetProgress(progress);

    ActiveRuntimeState previous;
    std::string error;
    if (!LoadActiveRuntimeState(
            runtime_root_ / "active-runtime.json", previous, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = error;
        SetProgress(progress);
        return {BackendPackStateStatus::InvalidRuntime, error};
    }
    if (previous.generation == std::numeric_limits<std::uint64_t>::max()) {
        return {BackendPackStateStatus::InvalidRuntime,
                "Runtime generation cannot be incremented"};
    }

    ActiveRuntimeState candidate = previous;
    if (operation == "activate") {
        const auto existing = std::find_if(
            candidate.packs.begin(), candidate.packs.end(),
            [&](const ActivePackState& value) {
                return value.backend == backend;
            });
        if (existing == candidate.packs.end()) {
            candidate.packs.push_back({backend, pack_id});
        } else {
            existing->pack_id = pack_id;
        }
    } else if (operation == "deactivate") {
        candidate.packs.erase(
            std::remove_if(
                candidate.packs.begin(), candidate.packs.end(),
                [&](const ActivePackState& value) {
                    return value.backend == backend;
                }),
            candidate.packs.end());
    }
    if (!build_candidate(previous, candidate, error)) {
        return {BackendPackStateStatus::InvalidRequest, error, previous};
    }
    candidate.generation = previous.generation + 1;

    progress.stage = BackendPackStateStage::Validating;
    progress.generation = candidate.generation;
    progress.message = "Validating the complete candidate runtime";
    SetProgress(progress);
    ActiveRuntime resolved;
    if (!ResolveRuntimeState(runtime_root_, candidate, resolved, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = error;
        SetProgress(progress);
        return {BackendPackStateStatus::InvalidRuntime,
                error, previous};
    }

    progress.stage = BackendPackStateStage::SavingRollback;
    progress.message = "Saving the previous complete runtime state";
    SetProgress(progress);
    if (!SaveActiveRuntimeStateAtomic(
            RollbackPath(previous), previous, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = error;
        SetProgress(progress);
        return {BackendPackStateStatus::PublishFailed,
                error, previous};
    }

    progress.stage = BackendPackStateStage::Publishing;
    progress.message = "Publishing the candidate runtime state";
    SetProgress(progress);
    if (!SaveActiveRuntimeStateAtomic(
            runtime_root_ / "active-runtime.json", candidate, error)) {
        progress.stage = BackendPackStateStage::Failed;
        progress.message = error;
        SetProgress(progress);
        return {BackendPackStateStatus::PublishFailed,
                error, previous};
    }

    progress.stage = BackendPackStateStage::Complete;
    progress.message = "Runtime state published atomically";
    SetProgress(progress);
    return {BackendPackStateStatus::Completed,
            progress.message, previous, candidate};
}

BackendPackStateProgress BackendPackStateService::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

void BackendPackStateService::SetProgress(
    BackendPackStateProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

std::filesystem::path BackendPackStateService::RollbackPath(
    const ActiveRuntimeState& state) const {
    return runtime_root_ / "rollback" / state.runtime_set_id /
           "previous-active-runtime.json";
}

const char* BackendPackStateStatusName(BackendPackStateStatus status) {
    switch (status) {
        case BackendPackStateStatus::Completed: return "completed";
        case BackendPackStateStatus::Busy: return "busy";
        case BackendPackStateStatus::ExecutionActive:
            return "execution_active";
        case BackendPackStateStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackStateStatus::InvalidRuntime:
            return "invalid_runtime";
        case BackendPackStateStatus::PublishFailed:
            return "publish_failed";
        default: return "unknown";
    }
}

const char* BackendPackStateStageName(BackendPackStateStage stage) {
    switch (stage) {
        case BackendPackStateStage::Idle: return "idle";
        case BackendPackStateStage::Reading: return "reading";
        case BackendPackStateStage::Validating: return "validating";
        case BackendPackStateStage::SavingRollback:
            return "saving_rollback";
        case BackendPackStateStage::Publishing: return "publishing";
        case BackendPackStateStage::Complete: return "complete";
        case BackendPackStateStage::Failed: return "failed";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
