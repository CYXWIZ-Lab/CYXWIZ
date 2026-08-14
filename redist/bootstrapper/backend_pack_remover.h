#pragma once

#include "backend_pack_state_service.h"

#include <atomic>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

namespace cyxwiz::runtime {

enum class BackendPackRemovalStage {
    Idle,
    Validating,
    Deactivating,
    UpdatingRollback,
    Quarantining,
    Deleting,
    Complete,
    Failed
};

enum class BackendPackRemovalStatus {
    Removed,
    AlreadyAbsent,
    Busy,
    ExecutionActive,
    InvalidRequest,
    InvalidRuntime,
    FilesystemFailure,
    Interrupted,
    CleanupPending
};

enum class BackendPackRemovalCheckpoint {
    AfterDeactivation,
    AfterRollbackUpdate,
    AfterQuarantine
};

struct BackendPackRemovalProgress {
    BackendPackRemovalStage stage = BackendPackRemovalStage::Idle;
    std::string backend;
    std::string pack_id;
    std::filesystem::path quarantined_directory;
    std::string message;
};

struct BackendPackRemovalResult {
    BackendPackRemovalStatus status =
        BackendPackRemovalStatus::InvalidRequest;
    std::string message;
    std::filesystem::path quarantined_directory;
    std::optional<BackendPackStateResult> deactivation;
};

using BackendPackRemovalObserver =
    std::function<void(const BackendPackRemovalProgress&)>;
using BackendPackRemovalCheckpointHook =
    std::function<bool(BackendPackRemovalCheckpoint)>;

class BackendPackRemover {
public:
    explicit BackendPackRemover(
        std::filesystem::path runtime_root,
        BackendPackExecutionActiveCheck execution_active = {},
        BackendPackRemovalObserver observer = {},
        BackendPackRemovalCheckpointHook checkpoint = {});

    BackendPackRemovalResult Remove(
        std::string backend,
        std::string pack_id);
    void Cancel();
    BackendPackRemovalProgress GetProgress() const;

private:
    BackendPackRemovalResult Finish(
        BackendPackRemovalStatus status,
        std::string message,
        std::filesystem::path quarantined_directory = {},
        std::optional<BackendPackStateResult> deactivation = std::nullopt);
    void SetProgress(BackendPackRemovalProgress progress);

    std::filesystem::path runtime_root_;
    BackendPackExecutionActiveCheck execution_active_;
    BackendPackRemovalObserver observer_;
    BackendPackRemovalCheckpointHook checkpoint_;
    std::atomic<bool> cancel_requested_{false};
    std::mutex removal_mutex_;
    mutable std::mutex progress_mutex_;
    BackendPackRemovalProgress progress_;
};

const char* BackendPackRemovalStatusName(BackendPackRemovalStatus status);
const char* BackendPackRemovalStageName(BackendPackRemovalStage stage);

}  // namespace cyxwiz::runtime
