#pragma once

#include "backend_pack_payload.h"
#include "backend_pack_state_service.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

enum class BackendPackInstallStage {
    Idle,
    Validating,
    Copying,
    Verifying,
    Deactivating,
    PublishingPack,
    Activating,
    Complete,
    Failed
};

enum class BackendPackInstallStatus {
    InstalledAndActivated,
    AlreadyInstalledAndActivated,
    InstalledUnqualified,
    Busy,
    ExecutionActive,
    InvalidRequest,
    DiskBudgetExceeded,
    IntegrityFailure,
    Interrupted,
    FilesystemFailure
};

enum class BackendPackInstallCheckpoint {
    AfterValidation,
    AfterCopy,
    BeforePackPublish,
    AfterDeactivation,
    AfterQuarantine,
    AfterPackPublish,
    BeforeActivation
};

struct BackendPackInstallProgress {
    BackendPackInstallStage stage = BackendPackInstallStage::Idle;
    std::string backend;
    std::string pack_id;
    std::size_t component_index = 0;
    std::size_t component_count = 0;
    std::uint64_t completed_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::string message;
};

struct BackendPackInstallResult {
    BackendPackInstallStatus status =
        BackendPackInstallStatus::InvalidRequest;
    std::string message;
    std::filesystem::path installed_directory;
    std::optional<BackendPackStateResult> activation;
};

using BackendPackInstallObserver =
    std::function<void(const BackendPackInstallProgress&)>;
using BackendPackInstallCheckpointHook =
    std::function<bool(BackendPackInstallCheckpoint)>;

class BackendPackInstaller {
public:
    explicit BackendPackInstaller(
        std::filesystem::path runtime_root,
        BackendPackExecutionActiveCheck execution_active = {},
        BackendPackInstallObserver observer = {},
        BackendPackInstallCheckpointHook checkpoint = {});

    BackendPackInstallResult InstallOrUpdate(
        const VerifiedBackendPackPayload& payload,
        std::uint64_t disk_budget_bytes);
    BackendPackInstallResult Repair(
        const VerifiedBackendPackPayload& payload,
        std::uint64_t disk_budget_bytes);
    void Cancel();
    BackendPackInstallProgress GetProgress() const;

private:
    BackendPackInstallResult Apply(
        const VerifiedBackendPackPayload& payload,
        std::uint64_t disk_budget_bytes,
        bool repair);
    BackendPackInstallResult Finish(
        BackendPackInstallStatus status,
        std::string message,
        std::filesystem::path installed_directory = {},
        std::optional<BackendPackStateResult> activation = std::nullopt);
    void SetProgress(BackendPackInstallProgress progress);

    std::filesystem::path runtime_root_;
    BackendPackExecutionActiveCheck execution_active_;
    BackendPackInstallObserver observer_;
    BackendPackInstallCheckpointHook checkpoint_;
    std::atomic<bool> cancel_requested_{false};
    mutable std::mutex progress_mutex_;
    std::mutex install_mutex_;
    BackendPackInstallProgress progress_;
};

const char* BackendPackInstallStatusName(BackendPackInstallStatus status);
const char* BackendPackInstallStageName(BackendPackInstallStage stage);

}  // namespace cyxwiz::runtime
