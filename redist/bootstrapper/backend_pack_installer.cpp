#include "backend_pack_installer.h"
#include "backend_pack_hash.h"
#include "backend_pack_path.h"
#include "backend_pack_progress_cadence.h"
#include "runtime_mutation_gate.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <memory>
#include <set>
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

bool IsWithin(
    const std::filesystem::path& root,
    const std::filesystem::path& child) {
    const auto relative = child.lexically_relative(root);
    return !relative.empty() && !relative.is_absolute() &&
           *relative.begin() != "..";
}

bool IsPrivateDeliveryExtraction(
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& source,
    std::string& error) {
    std::error_code filesystem_error;
    const auto delivery_root = std::filesystem::weakly_canonical(
        runtime_root / "staging" / "delivery", filesystem_error);
    if (filesystem_error) {
        error = "Cannot resolve the private delivery staging root";
        return false;
    }
    const auto canonical_source =
        std::filesystem::canonical(source, filesystem_error);
    if (filesystem_error || !IsWithin(delivery_root, canonical_source)) {
        error = "Private extraction is outside the delivery staging root";
        return false;
    }
    return true;
}

bool ValidateFile(
    const std::filesystem::path& path,
    const VerifiedPackComponent& component,
    std::string& error,
    const Sha256FileProgress& hash_progress = {}) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Component is missing or is not a regular file: " +
                component.relative_path;
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size != component.size) {
        error = "Component size differs from signed metadata: " +
                component.relative_path;
        return false;
    }
    std::string digest;
    if (!Sha256File(path, digest, error, hash_progress)) return false;
    if (digest != component.sha256) {
        error = "Component SHA-256 differs from signed metadata: " +
                component.relative_path;
        return false;
    }
    return true;
}

bool EnumerateExactPayload(
    const std::filesystem::path& root,
    const std::set<std::string>& expected,
    std::string& error) {
    std::error_code filesystem_error;
    const auto root_status =
        std::filesystem::symlink_status(root, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root_status) ||
        std::filesystem::is_symlink(root_status)) {
        error = "Pack payload root is missing or unsafe";
        return false;
    }
    std::set<std::string> observed;
    for (std::filesystem::recursive_directory_iterator iterator(root), end;
         iterator != end; ++iterator) {
        const auto status = iterator->symlink_status(filesystem_error);
        if (filesystem_error || std::filesystem::is_symlink(status)) {
            error = "Pack payload contains a link or unreadable entry";
            return false;
        }
        if (std::filesystem::is_directory(status)) continue;
        if (!std::filesystem::is_regular_file(status)) {
            error = "Pack payload contains an unsupported filesystem entry";
            return false;
        }
        const auto relative =
            std::filesystem::relative(iterator->path(), root, filesystem_error);
        if (filesystem_error) {
            error = "Cannot resolve a pack payload path";
            return false;
        }
        observed.insert(FoldBackendPackPath(relative.generic_string()));
    }
    if (observed != expected) {
        error = "Pack payload files differ from signed component inventory";
        return false;
    }
    return true;
}

}  // namespace

BackendPackInstaller::BackendPackInstaller(
    std::filesystem::path runtime_root,
    BackendPackExecutionActiveCheck execution_active,
    BackendPackInstallObserver observer,
    BackendPackInstallCheckpointHook checkpoint)
    : runtime_root_(std::move(runtime_root)),
      execution_active_(std::move(execution_active)),
      observer_(std::move(observer)),
      checkpoint_(std::move(checkpoint)) {}

BackendPackInstallResult BackendPackInstaller::InstallOrUpdate(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, false, true,
        InstallTarget::OptionalPack);
}

BackendPackInstallResult BackendPackInstaller::StageInstallOrUpdate(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, false, false,
        InstallTarget::OptionalPack);
}

BackendPackInstallResult BackendPackInstaller::StageBase(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, false, false,
        InstallTarget::FreshBase);
}

BackendPackInstallResult BackendPackInstaller::StageBaseUpdate(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, false, false,
        InstallTarget::BaseUpdate);
}

BackendPackInstallResult BackendPackInstaller::Repair(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, true, true,
        InstallTarget::OptionalPack);
}

BackendPackInstallResult BackendPackInstaller::StageRepair(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes) {
    return Apply(
        payload, disk_budget_bytes, true, false,
        InstallTarget::OptionalPack);
}

BackendPackInstallResult BackendPackInstaller::AdoptVerifiedPrivateExtraction(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes,
    PrivateExtractionAction action) {
    switch (action) {
        case PrivateExtractionAction::InstallOptionalPack:
            return Apply(
                payload, disk_budget_bytes, false, false,
                InstallTarget::OptionalPack,
                PayloadStagingMode::AdoptVerifiedPrivateExtraction);
        case PrivateExtractionAction::RepairOptionalPack:
            return Apply(
                payload, disk_budget_bytes, true, false,
                InstallTarget::OptionalPack,
                PayloadStagingMode::AdoptVerifiedPrivateExtraction);
        case PrivateExtractionAction::InstallFreshBase:
            return Apply(
                payload, disk_budget_bytes, false, false,
                InstallTarget::FreshBase,
                PayloadStagingMode::AdoptVerifiedPrivateExtraction);
        case PrivateExtractionAction::UpdateBase:
            return Apply(
                payload, disk_budget_bytes, false, false,
                InstallTarget::BaseUpdate,
                PayloadStagingMode::AdoptVerifiedPrivateExtraction);
    }
    return Finish(
        BackendPackInstallStatus::InvalidRequest,
        "Private extraction action is invalid");
}

BackendPackInstallResult BackendPackInstaller::Apply(
    const VerifiedBackendPackPayload& payload,
    std::uint64_t disk_budget_bytes,
    bool repair,
    bool activate,
    InstallTarget target,
    PayloadStagingMode staging_mode) {
    const bool base = target != InstallTarget::OptionalPack;
    std::unique_lock<std::mutex> install_lock(
        install_mutex_, std::try_to_lock);
    if (!install_lock.owns_lock()) {
        return {BackendPackInstallStatus::Busy,
                "A backend-pack installation is already running"};
    }
    if (execution_active_ && execution_active_()) {
        return {BackendPackInstallStatus::ExecutionActive,
                "Backend-pack installation is blocked while an execution context is active"};
    }
    cancel_requested_.store(false);

    BackendPackInstallProgress progress;
    progress.stage = BackendPackInstallStage::Validating;
    progress.backend = payload.backend;
    progress.pack_id = payload.pack_id;
    progress.component_count = payload.components.size();
    progress.message = "Validating verified pack payload metadata";
    SetProgress(progress);

    if (!IsIdentifier(payload.runtime_set_id) ||
        (base ? (!payload.companion_base_id.empty() ||
                 payload.backend != "cpu")
              : (!IsIdentifier(payload.companion_base_id) ||
                 !IsOptionalBackend(payload.backend))) ||
        !IsIdentifier(payload.pack_id) || payload.components.empty() ||
        payload.source_directory.empty()) {
        return Finish(
            BackendPackInstallStatus::InvalidRequest,
            "Verified backend-pack payload is incomplete");
    }
    ActiveRuntimeState active;
    std::string error;
    if (target == InstallTarget::FreshBase) {
        std::error_code active_error;
        if (std::filesystem::exists(
                runtime_root_ / "active-runtime.json", active_error) ||
            active_error) {
            return Finish(
                BackendPackInstallStatus::InvalidRequest,
                active_error
                    ? "Cannot inspect the active runtime state: " +
                          active_error.message()
                    : "A CPU base is already active");
        }
    } else if (target == InstallTarget::BaseUpdate) {
        if (!LoadActiveRuntimeState(
                runtime_root_ / "active-runtime.json", active, error)) {
            return Finish(
                BackendPackInstallStatus::InvalidRequest,
                error.empty() ? "A CPU base must be active before update"
                              : error);
        }
    } else if (!LoadActiveRuntimeState(
                   runtime_root_ / "active-runtime.json", active, error) ||
               active.runtime_set_id != payload.runtime_set_id ||
               active.base_pack_id != payload.companion_base_id) {
        return Finish(
            BackendPackInstallStatus::InvalidRequest,
            error.empty()
                ? "Pack requires a different active runtime set or base"
                : error);
    }

    std::set<std::string> expected_paths;
    std::uint64_t total_bytes = 0;
    for (const auto& component : payload.components) {
        if (!IsCanonicalBackendPackRelativePath(component.relative_path) ||
            !IsLowercaseSha256(component.sha256) ||
            total_bytes > std::numeric_limits<std::uint64_t>::max() -
                              component.size ||
            !expected_paths.insert(
                FoldBackendPackPath(component.relative_path)).second) {
            return Finish(
                BackendPackInstallStatus::InvalidRequest,
                "Pack component inventory is invalid");
        }
        total_bytes += component.size;
    }
    progress.total_bytes = total_bytes;
    SetProgress(progress);
    if ((disk_budget_bytes > 0 && total_bytes > disk_budget_bytes)) {
        return Finish(
            BackendPackInstallStatus::DiskBudgetExceeded,
            "Pack payload exceeds the approved disk budget");
    }
    std::error_code filesystem_error;
    if (staging_mode == PayloadStagingMode::Copy) {
        const auto disk =
            std::filesystem::space(runtime_root_, filesystem_error);
        if (filesystem_error || disk.available < total_bytes) {
            return Finish(
                BackendPackInstallStatus::DiskBudgetExceeded,
                "Insufficient free space for the staged pack payload");
        }
    } else if (!IsPrivateDeliveryExtraction(
                   runtime_root_, payload.source_directory, error)) {
        return Finish(BackendPackInstallStatus::InvalidRequest, error);
    }
    if (!EnumerateExactPayload(
            payload.source_directory, expected_paths, error)) {
        return Finish(BackendPackInstallStatus::IntegrityFailure, error);
    }
    if (staging_mode == PayloadStagingMode::Copy) {
        progress.message = "Validating extracted component hashes";
        progress.component_index = 0;
        progress.completed_bytes = 0;
        SetProgress(progress);
        BackendPackProgressCadence validation_cadence;
        for (std::size_t index = 0; index < payload.components.size();
             ++index) {
            const auto& component = payload.components[index];
            const auto completed_before = progress.completed_bytes;
            const auto publish_hash_progress = [&](std::uint64_t file_bytes) {
                progress.component_index = index + 1;
                progress.completed_bytes = completed_before + file_bytes;
                if (validation_cadence.ShouldPublish()) {
                    SetProgress(progress);
                }
                return !cancel_requested_.load();
            };
            if (!ValidateFile(
                    payload.source_directory /
                        BackendPackNativeRelativePath(
                            component.relative_path),
                    component, error, publish_hash_progress)) {
                return cancel_requested_.load()
                    ? Finish(
                          BackendPackInstallStatus::Interrupted,
                          "Pack installation cancelled during payload validation")
                    : Finish(
                          BackendPackInstallStatus::IntegrityFailure, error);
            }
            progress.component_index = index + 1;
            progress.completed_bytes = completed_before + component.size;
            if (validation_cadence.ShouldPublish(
                    progress.component_index == progress.component_count)) {
                SetProgress(progress);
            }
        }
    } else {
        progress.message = "Validated private extraction inventory";
        progress.component_index = progress.component_count;
        progress.completed_bytes = total_bytes;
        SetProgress(progress);
    }
    if (checkpoint_ &&
        !checkpoint_(BackendPackInstallCheckpoint::AfterValidation)) {
        return Finish(
            BackendPackInstallStatus::Interrupted,
            "Pack installation interrupted after validation");
    }

    const auto destination = base
        ? runtime_root_ / "base" / payload.pack_id
        : runtime_root_ / "packs" / payload.backend / payload.pack_id;
    const bool destination_exists =
        std::filesystem::exists(destination, filesystem_error) &&
        !filesystem_error;
    if (repair && !destination_exists) {
        return Finish(
            BackendPackInstallStatus::InvalidRequest,
            "The requested backend pack is not installed");
    }
    bool already_installed = false;
    bool replace_existing = false;
    std::unique_ptr<RuntimeMutationLease> mutation_lease;
    if (destination_exists) {
        bool valid = EnumerateExactPayload(destination, expected_paths, error);
        for (const auto& component : payload.components) {
            if (valid && !ValidateFile(
                    destination /
                        BackendPackNativeRelativePath(component.relative_path),
                    component, error)) {
                valid = false;
            }
        }
        if (!valid && !repair) {
            return Finish(BackendPackInstallStatus::IntegrityFailure, error);
        }
        if (!valid && repair) {
            const auto status = std::filesystem::symlink_status(
                destination, filesystem_error);
            const auto canonical_root = std::filesystem::canonical(
                runtime_root_, filesystem_error);
            const auto canonical_destination = std::filesystem::canonical(
                destination, filesystem_error);
            if (filesystem_error ||
                !std::filesystem::is_directory(status) ||
                std::filesystem::is_symlink(status) ||
                !IsWithin(canonical_root, canonical_destination)) {
                return Finish(
                    BackendPackInstallStatus::FilesystemFailure,
                    "The installed pack directory is unsafe to repair");
            }
        }
        already_installed = valid;
        replace_existing = !valid;
    }

    std::filesystem::path staging_root;
    if (!already_installed) {
        staging_root = runtime_root_ / "staging" /
            (payload.pack_id + "-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto staged_payload = staging_root / "payload";
        std::filesystem::create_directories(staging_root, filesystem_error);
        if (filesystem_error) {
            return Finish(
                BackendPackInstallStatus::FilesystemFailure,
                "Cannot create pack staging directory: " +
                    filesystem_error.message());
        }
        progress.stage = BackendPackInstallStage::Copying;
        progress.message = staging_mode == PayloadStagingMode::Copy
            ? "Copying verified components into staging"
            : "Adopting private extraction into versioned staging";
        progress.component_index = 0;
        progress.completed_bytes = 0;
        SetProgress(progress);
        if (staging_mode ==
            PayloadStagingMode::AdoptVerifiedPrivateExtraction) {
            if (cancel_requested_.load()) {
                std::filesystem::remove_all(staging_root, filesystem_error);
                return Finish(
                    BackendPackInstallStatus::Interrupted,
                    "Pack installation cancelled");
            }
            std::filesystem::rename(
                payload.source_directory, staged_payload, filesystem_error);
            if (filesystem_error) {
                const auto rename_error = filesystem_error.message();
                std::filesystem::remove_all(staging_root, filesystem_error);
                return Finish(
                    BackendPackInstallStatus::FilesystemFailure,
                    "Cannot atomically adopt the private extraction: " +
                        rename_error);
            }
            progress.component_index = progress.component_count;
            progress.completed_bytes = total_bytes;
            SetProgress(progress);
        } else {
            std::filesystem::create_directories(
                staged_payload, filesystem_error);
            if (filesystem_error) {
                std::filesystem::remove_all(staging_root, filesystem_error);
                return Finish(
                    BackendPackInstallStatus::FilesystemFailure,
                    "Cannot create payload staging directory: " +
                        filesystem_error.message());
            }
            BackendPackProgressCadence copy_cadence;
            for (std::size_t index = 0; index < payload.components.size();
                 ++index) {
                if (cancel_requested_.load()) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::Interrupted,
                        "Pack installation cancelled");
                }
                const auto& component = payload.components[index];
                const auto relative =
                    BackendPackNativeRelativePath(component.relative_path);
                const auto staged_component = staged_payload / relative;
                std::filesystem::create_directories(
                    staged_component.parent_path(), filesystem_error);
                if (filesystem_error || !std::filesystem::copy_file(
                        payload.source_directory / relative,
                        staged_component,
                        std::filesystem::copy_options::none,
                        filesystem_error)) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::FilesystemFailure,
                        "Cannot copy a verified pack component into staging");
                }
                progress.component_index = index + 1;
                progress.completed_bytes += component.size;
                if (copy_cadence.ShouldPublish(
                        progress.component_index ==
                            progress.component_count)) {
                    SetProgress(progress);
                }
            }
        }
        if (checkpoint_ &&
            !checkpoint_(BackendPackInstallCheckpoint::AfterCopy)) {
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::Interrupted,
                "Pack installation interrupted after staging");
        }

        if (staging_mode == PayloadStagingMode::Copy) {
            progress.stage = BackendPackInstallStage::Verifying;
            progress.message = "Verifying staged component hashes";
            progress.component_index = 0;
            progress.completed_bytes = 0;
            SetProgress(progress);
            BackendPackProgressCadence verification_cadence;
            for (std::size_t index = 0; index < payload.components.size();
                 ++index) {
                const auto& component = payload.components[index];
                const auto completed_before = progress.completed_bytes;
                const auto publish_hash_progress =
                    [&](std::uint64_t file_bytes) {
                        progress.component_index = index + 1;
                        progress.completed_bytes =
                            completed_before + file_bytes;
                        if (verification_cadence.ShouldPublish()) {
                            SetProgress(progress);
                        }
                        return !cancel_requested_.load();
                    };
                if (!ValidateFile(
                        staged_payload / BackendPackNativeRelativePath(
                            component.relative_path),
                        component, error, publish_hash_progress)) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    if (cancel_requested_.load()) {
                        return Finish(
                            BackendPackInstallStatus::Interrupted,
                            "Pack installation cancelled during staged verification");
                    }
                    return Finish(
                        BackendPackInstallStatus::IntegrityFailure, error);
                }
                progress.component_index = index + 1;
                progress.completed_bytes = completed_before + component.size;
                if (verification_cadence.ShouldPublish(
                        progress.component_index == progress.component_count)) {
                    SetProgress(progress);
                }
            }
        }
        if (checkpoint_ &&
            !checkpoint_(BackendPackInstallCheckpoint::BeforePackPublish)) {
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::Interrupted,
                "Pack installation interrupted before publication");
        }
        if (execution_active_ && execution_active_()) {
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::ExecutionActive,
                "Execution started before pack publication");
        }
        mutation_lease = std::make_unique<RuntimeMutationLease>();
        if (!mutation_lease->OwnsMutation()) {
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::ExecutionActive,
                "Execution started before pack publication");
        }

        progress.stage = BackendPackInstallStage::PublishingPack;
        progress.message = "Publishing the complete versioned pack directory";
        SetProgress(progress);
        std::filesystem::path quarantined;
        if (replace_existing) {
            const auto active_pack = std::find_if(
                active.packs.begin(), active.packs.end(),
                [&](const ActivePackState& pack) {
                    return pack.backend == payload.backend &&
                           pack.pack_id == payload.pack_id;
                });
            bool invalidate_rollback = false;
            if (active_pack != active.packs.end()) {
                progress.stage = BackendPackInstallStage::Deactivating;
                progress.message =
                    "Deactivating the corrupt pack before repair";
                SetProgress(progress);
                BackendPackStateService state_service(
                    runtime_root_, execution_active_);
                auto deactivation = state_service.DeactivateOptionalPack(
                    payload.backend);
                if (deactivation.status != BackendPackStateStatus::Completed ||
                    !deactivation.current) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::FilesystemFailure,
                        "Cannot deactivate the corrupt pack before repair: " +
                            deactivation.message);
                }
                active = *deactivation.current;
                invalidate_rollback = true;
                if (checkpoint_ &&
                    !checkpoint_(
                        BackendPackInstallCheckpoint::AfterDeactivation)) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::Interrupted,
                        "Pack repair interrupted after safe deactivation",
                        destination);
                }
            } else {
                ActiveRuntime resolved;
                if (!ResolveRuntimeState(
                        runtime_root_, active, resolved, error)) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::IntegrityFailure, error);
                }
            }
            const auto rollback_path = runtime_root_ / "rollback" /
                active.runtime_set_id / "previous-active-runtime.json";
            filesystem_error.clear();
            error.clear();
            if (!invalidate_rollback &&
                std::filesystem::exists(rollback_path, filesystem_error)) {
                ActiveRuntimeState rollback;
                if (!LoadActiveRuntimeState(
                        rollback_path, rollback, error)) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                    return Finish(
                        BackendPackInstallStatus::IntegrityFailure,
                        "Cannot validate the runtime rollback state: " +
                            error);
                }
                invalidate_rollback = std::any_of(
                    rollback.packs.begin(), rollback.packs.end(),
                    [&](const ActivePackState& pack) {
                        return pack.backend == payload.backend &&
                               pack.pack_id == payload.pack_id;
                    });
            }
            if (filesystem_error ||
                (invalidate_rollback && !SaveActiveRuntimeStateAtomic(
                    rollback_path, active, error))) {
                std::filesystem::remove_all(staging_root, filesystem_error);
                return Finish(
                    BackendPackInstallStatus::FilesystemFailure,
                    "Cannot invalidate the corrupt pack rollback reference: " +
                        (filesystem_error ? filesystem_error.message() :
                                            error));
            }
            quarantined = staging_root;
            quarantined += "-previous";
            std::filesystem::rename(
                destination, quarantined, filesystem_error);
            if (filesystem_error) {
                std::filesystem::remove_all(staging_root, filesystem_error);
                return Finish(
                    BackendPackInstallStatus::FilesystemFailure,
                    "Cannot quarantine the corrupt pack before repair: " +
                        filesystem_error.message());
            }
            if (checkpoint_ &&
                !checkpoint_(BackendPackInstallCheckpoint::AfterQuarantine)) {
                std::error_code restore_error;
                std::filesystem::rename(
                    quarantined, destination, restore_error);
                if (!restore_error) {
                    std::filesystem::remove_all(
                        staging_root, filesystem_error);
                }
                return Finish(
                    restore_error
                        ? BackendPackInstallStatus::FilesystemFailure
                        : BackendPackInstallStatus::Interrupted,
                    restore_error
                        ? "Pack repair was interrupted and the quarantined pack could not be restored"
                        : "Pack repair interrupted after quarantine");
            }
        }
        std::filesystem::create_directories(
            destination.parent_path(), filesystem_error);
        if (filesystem_error || std::filesystem::exists(destination)) {
            if (!quarantined.empty()) {
                std::error_code restore_error;
                std::filesystem::rename(
                    quarantined, destination, restore_error);
                if (restore_error) {
                    return Finish(
                        BackendPackInstallStatus::FilesystemFailure,
                        "Pack destination became unavailable and the quarantined pack could not be restored");
                }
            }
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::FilesystemFailure,
                "Pack destination became unavailable during publication");
        }
        std::filesystem::rename(staged_payload, destination, filesystem_error);
        if (filesystem_error) {
            if (!quarantined.empty()) {
                std::error_code restore_error;
                std::filesystem::rename(
                    quarantined, destination, restore_error);
                if (restore_error) {
                    return Finish(
                        BackendPackInstallStatus::FilesystemFailure,
                        "Cannot publish the repair or restore the quarantined pack");
                }
            }
            std::filesystem::remove_all(staging_root, filesystem_error);
            return Finish(
                BackendPackInstallStatus::FilesystemFailure,
                "Cannot atomically publish the pack directory: " +
                    filesystem_error.message());
        }
        if (IsWithin(
                std::filesystem::absolute(runtime_root_),
                std::filesystem::absolute(staging_root))) {
            std::filesystem::remove_all(staging_root, filesystem_error);
        }
        if (!quarantined.empty() && IsWithin(
                std::filesystem::absolute(runtime_root_),
                std::filesystem::absolute(quarantined))) {
            std::filesystem::remove_all(quarantined, filesystem_error);
        }
        if (checkpoint_ &&
            !checkpoint_(BackendPackInstallCheckpoint::AfterPackPublish)) {
            return Finish(
                BackendPackInstallStatus::Interrupted,
                "Pack installation interrupted after publication",
                destination);
        }
    }

    if (!mutation_lease) {
        mutation_lease = std::make_unique<RuntimeMutationLease>();
        if (!mutation_lease->OwnsMutation()) {
            return Finish(
                BackendPackInstallStatus::ExecutionActive,
                "Execution is active before pack activation",
                destination);
        }
    }

    if (checkpoint_ &&
        !checkpoint_(BackendPackInstallCheckpoint::BeforeActivation)) {
        return Finish(
            BackendPackInstallStatus::InstalledUnqualified,
            "Complete pack is installed but was not activated",
            destination);
    }
    if (!activate) {
        return Finish(
            BackendPackInstallStatus::InstalledUnqualified,
            "Complete pack is installed and awaiting route qualification",
            destination);
    }
    progress.stage = BackendPackInstallStage::Activating;
    progress.message = "Activating the complete installed pack";
    SetProgress(progress);
    BackendPackStateService state_service(runtime_root_, execution_active_);
    auto activation = state_service.ActivateOptionalPack(
        payload.backend, payload.pack_id);
    if (activation.status != BackendPackStateStatus::Completed) {
        return Finish(
            BackendPackInstallStatus::InstalledUnqualified,
            "Complete pack is installed but activation failed: " +
                activation.message,
            destination, std::move(activation));
    }

    return Finish(
        already_installed
            ? BackendPackInstallStatus::AlreadyInstalledAndActivated
            : BackendPackInstallStatus::InstalledAndActivated,
        "Pack installed and activated", destination, std::move(activation));
}

void BackendPackInstaller::Cancel() {
    cancel_requested_.store(true);
}

BackendPackInstallProgress BackendPackInstaller::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

BackendPackInstallResult BackendPackInstaller::Finish(
    BackendPackInstallStatus status,
    std::string message,
    std::filesystem::path installed_directory,
    std::optional<BackendPackStateResult> activation) {
    auto progress = GetProgress();
    progress.stage =
        status == BackendPackInstallStatus::InstalledAndActivated ||
                status ==
                    BackendPackInstallStatus::AlreadyInstalledAndActivated ||
                status == BackendPackInstallStatus::InstalledUnqualified
            ? BackendPackInstallStage::Complete
            : BackendPackInstallStage::Failed;
    progress.message = message;
    SetProgress(progress);
    return {status, std::move(message), std::move(installed_directory),
            std::move(activation)};
}

void BackendPackInstaller::SetProgress(
    BackendPackInstallProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

const char* BackendPackInstallStatusName(BackendPackInstallStatus status) {
    switch (status) {
        case BackendPackInstallStatus::InstalledAndActivated:
            return "installed_and_activated";
        case BackendPackInstallStatus::AlreadyInstalledAndActivated:
            return "already_installed_and_activated";
        case BackendPackInstallStatus::InstalledUnqualified:
            return "installed_unqualified";
        case BackendPackInstallStatus::Busy: return "busy";
        case BackendPackInstallStatus::ExecutionActive:
            return "execution_active";
        case BackendPackInstallStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackInstallStatus::DiskBudgetExceeded:
            return "disk_budget_exceeded";
        case BackendPackInstallStatus::IntegrityFailure:
            return "integrity_failure";
        case BackendPackInstallStatus::Interrupted: return "interrupted";
        case BackendPackInstallStatus::FilesystemFailure:
            return "filesystem_failure";
        default: return "unknown";
    }
}

const char* BackendPackInstallStageName(BackendPackInstallStage stage) {
    switch (stage) {
        case BackendPackInstallStage::Idle: return "idle";
        case BackendPackInstallStage::Validating: return "validating";
        case BackendPackInstallStage::Copying: return "copying";
        case BackendPackInstallStage::Verifying: return "verifying";
        case BackendPackInstallStage::Deactivating: return "deactivating";
        case BackendPackInstallStage::PublishingPack:
            return "publishing_pack";
        case BackendPackInstallStage::Activating: return "activating";
        case BackendPackInstallStage::Complete: return "complete";
        case BackendPackInstallStage::Failed: return "failed";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
