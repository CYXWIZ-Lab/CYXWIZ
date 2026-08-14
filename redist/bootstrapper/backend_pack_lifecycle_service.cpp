#include "backend_pack_lifecycle_service.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <utility>

namespace cyxwiz::runtime {
namespace {

class RemovePrivateStaging {
public:
    explicit RemovePrivateStaging(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~RemovePrivateStaging() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

private:
    std::filesystem::path path_;
};

bool IsTerminalInstallStatus(BackendPackInstallStatus status) {
    return status == BackendPackInstallStatus::InstalledUnqualified ||
           status == BackendPackInstallStatus::InstalledAndActivated ||
           status ==
               BackendPackInstallStatus::AlreadyInstalledAndActivated;
}

bool SameRuntimeState(
    const ActiveRuntimeState& left,
    const ActiveRuntimeState& right) {
    if (left.runtime_set_id != right.runtime_set_id ||
        left.generation != right.generation ||
        left.base_pack_id != right.base_pack_id ||
        left.packs.size() != right.packs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.packs.size(); ++index) {
        if (left.packs[index].backend != right.packs[index].backend ||
            left.packs[index].pack_id != right.packs[index].pack_id) {
            return false;
        }
    }
    return true;
}

}  // namespace

BackendPackLifecycleService::BackendPackLifecycleService(
    std::filesystem::path runtime_root,
    BackendPackMetadataVerifier metadata_verifier,
    BackendPackExecutionActiveCheck execution_active,
    BackendPackQualificationHook qualification,
    BackendPackLifecycleObserver observer)
    : runtime_root_(std::move(runtime_root)),
      metadata_verifier_(std::move(metadata_verifier)),
      execution_active_(std::move(execution_active)),
      qualification_(std::move(qualification)),
      observer_(std::move(observer)),
      acquirer_([this](const BackendPackAcquisitionProgress& progress) {
          SetStage(BackendPackLifecycleStage::Acquiring, progress.message);
      }),
      extractor_([this](const BackendPackExtractionProgress& progress) {
          SetStage(BackendPackLifecycleStage::Extracting, progress.message);
      }),
      installer_(
          runtime_root_, execution_active_,
          [this](const BackendPackInstallProgress& progress) {
              SetStage(BackendPackLifecycleStage::Installing,
                       progress.message);
          }),
      remover_(
          runtime_root_, execution_active_,
          [this](const BackendPackRemovalProgress& progress) {
              SetStage(BackendPackLifecycleStage::Removing,
                       progress.message);
          }) {}

bool BackendPackLifecycleService::ReadCatalog(
    const std::filesystem::path& catalog_path,
    const std::string& current_utc,
    VerifiedBackendPackCatalog& output,
    std::string& error) const {
    return metadata_verifier_.VerifyCatalog(
        catalog_path, current_utc, output, error);
}

BackendPackLifecycleResult BackendPackLifecycleService::Deliver(
    const BackendPackDeliveryRequest& request,
    BackendPackArtifactSource& source) {
    std::unique_lock<std::mutex> operation_lock(
        operation_mutex_, std::try_to_lock);
    if (!operation_lock.owns_lock()) {
        return {BackendPackLifecycleStatus::Busy,
                "A backend-pack lifecycle operation is already running"};
    }
    cancel_requested_.store(false);
    BackendPackLifecycleProgress progress;
    progress.stage = BackendPackLifecycleStage::VerifyingCatalog;
    progress.pack_id = request.pack_id;
    progress.message = "Verifying the signed backend-pack catalog";
    SetProgress(progress);

    if (runtime_root_.empty() || !runtime_root_.is_absolute() ||
        request.pack_id.empty() || request.current_utc.empty() ||
        request.catalog_path.empty() ||
        !request.catalog_path.is_absolute() ||
        request.manifest_path.empty() ||
        !request.manifest_path.is_absolute()) {
        return Finish(
            BackendPackLifecycleStatus::InvalidRequest,
            "Runtime root, signed metadata paths, time, and pack ID are required",
            request.pack_id);
    }

    VerifiedBackendPackCatalog catalog;
    std::string error;
    if (!metadata_verifier_.VerifyCatalog(
            request.catalog_path, request.current_utc, catalog, error)) {
        return Finish(
            BackendPackLifecycleStatus::MetadataFailure, error,
            request.pack_id);
    }
    const auto catalog_entry = std::find_if(
        catalog.packs.begin(), catalog.packs.end(),
        [&](const BackendPackCatalogEntry& entry) {
            return entry.pack_id == request.pack_id;
        });
    if (catalog_entry == catalog.packs.end()) {
        return Finish(
            BackendPackLifecycleStatus::MetadataFailure,
            "Requested pack is not present in the verified catalog",
            request.pack_id);
    }
    if (catalog_entry->support_status == BackendPackSupportStatus::Blocked ||
        catalog_entry->support_status == BackendPackSupportStatus::Revoked) {
        return Finish(
            BackendPackLifecycleStatus::PolicyRejected,
            "Requested pack is blocked by the verified release catalog",
            request.pack_id);
    }

    SetStage(
        BackendPackLifecycleStage::VerifyingManifest,
        "Verifying the signed backend-pack manifest");
    VerifiedBackendPackManifest manifest;
    if (!metadata_verifier_.VerifyManifest(
            request.manifest_path, *catalog_entry, manifest, error)) {
        return Finish(
            BackendPackLifecycleStatus::MetadataFailure, error,
            request.pack_id);
    }
    progress = GetProgress();
    progress.backend = manifest.backend;
    SetProgress(progress);

    const auto installed_directory = runtime_root_ / "packs" /
        manifest.backend / manifest.pack_id;
    std::error_code filesystem_error;
    const bool preexisting =
        std::filesystem::exists(installed_directory, filesystem_error) &&
        !filesystem_error;
    if (filesystem_error) {
        return Finish(
            BackendPackLifecycleStatus::InstallationFailure,
            "Cannot inspect the installed backend-pack directory",
            manifest.pack_id, manifest.backend);
    }
    const auto artifact = runtime_root_ / "cache" / "artifacts" /
        manifest.pack_id / manifest.archive.file_name;
    const auto acquired = acquirer_.Acquire(
        source, artifact, manifest.archive.size, manifest.archive.sha256,
        request.acquisition_disk_budget_bytes);
    if (acquired.status != BackendPackAcquisitionStatus::Downloaded &&
        acquired.status != BackendPackAcquisitionStatus::AlreadyPresent) {
        return Finish(
            acquired.status == BackendPackAcquisitionStatus::Interrupted
                ? BackendPackLifecycleStatus::Interrupted
                : BackendPackLifecycleStatus::AcquisitionFailure,
            acquired.message, manifest.pack_id, manifest.backend);
    }
    if (cancel_requested_.load()) {
        return Finish(
            BackendPackLifecycleStatus::Interrupted,
            "Backend-pack delivery cancelled after acquisition",
            manifest.pack_id, manifest.backend);
    }

    const auto extraction = runtime_root_ / "staging" / "delivery" /
        (manifest.pack_id + "-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto extracted = extractor_.Extract(
        artifact, manifest, extraction,
        request.extraction_disk_budget_bytes);
    if (extracted.status != BackendPackExtractionStatus::Extracted) {
        return Finish(
            extracted.status == BackendPackExtractionStatus::Interrupted
                ? BackendPackLifecycleStatus::Interrupted
                : BackendPackLifecycleStatus::ExtractionFailure,
            extracted.message, manifest.pack_id, manifest.backend);
    }
    RemovePrivateStaging cleanup(extraction);
    const auto payload = manifest.BindExtractedDirectory(extraction);
    const auto installed = request.repair
        ? installer_.StageRepair(
              payload, request.installation_disk_budget_bytes)
        : installer_.StageInstallOrUpdate(
              payload, request.installation_disk_budget_bytes);
    if (!IsTerminalInstallStatus(installed.status)) {
        return Finish(
            installed.status == BackendPackInstallStatus::Interrupted
                ? BackendPackLifecycleStatus::Interrupted
                : BackendPackLifecycleStatus::InstallationFailure,
            installed.message, manifest.pack_id, manifest.backend,
            installed.installed_directory);
    }
    if (cancel_requested_.load()) {
        return Finish(
            BackendPackLifecycleStatus::Interrupted,
            "Complete pack is installed but delivery was cancelled before qualification",
            manifest.pack_id, manifest.backend,
            installed.installed_directory);
    }

    ActiveRuntimeState qualification_base;
    if (!LoadActiveRuntimeState(
            runtime_root_ / "active-runtime.json", qualification_base,
            error) ||
        qualification_base.runtime_set_id != manifest.runtime_set_id ||
        qualification_base.base_pack_id != manifest.companion_base_id ||
        qualification_base.generation ==
            std::numeric_limits<std::uint64_t>::max()) {
        return Finish(
            BackendPackLifecycleStatus::InstallationFailure,
            error.empty()
                ? "Active runtime identity changed before qualification"
                : error,
            manifest.pack_id, manifest.backend,
            installed.installed_directory);
    }
    ActiveRuntimeState qualification_candidate = qualification_base;
    const auto candidate_pack = std::find_if(
        qualification_candidate.packs.begin(),
        qualification_candidate.packs.end(),
        [&](const ActivePackState& pack) {
            return pack.backend == manifest.backend;
        });
    if (candidate_pack == qualification_candidate.packs.end()) {
        qualification_candidate.packs.push_back(
            {manifest.backend, manifest.pack_id});
    } else {
        candidate_pack->pack_id = manifest.pack_id;
    }
    std::sort(
        qualification_candidate.packs.begin(),
        qualification_candidate.packs.end(),
        [](const ActivePackState& left, const ActivePackState& right) {
            return left.backend < right.backend;
        });
    qualification_candidate.generation = qualification_base.generation + 1;

    SetStage(
        BackendPackLifecycleStage::Qualifying,
        "Requesting shared route qualification for the staged runtime");
    if (!qualification_) {
        return Finish(
            BackendPackLifecycleStatus::InstalledUnqualified,
            "Complete pack is installed but no route qualification service was supplied",
            manifest.pack_id, manifest.backend,
            installed.installed_directory);
    }
    BackendPackQualificationDecision qualification;
    try {
        qualification = qualification_(
            manifest, installed.installed_directory,
            qualification_candidate);
    } catch (const std::exception& exception) {
        qualification.message =
            std::string("Route qualification failed: ") + exception.what();
        return Finish(
            BackendPackLifecycleStatus::QualificationFailure,
            qualification.message, manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    } catch (...) {
        qualification.message = "Route qualification failed unexpectedly";
        return Finish(
            BackendPackLifecycleStatus::QualificationFailure,
            qualification.message, manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    }

    ActiveRuntimeState current;
    if (!LoadActiveRuntimeState(
            runtime_root_ / "active-runtime.json", current, error) ||
        !SameRuntimeState(qualification_base, current)) {
        return Finish(
            BackendPackLifecycleStatus::InstalledUnqualified,
            "Runtime state changed during qualification; stale evidence was not used for activation or cleanup",
            manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    }

    if (catalog_entry->support_status ==
            BackendPackSupportStatus::Diagnostic ||
        manifest.compatibility.support_status ==
            BackendPackSupportStatus::Diagnostic ||
        qualification.disposition ==
            BackendPackQualificationDisposition::InstalledUnqualified) {
        return Finish(
            BackendPackLifecycleStatus::InstalledUnqualified,
            qualification.message.empty()
                ? "Complete pack is installed but not authorized for normal training"
                : qualification.message,
            manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    }
    if (qualification.disposition ==
        BackendPackQualificationDisposition::RollbackRequired) {
        if (!preexisting) {
            const auto removal = remover_.Remove(
                manifest.backend, manifest.pack_id);
            if (removal.status != BackendPackRemovalStatus::Removed &&
                removal.status != BackendPackRemovalStatus::AlreadyAbsent) {
                return Finish(
                    BackendPackLifecycleStatus::MaintenanceFailure,
                    "Qualification rejected the pack and cleanup failed: " +
                        removal.message,
                    manifest.pack_id, manifest.backend,
                    installed.installed_directory, qualification);
            }
        }
        return Finish(
            BackendPackLifecycleStatus::RolledBack,
            qualification.message.empty()
                ? "Qualification rejected the candidate; the previous runtime remains active"
                : qualification.message,
            manifest.pack_id, manifest.backend,
            preexisting ? installed.installed_directory :
                          std::filesystem::path{},
            qualification);
    }
    if (cancel_requested_.load()) {
        return Finish(
            BackendPackLifecycleStatus::Interrupted,
            "Complete pack is qualified but activation was cancelled",
            manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    }

    SetStage(
        BackendPackLifecycleStage::Activating,
        "Activating the locally qualified backend pack");
    BackendPackStateService state_service(
        runtime_root_, execution_active_);
    const auto activation = state_service.ActivateOptionalPack(
        manifest.backend, manifest.pack_id);
    if (activation.status != BackendPackStateStatus::Completed) {
        return Finish(
            BackendPackLifecycleStatus::ActivationFailure,
            "Qualified pack remains installed but activation failed: " +
                activation.message,
            manifest.pack_id, manifest.backend,
            installed.installed_directory, qualification);
    }
    return Finish(
        BackendPackLifecycleStatus::InstalledAndActivated,
        qualification.message.empty()
            ? "Backend pack installed, qualified, and activated"
            : qualification.message,
        manifest.pack_id, manifest.backend,
        installed.installed_directory, qualification);
}

BackendPackLifecycleResult BackendPackLifecycleService::Remove(
    std::string backend,
    std::string pack_id) {
    std::unique_lock<std::mutex> operation_lock(
        operation_mutex_, std::try_to_lock);
    if (!operation_lock.owns_lock()) {
        return {BackendPackLifecycleStatus::Busy,
                "A backend-pack lifecycle operation is already running"};
    }
    cancel_requested_.store(false);
    BackendPackLifecycleProgress progress;
    progress.stage = BackendPackLifecycleStage::Removing;
    progress.backend = backend;
    progress.pack_id = pack_id;
    progress.message = "Removing the optional backend pack";
    SetProgress(progress);
    const auto removal = remover_.Remove(backend, pack_id);
    if (removal.status == BackendPackRemovalStatus::Removed ||
        removal.status == BackendPackRemovalStatus::AlreadyAbsent) {
        return Finish(
            BackendPackLifecycleStatus::Removed, removal.message,
            std::move(pack_id), std::move(backend));
    }
    return Finish(
        removal.status == BackendPackRemovalStatus::Interrupted
            ? BackendPackLifecycleStatus::Interrupted
            : BackendPackLifecycleStatus::MaintenanceFailure,
        removal.message, std::move(pack_id), std::move(backend),
        removal.quarantined_directory);
}

BackendPackLifecycleResult BackendPackLifecycleService::Rollback() {
    std::unique_lock<std::mutex> operation_lock(
        operation_mutex_, std::try_to_lock);
    if (!operation_lock.owns_lock()) {
        return {BackendPackLifecycleStatus::Busy,
                "A backend-pack lifecycle operation is already running"};
    }
    cancel_requested_.store(false);
    BackendPackLifecycleProgress progress;
    progress.stage = BackendPackLifecycleStage::RollingBack;
    progress.message = "Restoring the previous complete runtime state";
    SetProgress(progress);
    BackendPackStateService state_service(
        runtime_root_, execution_active_);
    const auto rollback = state_service.Rollback();
    return Finish(
        rollback.status == BackendPackStateStatus::Completed
            ? BackendPackLifecycleStatus::RolledBack
            : BackendPackLifecycleStatus::MaintenanceFailure,
        rollback.message);
}

void BackendPackLifecycleService::Cancel() {
    cancel_requested_.store(true);
    acquirer_.Cancel();
    extractor_.Cancel();
    installer_.Cancel();
    remover_.Cancel();
}

BackendPackLifecycleProgress
BackendPackLifecycleService::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

BackendPackLifecycleResult BackendPackLifecycleService::Finish(
    BackendPackLifecycleStatus status,
    std::string message,
    std::string pack_id,
    std::string backend,
    std::filesystem::path installed_directory,
    std::optional<BackendPackQualificationDecision> qualification) {
    auto progress = GetProgress();
    progress.stage =
        status == BackendPackLifecycleStatus::InstalledAndActivated ||
                status == BackendPackLifecycleStatus::InstalledUnqualified ||
                status == BackendPackLifecycleStatus::Removed ||
                status == BackendPackLifecycleStatus::RolledBack
            ? BackendPackLifecycleStage::Complete
            : BackendPackLifecycleStage::Failed;
    progress.message = message;
    if (!pack_id.empty()) progress.pack_id = pack_id;
    if (!backend.empty()) progress.backend = backend;
    SetProgress(progress);
    return {status, std::move(message), std::move(pack_id),
            std::move(backend), std::move(installed_directory),
            std::move(qualification)};
}

void BackendPackLifecycleService::SetProgress(
    BackendPackLifecycleProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

void BackendPackLifecycleService::SetStage(
    BackendPackLifecycleStage stage,
    std::string message) {
    auto progress = GetProgress();
    progress.stage = stage;
    progress.message = std::move(message);
    SetProgress(std::move(progress));
}

const char* BackendPackLifecycleStatusName(
    BackendPackLifecycleStatus status) {
    switch (status) {
        case BackendPackLifecycleStatus::InstalledAndActivated:
            return "installed_and_activated";
        case BackendPackLifecycleStatus::InstalledUnqualified:
            return "installed_unqualified";
        case BackendPackLifecycleStatus::Removed: return "removed";
        case BackendPackLifecycleStatus::RolledBack: return "rolled_back";
        case BackendPackLifecycleStatus::Busy: return "busy";
        case BackendPackLifecycleStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackLifecycleStatus::PolicyRejected:
            return "policy_rejected";
        case BackendPackLifecycleStatus::MetadataFailure:
            return "metadata_failure";
        case BackendPackLifecycleStatus::AcquisitionFailure:
            return "acquisition_failure";
        case BackendPackLifecycleStatus::ExtractionFailure:
            return "extraction_failure";
        case BackendPackLifecycleStatus::InstallationFailure:
            return "installation_failure";
        case BackendPackLifecycleStatus::QualificationFailure:
            return "qualification_failure";
        case BackendPackLifecycleStatus::ActivationFailure:
            return "activation_failure";
        case BackendPackLifecycleStatus::MaintenanceFailure:
            return "maintenance_failure";
        case BackendPackLifecycleStatus::Interrupted: return "interrupted";
        default: return "unknown";
    }
}

const char* BackendPackLifecycleStageName(
    BackendPackLifecycleStage stage) {
    switch (stage) {
        case BackendPackLifecycleStage::Idle: return "idle";
        case BackendPackLifecycleStage::VerifyingCatalog:
            return "verifying_catalog";
        case BackendPackLifecycleStage::VerifyingManifest:
            return "verifying_manifest";
        case BackendPackLifecycleStage::Acquiring: return "acquiring";
        case BackendPackLifecycleStage::Extracting: return "extracting";
        case BackendPackLifecycleStage::Installing: return "installing";
        case BackendPackLifecycleStage::Qualifying: return "qualifying";
        case BackendPackLifecycleStage::Activating: return "activating";
        case BackendPackLifecycleStage::Removing: return "removing";
        case BackendPackLifecycleStage::RollingBack: return "rolling_back";
        case BackendPackLifecycleStage::Complete: return "complete";
        case BackendPackLifecycleStage::Failed: return "failed";
        default: return "unknown";
    }
}

const char* BackendPackQualificationDispositionName(
    BackendPackQualificationDisposition disposition) {
    switch (disposition) {
        case BackendPackQualificationDisposition::Qualified:
            return "qualified";
        case BackendPackQualificationDisposition::InstalledUnqualified:
            return "installed_unqualified";
        case BackendPackQualificationDisposition::RollbackRequired:
            return "rollback_required";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
