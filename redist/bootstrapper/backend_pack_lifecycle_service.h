#pragma once

#include "backend_pack_acquisition.h"
#include "backend_pack_archive_extractor.h"
#include "backend_pack_installer.h"
#include "backend_pack_metadata_verifier.h"
#include "backend_pack_remover.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::runtime {

enum class BackendPackQualificationDisposition {
    Qualified,
    InstalledUnqualified,
    RollbackRequired
};

struct BackendPackQualificationDecision {
    BackendPackQualificationDisposition disposition =
        BackendPackQualificationDisposition::InstalledUnqualified;
    std::string message;
};

using BackendPackQualificationHook = std::function<
    BackendPackQualificationDecision(
        const VerifiedBackendPackManifest&,
        const std::filesystem::path&,
        const ActiveRuntimeState&)>;

enum class BackendPackDeliverySource {
    CatalogHttps,
    OfflineSibling
};

enum class BackendPackRouteVerificationPolicy {
    RequiredBeforeActivation,
    DeferredToEngine
};

struct BackendPackDeliveryRequest {
    std::filesystem::path catalog_path;
    std::filesystem::path manifest_path;
    std::string current_utc;
    std::string pack_id;
    std::uint64_t acquisition_disk_budget_bytes = 0;
    std::uint64_t extraction_disk_budget_bytes = 0;
    std::uint64_t installation_disk_budget_bytes = 0;
    BackendPackAcquisitionRetryPolicy acquisition_retry;
    BackendPackDeliverySource source =
        BackendPackDeliverySource::CatalogHttps;
    BackendPackRouteVerificationPolicy route_verification =
        BackendPackRouteVerificationPolicy::RequiredBeforeActivation;
    bool repair = false;
    bool discard_operation_data_on_cancel = false;
    bool discard_artifact_on_success = false;
};

struct VerifiedBackendPackCatalogRecord {
    BackendPackCatalogEntry catalog_entry;
    std::filesystem::path manifest_path;
    std::optional<VerifiedBackendPackManifest> manifest;
    std::string manifest_error;
};

struct VerifiedBackendPackCatalogSnapshot {
    std::filesystem::path catalog_path;
    VerifiedBackendPackCatalog catalog;
    std::vector<VerifiedBackendPackCatalogRecord> records;
};

enum class BackendPackLifecycleStage {
    Idle,
    VerifyingCatalog,
    VerifyingManifest,
    Acquiring,
    Extracting,
    Installing,
    Qualifying,
    Activating,
    Removing,
    RollingBack,
    Complete,
    Failed
};

enum class BackendPackLifecycleStatus {
    InstalledAndActivated,
    InstalledUnqualified,
    Removed,
    RolledBack,
    Busy,
    InvalidRequest,
    PolicyRejected,
    MetadataFailure,
    AcquisitionFailure,
    ExtractionFailure,
    InstallationFailure,
    QualificationFailure,
    ActivationFailure,
    MaintenanceFailure,
    Interrupted
};

struct BackendPackLifecycleProgress {
    BackendPackLifecycleStage stage = BackendPackLifecycleStage::Idle;
    std::string pack_id;
    std::string backend;
    std::uint64_t completed_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::size_t component_index = 0;
    std::size_t component_count = 0;
    std::string message;
};

struct BackendPackLifecycleResult {
    BackendPackLifecycleResult() = default;

    BackendPackLifecycleResult(
        BackendPackLifecycleStatus status_value,
        std::string message_value,
        std::string pack_id_value = {},
        std::string backend_value = {},
        std::filesystem::path installed_directory_value = {},
        std::optional<BackendPackQualificationDecision> qualification_value =
            std::nullopt)
        : status(status_value),
          message(std::move(message_value)),
          pack_id(std::move(pack_id_value)),
          backend(std::move(backend_value)),
          installed_directory(std::move(installed_directory_value)),
          qualification(std::move(qualification_value)) {}

    BackendPackLifecycleStatus status =
        BackendPackLifecycleStatus::InvalidRequest;
    std::string message;
    std::string pack_id;
    std::string backend;
    std::filesystem::path installed_directory;
    std::optional<BackendPackQualificationDecision> qualification;
};

using BackendPackLifecycleObserver =
    std::function<void(const BackendPackLifecycleProgress&)>;

class BackendPackLifecycleService {
public:
    BackendPackLifecycleService(
        std::filesystem::path runtime_root,
        BackendPackMetadataVerifier metadata_verifier,
        BackendPackExecutionActiveCheck execution_active = {},
        BackendPackQualificationHook qualification = {},
        BackendPackLifecycleObserver observer = {});

    bool ReadCatalog(
        const std::filesystem::path& catalog_path,
        const std::string& current_utc,
        VerifiedBackendPackCatalog& output,
        std::string& error) const;
    bool ReadManifest(
        const std::filesystem::path& manifest_path,
        const BackendPackCatalogEntry& catalog_entry,
        VerifiedBackendPackManifest& output,
        std::string& error) const;
    bool ReadCatalogSnapshot(
        const std::string& current_utc,
        VerifiedBackendPackCatalogSnapshot& output,
        std::string& error) const;
    BackendPackLifecycleResult Deliver(
        const BackendPackDeliveryRequest& request,
        BackendPackArtifactSource& source);
    BackendPackLifecycleResult Deliver(
        const BackendPackDeliveryRequest& request);
    BackendPackLifecycleResult DeliverBase(
        const BackendPackDeliveryRequest& request,
        BackendPackArtifactSource& source);
    BackendPackLifecycleResult DeliverBase(
        const BackendPackDeliveryRequest& request);
    BackendPackLifecycleResult DeliverBaseUpdate(
        const BackendPackDeliveryRequest& request,
        BackendPackArtifactSource& source);
    BackendPackLifecycleResult DeliverBaseUpdate(
        const BackendPackDeliveryRequest& request);
    BackendPackLifecycleResult Remove(
        std::string backend,
        std::string pack_id);
    BackendPackLifecycleResult Rollback();

    void Cancel();
    BackendPackLifecycleProgress GetProgress() const;

private:
    enum class DeliveryTarget {
        OptionalPack,
        FreshBase,
        BaseUpdate
    };

    BackendPackLifecycleResult DeliverInternal(
        const BackendPackDeliveryRequest& request,
        BackendPackArtifactSource* source,
        DeliveryTarget target);
    BackendPackLifecycleResult Finish(
        BackendPackLifecycleStatus status,
        std::string message,
        std::string pack_id = {},
        std::string backend = {},
        std::filesystem::path installed_directory = {},
        std::optional<BackendPackQualificationDecision> qualification =
            std::nullopt);
    void SetProgress(BackendPackLifecycleProgress progress);
    void SetStage(BackendPackLifecycleStage stage, std::string message);

    std::filesystem::path runtime_root_;
    BackendPackMetadataVerifier metadata_verifier_;
    BackendPackExecutionActiveCheck execution_active_;
    BackendPackQualificationHook qualification_;
    BackendPackLifecycleObserver observer_;
    BackendPackArtifactAcquirer acquirer_;
    BackendPackArchiveExtractor extractor_;
    BackendPackInstaller installer_;
    BackendPackRemover remover_;
    std::atomic<bool> cancel_requested_{false};
    std::mutex operation_mutex_;
    mutable std::mutex progress_mutex_;
    BackendPackLifecycleProgress progress_;
};

const char* BackendPackLifecycleStatusName(
    BackendPackLifecycleStatus status);
const char* BackendPackLifecycleStageName(
    BackendPackLifecycleStage stage);
const char* BackendPackQualificationDispositionName(
    BackendPackQualificationDisposition disposition);

std::filesystem::path BackendPackCurrentCatalogPath(
    const std::filesystem::path& runtime_root);
std::filesystem::path BackendPackCachedManifestPath(
    const std::filesystem::path& runtime_root,
    const std::string& pack_id);

}  // namespace cyxwiz::runtime
