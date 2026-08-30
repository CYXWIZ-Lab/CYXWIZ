#pragma once

#include "core/backend_pack_manager_model.h"
#include "core/installer_verification_summary.h"
#include "installer_cuda_prerequisite.h"
#include "installer_progress_channel.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::installer {

enum class InstallerPackageSource {
    CatalogHttps,
    OfflineSibling,
};

struct InstallerCatalogState {
    CyxWizInstallerMode mode = CyxWizInstallerMode::Maintenance;
    bool available = false;
    std::string catalog_id;
    std::string message;
    std::vector<BackendPackManagerRecord> records;
    InstallerVerificationSummary verification;
    InstallerCudaPrerequisiteState cuda_prerequisite;
};

struct InstallerOperationResult {
    bool succeeded = false;
    bool activated = false;
    std::string message;
};

struct InstallerCatalogRefreshResult {
    bool succeeded = false;
    std::string message;
};

using InstallerOperationDetailObserver =
    std::function<void(const InstallerHelperProgress &)>;

class BackendPackInstallerPlatform {
public:
    virtual ~BackendPackInstallerPlatform() = default;

    virtual void BeginPlanExecution() = 0;
    virtual void EndPlanExecution() = 0;
    virtual InstallerCatalogState Refresh() = 0;
    virtual InstallerCatalogRefreshResult RefreshOnline() = 0;
    virtual InstallerOperationResult InstallBase(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer = {}) = 0;
    virtual InstallerOperationResult UpdateBase(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer = {}) = 0;
    virtual InstallerOperationResult InstallOrUpdate(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer = {}) = 0;
    virtual InstallerOperationResult DeactivateBackend(
        const std::string& backend,
        const InstallerOperationDetailObserver& observer = {}) = 0;
    virtual InstallerOperationResult RequestCancellation() = 0;
    virtual InstallerOperationResult LaunchEngine() = 0;
    virtual InstallerOperationResult OpenInstalledManager() = 0;
    virtual std::string PlatformName() const = 0;
};

std::unique_ptr<BackendPackInstallerPlatform>
CreateBackendPackInstallerPlatform(
    std::filesystem::path runtime_root,
    std::filesystem::path metadata_root,
    std::filesystem::path executable_directory,
    CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser,
    std::string catalog_url = {},
    InstallerPackageSource package_source =
        InstallerPackageSource::CatalogHttps);

std::filesystem::path DefaultCyxWizInstallRoot(
    CyxWizInstallScope scope);

}  // namespace cyxwiz::installer
