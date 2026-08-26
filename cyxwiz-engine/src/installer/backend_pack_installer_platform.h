#pragma once

#include "core/backend_pack_manager_model.h"
#include "core/installer_verification_summary.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::installer {

struct InstallerCatalogState {
    CyxWizInstallerMode mode = CyxWizInstallerMode::Maintenance;
    bool available = false;
    std::string catalog_id;
    std::string message;
    std::vector<BackendPackManagerRecord> records;
    InstallerVerificationSummary verification;
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

class BackendPackInstallerPlatform {
public:
    virtual ~BackendPackInstallerPlatform() = default;

    virtual InstallerCatalogState Refresh() = 0;
    virtual InstallerCatalogRefreshResult RefreshOnline() = 0;
    virtual InstallerOperationResult InstallBase(
        const std::string& pack_id) = 0;
    virtual InstallerOperationResult InstallOrUpdate(
        const std::string& pack_id) = 0;
    virtual InstallerOperationResult DeactivateBackend(
        const std::string& backend) = 0;
    virtual std::string PlatformName() const = 0;
};

std::unique_ptr<BackendPackInstallerPlatform>
CreateBackendPackInstallerPlatform(
    std::filesystem::path runtime_root,
    std::filesystem::path metadata_root,
    std::filesystem::path executable_directory,
    CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser,
    std::string catalog_url = {});

std::filesystem::path DefaultCyxWizInstallRoot(
    CyxWizInstallScope scope);

}  // namespace cyxwiz::installer
