#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {

enum class BackendPackInstallChoice {
    Recommended,
    CpuOnly,
    Custom
};

enum class CyxWizInstallerMode {
    FreshInstall,
    Maintenance
};

enum class CyxWizInstallScope {
    CurrentUser,
    AllUsers
};

struct CyxWizInstallLocation {
    bool valid = false;
    std::filesystem::path install_root;
    std::filesystem::path runtime_root;
    CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser;
    bool requires_elevation = false;
    std::string message;
};

enum class BackendPackCatalogSupport {
    Unavailable,
    Supported,
    Diagnostic,
    Blocked,
    Revoked
};

enum class BackendPackAction {
    Install,
    Verify,
    Repair,
    Update,
    Remove,
    Details,
    Rollback
};

struct BackendPackManagerRecord {
    std::string backend;
    std::string pack_id;
    std::string installed_pack_id;
    std::string package_version;
    std::string runtime_set_id;
    std::string companion_base_id;
    std::filesystem::path catalog_path;
    std::filesystem::path manifest_path;
    std::uint64_t download_size_bytes = 0;
    std::vector<std::string> licenses;
    std::vector<std::string> provider_requirements;
    BackendPackCatalogSupport catalog_support =
        BackendPackCatalogSupport::Unavailable;
    bool installed = false;
    bool active = false;
    bool integrity_verified = false;
    bool qualification_evidence_available = false;
    bool training_authorized = false;
    bool update_available = false;
    bool recommended = false;
    bool delivery_metadata_available = false;
    std::string delivery_metadata_error;
};

struct BackendPackManagerContext {
    bool packaged_runtime = false;
    bool catalog_available = false;
    bool delivery_available = false;
    bool repair_available = false;
    bool maintenance_available = false;
    bool maintenance_identity_matches = true;
    bool maintenance_pending = false;
    bool operation_running = false;
    bool training_active = false;
    bool rollback_available = false;
};

struct BackendPackActionDecision {
    bool enabled = false;
    std::string reason;
};

struct BackendPackInstallerSelection {
    bool valid = false;
    bool deactivate_optional_backends = false;
    std::vector<std::string> pack_ids;
    std::string message;
};

struct BackendPackInstallerPlan {
    bool valid = false;
    bool install_base = false;
    std::string base_pack_id;
    std::vector<std::string> pack_ids;
    std::vector<std::string> deactivate_backends;
    std::uint64_t download_size_bytes = 0;
    std::string message;
};

BackendPackActionDecision EvaluateBackendPackAction(
    BackendPackAction action,
    const BackendPackManagerContext& context,
    const BackendPackManagerRecord* record = nullptr);

BackendPackInstallerSelection ResolveBackendPackInstallerSelection(
    BackendPackInstallChoice choice,
    const std::vector<BackendPackManagerRecord>& catalog_records,
    const std::vector<std::string>& custom_pack_ids = {});

BackendPackInstallerPlan BuildBackendPackInstallerPlan(
    const BackendPackInstallerSelection& selection,
    const std::vector<BackendPackManagerRecord>& catalog_records,
    CyxWizInstallerMode mode = CyxWizInstallerMode::Maintenance);

CyxWizInstallLocation ResolveCyxWizInstallLocation(
    std::filesystem::path install_root,
    CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser);

const char* BackendPackCatalogSupportName(
    BackendPackCatalogSupport support);
std::string FormatBackendPackByteSize(std::uint64_t bytes);

}  // namespace cyxwiz
