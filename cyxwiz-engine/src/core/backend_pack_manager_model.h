#pragma once

#include "backend_pack_compatibility.h"
#include "backend_pack_update_policy.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

enum class BackendPackInstallChoice { Recommended, CpuOnly, Custom };

enum class CyxWizInstallerMode { FreshInstall, Maintenance, RecoveryRequired };

enum class CyxWizInstallScope { CurrentUser, AllUsers };

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

// A compute pack (the CPU base or an ArrayFire backend pack) or a plugin pack
// (an add-on such as Simulation (MuJoCo)): plugin packs come from schema-2
// manifests, have no backend, device or route, and are listed by plugin id
// in active-plugins.json.
enum class BackendPackRecordKind { Compute, Plugin };

struct BackendPackManagerRecord {
  BackendPackRecordKind pack_kind = BackendPackRecordKind::Compute;
  std::string backend;     // empty for a plugin pack
  std::string plugin_id;   // set for a plugin pack only
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
  // An add-on that active-plugins.json lists for another CPU base: the Engine
  // no longer loads it, and it can be uninstalled.
  bool installed_for_other_base = false;
  bool integrity_verified = false;
  bool qualification_evidence_available = false;
  bool training_authorized = false;
  bool update_available = false;
  std::optional<runtime::BackendPackUpdateDecision> update_decision;
  bool delivery_metadata_available = false;
  std::string delivery_metadata_error;
  std::optional<runtime::BackendPackCompatibilityDecision> compatibility;
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
  bool update_base = false;
  std::string base_pack_id;
  std::vector<std::string> pack_ids;
  // Plugin packs (add-ons) to install or update; installed after the base
  // and the backend packs, with no route qualification. A base update also
  // queues the new base's pack of every installed add-on, so none is lost.
  std::vector<std::string> plugin_pack_ids;
  std::vector<std::string> remove_pack_ids;
  std::vector<std::string> deactivate_backends;
  std::uint64_t download_size_bytes = 0;
  std::string message;
};

BackendPackActionDecision
EvaluateBackendPackAction(BackendPackAction action,
                          const BackendPackManagerContext &context,
                          const BackendPackManagerRecord *record = nullptr);

BackendPackInstallerSelection ResolveBackendPackInstallerSelection(
    BackendPackInstallChoice choice,
    const std::vector<BackendPackManagerRecord> &catalog_records,
    const std::vector<std::string> &custom_pack_ids = {});

bool HasSelectableCustomBackendPack(
    const std::vector<BackendPackManagerRecord> &catalog_records);

bool IsBackendPackSelectableForInstaller(
    const BackendPackManagerRecord &record);

bool IsBackendPackRecommended(const BackendPackManagerRecord &record);

BackendPackInstallerPlan BuildBackendPackInstallerPlan(
    const BackendPackInstallerSelection &selection,
    const std::vector<BackendPackManagerRecord> &catalog_records,
    CyxWizInstallerMode mode = CyxWizInstallerMode::Maintenance);

CyxWizInstallLocation ResolveCyxWizInstallLocation(
    std::filesystem::path install_root,
    CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser);

// GPU packs that suit this computer (compatible, a recommendation target)
// and are neither installed nor chosen: the installer names them so a
// CPU-only Recommended choice does not hide them. Backend names, catalog order.
std::vector<std::string> OptionalPacksForThisComputer(
    const std::vector<BackendPackManagerRecord> &records,
    const std::vector<std::string> &selected_pack_ids);

// The Backend Manager's catalog line: "Signed catalog <id> verified; N
// optional pack(s) for this computer." Add-ons are named (" and 1 add-on")
// only when the catalog offers some; the CPU base is not counted.
std::string BackendPackCatalogSummary(
    const std::string &catalog_id,
    const std::vector<BackendPackManagerRecord> &records);

// The installation folder as typed or pasted: surrounding spaces and one pair
// of surrounding quotes (as Explorer's "Copy as path" adds) are removed.
std::filesystem::path ParseInstallLocationText(std::string_view text);

const char *BackendPackCatalogSupportName(BackendPackCatalogSupport support);
std::string FormatBackendPackByteSize(std::uint64_t bytes);

} // namespace cyxwiz
