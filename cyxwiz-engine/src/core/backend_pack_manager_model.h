#pragma once

#include "../plugin/plugin_pack.h"
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
  // A plugin pack's own name and description, from its signed manifest;
  // empty for a pack made before manifests carried them (AddOnTitle and
  // AddOnDescription then fall back).
  std::string plugin_name;
  std::string plugin_description;
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
  // An installed add-on's files against its signed inventory
  // (CheckInstalledAddOnFiles, core/addon_pack_files.h); nothing is known
  // when files_checked is false. integrity_verified is set when they match.
  bool files_checked = false;
  std::size_t files_listed = 0;
  std::vector<plugin::PackFileIssue> file_issues;
  std::string files_error;       // its signed metadata or the check itself failed
  std::string files_checked_at;  // "14:02"
  bool files_check_pending = false;  // being checked off the UI thread
  std::string files_check_note;      // why its files could not be checked (not damaged)
};

// An installed add-on whose files differ from its signed inventory: the
// Engine does not load it, and Repair puts the signed files back.
bool IsDamagedAddOn(const BackendPackManagerRecord &record);

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
  // The ones of plugin_pack_ids that are installed and damaged: repaired
  // from their signed pack (the helper's --plugin-pack-id with --repair).
  std::vector<std::string> repair_plugin_pack_ids;
  std::vector<std::string> remove_pack_ids;
  std::vector<std::string> deactivate_backends;
  std::uint64_t download_size_bytes = 0;
  std::string message;
  // What the user loses by applying the plan, shown apart from the message
  // in the warning tone (a base update leaves an installed add-on with no
  // version for the new base). Empty when nothing is lost.
  std::string warning;
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

// The CPU base update an installation's plan applies: the catalog's one CPU
// base with an update, when it can be delivered and the mode updates the base
// (Maintenance; a fresh install installs it, recovery changes nothing).
// nullptr when there is none, more than one, or it cannot be delivered (the
// plan then refuses it).
const BackendPackManagerRecord *
FindBaseUpdate(const std::vector<BackendPackManagerRecord> &records,
               CyxWizInstallerMode mode);

// The catalog's pack of an add-on built for a CPU base, as a base update
// installs it (supported, with verified metadata). nullptr when there is none.
const BackendPackManagerRecord *
FindAddOnPackForBase(const std::vector<BackendPackManagerRecord> &records,
                     const std::string &plugin_id,
                     const std::string &base_pack_id);

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
