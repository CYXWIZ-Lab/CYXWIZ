#include "backend_pack_manager_model.h"
#include "addon_identity.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace cyxwiz {
namespace {

bool CatalogAllowsConsent(BackendPackCatalogSupport support) {
  return support == BackendPackCatalogSupport::Supported ||
         support == BackendPackCatalogSupport::Diagnostic;
}

// Plugin packs are delivered only when the signed catalog supports them (no
// diagnostic add-ons); compute packs may also be diagnostic.
bool CatalogAllowsConsent(const BackendPackManagerRecord &record) {
  return record.pack_kind == BackendPackRecordKind::Plugin
             ? record.catalog_support == BackendPackCatalogSupport::Supported
             : CatalogAllowsConsent(record.catalog_support);
}

bool CompatibilityAllowsConsent(const BackendPackManagerRecord &record) {
  return record.compatibility.has_value() &&
         record.compatibility->eligibility !=
             runtime::BackendPackEligibility::Incompatible &&
         record.compatibility->install_recommendation !=
             runtime::BackendPackInstallRecommendation::NotOffered;
}

BackendPackActionDecision Disabled(std::string reason) {
  return {false, std::move(reason)};
}

bool IsPluginRecord(const BackendPackManagerRecord &record) {
  return record.pack_kind == BackendPackRecordKind::Plugin;
}

std::string AddOnCount(std::size_t count) {
  return std::to_string(count) + (count == 1 ? " add-on" : " add-ons");
}

// What a base update does to the installed add-ons it keeps: they are
// reinstalled for the new base.
std::string KeptAddOnNote(std::size_t kept) {
  if (kept == 0)
    return {};
  return ". " + std::to_string(kept) + " installed add-on" +
         (kept == 1 ? " is" : "s are") +
         " reinstalled for the new base, so you keep " +
         (kept == 1 ? "it" : "them");
}

// The add-ons a base update loses: they have no pack for the new base yet.
std::string LostAddOnWarning(std::size_t lost) {
  if (lost == 0)
    return {};
  const char *object = lost == 1 ? "it" : "them";
  return std::to_string(lost) + " installed add-on" +
         (lost == 1 ? " has" : "s have") +
         " no version for the new CPU Engine yet. The Engine stops loading " +
         object + " until one is published; you can uninstall " + object +
         " under Add-ons.";
}

// The plan updates the CPU base only in Maintenance: a fresh install
// installs it and recovery changes nothing.
bool ModeUpdatesBase(CyxWizInstallerMode mode) {
  return mode == CyxWizInstallerMode::Maintenance;
}

// A CPU base update the installer can deliver from verified metadata.
bool IsDeliverableBase(const BackendPackManagerRecord &record) {
  return !record.pack_id.empty() &&
         record.catalog_support == BackendPackCatalogSupport::Supported &&
         record.delivery_metadata_available;
}

BackendPackActionDecision
CheckOperationBoundary(BackendPackAction action,
                       const BackendPackManagerContext &context) {
  if (action == BackendPackAction::Details)
    return {true, {}};
  if (context.operation_running) {
    return Disabled("Another backend operation is already running");
  }
  if (context.training_active) {
    return Disabled("Backend operations are disabled while training is active");
  }
  return {true, {}};
}

} // namespace

bool IsDamagedAddOn(const BackendPackManagerRecord &record) {
  return record.pack_kind == BackendPackRecordKind::Plugin && record.installed &&
         !record.installed_for_other_base && record.files_checked &&
         (!record.file_issues.empty() || !record.files_error.empty());
}

BackendPackActionDecision
EvaluateBackendPackAction(BackendPackAction action,
                          const BackendPackManagerContext &context,
                          const BackendPackManagerRecord *record) {
  const auto boundary = CheckOperationBoundary(action, context);
  if (!boundary.enabled)
    return boundary;

  if (action == BackendPackAction::Rollback) {
    if (!context.packaged_runtime) {
      return Disabled("Rollback is available only in a packaged runtime");
    }
    if (!context.maintenance_available) {
      return Disabled("Backend-pack maintenance is not connected");
    }
    if (!context.maintenance_identity_matches) {
      return Disabled(
          "Restart into the next-launch runtime before maintenance");
    }
    if (context.maintenance_pending) {
      return Disabled("A backend maintenance action is already queued");
    }
    return context.rollback_available
               ? BackendPackActionDecision{true, {}}
               : Disabled("No validated rollback state is available");
  }
  if (!record)
    return Disabled("Select a backend pack first");
  if (action == BackendPackAction::Details)
    return {true, {}};

  if (action == BackendPackAction::Verify) {
    if (!record->installed) {
      return Disabled("Install the backend pack before local verification");
    }
    return {true, {}};
  }

  if (!context.packaged_runtime) {
    return Disabled("Pack changes are available only in a packaged runtime");
  }
  if (action == BackendPackAction::Remove) {
    if (!record->installed)
      return Disabled("The backend pack is not installed");
    if (record->backend == "cpu") {
      return Disabled("The required CPU base cannot be removed");
    }
    if (context.maintenance_pending) {
      return Disabled("A backend maintenance action is already queued");
    }
    if (!context.maintenance_identity_matches) {
      return Disabled(
          "Restart into the next-launch runtime before maintenance");
    }
    return context.maintenance_available
               ? BackendPackActionDecision{true, {}}
               : Disabled("Backend-pack maintenance is not connected");
  }

  if (!context.catalog_available) {
    return Disabled("No current signed backend-pack catalog is available");
  }
  if (!CatalogAllowsConsent(record->catalog_support)) {
    return Disabled("The signed catalog does not authorize this pack");
  }
  if (!CompatibilityAllowsConsent(*record)) {
    return Disabled("Compatibility evidence is unavailable or this pack is "
                    "incompatible with the selected runtime or known machine "
                    "capabilities");
  }
  if (!record->delivery_metadata_available) {
    return Disabled(record->delivery_metadata_error.empty()
                        ? "The signed pack manifest is unavailable"
                        : record->delivery_metadata_error);
  }
  if (!context.delivery_available) {
    return Disabled("Signed backend-pack delivery is not connected");
  }
  switch (action) {
  case BackendPackAction::Install:
    return record->installed
               ? Disabled("This backend pack is already installed")
               : BackendPackActionDecision{true, {}};
  case BackendPackAction::Repair:
    if (!context.repair_available) {
      return Disabled("Repair must be applied safely after the Engine exits");
    }
    if (context.maintenance_pending) {
      return Disabled("A backend maintenance action is already queued");
    }
    if (!context.maintenance_identity_matches) {
      return Disabled("Restart into the next-launch runtime before repair");
    }
    return record->installed && record->installed_pack_id == record->pack_id
               ? BackendPackActionDecision{true, {}}
               : Disabled("Repair requires the exact installed catalog pack");
  case BackendPackAction::Update:
    if (!record->installed) {
      return Disabled("Install the backend pack before updating");
    }
    return record->update_available
               ? BackendPackActionDecision{true, {}}
               : Disabled(record->update_decision ? record->update_decision->message
                           : "No catalog-authorized update is available");
  case BackendPackAction::Verify:
  case BackendPackAction::Remove:
  case BackendPackAction::Details:
  case BackendPackAction::Rollback:
    break;
  }
  return Disabled("The requested backend-pack action is unavailable");
}

BackendPackInstallerSelection ResolveBackendPackInstallerSelection(
    BackendPackInstallChoice choice,
    const std::vector<BackendPackManagerRecord> &catalog_records,
    const std::vector<std::string> &custom_pack_ids) {
  BackendPackInstallerSelection result;
  if (choice == BackendPackInstallChoice::CpuOnly) {
    result.valid = true;
    result.deactivate_optional_backends = true;
    result.message = "Required CPU base only";
    return result;
  }

  std::set<std::string> selected;
  if (choice == BackendPackInstallChoice::Recommended) {
    for (const auto &record : catalog_records) {
      // Add-ons are never part of a profile: the user ticks them.
      if (record.backend != "cpu" && !IsPluginRecord(record) &&
          IsBackendPackRecommended(record) &&
          record.catalog_support == BackendPackCatalogSupport::Supported &&
          CompatibilityAllowsConsent(record) && !record.pack_id.empty()) {
        selected.insert(record.pack_id);
      }
    }
    result.valid = true;
    result.pack_ids.assign(selected.begin(), selected.end());
    result.message = result.pack_ids.empty()
                         ? "No catalog-supported optional pack is recommended; "
                           "CPU base remains selected"
                         : "Catalog-supported recommendations require local "
                           "verification after installation";
    return result;
  }

  for (const auto &pack_id : custom_pack_ids) {
    const auto record =
        std::find_if(catalog_records.begin(), catalog_records.end(),
                     [&](const BackendPackManagerRecord &candidate) {
                       return candidate.pack_id == pack_id;
                     });
    if (record == catalog_records.end() || record->backend == "cpu" ||
        !CatalogAllowsConsent(*record) ||
        !CompatibilityAllowsConsent(*record)) {
      result.message = "Custom selection contains a pack not authorized by the "
                       "signed catalog";
      return result;
    }
    selected.insert(pack_id);
  }
  result.pack_ids.assign(selected.begin(), selected.end());
  result.valid = !result.pack_ids.empty();
  result.message =
      result.valid
          ? "Custom packs require explicit consent and local verification"
          : "Choose at least one optional backend pack, or use CPU only";
  return result;
}

bool HasSelectableCustomBackendPack(
    const std::vector<BackendPackManagerRecord> &catalog_records) {
  return std::any_of(catalog_records.begin(), catalog_records.end(),
                     [](const BackendPackManagerRecord &record) {
                       return IsBackendPackSelectableForInstaller(record);
                     });
}

bool IsBackendPackSelectableForInstaller(
    const BackendPackManagerRecord &record) {
  return record.backend != "cpu" && record.delivery_metadata_available &&
         CatalogAllowsConsent(record) &&
         CompatibilityAllowsConsent(record);
}

bool IsBackendPackRecommended(const BackendPackManagerRecord &record) {
  return record.compatibility.has_value() &&
         record.compatibility->install_recommendation ==
             runtime::BackendPackInstallRecommendation::Recommended;
}

AddOnWords AddOnWordsFor(const std::vector<BackendPackManagerRecord> &records,
                         const std::string &plugin_id) {
  const BackendPackManagerRecord *newest = nullptr;
  for (const auto &record : records) {
    if (!IsPluginRecord(record) || record.plugin_id != plugin_id ||
        record.plugin_name.empty()) {
      continue;
    }
    // Signed UTC times of one format compare as text. On a tie a pack for
    // this CPU Engine wins over one left from another, then the first listed.
    if (newest == nullptr ||
        record.plugin_words_utc > newest->plugin_words_utc ||
        (record.plugin_words_utc == newest->plugin_words_utc &&
         newest->installed_for_other_base && !record.installed_for_other_base)) {
      newest = &record;
    }
  }
  return {AddOnTitle(plugin_id, newest ? newest->plugin_name : std::string{}),
          AddOnDescription(plugin_id, newest ? newest->plugin_description
                                             : std::string{})};
}

const BackendPackManagerRecord *
FindBaseUpdate(const std::vector<BackendPackManagerRecord> &records,
               CyxWizInstallerMode mode) {
  if (!ModeUpdatesBase(mode))
    return nullptr;
  const BackendPackManagerRecord *base = nullptr;
  for (const auto &record : records) {
    if (record.backend != "cpu" || !record.update_available)
      continue;
    if (base != nullptr)
      return nullptr;
    base = &record;
  }
  return base && IsDeliverableBase(*base) ? base : nullptr;
}

const BackendPackManagerRecord *
FindAddOnPackForBase(const std::vector<BackendPackManagerRecord> &records,
                     const std::string &plugin_id,
                     const std::string &base_pack_id) {
  const auto match = std::find_if(
      records.begin(), records.end(), [&](const auto &candidate) {
        return IsPluginRecord(candidate) && candidate.plugin_id == plugin_id &&
               !candidate.pack_id.empty() && !base_pack_id.empty() &&
               candidate.companion_base_id == base_pack_id &&
               candidate.catalog_support ==
                   BackendPackCatalogSupport::Supported &&
               candidate.delivery_metadata_available;
      });
  return match == records.end() ? nullptr : &*match;
}

BackendPackInstallerPlan BuildBackendPackInstallerPlan(
    const BackendPackInstallerSelection &selection,
    const std::vector<BackendPackManagerRecord> &catalog_records,
    CyxWizInstallerMode mode) {
  BackendPackInstallerPlan plan;
  if (mode == CyxWizInstallerMode::RecoveryRequired) {
    plan.message = "Recovery is required before this installation can be modified";
    return plan;
  }
  if (!selection.valid) {
    plan.message = selection.message.empty()
                       ? "The installer selection is invalid"
                       : selection.message;
    return plan;
  }
  const BackendPackManagerRecord *base = nullptr;
  if (mode == CyxWizInstallerMode::FreshInstall) {
    for (const auto &record : catalog_records) {
      if (record.backend != "cpu")
        continue;
      if (base != nullptr) {
        plan.message = "The signed catalog contains more than one CPU base for "
                       "this target";
        return plan;
      }
      base = &record;
    }
    if (!base || base->pack_id.empty() ||
        base->catalog_support != BackendPackCatalogSupport::Supported ||
        !CompatibilityAllowsConsent(*base) ||
        !base->delivery_metadata_available) {
      plan.message = "A supported signed CyxWiz Engine/CPU base is required "
                     "for a fresh installation";
      return plan;
    }
    plan.install_base = true;
    plan.base_pack_id = base->pack_id;
    plan.download_size_bytes = base->download_size_bytes;
  } else if (ModeUpdatesBase(mode)) {
    for (const auto &record : catalog_records) {
      if (record.backend != "cpu" || !record.update_available)
        continue;
      if (base != nullptr) {
        plan.message = "The signed catalog contains more than one CPU base "
                       "update for this target";
        return plan;
      }
      base = &record;
    }
    if (base) {
      if (!IsDeliverableBase(*base)) {
        plan.message = "The available CyxWiz Engine update is not "
                       "deliverable from verified metadata";
        return plan;
      }
      plan.update_base = true;
      plan.base_pack_id = base->pack_id;
      plan.download_size_bytes = base->download_size_bytes;
    }
  }
  if (selection.deactivate_optional_backends) {
    std::set<std::string> active_backends;
    for (const auto &record : catalog_records) {
      if (record.installed && !record.installed_pack_id.empty() &&
          (record.backend == "cuda" || record.backend == "opencl" ||
           record.backend == "oneapi")) {
        active_backends.insert(record.backend);
      }
    }
    plan.deactivate_backends.assign(active_backends.begin(),
                                    active_backends.end());
  }
  for (const auto &pack_id : selection.pack_ids) {
    const auto record = std::find_if(
        catalog_records.begin(), catalog_records.end(),
        [&](const auto &candidate) { return candidate.pack_id == pack_id; });
    if (record == catalog_records.end() || record->backend == "cpu" ||
        !CatalogAllowsConsent(*record) ||
        !CompatibilityAllowsConsent(*record) ||
        !record->delivery_metadata_available) {
      plan.message =
          "A selected pack is not deliverable from the signed catalog";
      return plan;
    }
    if (base && IsPluginRecord(*record) &&
        record->companion_base_id != base->pack_id) {
      plan.message = "A selected add-on is built for another CyxWiz Engine "
                     "release";
      return plan;
    }
    if (base && !IsPluginRecord(*record) &&
        (record->runtime_set_id != base->runtime_set_id ||
         record->companion_base_id != base->pack_id)) {
      plan.message = "A selected backend pack does not belong to the required "
                     "CPU base runtime set";
      return plan;
    }
    if (record->installed && record->update_decision &&
        record->update_decision->disposition != runtime::BackendPackUpdateDisposition::Upgrade &&
        record->update_decision->disposition != runtime::BackendPackUpdateDisposition::SamePackage) {
      plan.message = record->update_decision->message;
      return plan;
    }
    if (record->installed && record->active && !record->update_available) {
      if (!IsDamagedAddOn(*record)) continue;
      plan.repair_plugin_pack_ids.push_back(record->pack_id);
    }
    if (record->download_size_bytes >
        std::numeric_limits<std::uint64_t>::max() - plan.download_size_bytes) {
      plan.message = "Selected pack download size is too large";
      return plan;
    }
    plan.download_size_bytes += record->download_size_bytes;
    (IsPluginRecord(*record) ? plan.plugin_pack_ids : plan.pack_ids)
        .push_back(record->pack_id);
  }
  // The Engine loads an add-on only on the base it was built for: a base
  // update queues the new base's pack of every installed add-on, so the user
  // keeps it. One with no such pack is counted for the warning.
  std::size_t kept_add_ons = 0;
  std::size_t lost_add_ons = 0;
  if (plan.update_base) {
    std::set<std::string> installed_add_ons;
    for (const auto &record : catalog_records) {
      if (IsPluginRecord(record) && record.installed &&
          !record.plugin_id.empty()) {
        installed_add_ons.insert(record.plugin_id);
      }
    }
    for (const auto &plugin_id : installed_add_ons) {
      const auto *match =
          FindAddOnPackForBase(catalog_records, plugin_id, base->pack_id);
      if (!match) {
        ++lost_add_ons;
        continue;
      }
      ++kept_add_ons;
      if (std::find(plan.plugin_pack_ids.begin(), plan.plugin_pack_ids.end(),
                    match->pack_id) != plan.plugin_pack_ids.end()) {
        continue;
      }
      if (match->download_size_bytes >
          std::numeric_limits<std::uint64_t>::max() -
              plan.download_size_bytes) {
        plan.message = "Selected pack download size is too large";
        return plan;
      }
      plan.download_size_bytes += match->download_size_bytes;
      plan.plugin_pack_ids.push_back(match->pack_id);
    }
  }
  plan.valid = true;
  const std::string add_ons =
      plan.plugin_pack_ids.empty()
          ? std::string{}
          : " and " + AddOnCount(plan.plugin_pack_ids.size());
  if (plan.install_base) {
    plan.message = "Install the required CyxWiz Engine/CPU base";
    if (!plan.pack_ids.empty()) {
      plan.message += " and " + std::to_string(plan.pack_ids.size()) +
                      " signed optional backend pack(s)";
    }
    plan.message += add_ons;
    plan.message += "; every compute route requires local verification";
  } else if (plan.update_base) {
    plan.message = "Update the signed CyxWiz Engine/CPU base";
    if (!plan.pack_ids.empty()) {
      plan.message += " and " + std::to_string(plan.pack_ids.size()) +
                      " compatible backend pack(s)";
    }
    plan.message += add_ons;
    plan.message += "; the new base activates CPU-only before optional packs";
    plan.message += KeptAddOnNote(kept_add_ons);
    plan.warning = LostAddOnWarning(lost_add_ons);
  } else if (!plan.deactivate_backends.empty()) {
    plan.message = std::to_string(plan.deactivate_backends.size()) +
                   " optional backend route(s) will be deactivated; package "
                   "files will be kept";
  } else if (selection.deactivate_optional_backends) {
    plan.message = "CPU base is already the only active compute route";
  } else if (plan.pack_ids.empty() && plan.plugin_pack_ids.empty()) {
    plan.message = "No optional backend-pack download is required";
  } else if (plan.pack_ids.empty() &&
             plan.repair_plugin_pack_ids.size() == plan.plugin_pack_ids.size()) {
    // Only damaged add-ons ticked: they are repaired.
    const auto record = std::find_if(
        catalog_records.begin(), catalog_records.end(), [&](const auto &item) {
          return item.pack_id == plan.repair_plugin_pack_ids.front();
        });
    plan.message =
        plan.repair_plugin_pack_ids.size() == 1 && record != catalog_records.end()
            ? "Repair the " +
                  AddOnWordsFor(catalog_records, record->plugin_id).title +
                  " add-on from its signed package"
            : "Repair " + AddOnCount(plan.repair_plugin_pack_ids.size()) +
                  " from their signed packages";
    plan.message += "; the CPU Engine and compute packs are unchanged";
  } else if (plan.pack_ids.empty()) {
    plan.message = AddOnCount(plan.plugin_pack_ids.size()) +
                   " will be downloaded and installed; no device check is "
                   "needed";
  } else {
    plan.message =
        std::to_string(plan.pack_ids.size()) +
        " signed backend pack(s) will be downloaded and locally qualified";
    if (!plan.plugin_pack_ids.empty()) {
      plan.message += ", and " + AddOnCount(plan.plugin_pack_ids.size()) +
                      " installed";
    }
  }
  return plan;
}

std::vector<std::string> OptionalPacksForThisComputer(
    const std::vector<BackendPackManagerRecord> &records,
    const std::vector<std::string> &selected_pack_ids) {
  std::vector<std::string> backends;
  for (const auto &record : records) {
    if (record.backend == "cpu" || IsPluginRecord(record) || record.installed ||
        !record.compatibility)
      continue;
    if (record.compatibility->eligibility != runtime::BackendPackEligibility::Compatible ||
        !record.compatibility->recommendation_target_eligible) {
      continue;
    }
    if (std::find(selected_pack_ids.begin(), selected_pack_ids.end(), record.pack_id) !=
        selected_pack_ids.end()) {
      continue;
    }
    if (std::find(backends.begin(), backends.end(), record.backend) == backends.end()) {
      backends.push_back(record.backend);
    }
  }
  return backends;
}

std::string BackendPackCatalogSummary(
    const std::string &catalog_id,
    const std::vector<BackendPackManagerRecord> &records) {
  const auto optional_packs = std::count_if(
      records.begin(), records.end(), [](const auto &record) {
        return record.backend != "cpu" && !IsPluginRecord(record);
      });
  const auto add_ons =
      static_cast<std::size_t>(std::count_if(records.begin(), records.end(),
                                             IsPluginRecord));
  return "Signed catalog " + catalog_id + " verified; " +
         std::to_string(optional_packs) + " optional pack(s)" +
         (add_ons > 0 ? " and " + AddOnCount(add_ons) : std::string{}) +
         " for this computer.";
}

std::filesystem::path ParseInstallLocationText(std::string_view text) {
  const auto trim = [](std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return std::string_view{};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
  };
  text = trim(text);
  if (text.size() >= 2 && (text.front() == '"' || text.front() == '\'') &&
      text.back() == text.front()) {
    text = trim(text.substr(1, text.size() - 2));
  }
  return std::filesystem::path(std::u8string(
      reinterpret_cast<const char8_t *>(text.data()), text.size()));
}

CyxWizInstallLocation
ResolveCyxWizInstallLocation(std::filesystem::path install_root,
                             CyxWizInstallScope scope) {
  CyxWizInstallLocation result;
  result.scope = scope;
  result.requires_elevation = scope == CyxWizInstallScope::AllUsers;
  if (install_root.empty()) {
    result.message = "Choose an installation location";
    return result;
  }
  if (!install_root.is_absolute()) {
    result.message = "The installation location must be an absolute path";
    return result;
  }
  install_root = install_root.lexically_normal();
  if (install_root != install_root.root_path() &&
      install_root.filename().empty()) {
    install_root = install_root.parent_path();
  }
  if (install_root == install_root.root_path()) {
    result.message =
        "The filesystem root cannot be used as the installation location";
    return result;
  }
  result.valid = true;
  result.install_root = std::move(install_root);
  result.runtime_root = result.install_root / "runtime";
  result.message =
      result.requires_elevation
          ? "Install for all users; platform authorization will be requested "
            "when changes are applied"
          : "Install for the current user without system-wide changes";
  return result;
}

const char *BackendPackCatalogSupportName(BackendPackCatalogSupport support) {
  switch (support) {
  case BackendPackCatalogSupport::Supported:
    return "Supported";
  case BackendPackCatalogSupport::Diagnostic:
    return "Diagnostic only";
  case BackendPackCatalogSupport::Blocked:
    return "Blocked";
  case BackendPackCatalogSupport::Revoked:
    return "Revoked";
  case BackendPackCatalogSupport::Unavailable:
    return "Unavailable";
  }
  return "Unavailable";
}

std::string FormatBackendPackByteSize(std::uint64_t bytes) {
  if (bytes == 0)
    return "Unavailable";
  constexpr double kKiB = 1024.0;
  constexpr double kMiB = kKiB * 1024.0;
  constexpr double kGiB = kMiB * 1024.0;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1);
  if (bytes >= kGiB)
    stream << bytes / kGiB << " GiB";
  else if (bytes >= kMiB)
    stream << bytes / kMiB << " MiB";
  else if (bytes >= kKiB)
    stream << bytes / kKiB << " KiB";
  else
    return std::to_string(bytes) + " B";
  return stream.str();
}

} // namespace cyxwiz
