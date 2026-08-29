#include "backend_pack_manager_model.h"

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

bool CompatibilityAllowsConsent(const BackendPackManagerRecord &record) {
  if (!record.compatibility)
    return true;
  return record.compatibility->eligibility !=
             runtime::BackendPackEligibility::Incompatible &&
         record.compatibility->install_recommendation !=
             runtime::BackendPackInstallRecommendation::NotOffered;
}

BackendPackActionDecision Disabled(std::string reason) {
  return {false, std::move(reason)};
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
    return Disabled("This pack is incompatible with the selected runtime or "
                    "known machine capabilities");
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
               : Disabled("No catalog-authorized update is available");
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
      if (record.backend != "cpu" && IsBackendPackRecommended(record) &&
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
        !CatalogAllowsConsent(record->catalog_support) ||
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
         CatalogAllowsConsent(record.catalog_support) &&
         CompatibilityAllowsConsent(record);
}

bool IsBackendPackRecommended(const BackendPackManagerRecord &record) {
  return record.compatibility.has_value() &&
         record.compatibility->install_recommendation ==
             runtime::BackendPackInstallRecommendation::Recommended;
}

BackendPackInstallerPlan BuildBackendPackInstallerPlan(
    const BackendPackInstallerSelection &selection,
    const std::vector<BackendPackManagerRecord> &catalog_records,
    CyxWizInstallerMode mode) {
  BackendPackInstallerPlan plan;
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
        !base->delivery_metadata_available) {
      plan.message = "A supported signed CyxWiz Engine/CPU base is required "
                     "for a fresh installation";
      return plan;
    }
    plan.install_base = true;
    plan.base_pack_id = base->pack_id;
    plan.download_size_bytes = base->download_size_bytes;
  } else {
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
      if (base->pack_id.empty() ||
          base->catalog_support != BackendPackCatalogSupport::Supported ||
          !base->delivery_metadata_available) {
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
        !CatalogAllowsConsent(record->catalog_support) ||
        !CompatibilityAllowsConsent(*record) ||
        !record->delivery_metadata_available) {
      plan.message =
          "A selected pack is not deliverable from the signed catalog";
      return plan;
    }
    if (base && (record->runtime_set_id != base->runtime_set_id ||
                 record->companion_base_id != base->pack_id)) {
      plan.message = "A selected backend pack does not belong to the required "
                     "CPU base runtime set";
      return plan;
    }
    if (record->installed && record->active && !record->update_available) {
      continue;
    }
    if (record->download_size_bytes >
        std::numeric_limits<std::uint64_t>::max() - plan.download_size_bytes) {
      plan.message = "Selected pack download size is too large";
      return plan;
    }
    plan.download_size_bytes += record->download_size_bytes;
    plan.pack_ids.push_back(record->pack_id);
  }
  plan.valid = true;
  if (plan.install_base) {
    plan.message = "Install the required CyxWiz Engine/CPU base";
    if (!plan.pack_ids.empty()) {
      plan.message += " and " + std::to_string(plan.pack_ids.size()) +
                      " signed optional backend pack(s)";
    }
    plan.message += "; every compute route requires local verification";
  } else if (plan.update_base) {
    plan.message = "Update the signed CyxWiz Engine/CPU base";
    if (!plan.pack_ids.empty()) {
      plan.message += " and " + std::to_string(plan.pack_ids.size()) +
                      " compatible backend pack(s)";
    }
    plan.message += "; the new base activates CPU-only before optional packs";
  } else if (!plan.deactivate_backends.empty()) {
    plan.message = std::to_string(plan.deactivate_backends.size()) +
                   " optional backend route(s) will be deactivated; package "
                   "files will be kept";
  } else if (selection.deactivate_optional_backends) {
    plan.message = "CPU base is already the only active compute route";
  } else if (plan.pack_ids.empty()) {
    plan.message = "No optional backend-pack download is required";
  } else {
    plan.message =
        std::to_string(plan.pack_ids.size()) +
        " signed backend pack(s) will be downloaded and locally qualified";
  }
  return plan;
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
