#include "backend_pack_manager_model.h"

#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>

namespace cyxwiz {
namespace {

bool CatalogAllowsConsent(BackendPackCatalogSupport support) {
    return support == BackendPackCatalogSupport::Supported ||
           support == BackendPackCatalogSupport::Diagnostic;
}

BackendPackActionDecision Disabled(std::string reason) {
    return {false, std::move(reason)};
}

BackendPackActionDecision CheckOperationBoundary(
    BackendPackAction action,
    const BackendPackManagerContext& context) {
    if (action == BackendPackAction::Details) return {true, {}};
    if (context.operation_running) {
        return Disabled("Another backend operation is already running");
    }
    if (context.training_active) {
        return Disabled("Backend operations are disabled while training is active");
    }
    return {true, {}};
}

}  // namespace

BackendPackActionDecision EvaluateBackendPackAction(
    BackendPackAction action,
    const BackendPackManagerContext& context,
    const BackendPackManagerRecord* record) {
    const auto boundary = CheckOperationBoundary(action, context);
    if (!boundary.enabled) return boundary;

    if (action == BackendPackAction::Rollback) {
        if (!context.packaged_runtime) {
            return Disabled("Rollback is available only in a packaged runtime");
        }
        if (!context.maintenance_available) {
            return Disabled("Backend-pack maintenance is not connected");
        }
        return context.rollback_available
            ? BackendPackActionDecision{true, {}}
            : Disabled("No validated rollback state is available");
    }
    if (!record) return Disabled("Select a backend pack first");
    if (action == BackendPackAction::Details) return {true, {}};

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
        if (!record->installed) return Disabled("The backend pack is not installed");
        if (record->backend == "cpu") {
            return Disabled("The required CPU base cannot be removed");
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
    if (!context.delivery_available) {
        return Disabled("Signed backend-pack delivery is not connected");
    }
    switch (action) {
        case BackendPackAction::Install:
            return record->installed
                ? Disabled("This backend pack is already installed")
                : BackendPackActionDecision{true, {}};
        case BackendPackAction::Repair:
            return record->installed
                ? BackendPackActionDecision{true, {}}
                : Disabled("Install the backend pack before repair");
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
    const std::vector<BackendPackManagerRecord>& catalog_records,
    const std::vector<std::string>& custom_pack_ids) {
    BackendPackInstallerSelection result;
    if (choice == BackendPackInstallChoice::CpuOnly) {
        result.valid = true;
        result.message = "Required CPU base only";
        return result;
    }

    std::set<std::string> selected;
    if (choice == BackendPackInstallChoice::Recommended) {
        for (const auto& record : catalog_records) {
            if (record.recommended &&
                record.catalog_support == BackendPackCatalogSupport::Supported &&
                !record.pack_id.empty()) {
                selected.insert(record.pack_id);
            }
        }
        result.valid = true;
        result.pack_ids.assign(selected.begin(), selected.end());
        result.message = result.pack_ids.empty()
            ? "No catalog-supported optional pack is recommended; CPU base remains selected"
            : "Catalog-supported recommendations require local verification after installation";
        return result;
    }

    for (const auto& pack_id : custom_pack_ids) {
        const auto record = std::find_if(
            catalog_records.begin(), catalog_records.end(),
            [&](const BackendPackManagerRecord& candidate) {
                return candidate.pack_id == pack_id;
            });
        if (record == catalog_records.end() ||
            !CatalogAllowsConsent(record->catalog_support)) {
            result.message =
                "Custom selection contains a pack not authorized by the signed catalog";
            return result;
        }
        selected.insert(pack_id);
    }
    result.pack_ids.assign(selected.begin(), selected.end());
    result.valid = !result.pack_ids.empty();
    result.message = result.valid
        ? "Custom packs require explicit consent and local verification"
        : "Choose at least one optional backend pack, or use CPU only";
    return result;
}

const char* BackendPackCatalogSupportName(
    BackendPackCatalogSupport support) {
    switch (support) {
        case BackendPackCatalogSupport::Supported: return "Supported";
        case BackendPackCatalogSupport::Diagnostic: return "Diagnostic only";
        case BackendPackCatalogSupport::Blocked: return "Blocked";
        case BackendPackCatalogSupport::Revoked: return "Revoked";
        case BackendPackCatalogSupport::Unavailable: return "Unavailable";
    }
    return "Unavailable";
}

std::string FormatBackendPackByteSize(std::uint64_t bytes) {
    if (bytes == 0) return "Unavailable";
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = kKiB * 1024.0;
    constexpr double kGiB = kMiB * 1024.0;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1);
    if (bytes >= kGiB) stream << bytes / kGiB << " GiB";
    else if (bytes >= kMiB) stream << bytes / kMiB << " MiB";
    else if (bytes >= kKiB) stream << bytes / kKiB << " KiB";
    else return std::to_string(bytes) + " B";
    return stream.str();
}

}  // namespace cyxwiz
