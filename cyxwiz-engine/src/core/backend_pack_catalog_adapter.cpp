#include "backend_pack_catalog_adapter.h"

#include "backend_pack_platform.h"

#include <algorithm>
#include <utility>

namespace cyxwiz {
namespace {

BackendPackCatalogSupport MapSupport(
    runtime::BackendPackSupportStatus support) {
    switch (support) {
        case runtime::BackendPackSupportStatus::Supported:
            return BackendPackCatalogSupport::Supported;
        case runtime::BackendPackSupportStatus::Diagnostic:
            return BackendPackCatalogSupport::Diagnostic;
        case runtime::BackendPackSupportStatus::Blocked:
            return BackendPackCatalogSupport::Blocked;
        case runtime::BackendPackSupportStatus::Revoked:
            return BackendPackCatalogSupport::Revoked;
    }
    return BackendPackCatalogSupport::Unavailable;
}

const runtime::ActivePackState* FindActiveBackend(
    const runtime::ActiveRuntimeState& active_runtime,
    const std::string& backend) {
    const auto match = std::find_if(
        active_runtime.packs.begin(), active_runtime.packs.end(),
        [&](const auto& pack) { return pack.backend == backend; });
    return match == active_runtime.packs.end() ? nullptr : &*match;
}

const runtime::ActivePackState* FindActivePack(
    const runtime::ActiveRuntimeState& active_runtime,
    const std::string& pack_id) {
    const auto match = std::find_if(
        active_runtime.packs.begin(), active_runtime.packs.end(),
        [&](const auto& pack) { return pack.pack_id == pack_id; });
    return match == active_runtime.packs.end() ? nullptr : &*match;
}

runtime::BackendPackCompatibilityContext BuildDefaultCompatibilityContext(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime) {
    runtime::BackendPackCompatibilityContext context;
    context.platform = runtime::CurrentBackendPackPlatformId();
    context.architecture = runtime::CurrentBackendPackArchitectureId();
    context.runtime_set_id = active_runtime.runtime_set_id;
    context.base_pack_id = active_runtime.base_pack_id;

    if (!active_runtime.base_pack_id.empty()) {
        for (const auto& candidate : catalog.records) {
            if (!candidate.manifest) continue;
            const auto& manifest = *candidate.manifest;
            if (manifest.runtime_set_id != active_runtime.runtime_set_id) {
                continue;
            }
            if ((manifest.kind == runtime::BackendPackManifestKind::Base &&
                 manifest.pack_id == active_runtime.base_pack_id) ||
                (manifest.kind ==
                     runtime::BackendPackManifestKind::BackendPack &&
                 manifest.companion_base_id == active_runtime.base_pack_id)) {
                if (context.arrayfire_abi.empty()) {
                    context.arrayfire_abi = manifest.arrayfire_abi;
                } else if (context.arrayfire_abi != manifest.arrayfire_abi) {
                    context.arrayfire_abi.clear();
                    break;
                }
            }
        }
        return context;
    }

    const runtime::VerifiedBackendPackManifest* selected_base = nullptr;
    for (const auto& candidate : catalog.records) {
        if (!candidate.manifest ||
            candidate.manifest->kind !=
                runtime::BackendPackManifestKind::Base ||
            candidate.manifest->compatibility.support_status !=
                runtime::BackendPackSupportStatus::Supported) {
            continue;
        }
        if (selected_base != nullptr) return context;
        selected_base = &*candidate.manifest;
    }
    if (selected_base != nullptr) {
        context.runtime_set_id = selected_base->runtime_set_id;
        context.base_pack_id = selected_base->pack_id;
        context.arrayfire_abi = selected_base->arrayfire_abi;
    }
    return context;
}

}  // namespace

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime) {
    return BuildBackendPackCatalogRecords(
        catalog, active_runtime,
        BuildDefaultCompatibilityContext(catalog, active_runtime));
}

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime,
    const runtime::BackendPackCompatibilityContext& compatibility_context) {
    std::vector<BackendPackManagerRecord> records;
    records.reserve(catalog.records.size() + active_runtime.packs.size());
    for (const auto& candidate : catalog.records) {
        BackendPackManagerRecord record;
        record.pack_id = candidate.catalog_entry.pack_id;
        record.catalog_path = catalog.catalog_path;
        record.manifest_path = candidate.manifest_path;
        record.catalog_support =
            MapSupport(candidate.catalog_entry.support_status);
        record.delivery_metadata_available = candidate.manifest.has_value();
        record.delivery_metadata_error = candidate.manifest_error;
        const runtime::ActivePackState* installed = nullptr;
        if (candidate.manifest) {
            record.compatibility = runtime::EvaluateBackendPackCompatibility(
                *candidate.manifest, compatibility_context);
            record.backend = candidate.manifest->backend;
            record.package_version = candidate.manifest->package_version;
            record.runtime_set_id = candidate.manifest->runtime_set_id;
            record.companion_base_id =
                candidate.manifest->companion_base_id;
            record.download_size_bytes = candidate.manifest->archive.size;
            record.licenses = candidate.manifest->licenses;
            record.provider_requirements =
                candidate.manifest->compatibility.provider_types;
            if (record.backend == "cpu") {
                if (!active_runtime.base_pack_id.empty()) {
                    record.installed = true;
                    record.installed_pack_id = active_runtime.base_pack_id;
                    record.active =
                        record.installed_pack_id == record.pack_id;
                    record.update_available =
                        record.installed_pack_id != record.pack_id;
                }
            } else {
                installed = FindActiveBackend(active_runtime, record.backend);
            }
        } else {
            installed = FindActivePack(active_runtime, record.pack_id);
            if (installed) record.backend = installed->backend;
        }
        if (installed) {
            record.installed = true;
            record.installed_pack_id = installed->pack_id;
            record.active = record.installed_pack_id == record.pack_id;
            record.update_available =
                record.delivery_metadata_available &&
                record.installed_pack_id != record.pack_id;
        }
        records.push_back(std::move(record));
    }

    if (!active_runtime.base_pack_id.empty()) {
        const bool base_represented = std::any_of(
            records.begin(), records.end(), [&](const auto& record) {
                return record.backend == "cpu" &&
                       record.installed_pack_id == active_runtime.base_pack_id;
            });
        if (!base_represented) {
            BackendPackManagerRecord record;
            record.backend = "cpu";
            record.pack_id = active_runtime.base_pack_id;
            record.installed_pack_id = active_runtime.base_pack_id;
            record.installed = true;
            record.active = true;
            records.push_back(std::move(record));
        }
    }

    for (const auto& installed : active_runtime.packs) {
        const bool represented = std::any_of(
            records.begin(), records.end(), [&](const auto& record) {
                return record.installed_pack_id == installed.pack_id;
            });
        if (represented) continue;
        BackendPackManagerRecord record;
        record.backend = installed.backend;
        record.pack_id = installed.pack_id;
        record.installed_pack_id = installed.pack_id;
        record.installed = true;
        record.active = true;
        records.push_back(std::move(record));
    }
    return records;
}

}  // namespace cyxwiz
