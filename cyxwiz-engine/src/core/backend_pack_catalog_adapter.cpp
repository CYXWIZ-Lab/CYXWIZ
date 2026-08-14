#include "backend_pack_catalog_adapter.h"

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

}  // namespace

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime) {
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
            record.backend = candidate.manifest->backend;
            record.package_version = candidate.manifest->package_version;
            record.download_size_bytes = candidate.manifest->archive.size;
            record.licenses = candidate.manifest->licenses;
            record.provider_requirements =
                candidate.manifest->compatibility.provider_types;
            installed = FindActiveBackend(active_runtime, record.backend);
        } else {
            installed = FindActivePack(active_runtime, record.pack_id);
            if (installed) record.backend = installed->backend;
        }
        if (installed) {
            record.installed = true;
            record.active = true;
            record.installed_pack_id = installed->pack_id;
            record.update_available =
                record.delivery_metadata_available &&
                record.installed_pack_id != record.pack_id;
        }
        records.push_back(std::move(record));
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
