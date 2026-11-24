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

// A plugin pack active for this base, by plugin id (or by pack id when the
// plugin id is not known). A list written for another base is inactive: the
// Engine skips it.
const runtime::ActivePluginPackState* FindActivePlugin(
    const runtime::ActivePluginsState& active_plugins,
    const runtime::ActiveRuntimeState& active_runtime,
    const std::string& plugin_id, const std::string& pack_id) {
    if (active_runtime.base_pack_id.empty() ||
        active_plugins.base_pack_id != active_runtime.base_pack_id) {
        return nullptr;
    }
    const auto match = std::find_if(
        active_plugins.plugins.begin(), active_plugins.plugins.end(),
        [&](const auto& plugin) {
            return plugin_id.empty() ? plugin.pack_id == pack_id
                                     : plugin.plugin_id == plugin_id;
        });
    return match == active_plugins.plugins.end() ? nullptr : &*match;
}

void ApplyUpdatePolicy(BackendPackManagerRecord& record,
    const runtime::VerifiedBackendPackManifest& candidate,
    const runtime::VerifiedBackendPackCatalogSnapshot& snapshot) {
    const auto previous = std::find_if(snapshot.installed_manifests.begin(),
        snapshot.installed_manifests.end(), [&](const auto& manifest) {
            return manifest.pack_id == record.installed_pack_id;
        });
    if (previous != snapshot.installed_manifests.end()) {
        record.update_decision = runtime::EvaluateBackendPackUpdate(*previous, candidate);
    } else if (record.installed_pack_id == record.pack_id) {
        record.update_decision = runtime::BackendPackUpdateDecision{
            runtime::BackendPackUpdateDisposition::SamePackage,
            "Already installed; no package download is needed"};
    } else {
        record.update_decision = runtime::BackendPackUpdateDecision{
            runtime::BackendPackUpdateDisposition::Unknown,
            "Installed version evidence is unavailable; update is blocked"};
    }
    record.update_available = record.update_decision->disposition ==
        runtime::BackendPackUpdateDisposition::Upgrade;
}

// An installed add-on the catalog does not list: its version, base and
// licences come from the signed manifest kept when it was installed.
void ApplyInstalledManifest(BackendPackManagerRecord& record,
    const runtime::VerifiedBackendPackCatalogSnapshot& snapshot) {
    const auto manifest = std::find_if(snapshot.installed_manifests.begin(),
        snapshot.installed_manifests.end(), [&](const auto& candidate) {
            return candidate.pack_id == record.installed_pack_id;
        });
    if (manifest == snapshot.installed_manifests.end()) return;
    record.package_version = manifest->package_version;
    record.runtime_set_id = manifest->runtime_set_id;
    record.companion_base_id = manifest->companion_base_id;
    record.licenses = manifest->licenses;
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
            candidate.manifest->platform != context.platform ||
            candidate.manifest->architecture != context.architecture ||
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
    const runtime::ActiveRuntimeState& active_runtime,
    const runtime::ActivePluginsState& active_plugins) {
    return BuildBackendPackCatalogRecords(
        catalog, active_runtime,
        BuildDefaultCompatibilityContext(catalog, active_runtime),
        active_plugins);
}

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime,
    const runtime::BackendPackCompatibilityContext& compatibility_context,
    const runtime::ActivePluginsState& active_plugins) {
    std::vector<BackendPackManagerRecord> records;
    records.reserve(catalog.records.size() + active_runtime.packs.size());
    for (const auto& candidate : catalog.records) {
        if (candidate.manifest &&
            (candidate.manifest->platform != compatibility_context.platform ||
             candidate.manifest->architecture != compatibility_context.architecture)) {
            continue;
        }
        BackendPackManagerRecord record;
        record.pack_id = candidate.catalog_entry.pack_id;
        record.catalog_path = catalog.catalog_path;
        record.manifest_path = candidate.manifest_path;
        record.catalog_support =
            MapSupport(candidate.catalog_entry.support_status);
        record.delivery_metadata_available = candidate.manifest.has_value();
        record.delivery_metadata_error = candidate.manifest_error;
        const runtime::ActivePackState* installed = nullptr;
        if (candidate.manifest &&
            candidate.manifest->kind ==
                runtime::BackendPackManifestKind::PluginPack) {
            const auto& manifest = *candidate.manifest;
            record.pack_kind = BackendPackRecordKind::Plugin;
            record.plugin_id = manifest.plugin_id;
            record.compatibility = runtime::EvaluateBackendPackCompatibility(
                manifest, compatibility_context);
            record.package_version = manifest.package_version;
            record.runtime_set_id = manifest.runtime_set_id;
            record.companion_base_id = manifest.companion_base_id;
            record.download_size_bytes = manifest.archive.size;
            record.licenses = manifest.licenses;
            if (const auto* plugin = FindActivePlugin(
                    active_plugins, active_runtime, manifest.plugin_id, {})) {
                record.installed = true;
                record.installed_pack_id = plugin->pack_id;
                record.active = record.installed_pack_id == record.pack_id;
                ApplyUpdatePolicy(record, manifest, catalog);
            }
            records.push_back(std::move(record));
            continue;
        }
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
                    ApplyUpdatePolicy(record, *candidate.manifest, catalog);
                }
            } else {
                installed = FindActiveBackend(active_runtime, record.backend);
            }
        } else {
            installed = FindActivePack(active_runtime, record.pack_id);
            if (installed) {
                record.backend = installed->backend;
            } else if (const auto* plugin = FindActivePlugin(
                           active_plugins, active_runtime, {},
                           record.pack_id)) {
                // An installed add-on whose catalog metadata cannot be
                // verified stays an add-on, so it can still be uninstalled.
                record.pack_kind = BackendPackRecordKind::Plugin;
                record.plugin_id = plugin->plugin_id;
                record.installed = true;
                record.installed_pack_id = plugin->pack_id;
                record.active = true;
            }
        }
        if (installed) {
            record.installed = true;
            record.installed_pack_id = installed->pack_id;
            record.active = record.installed_pack_id == record.pack_id;
            if (candidate.manifest) ApplyUpdatePolicy(record, *candidate.manifest, catalog);
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

    // Active add-ons the catalog no longer lists stay visible (and removable).
    if (!active_runtime.base_pack_id.empty() &&
        active_plugins.base_pack_id == active_runtime.base_pack_id) {
        for (const auto& installed : active_plugins.plugins) {
            const bool represented = std::any_of(
                records.begin(), records.end(), [&](const auto& record) {
                    return record.pack_kind == BackendPackRecordKind::Plugin &&
                           record.installed_pack_id == installed.pack_id;
                });
            if (represented) continue;
            BackendPackManagerRecord record;
            record.pack_kind = BackendPackRecordKind::Plugin;
            record.plugin_id = installed.plugin_id;
            record.pack_id = installed.pack_id;
            record.installed_pack_id = installed.pack_id;
            record.installed = true;
            record.active = true;
            ApplyInstalledManifest(record, catalog);
            records.push_back(std::move(record));
        }
    }

    // Add-ons installed for another base (the base was updated since): the
    // Engine no longer loads them. Each gets its own record, after the
    // catalog's, so it can be uninstalled while the catalog's packs stay
    // offered for this base.
    if (!active_runtime.base_pack_id.empty() &&
        !active_plugins.base_pack_id.empty() &&
        active_plugins.base_pack_id != active_runtime.base_pack_id) {
        for (const auto& installed : active_plugins.plugins) {
            BackendPackManagerRecord record;
            record.pack_kind = BackendPackRecordKind::Plugin;
            record.plugin_id = installed.plugin_id;
            record.pack_id = installed.pack_id;
            record.installed_pack_id = installed.pack_id;
            ApplyInstalledManifest(record, catalog);
            record.companion_base_id = active_plugins.base_pack_id;
            record.installed = true;
            record.installed_for_other_base = true;
            records.push_back(std::move(record));
        }
    }
    return records;
}

}  // namespace cyxwiz
