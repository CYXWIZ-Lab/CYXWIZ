#include "core/backend_pack_catalog_adapter.h"
#include "core/backend_pack_manager_model.h"
#include "backend_pack_lifecycle_service.h"
#include "backend_pack_platform.h"

#include <filesystem>
#include <iostream>

// Read-only acceptance path for the actual signed release metadata. Shares the
// native verifier, catalog adapter, selection and plan used by the GUI.
int RunBackendPackCatalogAcceptance(const char* root, const char* current_utc,
                                   const char* archive, const char* destination) {
    using namespace cyxwiz;
    const auto metadata_root = std::filesystem::absolute(root);
    std::string error;
    auto trust = runtime::BackendPackTrustStore::Load(
        metadata_root / "trust/trusted-keys.json", error);
    if (!trust) {
        std::cerr << error << '\n';
        return 1;
    }
    runtime::BackendPackLifecycleService service(metadata_root,
        runtime::BackendPackMetadataVerifier(std::move(*trust), "0.2.0",
            std::string(runtime::CurrentBackendPackPlatformId()),
            std::string(runtime::CurrentBackendPackArchitectureId())));
    runtime::VerifiedBackendPackCatalogSnapshot snapshot;
    if (!service.ReadCatalogSnapshot(current_utc, snapshot, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    for (const auto& candidate : snapshot.records) {
        if (!candidate.manifest) {
            std::cerr << candidate.catalog_entry.pack_id << ": "
                      << candidate.manifest_error << '\n';
            return 1;
        }
    }
    const auto records = BuildBackendPackCatalogRecords(snapshot, {});
    for (const auto& record : records) {
        std::cout << record.pack_id << " backend=" << record.backend
                  << " version=" << record.package_version << '\n';
    }
    const auto selection = ResolveBackendPackInstallerSelection(
        BackendPackInstallChoice::CpuOnly, records);
    const auto plan = BuildBackendPackInstallerPlan(
        selection, records, CyxWizInstallerMode::FreshInstall);
    if (!plan.valid || !plan.install_base) {
        std::cerr << "No fresh CPU installation plan: " << plan.message << '\n';
        return 1;
    }
    std::cout << "PASS: signed catalog yields fresh CPU plan for "
              << plan.base_pack_id << '\n';
    if (archive && destination) {
        for (const auto& candidate : snapshot.records) {
            if (candidate.manifest->pack_id != plan.base_pack_id) continue;
            runtime::BackendPackArchiveExtractor extractor;
            const auto extracted = extractor.Extract(
                std::filesystem::absolute(archive), *candidate.manifest,
                std::filesystem::absolute(destination), 2ULL * 1024 * 1024 * 1024);
            std::cout << extracted.message << '\n';
            return extracted.status == runtime::BackendPackExtractionStatus::Extracted ? 0 : 1;
        }
        return 1;
    }
    return 0;
}
