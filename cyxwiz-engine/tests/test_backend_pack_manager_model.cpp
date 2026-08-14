#include "../src/core/backend_pack_manager_model.h"
#include "../src/core/backend_pack_catalog_adapter.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::BackendPackManagerRecord Pack(
    std::string id,
    cyxwiz::BackendPackCatalogSupport support,
    bool installed = false) {
    cyxwiz::BackendPackManagerRecord pack;
    pack.backend = "opencl";
    pack.pack_id = std::move(id);
    pack.catalog_support = support;
    pack.installed = installed;
    if (installed) pack.installed_pack_id = pack.pack_id;
    return pack;
}

void TestInstallerChoices() {
    auto recommended = Pack(
        "opencl-v1", cyxwiz::BackendPackCatalogSupport::Supported);
    recommended.recommended = true;
    auto diagnostic = Pack(
        "oneapi-v1", cyxwiz::BackendPackCatalogSupport::Diagnostic);
    diagnostic.recommended = true;
    auto blocked = Pack(
        "cuda-old", cyxwiz::BackendPackCatalogSupport::Blocked);
    blocked.recommended = true;
    const std::vector records{recommended, diagnostic, blocked};

    const auto automatic = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Recommended, records);
    Check(automatic.valid &&
              automatic.pack_ids == std::vector<std::string>{"opencl-v1"},
          "Recommended must include only supported recommended packs");

    const auto cpu = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::CpuOnly, records);
    Check(cpu.valid && cpu.pack_ids.empty(),
          "CPU only must not silently select an optional pack");

    const auto custom = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records,
        {"oneapi-v1"});
    Check(custom.valid && custom.pack_ids.size() == 1,
          "Custom may explicitly consent to a diagnostic pack");
    const auto rejected = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records,
        {"cuda-old"});
    Check(!rejected.valid,
          "Custom must reject catalog-blocked packs");
    const auto empty_custom = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records);
    Check(!empty_custom.valid,
          "Custom must require an explicit optional-pack choice");
}

void TestActionPolicy() {
    cyxwiz::BackendPackManagerContext context;
    context.packaged_runtime = true;
    context.catalog_available = true;
    context.delivery_available = true;
    context.maintenance_available = true;
    context.rollback_available = true;
    auto installed = Pack(
        "opencl-v1", cyxwiz::BackendPackCatalogSupport::Supported, true);

    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Verify, context, &installed).enabled,
          "An installed pack may be locally verified");
    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Remove, context, &installed).enabled,
          "An installed optional pack may be removed");
    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Rollback, context).enabled,
          "Validated rollback state should enable rollback");
    installed.delivery_metadata_available = true;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Repair, context, &installed).enabled,
          "In-process UI must keep active-pack repair exit-only");
    context.repair_available = true;
    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Repair, context, &installed).enabled,
          "An exit-safe delivery host may repair the exact installed pack");
    context.maintenance_pending = true;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Repair, context, &installed).enabled,
          "A queued maintenance action must block repair");
    context.maintenance_pending = false;
    context.repair_available = false;

    context.training_active = true;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Verify, context, &installed).enabled,
          "Verification must remain disabled during training");
    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Details, context, &installed).enabled,
          "Read-only details should remain available during training");

    context.training_active = false;
    context.maintenance_available = false;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Remove, context, &installed).enabled,
          "UI must not offer a fake removal without lifecycle wiring");
    context.maintenance_available = true;
    context.maintenance_pending = true;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Remove, context, &installed).enabled,
          "A queued maintenance request must block a second mutation");
    context.maintenance_pending = false;
    context.maintenance_identity_matches = false;
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Remove, context, &installed).enabled,
          "Maintenance must reject a process running a stale runtime identity");
}

void TestCatalogAdapter() {
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot snapshot;
    snapshot.catalog_path = "C:/CyxWiz/runtime/catalogs/current.json";
    cyxwiz::runtime::VerifiedBackendPackCatalogRecord candidate;
    candidate.catalog_entry.pack_id = "opencl-v2";
    candidate.catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    candidate.manifest_path =
        "C:/CyxWiz/runtime/catalogs/manifests/opencl-v2.json";
    candidate.manifest.emplace();
    candidate.manifest->pack_id = "opencl-v2";
    candidate.manifest->backend = "opencl";
    candidate.manifest->package_version = "2.0.0";
    candidate.manifest->archive.size = 5 * 1024 * 1024;
    candidate.manifest->licenses = {"arrayfire"};
    candidate.manifest->compatibility.provider_types = {"opencl-icd"};
    snapshot.records.push_back(candidate);

    cyxwiz::runtime::VerifiedBackendPackCatalogRecord unavailable;
    unavailable.catalog_entry.pack_id = "cuda-v2";
    unavailable.catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    unavailable.manifest_error = "Manifest SHA-256 differs from the signed catalog";
    snapshot.records.push_back(unavailable);

    cyxwiz::runtime::ActiveRuntimeState active;
    active.packs.push_back({"opencl", "opencl-v1"});
    active.packs.push_back({"oneapi", "oneapi-local"});
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(
        snapshot, active);
    Check(records.size() == 3,
          "Catalog view must retain current packs absent from the catalog");
    const auto& update = records[0];
    Check(update.pack_id == "opencl-v2" && update.installed &&
              !update.active &&
              update.installed_pack_id == "opencl-v1" &&
              update.update_available &&
              update.delivery_metadata_available &&
              update.licenses == std::vector<std::string>{"arrayfire"} &&
              update.provider_requirements ==
                  std::vector<std::string>{"opencl-icd"},
          "Catalog view must expose signed consent data and exact update identity");
    Check(!records[1].delivery_metadata_available &&
              !records[1].delivery_metadata_error.empty(),
          "Invalid per-pack metadata must stay visible but unavailable");

    cyxwiz::BackendPackManagerContext context;
    context.packaged_runtime = true;
    context.catalog_available = true;
    context.delivery_available = true;
    Check(cyxwiz::EvaluateBackendPackAction(
              cyxwiz::BackendPackAction::Update, context, &update).enabled,
          "A verified catalog target may update an installed backend");
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Repair, context, &update).enabled,
          "Repair must not relabel an older installed pack as the catalog target");
    Check(!cyxwiz::EvaluateBackendPackAction(
               cyxwiz::BackendPackAction::Install, context,
               &records[1]).enabled,
          "Missing verified manifest metadata must disable delivery");
}

void TestDisplayFormatting() {
    Check(cyxwiz::FormatBackendPackByteSize(0) == "Unavailable",
          "Unknown signed size needs an explicit label");
    Check(cyxwiz::FormatBackendPackByteSize(5 * 1024 * 1024) == "5.0 MiB",
          "Signed download size should be human readable");
}

}  // namespace

int main() {
    TestInstallerChoices();
    TestActionPolicy();
    TestCatalogAdapter();
    TestDisplayFormatting();
    std::cout << "Backend pack manager model tests passed\n";
    return 0;
}
