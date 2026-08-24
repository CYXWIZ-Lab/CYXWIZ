#include "../src/core/backend_pack_manager_model.h"
#include "../src/core/backend_pack_catalog_adapter.h"
#include "backend_pack_platform.h"

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
    auto base = Pack(
        "base-v1", cyxwiz::BackendPackCatalogSupport::Supported);
    base.backend = "cpu";
    base.recommended = true;
    const std::vector records{recommended, diagnostic, blocked, base};

    const auto automatic = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Recommended, records);
    Check(automatic.valid &&
              automatic.pack_ids == std::vector<std::string>{"opencl-v1"},
          "Recommended must include only supported recommended packs");

    const auto cpu = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::CpuOnly, records);
    Check(cpu.valid && cpu.deactivate_optional_backends &&
              cpu.pack_ids.empty(),
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
    const auto explicit_base = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records, {"base-v1"});
    Check(!explicit_base.valid,
          "The required CPU base must not enter the optional-pack selection");
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

void TestInstallerPlan() {
    auto current = Pack(
        "cuda-v1", cyxwiz::BackendPackCatalogSupport::Supported, true);
    current.backend = "cuda";
    current.installed = true;
    current.active = true;
    current.delivery_metadata_available = true;
    current.download_size_bytes = 100;
    auto missing = Pack(
        "opencl-v1", cyxwiz::BackendPackCatalogSupport::Supported, false);
    missing.backend = "opencl";
    missing.delivery_metadata_available = true;
    missing.download_size_bytes = 250;
    const std::vector records{current, missing};

    cyxwiz::BackendPackInstallerSelection selection;
    selection.valid = true;
    selection.pack_ids = {"cuda-v1", "opencl-v1"};
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, records);
    Check(plan.valid && plan.pack_ids.size() == 1 &&
              plan.pack_ids.front() == "opencl-v1" &&
              plan.download_size_bytes == 250,
          "Installer plan must skip the exact active pack and size downloads");

    missing.delivery_metadata_available = false;
    const auto unavailable = cyxwiz::BuildBackendPackInstallerPlan(
        selection, std::vector{current, missing});
    Check(!unavailable.valid,
          "Installer plan must reject missing signed delivery metadata");

    auto update = current;
    update.pack_id = "cuda-v2";
    update.active = false;
    update.installed_pack_id = "cuda-v1";
    update.update_available = true;
    cyxwiz::BackendPackInstallerSelection cpu_only;
    cpu_only.valid = true;
    cpu_only.deactivate_optional_backends = true;
    const auto cpu_plan = cyxwiz::BuildBackendPackInstallerPlan(
        cpu_only, std::vector{current, update, missing});
    Check(cpu_plan.valid && cpu_plan.pack_ids.empty() &&
              cpu_plan.deactivate_backends ==
                  std::vector<std::string>{"cuda"},
          "CPU-only plan must deactivate each active optional backend once");

    current.installed = false;
    current.active = false;
    current.installed_pack_id.clear();
    const auto already_cpu = cyxwiz::BuildBackendPackInstallerPlan(
        cpu_only, std::vector{current, missing});
    Check(already_cpu.valid && already_cpu.deactivate_backends.empty(),
          "CPU-only plan must be a no-op when no optional route is active");
}

void TestFreshInstallerPlan() {
    cyxwiz::BackendPackManagerRecord base;
    base.backend = "cpu";
    base.pack_id = "base-v1";
    base.runtime_set_id = "set-v1";
    base.catalog_support = cyxwiz::BackendPackCatalogSupport::Supported;
    base.delivery_metadata_available = true;
    base.download_size_bytes = 1000;

    auto optional = Pack(
        "opencl-v1", cyxwiz::BackendPackCatalogSupport::Supported);
    optional.runtime_set_id = "set-v1";
    optional.companion_base_id = "base-v1";
    optional.delivery_metadata_available = true;
    optional.download_size_bytes = 250;

    cyxwiz::BackendPackInstallerSelection selection;
    selection.valid = true;
    selection.pack_ids = {"opencl-v1"};
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, std::vector{base, optional},
        cyxwiz::CyxWizInstallerMode::FreshInstall);
    Check(plan.valid && plan.install_base &&
              plan.base_pack_id == "base-v1" &&
              plan.pack_ids == std::vector<std::string>{"opencl-v1"} &&
              plan.download_size_bytes == 1250,
          "Fresh plan must compose one required base with compatible optional packs");

    auto foreign = optional;
    foreign.pack_id = "opencl-v2";
    foreign.companion_base_id = "base-v2";
    selection.pack_ids = {"opencl-v2"};
    const auto incompatible = cyxwiz::BuildBackendPackInstallerPlan(
        selection, std::vector{base, foreign},
        cyxwiz::CyxWizInstallerMode::FreshInstall);
    Check(!incompatible.valid,
          "Fresh plan must reject a backend pack from another base runtime set");

    const auto missing_base = cyxwiz::BuildBackendPackInstallerPlan(
        selection, std::vector{foreign},
        cyxwiz::CyxWizInstallerMode::FreshInstall);
    Check(!missing_base.valid,
          "Fresh plan must fail closed without a signed CPU base");
}

void TestPackPlatformIdentity() {
    const auto platform = cyxwiz::runtime::CurrentBackendPackPlatformId();
    const auto architecture =
        cyxwiz::runtime::CurrentBackendPackArchitectureId();
    Check(platform == "win64" || platform == "linux64" ||
              platform == "macos",
          "Desktop pack platform must use a signed manifest identifier");
    Check(architecture == "x86_64" || architecture == "arm64",
          "Desktop pack architecture must use a signed manifest identifier");
#ifdef _WIN32
    Check(cyxwiz::runtime::CurrentEngineExecutableName() ==
              "cyxwiz-engine.exe" &&
              cyxwiz::runtime::CurrentRouteProbeExecutableName() ==
                  "cyxwiz-route-probe.exe" &&
              cyxwiz::runtime::CurrentBackendPackInstallerExecutableName() ==
                  "cyxwiz-backend-pack-installer.exe",
          "Windows package tools must use executable suffixes");
#else
    Check(cyxwiz::runtime::CurrentEngineExecutableName() ==
              "cyxwiz-engine" &&
              cyxwiz::runtime::CurrentRouteProbeExecutableName() ==
                  "cyxwiz-route-probe" &&
              cyxwiz::runtime::CurrentBackendPackInstallerExecutableName() ==
                  "cyxwiz-backend-pack-installer",
          "Unix package tools must use suffix-free executable names");
#endif
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
    active.runtime_set_id = "set-v1";
    active.base_pack_id = "base-v1";
    active.packs.push_back({"opencl", "opencl-v1"});
    active.packs.push_back({"oneapi", "oneapi-local"});
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(
        snapshot, active);
    Check(records.size() == 4,
          "Catalog view must retain the required base and current packs absent from the catalog");
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

void TestInstallLocation() {
#ifdef _WIN32
    const std::filesystem::path absolute_root = "C:\\Users\\test\\CyxWiz";
    const std::filesystem::path filesystem_root = "C:\\";
#else
    const std::filesystem::path absolute_root = "/home/test/CyxWiz";
    const std::filesystem::path filesystem_root = "/";
#endif
    const auto current_user = cyxwiz::ResolveCyxWizInstallLocation(
        absolute_root, cyxwiz::CyxWizInstallScope::CurrentUser);
    const auto all_users = cyxwiz::ResolveCyxWizInstallLocation(
        absolute_root / "." / "product" / "..",
        cyxwiz::CyxWizInstallScope::AllUsers);
    const auto relative = cyxwiz::ResolveCyxWizInstallLocation(
        "relative/CyxWiz");
    const auto root = cyxwiz::ResolveCyxWizInstallLocation(filesystem_root);
    Check(
        current_user.valid && current_user.install_root == absolute_root &&
            current_user.runtime_root == absolute_root / "runtime" &&
            !current_user.requires_elevation && all_users.valid &&
            all_users.install_root == absolute_root &&
            all_users.requires_elevation && !relative.valid && !root.valid,
        "Installation locations must be absolute, normalized, non-root, and least-privilege by default");
}

}  // namespace

int main() {
    TestInstallerChoices();
    TestActionPolicy();
    TestInstallerPlan();
    TestFreshInstallerPlan();
    TestPackPlatformIdentity();
    TestCatalogAdapter();
    TestDisplayFormatting();
    TestInstallLocation();
    std::cout << "Backend pack manager model tests passed\n";
    return 0;
}
