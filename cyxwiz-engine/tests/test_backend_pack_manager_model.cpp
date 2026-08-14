#include "../src/core/backend_pack_manager_model.h"

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
    TestDisplayFormatting();
    std::cout << "Backend pack manager model tests passed\n";
    return 0;
}
