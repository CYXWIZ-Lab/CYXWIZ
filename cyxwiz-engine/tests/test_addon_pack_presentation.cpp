// The add-on card (Simulation (MuJoCo)) shared by the CyxWiz Installer's
// Individual components and the Engine's Installed packs: the record path a
// plugin-pack manifest takes through the catalog adapter, the installer
// selection and plan, and the card presentation. Main scenario: the signed
// MuJoCo plugin pack of the 2026-11-24 local end-to-end run.

#include "../src/core/addon_pack_presentation.h"
#include "../src/core/backend_pack_catalog_adapter.h"
#include "../src/core/backend_pack_decision_reconciliation.h"
#include "../src/core/backend_pack_manager_model.h"
#include "backend_pack_platform.h"

#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

constexpr const char* kBase = "cyxwiz-base-0.2.0-1-win64-x86_64";
constexpr const char* kMuJoCoPack = "cyxwiz-plugin-mujoco-1.0.0-1-win64-x86_64";
constexpr const char* kMuJoCoPlugin = "com.cyxwiz.simulation.mujoco";
constexpr std::uint64_t kMuJoCoArchiveBytes = 3327509;  // the signed zip

cyxwiz::runtime::VerifiedBackendPackCatalogRecord BaseRecord() {
    cyxwiz::runtime::VerifiedBackendPackCatalogRecord record;
    record.catalog_entry.pack_id = kBase;
    record.catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    record.manifest.emplace();
    auto& manifest = *record.manifest;
    manifest.kind = cyxwiz::runtime::BackendPackManifestKind::Base;
    manifest.pack_id = kBase;
    manifest.backend = "cpu";
    manifest.package_version = "1";
    manifest.platform = cyxwiz::runtime::CurrentBackendPackPlatformId();
    manifest.architecture = cyxwiz::runtime::CurrentBackendPackArchitectureId();
    manifest.runtime_set_id = "e2e-cpu";
    manifest.arrayfire_abi = "arrayfire-3.10";
    manifest.archive.size = 54;
    manifest.licenses = {"arrayfire"};
    manifest.compatibility.device_kinds = {"cpu"};
    manifest.compatibility.recommendation_targets = {"cpu"};
    manifest.compatibility.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    return record;
}

// The MuJoCo pack's signed schema-2 manifest, as the verifier reads it.
cyxwiz::runtime::VerifiedBackendPackCatalogRecord MuJoCoRecord(
    std::string pack_id = kMuJoCoPack, std::string package_version = "1",
    std::string companion_base = kBase) {
    cyxwiz::runtime::VerifiedBackendPackCatalogRecord record;
    record.catalog_entry.pack_id = pack_id;
    record.catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    record.manifest.emplace();
    auto& manifest = *record.manifest;
    manifest.kind = cyxwiz::runtime::BackendPackManifestKind::PluginPack;
    manifest.pack_id = std::move(pack_id);
    manifest.plugin_id = kMuJoCoPlugin;
    manifest.package_version = std::move(package_version);
    manifest.platform = cyxwiz::runtime::CurrentBackendPackPlatformId();
    manifest.architecture = cyxwiz::runtime::CurrentBackendPackArchitectureId();
    manifest.minimum_cyxwiz_release = "0.2.0";
    manifest.maximum_cyxwiz_release = "0.2.0";
    manifest.companion_base_id = std::move(companion_base);
    manifest.archive.size = kMuJoCoArchiveBytes;
    manifest.licenses = {"cyxwiz-package", "gymnasium", "mujoco", "mujoco-third-party"};
    manifest.compatibility.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    return record;
}

cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot Catalog() {
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot snapshot;
    snapshot.records = {BaseRecord(), MuJoCoRecord()};
    return snapshot;
}

cyxwiz::runtime::ActiveRuntimeState ActiveBase() {
    cyxwiz::runtime::ActiveRuntimeState active;
    active.runtime_set_id = "e2e-cpu";
    active.generation = 1;
    active.base_pack_id = kBase;
    return active;
}

cyxwiz::runtime::ActivePluginsState ActiveMuJoCo(std::string pack_id = kMuJoCoPack,
                                                 std::string base = kBase) {
    cyxwiz::runtime::ActivePluginsState plugins;
    plugins.base_pack_id = std::move(base);
    plugins.plugins.push_back({kMuJoCoPlugin, std::move(pack_id)});
    return plugins;
}

cyxwiz::AddOnCardView MuJoCoCard(const std::vector<cyxwiz::AddOnCardView>& cards) {
    const auto card = std::find_if(cards.begin(), cards.end(), [](const auto& candidate) {
        return candidate.key == kMuJoCoPlugin;
    });
    Check(card != cards.end(), "The known MuJoCo add-on always has a card");
    return *card;
}

const cyxwiz::BackendPackManagerRecord* FindRecord(
    const std::vector<cyxwiz::BackendPackManagerRecord>& records, const std::string& pack_id) {
    const auto record = std::find_if(records.begin(), records.end(),
                                     [&](const auto& candidate) { return candidate.pack_id == pack_id; });
    return record == records.end() ? nullptr : &*record;
}

// The Installer's cards and --select for an installation (Maintenance).
std::vector<cyxwiz::AddOnCardView> InstallerCards(
    const std::vector<cyxwiz::BackendPackManagerRecord>& records,
    const std::set<std::string>& selected_pack_ids) {
    return cyxwiz::BuildAddOnCards(records, selected_pack_ids, cyxwiz::AddOnCardContext::Installer,
                                   cyxwiz::CyxWizInstallerMode::Maintenance);
}

std::string Resolve(const std::vector<cyxwiz::BackendPackManagerRecord>& records,
                    const std::string& requested) {
    return cyxwiz::ResolvePackToSelect(records, requested,
                                       cyxwiz::CyxWizInstallerMode::Maintenance);
}

// The Engine's Installed packs cards.
std::vector<cyxwiz::AddOnCardView> EngineCards(
    const std::vector<cyxwiz::BackendPackManagerRecord>& records) {
    return cyxwiz::BuildAddOnCards(records, {}, cyxwiz::AddOnCardContext::EngineManager,
                                   cyxwiz::CyxWizInstallerMode::Maintenance);
}

std::string Detail(const cyxwiz::AddOnCardView& card, const std::string& key) {
    for (const auto& [name, value] : card.details) {
        if (name == key) return value;
    }
    return {};
}

void TestPluginManifestBecomesAnAddOnRecord() {
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), {});
    Check(records.size() == 2, "Fresh install lists the base and the add-on");
    const auto* mujoco = FindRecord(records, kMuJoCoPack);
    Check(mujoco && mujoco->pack_kind == cyxwiz::BackendPackRecordKind::Plugin &&
              mujoco->plugin_id == kMuJoCoPlugin && mujoco->backend.empty() &&
              mujoco->download_size_bytes == kMuJoCoArchiveBytes &&
              mujoco->companion_base_id == kBase && mujoco->provider_requirements.empty() &&
              !mujoco->installed,
          "A plugin-pack manifest is an add-on record, never a backend record");
    Check(mujoco->compatibility &&
              mujoco->compatibility->eligibility ==
                  cyxwiz::runtime::BackendPackEligibility::Compatible &&
              mujoco->compatibility->verification_requirement ==
                  cyxwiz::runtime::BackendPackVerificationRequirement::NotRequired,
          "The add-on matches the catalog's one base and needs no device check");

    // Verification evidence never touches an add-on (it has no route).
    auto reconciled = records;
    cyxwiz::InstallerVerificationSummary stale;
    stale.evidence_available = true;
    stale.evidence_matches_runtime = false;
    cyxwiz::ReconcileBackendPackDecisionEvidence(reconciled, stale);
    const auto* after = FindRecord(reconciled, kMuJoCoPack);
    Check(after->compatibility->verification_requirement ==
                  cyxwiz::runtime::BackendPackVerificationRequirement::NotRequired &&
              after->compatibility->verification_status ==
                  mujoco->compatibility->verification_status,
          "Route evidence must not mark an add-on as needing verification");
}

void TestProfilesNeverIncludeAnAddOn() {
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), {});
    const auto recommended = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Recommended, records);
    const auto cpu_only = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::CpuOnly, records);
    Check(recommended.valid && recommended.pack_ids.empty() && cpu_only.valid &&
              cpu_only.pack_ids.empty(),
          "Neither Recommended nor CPU only installs an add-on");
    Check(cyxwiz::OptionalPacksForThisComputer(records, {}).empty(),
          "The summary's optional-pack hint names compute backends only");
    Check(cyxwiz::IsBackendPackSelectableForInstaller(*FindRecord(records, kMuJoCoPack)),
          "The add-on can be ticked");
}

void TestNotInstalledCard() {
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), {});
    const auto cards = InstallerCards(records, {});
    Check(cards.size() == 1, "One add-on card for one add-on");
    const auto card = MuJoCoCard(cards);
    Check(card.title == "Simulation (MuJoCo)" &&
              card.description ==
                  "Physics simulation for reinforcement learning: MuJoCo environments, the 3D "
                  "viewport and the Environment Library.",
          "The card names the add-on in plain words");
    Check(card.status == cyxwiz::AddOnCardStatus::Optional && card.status_text == "Optional" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Neutral && !card.checked &&
              card.can_select && card.can_install && !card.can_uninstall,
          "Unticked by default and offered");
    Check(card.facts == "Version 1  |  3.2 MiB  |  Requires: CPU Engine (this release)",
          "Facts: version, the signed archive size, the Engine it is built for: " + card.facts);
    Check(card.facts.find("provider") == std::string::npos &&
              card.explanation ==
                  "Ready to use when the Engine next starts; no device check is needed.",
          "No provider or device-check wording on an add-on");
    Check(card.licence ==
              "Includes MuJoCo by Google DeepMind (Apache-2.0 licence) and Gymnasium task "
              "models (MIT licence).",
          "The licence line names the third-party parts: " + card.licence);
    Check(card.pack_id == kMuJoCoPack && card.installed_pack_id.empty() &&
              Detail(card, "Licence files") == "cyxwiz-package, gymnasium, mujoco, mujoco-third-party" &&
              Detail(card, "Built for CPU Engine") == kBase && Detail(card, "Plugin") == kMuJoCoPlugin,
          "Details keep every signed fact");
}

void TestTickedCardAndPlan() {
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), {});
    const std::set<std::string> ticked{kMuJoCoPack};
    const auto card = MuJoCoCard(InstallerCards(records, ticked));
    Check(card.status == cyxwiz::AddOnCardStatus::Selected && card.status_text == "Selected" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Accent && card.checked,
          "Ticking selects the add-on");

    const auto selection = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records, {kMuJoCoPack});
    Check(selection.valid, "A ticked add-on alone is a valid Custom choice");
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, records, cyxwiz::CyxWizInstallerMode::FreshInstall);
    Check(plan.valid && plan.install_base && plan.base_pack_id == kBase &&
              plan.pack_ids.empty() &&
              plan.plugin_pack_ids == std::vector<std::string>{kMuJoCoPack} &&
              plan.download_size_bytes == 54 + kMuJoCoArchiveBytes,
          "The add-on installs through its own plan list, after the base");
    Check(plan.message ==
              "Install the required CyxWiz Engine/CPU base and 1 add-on; every compute route "
              "requires local verification",
          "The plan says an add-on is installed: " + plan.message);

    auto maintenance_records =
        cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase());
    const auto maintenance = cyxwiz::BuildBackendPackInstallerPlan(
        selection, maintenance_records, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(maintenance.valid && !maintenance.install_base &&
              maintenance.plugin_pack_ids == std::vector<std::string>{kMuJoCoPack} &&
              maintenance.message ==
                  "1 add-on will be downloaded and installed; no device check is needed",
          "Adding the add-on to an installation: " + maintenance.message);
}

void TestInstalledCard() {
    const auto records =
        cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), ActiveMuJoCo());
    const auto* record = FindRecord(records, kMuJoCoPack);
    Check(record && record->installed && record->active &&
              record->installed_pack_id == kMuJoCoPack,
          "active-plugins.json marks the add-on installed");
    const auto card = MuJoCoCard(InstallerCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::Installed && card.status_text == "Installed" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Success && card.checked &&
              !card.can_select && card.can_uninstall && !card.can_install &&
              card.installed_pack_id == kMuJoCoPack,
          "An installed add-on offers Uninstall add-on");
    Check(card.explanation ==
              "Loaded by the Engine at start. Turn it off for a while in Apps > Plugin Manager.",
          "Installed wording");
    Check(card.select_disabled_reason == "Installed. Use Uninstall add-on to remove it.",
          "The disabled checkbox's reason is shown on the card");
    const auto wording = cyxwiz::BuildPackRemovalWording(records, kMuJoCoPack);
    Check(wording.title == "Uninstall add-on" && wording.subject == "Add-on: Simulation (MuJoCo)" &&
              wording.confirm == "Uninstall add-on" && wording.keep == "Keep add-on" &&
              wording.body.find("compute route") == std::string::npos,
          "The uninstall confirmation speaks of an add-on");
    Check(cyxwiz::BuildPackRemovalWording(records, "opencl-v1").title ==
              "Uninstall backend component",
          "A compute component keeps its own confirmation");

    const auto selection = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records, {kMuJoCoPack});
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, records, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(plan.valid && plan.plugin_pack_ids.empty() && plan.download_size_bytes == 0,
          "An installed add-on is not downloaded again");

    const auto other_base = cyxwiz::BuildBackendPackCatalogRecords(
        Catalog(), ActiveBase(), ActiveMuJoCo(kMuJoCoPack, "cyxwiz-base-0.1.0-1-win64-x86_64"));
    Check(!FindRecord(other_base, kMuJoCoPack)->installed,
          "Add-ons activated with another base are not installed for this one");

    // An installed add-on the catalog no longer lists stays removable.
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot base_only;
    base_only.records = {BaseRecord()};
    const auto unlisted =
        cyxwiz::BuildBackendPackCatalogRecords(base_only, ActiveBase(), ActiveMuJoCo());
    const auto unlisted_card = MuJoCoCard(InstallerCards(unlisted, {}));
    Check(unlisted_card.status == cyxwiz::AddOnCardStatus::Installed &&
              unlisted_card.can_uninstall && unlisted_card.installed_pack_id == kMuJoCoPack,
          "An installed add-on missing from the catalog can still be uninstalled");
}

void TestUpdateAvailableCard() {
    constexpr const char* kNext = "cyxwiz-plugin-mujoco-1.0.0-2-win64-x86_64";
    auto snapshot = Catalog();
    snapshot.records[1] = MuJoCoRecord(kNext, "2");
    snapshot.installed_manifests.push_back(*MuJoCoRecord().manifest);
    const auto records =
        cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase(), ActiveMuJoCo());
    const auto* record = FindRecord(records, kNext);
    Check(record && record->installed && !record->active && record->update_available,
          "A newer signed package of the installed add-on is an update");
    const auto card = MuJoCoCard(InstallerCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::UpdateAvailable &&
              card.status_text == "Update available" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Warning && card.can_select &&
              !card.checked && !card.can_uninstall &&
              card.action_hint == "Select this add-on to review and install its signed update." &&
              card.facts.rfind("Version 2", 0) == 0,
          "Update available asks to select the update");
    const auto ticked = MuJoCoCard(InstallerCards(records, {kNext}));
    Check(ticked.checked && ticked.status_text == "Update selected" && ticked.action_hint.empty(),
          "A ticked update says it installs with the other changes");
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                     records, {kNext}),
        records, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(plan.valid && plan.plugin_pack_ids == std::vector<std::string>{kNext},
          "The update installs through the add-on plan list");

    auto older = Catalog();
    older.installed_manifests.push_back(*MuJoCoRecord(kNext, "2").manifest);
    const auto blocked = cyxwiz::BuildBackendPackCatalogRecords(
        older, ActiveBase(), ActiveMuJoCo(kNext));
    const auto blocked_card = MuJoCoCard(InstallerCards(blocked, {}));
    Check(blocked_card.status == cyxwiz::AddOnCardStatus::UpdateBlocked &&
              !blocked_card.can_select && blocked_card.can_uninstall,
          "An older catalog package never replaces the installed add-on");
}

void TestBuiltForAnotherEngineCard() {
    auto snapshot = Catalog();
    snapshot.records[1] = MuJoCoRecord(kMuJoCoPack, "1", "cyxwiz-base-0.2.0-2-win64-x86_64");
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase());
    const auto card = MuJoCoCard(InstallerCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::NeedsEngineUpdate &&
              card.status_text == "Needs Engine update" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Warning && !card.can_select &&
              !card.can_install && card.facts.empty() &&
              card.explanation ==
                  "This add-on is built for a different CyxWiz Engine release. Update the CPU "
                  "Engine first, then add it.",
          "Companion base mismatch: built for another Engine");
    Check(Detail(card, "Built for CPU Engine") == "cyxwiz-base-0.2.0-2-win64-x86_64",
          "Details name the Engine it is built for");
    Check(!cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                        records, {kMuJoCoPack})
               .valid,
          "It cannot be selected for installation");

    // A fresh install whose catalog base differs: the plan refuses it too.
    cyxwiz::BackendPackInstallerSelection forced;
    forced.valid = true;
    forced.pack_ids = {kMuJoCoPack};
    auto fresh = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), {});
    for (auto& record : fresh) {
        if (record.pack_id == kMuJoCoPack) record.companion_base_id = "another-base";
    }
    Check(!cyxwiz::BuildBackendPackInstallerPlan(forced, fresh,
                                                 cyxwiz::CyxWizInstallerMode::FreshInstall)
               .valid,
          "The plan refuses an add-on for another base");
}

void TestNotPublishedAndUnavailableCards() {
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot base_only;
    base_only.records = {BaseRecord()};
    const auto missing =
        MuJoCoCard(InstallerCards(cyxwiz::BuildBackendPackCatalogRecords(base_only, {}), {}));
    Check(missing.status == cyxwiz::AddOnCardStatus::NotPublished &&
              missing.status_text == "Not published" && !missing.can_select &&
              missing.pack_id.empty() &&
              missing.explanation ==
                  "No signed MuJoCo add-on is published for this platform and release." &&
              missing.select_disabled_reason.empty(),
          "No pack for this platform: Not published (the explanation is the reason)");

    auto other_platform = Catalog();
    other_platform.records[1].manifest->platform =
        cyxwiz::runtime::CurrentBackendPackPlatformId() == std::string_view("win64") ? "linux64"
                                                                                      : "win64";
    Check(MuJoCoCard(InstallerCards(
                         cyxwiz::BuildBackendPackCatalogRecords(other_platform, {}), {}))
                  .status == cyxwiz::AddOnCardStatus::NotPublished,
          "Another platform's pack does not count as published here");

    auto diagnostic = Catalog();
    diagnostic.records[1].catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Diagnostic;
    diagnostic.records[1].manifest->compatibility.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Diagnostic;
    const auto diagnostic_records = cyxwiz::BuildBackendPackCatalogRecords(diagnostic, {});
    const auto diagnostic_card = MuJoCoCard(InstallerCards(diagnostic_records, {}));
    Check(diagnostic_card.status == cyxwiz::AddOnCardStatus::Unavailable &&
              !diagnostic_card.can_select &&
              !cyxwiz::ResolveBackendPackInstallerSelection(
                   cyxwiz::BackendPackInstallChoice::Custom, diagnostic_records, {kMuJoCoPack})
                   .valid,
          "A diagnostic add-on is never offered (plugin delivery is supported-only)");
}

void TestUnknownAddOnGetsAGenericCard() {
    auto snapshot = Catalog();
    auto other = MuJoCoRecord("cyxwiz-plugin-other-1.0.0-1-win64-x86_64");
    other.manifest->plugin_id = "com.example.other";
    other.manifest->licenses = {"cyxwiz-package"};
    snapshot.records.push_back(other);
    const auto cards =
        InstallerCards(cyxwiz::BuildBackendPackCatalogRecords(snapshot, {}), {});
    Check(cards.size() == 2 && cards[0].key == kMuJoCoPlugin && cards[1].key == "com.example.other" &&
              cards[1].title == "com.example.other" &&
              cards[1].licence == "Its licence files come with the add-on." &&
              cards[1].status == cyxwiz::AddOnCardStatus::Optional,
          "Known add-ons come first; another plugin pack still gets a card");
    Check(cyxwiz::AddOnTitle(kMuJoCoPlugin) == "Simulation (MuJoCo)" &&
              cyxwiz::KnownAddOnPluginIds() == std::vector<std::string>{kMuJoCoPlugin},
          "Shared add-on names");
}

// An add-on is called what its plugin calls itself: the signed pack carries
// the plugin.json name and description, for any plugin, with no table entry.
void TestAddOnWordsComeFromThePack() {
    constexpr const char* kFluidPlugin = "org.example.fluid-lab";
    constexpr const char* kFluidPack = "cyxwiz-plugin-fluid-lab-2.0.0-1-win64-x86_64";
    const auto fluid_card = [&](const std::vector<cyxwiz::AddOnCardView>& cards) {
        const auto card = std::find_if(cards.begin(), cards.end(), [&](const auto& candidate) {
            return candidate.key == kFluidPlugin;
        });
        Check(card != cards.end(), "A plugin pack of the catalog has a card");
        return *card;
    };
    auto fluid = MuJoCoRecord(kFluidPack);
    fluid.manifest->plugin_id = kFluidPlugin;
    fluid.manifest->plugin_name = "Fluid Lab";
    fluid.manifest->plugin_description = "Fluid simulation: tanks, pipes and pumps.";
    auto snapshot = Catalog();
    snapshot.records.push_back(fluid);

    const auto offered = cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase());
    const auto* record = FindRecord(offered, kFluidPack);
    Check(record && record->plugin_name == "Fluid Lab" &&
              record->plugin_description == "Fluid simulation: tanks, pipes and pumps.",
          "The record carries the pack's name and description");
    const auto card = fluid_card(InstallerCards(offered, {}));
    Check(card.title == "Fluid Lab" &&
              card.description == "Fluid simulation: tanks, pipes and pumps." &&
              Detail(card, "Plugin") == kFluidPlugin,
          "The card shows the plugin's own name and description: " + card.title);
    Check(fluid_card(EngineCards(offered)).title == "Fluid Lab", "So does the Engine's card");

    // A pack made before manifests carried them: this release's wording for an
    // add-on it expects, and the plugin id for any other, as before.
    const auto old_mujoco = MuJoCoCard(InstallerCards(offered, {}));
    Check(old_mujoco.title == "Simulation (MuJoCo)" &&
              old_mujoco.description.find("MuJoCo environments") != std::string::npos,
          "An older pack of an expected add-on keeps this release's wording");
    auto old_fluid = snapshot;
    old_fluid.records.back().manifest->plugin_name.clear();
    old_fluid.records.back().manifest->plugin_description.clear();
    const auto old_card = fluid_card(
        InstallerCards(cyxwiz::BuildBackendPackCatalogRecords(old_fluid, ActiveBase()), {}));
    Check(old_card.title == kFluidPlugin &&
              old_card.description == "An optional add-on from the signed catalog.",
          "An older pack of any other plugin shows under its id");

    // The pack's words win over this release's wording
    auto renamed = Catalog();
    renamed.records[1].manifest->plugin_name = "Simulation (MuJoCo 4)";
    renamed.records[1].manifest->plugin_description = "Physics for robots.";
    const auto renamed_card = MuJoCoCard(
        InstallerCards(cyxwiz::BuildBackendPackCatalogRecords(renamed, ActiveBase()), {}));
    Check(renamed_card.title == "Simulation (MuJoCo 4)" &&
              renamed_card.description == "Physics for robots.",
          "A named pack of an expected add-on shows its own words");

    // Installed: the summary lines and the uninstall confirmation use them too
    cyxwiz::runtime::ActivePluginsState active;
    active.base_pack_id = kBase;
    active.plugins.push_back({kFluidPlugin, kFluidPack});
    const auto installed = cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase(), active);
    Check(cyxwiz::BuildPackRemovalWording(installed, kFluidPack).subject == "Add-on: Fluid Lab",
          "The uninstall confirmation names the add-on by its own name");
    Check(cyxwiz::SummaryAddOnLines({}, {kFluidPack}, offered) ==
              std::vector<std::string>{"Fluid Lab"},
          "The summary names it by its own name");
    const auto selection = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, offered, {kFluidPack});
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, offered, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(plan.valid && cyxwiz::PlannedAddOnLines(plan, offered) ==
                            std::vector<std::string>{"Fluid Lab"},
          "The plan names it by its own name: " + plan.message);

    // No longer in the catalog: the name comes from the signed manifest kept
    // when it was installed
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot unlisted;
    unlisted.records = {BaseRecord()};
    unlisted.installed_manifests.push_back(*fluid.manifest);
    const auto kept = cyxwiz::BuildBackendPackCatalogRecords(unlisted, ActiveBase(), active);
    Check(fluid_card(InstallerCards(kept, {})).title == "Fluid Lab",
          "An installed add-on the catalog dropped keeps its name");

    Check(cyxwiz::AddOnTitle(kFluidPlugin, "Fluid Lab") == "Fluid Lab" &&
              cyxwiz::AddOnTitle(kFluidPlugin) == kFluidPlugin &&
              cyxwiz::AddOnTitle(kMuJoCoPlugin, "Other") == "Other" &&
              cyxwiz::AddOnDescription(kFluidPlugin, "Fluids.") == "Fluids." &&
              cyxwiz::AddOnDescription(kFluidPlugin) ==
                  "An optional add-on from the signed catalog.",
          "Name order: the pack's, this release's, the plugin id");
}

// After a CPU base update the add-on is still listed for the old base: the
// Engine no longer loads it. Its card says so, offers this base's pack when
// the catalog has one, and always offers Uninstall add-on.
void TestAddOnInstalledForAnotherBase() {
    constexpr const char* kOldBase = "cyxwiz-base-0.1.0-1-win64-x86_64";
    constexpr const char* kOldPack = "cyxwiz-plugin-mujoco-0.1.0-1-win64-x86_64";
    const auto old_list = ActiveMuJoCo(kOldPack, kOldBase);

    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot base_only;
    base_only.records = {BaseRecord()};
    const auto lost = cyxwiz::BuildBackendPackCatalogRecords(base_only, ActiveBase(), old_list);
    const auto* stale = FindRecord(lost, kOldPack);
    Check(stale && stale->installed && !stale->active && stale->installed_for_other_base &&
              stale->plugin_id == kMuJoCoPlugin && stale->companion_base_id == kOldBase,
          "An add-on listed for another base is an installed, inactive record");
    const auto lost_card = MuJoCoCard(InstallerCards(lost, {}));
    Check(lost_card.status == cyxwiz::AddOnCardStatus::NeedsReinstall &&
              lost_card.status_text == "Needs reinstall" &&
              lost_card.tone == cyxwiz::InstallerPackPresentationTone::Warning &&
              lost_card.can_uninstall && lost_card.installed_pack_id == kOldPack &&
              !lost_card.can_select && !lost_card.can_install && !lost_card.checked &&
              lost_card.pack_id.empty() &&
              lost_card.explanation ==
                  "Installed for another CyxWiz Engine release, so the Engine no longer loads "
                  "it. No version for this release is published yet; Uninstall add-on removes "
                  "its files.",
          "With no pack for this base the card offers only Uninstall add-on");
    Check(cyxwiz::BuildPackRemovalWording(lost, kOldPack).title == "Uninstall add-on",
          "Its uninstall confirmation speaks of an add-on");
    Check(Resolve(lost, kMuJoCoPlugin).empty(),
          "Nothing is ticked for it");

    const auto offered = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), old_list);
    const auto offered_card = MuJoCoCard(InstallerCards(offered, {}));
    Check(offered_card.status == cyxwiz::AddOnCardStatus::NeedsReinstall &&
              offered_card.status_text == "Needs reinstall" && offered_card.can_select &&
              offered_card.can_install && offered_card.can_uninstall &&
              offered_card.pack_id == kMuJoCoPack && offered_card.installed_pack_id == kOldPack &&
              Detail(offered_card, "Installed pack") == kOldPack &&
              Detail(offered_card, "Installed for CPU Engine") == kOldBase,
          "With a pack for this base the card offers to reinstall it");
    const auto ticked = MuJoCoCard(InstallerCards(offered, {kMuJoCoPack}));
    Check(ticked.checked && ticked.status_text == "Reinstall selected" &&
              ticked.tone == cyxwiz::InstallerPackPresentationTone::Accent,
          "A ticked reinstall says so");
    Check(offered_card.explanation.find("Select it to install") != std::string::npos &&
              ticked.explanation ==
                  "Installed for another CyxWiz Engine release, so the Engine no longer loads "
                  "it. The version for this release installs with the other changes.",
          "Select it ... shows only while the card is unticked: " + ticked.explanation);
    Check(Resolve(offered, kMuJoCoPlugin) == kMuJoCoPack,
          "The Engine's Reinstall add-on ticks this base's pack");
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                     offered, {kMuJoCoPack}),
        offered, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(plan.valid && plan.plugin_pack_ids == std::vector<std::string>{kMuJoCoPack},
          "The reinstall goes through the add-on plan list");
}

// --select <id>: an add-on follows its card, a compute pack its own rule.
void TestResolvePackToSelect() {
    const auto fresh = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase());
    Check(Resolve(fresh, kMuJoCoPlugin) == kMuJoCoPack &&
              Resolve(fresh, kMuJoCoPack) == kMuJoCoPack,
          "An add-on not installed is ticked by plugin id or pack id");
    const auto installed =
        cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), ActiveMuJoCo());
    Check(Resolve(installed, kMuJoCoPlugin).empty(),
          "An installed add-on is not ticked again");

    // Two packs of the add-on: the card's pack (the one this base takes) is
    // ticked, never just the first one listed.
    auto two = Catalog();
    two.records.insert(two.records.begin() + 1,
                       MuJoCoRecord("cyxwiz-plugin-mujoco-1.0.0-0-win64-x86_64", "1",
                                    "cyxwiz-base-0.1.0-1-win64-x86_64"));
    const auto two_records = cyxwiz::BuildBackendPackCatalogRecords(two, ActiveBase());
    Check(MuJoCoCard(InstallerCards(two_records, {})).pack_id == kMuJoCoPack &&
              Resolve(two_records, kMuJoCoPlugin) == kMuJoCoPack,
          "The add-on is ticked through its card's pack");
    Check(Resolve(two_records, kBase).empty() &&
              Resolve(two_records, "missing").empty() &&
              Resolve(two_records, "").empty(),
          "The CPU base and unknown ids are never ticked");
}

// A CPU base update with the add-on installed, as the adapter lists it: the
// base with its update, the add-on's pack for the installed base and, when
// the catalog has one, its pack for the new base (installed for the old base,
// so its update is blocked and it does not fit the running base).
constexpr const char* kNewBase = "cyxwiz-base-0.2.0-2-win64-x86_64";
constexpr const char* kNewBasePack = "cyxwiz-plugin-mujoco-1.0.0-3-win64-x86_64";

std::vector<cyxwiz::BackendPackManagerRecord> BaseUpdateRecords(bool old_pack_listed,
                                                                bool new_pack_listed) {
    cyxwiz::BackendPackManagerRecord base;
    base.backend = "cpu";
    base.pack_id = kNewBase;
    base.installed = true;
    base.installed_pack_id = kBase;
    base.update_available = true;
    base.catalog_support = cyxwiz::BackendPackCatalogSupport::Supported;
    base.delivery_metadata_available = true;
    base.download_size_bytes = 54;

    cyxwiz::BackendPackManagerRecord old_pack;
    old_pack.pack_kind = cyxwiz::BackendPackRecordKind::Plugin;
    old_pack.plugin_id = kMuJoCoPlugin;
    old_pack.pack_id = kMuJoCoPack;
    old_pack.installed_pack_id = kMuJoCoPack;
    old_pack.package_version = "1";
    old_pack.companion_base_id = kBase;
    old_pack.download_size_bytes = kMuJoCoArchiveBytes;
    old_pack.licenses = {"gymnasium", "mujoco"};
    old_pack.installed = true;
    old_pack.active = true;
    old_pack.catalog_support = cyxwiz::BackendPackCatalogSupport::Supported;
    old_pack.delivery_metadata_available = true;
    old_pack.compatibility.emplace();
    old_pack.compatibility->eligibility = cyxwiz::runtime::BackendPackEligibility::Compatible;
    old_pack.compatibility->install_recommendation =
        cyxwiz::runtime::BackendPackInstallRecommendation::AvailableAfterVerification;
    old_pack.update_decision = cyxwiz::runtime::BackendPackUpdateDecision{
        cyxwiz::runtime::BackendPackUpdateDisposition::SamePackage,
        "Already installed; no package download is needed"};

    auto new_pack = old_pack;
    new_pack.pack_id = kNewBasePack;
    new_pack.package_version = "3";
    new_pack.companion_base_id = kNewBase;
    new_pack.active = false;
    new_pack.compatibility->eligibility = cyxwiz::runtime::BackendPackEligibility::Incompatible;
    new_pack.compatibility->rule = cyxwiz::runtime::BackendPackCompatibilityRule::CompanionBase;
    new_pack.compatibility->install_recommendation =
        cyxwiz::runtime::BackendPackInstallRecommendation::NotOffered;
    new_pack.update_decision = cyxwiz::runtime::BackendPackUpdateDecision{
        cyxwiz::runtime::BackendPackUpdateDisposition::Unknown,
        "Backend updates require the same companion CPU base"};

    std::vector<cyxwiz::BackendPackManagerRecord> records{base};
    if (old_pack_listed) records.push_back(old_pack);
    if (new_pack_listed) records.push_back(new_pack);
    return records;
}

// Every word on a card, details included.
std::string CardWords(const cyxwiz::AddOnCardView& card) {
    std::string words = card.status_text + '\n' + card.facts + '\n' + card.explanation + '\n' +
                        card.action_hint + '\n' + card.select_disabled_reason;
    for (const auto& [name, value] : card.details) words += '\n' + name + ": " + value;
    return words;
}

// The plan's base update reinstalls the add-on: the card says so in add-on
// words, never in the compute update policy's.
void TestBaseUpdateReinstallsAddOn() {
    for (const bool old_pack_listed : {true, false}) {
        const auto records = BaseUpdateRecords(old_pack_listed, true);
        const auto card = MuJoCoCard(InstallerCards(records, {}));
        Check(card.status == cyxwiz::AddOnCardStatus::WithEngineUpdate &&
                  card.status_text == "Comes with the Engine update" &&
                  card.tone == cyxwiz::InstallerPackPresentationTone::Accent &&
                  card.explanation == "Reinstalled for the new CPU Engine with this update." &&
                  card.checked && !card.can_select && !card.can_install && card.can_uninstall &&
                  card.action_hint.empty(),
              "A base update that reinstalls the add-on says so on its card");
        Check(card.pack_id == kNewBasePack && card.installed_pack_id == kMuJoCoPack &&
                  card.facts == "Version 3  |  3.2 MiB  |  Requires: the CPU Engine update" &&
                  Detail(card, "Installed pack") == kMuJoCoPack &&
                  Detail(card, "Built for CPU Engine") == kNewBase,
              "The card shows the pack the update installs: " + card.facts);
        const auto words = CardWords(card);
        Check(words.find("Backend") == std::string::npos &&
                  words.find("companion") == std::string::npos &&
                  words.find("Update blocked") == std::string::npos,
              "No compute-pack wording on an add-on card: " + words);
        Check(Resolve(records, kMuJoCoPlugin).empty(),
              "Nothing to tick: it comes with the update");
    }

    // Without a base update the blocked pack keeps add-on words too.
    auto records = BaseUpdateRecords(false, true);
    records.front().update_available = false;
    const auto blocked = MuJoCoCard(InstallerCards(records, {}));
    Check(blocked.status == cyxwiz::AddOnCardStatus::UpdateBlocked &&
              blocked.explanation ==
                  "The catalog's version is built for a different CyxWiz Engine release, so the "
                  "installed version stays until the CPU Engine is updated." &&
              CardWords(blocked).find("Backend") == std::string::npos,
          "A blocked add-on update is explained in add-on words: " + blocked.explanation);
}

// The plan's base update has no pack of the add-on for the new base: the card
// warns that the Engine stops loading it, and offers Uninstall add-on.
void TestBaseUpdateLosesAddOn() {
    const auto records = BaseUpdateRecords(true, false);
    const auto card = MuJoCoCard(InstallerCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::Installed && card.can_uninstall &&
              card.installed_pack_id == kMuJoCoPack &&
              card.action_hint ==
                  "The CPU Engine update stops loading this add-on until a version for the new "
                  "Engine is published. You can uninstall it.",
          "The card warns that the base update stops the add-on loading: " + card.action_hint);

    // The installed version comes from the manifest kept at installation
    // when the catalog no longer lists the add-on.
    cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot base_only;
    base_only.records = {BaseRecord()};
    base_only.installed_manifests.push_back(*MuJoCoRecord().manifest);
    const auto unlisted = MuJoCoCard(InstallerCards(
        cyxwiz::BuildBackendPackCatalogRecords(base_only, ActiveBase(), ActiveMuJoCo()), {}));
    Check(unlisted.facts == "Version 1  |  Requires: CPU Engine (this release)" &&
              unlisted.licence ==
                  "Includes MuJoCo by Google DeepMind (Apache-2.0 licence) and Gymnasium task "
                  "models (MIT licence).",
          "An unlisted installed add-on keeps its version: " + unlisted.facts);
}

// The add-on records of a base update with the add-on not installed.
std::vector<cyxwiz::BackendPackManagerRecord> NotInstalled(
    std::vector<cyxwiz::BackendPackManagerRecord> records) {
    for (auto& record : records) {
        if (record.pack_kind != cyxwiz::BackendPackRecordKind::Plugin) continue;
        record.installed = false;
        record.active = false;
        record.installed_pack_id.clear();
        record.update_decision.reset();
    }
    return records;
}

// An add-on not installed during a base update: the plan refuses the old
// base's pack, so the card is locked and says it can be added after the update.
void TestBaseUpdateLocksAddOnNotInstalled() {
    for (const bool new_pack_listed : {true, false}) {
        const auto records = NotInstalled(BaseUpdateRecords(true, new_pack_listed));
        cyxwiz::BackendPackInstallerSelection old_pack;
        old_pack.valid = true;
        old_pack.pack_ids = {kMuJoCoPack};
        Check(!cyxwiz::BuildBackendPackInstallerPlan(old_pack, records,
                                                     cyxwiz::CyxWizInstallerMode::Maintenance)
                   .valid,
              "The plan refuses the old base's pack during a base update");
        const auto card = MuJoCoCard(InstallerCards(records, {}));
        Check(card.status == cyxwiz::AddOnCardStatus::AfterEngineUpdate &&
                  card.status_text == "After the Engine update" && !card.checked &&
                  !card.can_select && !card.can_install && !card.can_uninstall &&
                  card.installed_pack_id.empty(),
              "The card is locked during a base update: " + card.status_text);
        Check(card.explanation ==
                  (new_pack_listed
                       ? "This add-on can be added after the CPU Engine update. Install the "
                         "update first, then open the installer again to add it."
                       : "This add-on can be added after the CPU Engine update, once a version "
                         "for the new Engine is published."),
              "The card says it can be added after the update: " + card.explanation);
        Check(new_pack_listed
                  ? card.pack_id == kNewBasePack &&
                        card.facts == "Version 3  |  3.2 MiB  |  Requires: the CPU Engine update"
                  : card.facts.empty(),
              "The facts name the pack for the new Engine, or nothing: " + card.facts);
        Check(Resolve(records, kMuJoCoPlugin).empty() && Resolve(records, kMuJoCoPack).empty(),
              "--select ticks nothing during a base update");
    }
}

// The Engine's Installed packs: a CPU Engine update is only available there,
// so the cards say the Installer applies it; the Installer keeps its words.
void TestEngineManagerWording() {
    const auto kept = BaseUpdateRecords(true, true);
    const auto engine = MuJoCoCard(EngineCards(kept));
    Check(engine.status == cyxwiz::AddOnCardStatus::WithEngineUpdate &&
              engine.status_text == "CPU Engine update available" &&
              engine.explanation ==
                  "A CPU Engine update is available. The CyxWiz Installer reinstalls this "
                  "add-on with it." &&
              engine.can_uninstall && !engine.can_install,
          "The Engine says a CPU Engine update is available: " + engine.explanation);
    const auto engine_words = CardWords(engine);
    Check(engine_words.find("Comes with the Engine update") == std::string::npos &&
              engine_words.find("with this update") == std::string::npos,
          "No installer-time wording in the Engine: " + engine_words);
    const auto installer = MuJoCoCard(InstallerCards(kept, {}));
    Check(installer.status_text == "Comes with the Engine update" &&
              installer.explanation == "Reinstalled for the new CPU Engine with this update.",
          "The Installer keeps its wording: " + installer.explanation);

    const auto lost = MuJoCoCard(EngineCards(BaseUpdateRecords(true, false)));
    Check(lost.status == cyxwiz::AddOnCardStatus::Installed && lost.can_uninstall &&
              lost.action_hint ==
                  "A CPU Engine update is available. Once it is installed, the Engine stops "
                  "loading this add-on until a version for the new Engine is published. You "
                  "can uninstall it.",
          "The Engine warns what the available update does: " + lost.action_hint);

    const auto later = MuJoCoCard(EngineCards(NotInstalled(BaseUpdateRecords(true, true))));
    Check(later.status == cyxwiz::AddOnCardStatus::AfterEngineUpdate && !later.can_install &&
              later.explanation ==
                  "A CPU Engine update is available. This add-on can be added after it: update "
                  "the CPU Engine in the CyxWiz Installer first.",
          "The Engine says a new add-on comes after the update: " + later.explanation);
}

// Recovery and a fresh install never update the base: no base-update state.
void TestNoBaseUpdateStatesWithoutBaseUpdate() {
    cyxwiz::BackendPackInstallerSelection none;
    none.valid = true;
    for (const auto mode :
         {cyxwiz::CyxWizInstallerMode::RecoveryRequired, cyxwiz::CyxWizInstallerMode::FreshInstall}) {
        const auto kept = BaseUpdateRecords(true, true);
        Check(!cyxwiz::BuildBackendPackInstallerPlan(none, kept, mode).update_base &&
                  cyxwiz::FindBaseUpdate(kept, mode) == nullptr,
              "This mode's plan does not update the base");
        const auto card = MuJoCoCard(
            cyxwiz::BuildAddOnCards(kept, {}, cyxwiz::AddOnCardContext::Installer, mode));
        Check(card.status == cyxwiz::AddOnCardStatus::Installed && card.action_hint.empty() &&
                  CardWords(card).find("Engine update") == std::string::npos,
              "No base-update state without a base update: " + card.status_text);
        const auto lost = MuJoCoCard(cyxwiz::BuildAddOnCards(
            BaseUpdateRecords(true, false), {}, cyxwiz::AddOnCardContext::Installer, mode));
        Check(lost.action_hint.empty(), "No loss warning without a base update");
        const auto fresh = MuJoCoCard(cyxwiz::BuildAddOnCards(
            NotInstalled(BaseUpdateRecords(true, true)), {}, cyxwiz::AddOnCardContext::Installer,
            mode));
        Check(fresh.status != cyxwiz::AddOnCardStatus::AfterEngineUpdate,
              "No after-the-update lock without a base update");
    }
}

// The summary lists the add-ons a plan installs, with what happens to each.
void TestPlannedAddOnLines() {
    cyxwiz::BackendPackInstallerSelection none;
    none.valid = true;
    const auto kept = BaseUpdateRecords(true, true);
    const auto update_plan = cyxwiz::BuildBackendPackInstallerPlan(
        none, kept, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(update_plan.plugin_pack_ids == std::vector<std::string>{kNewBasePack} &&
              cyxwiz::PlannedAddOnLines(update_plan, kept) ==
                  std::vector<std::string>{
                      "Simulation (MuJoCo), reinstalled for the new CPU Engine"},
          "A base update lists the add-on it reinstalls");
    const auto lost = BaseUpdateRecords(true, false);
    Check(cyxwiz::PlannedAddOnLines(
              cyxwiz::BuildBackendPackInstallerPlan(none, lost,
                                                    cyxwiz::CyxWizInstallerMode::Maintenance),
              lost)
              .empty(),
          "An add-on with no pack for the new base is not listed (the warning counts it)");

    const auto fresh = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase());
    const auto fresh_plan = cyxwiz::BuildBackendPackInstallerPlan(
        cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                     fresh, {kMuJoCoPack}),
        fresh, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(cyxwiz::PlannedAddOnLines(fresh_plan, fresh) ==
              std::vector<std::string>{"Simulation (MuJoCo)"},
          "A new add-on is listed by name");

    const auto stale = cyxwiz::BuildBackendPackCatalogRecords(
        Catalog(), ActiveBase(),
        ActiveMuJoCo("cyxwiz-plugin-mujoco-0.1.0-1-win64-x86_64",
                     "cyxwiz-base-0.1.0-1-win64-x86_64"));
    const auto stale_plan = cyxwiz::BuildBackendPackInstallerPlan(
        cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                     stale, {kMuJoCoPack}),
        stale, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(cyxwiz::PlannedAddOnLines(stale_plan, stale) ==
              std::vector<std::string>{"Simulation (MuJoCo), reinstalled for this CPU Engine"},
          "A reinstall for this base says so");

    constexpr const char* kNext = "cyxwiz-plugin-mujoco-1.0.0-2-win64-x86_64";
    auto snapshot = Catalog();
    snapshot.records[1] = MuJoCoRecord(kNext, "2");
    snapshot.installed_manifests.push_back(*MuJoCoRecord().manifest);
    const auto updated =
        cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase(), ActiveMuJoCo());
    const auto updated_plan = cyxwiz::BuildBackendPackInstallerPlan(
        cyxwiz::ResolveBackendPackInstallerSelection(cyxwiz::BackendPackInstallChoice::Custom,
                                                     updated, {kNext}),
        updated, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(cyxwiz::PlannedAddOnLines(updated_plan, updated) ==
              std::vector<std::string>{"Simulation (MuJoCo), updated"},
          "An add-on update says so");

    // A refused plan still lists the ticked add-ons by name; a valid plan
    // lists its own lines.
    const auto refused_records = NotInstalled(BaseUpdateRecords(true, false));
    cyxwiz::BackendPackInstallerSelection ticked;
    ticked.valid = true;
    ticked.pack_ids = {kBase, kMuJoCoPack};
    const auto refused = cyxwiz::BuildBackendPackInstallerPlan(
        ticked, refused_records, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(!refused.valid &&
              cyxwiz::SummaryAddOnLines(refused, ticked.pack_ids, refused_records) ==
                  std::vector<std::string>{"Simulation (MuJoCo)"},
          "A refused plan's summary lists the ticked add-on by name");
    Check(cyxwiz::SummaryAddOnLines(update_plan, {}, kept) ==
              cyxwiz::PlannedAddOnLines(update_plan, kept),
          "A valid plan's summary lists the plan's add-on lines");
}

// An installed add-on whose files differ from its signed package (the check
// the Engine runs before it loads one): Damaged, with Repair (mockup of
// 2026-11-25: Engine Installed packs, Installer Add-ons).
void TestDamagedCard() {
    auto records =
        cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), ActiveMuJoCo());
    for (auto& record : records) {
        if (record.pack_id != kMuJoCoPack) continue;
        record.files_checked = true;
        record.files_listed = 15;
        record.files_checked_at = "14:02";
    }
    Check(MuJoCoCard(InstallerCards(records, {})).status == cyxwiz::AddOnCardStatus::Installed &&
              Detail(MuJoCoCard(EngineCards(records)), "Files") == "15 match the signed list" &&
              Detail(MuJoCoCard(EngineCards(records)), "Signed metadata") == "Verified",
          "An add-on whose files match stays Installed, its check under Details");

    const std::string dll =
        std::filesystem::path("bin/mujoco.dll").make_preferred().string();
    for (auto& record : records) {
        if (record.pack_id == kMuJoCoPack)
            record.file_issues = {{"bin/mujoco.dll", cyxwiz::plugin::PackFileProblem::ContentDiffers}};
    }
    const auto engine = MuJoCoCard(EngineCards(records));
    Check(engine.status == cyxwiz::AddOnCardStatus::Damaged && engine.status_text == "Damaged" &&
              engine.tone == cyxwiz::InstallerPackPresentationTone::Danger && engine.can_repair &&
              engine.can_uninstall && !engine.can_install,
          "The Engine's card reads Damaged with Repair add-on... and Uninstall add-on...");
    Check(engine.explanation ==
              dll + " does not match its signed package, so the Engine does not load it. "
                    "Repair puts the signed files back once the Engine is closed; your graphs and projects are "
                    "not touched.",
          "The Engine's wording: " + engine.explanation);
    Check(Detail(engine, "Files") == "1 of 15 differs: " + dll && Detail(engine, "Checked") == "at 14:02" &&
              Detail(engine, "Signed metadata") == "Verified",
          "Details say which file differs and when it was checked");

    const auto unticked = MuJoCoCard(InstallerCards(records, {}));
    Check(unticked.status == cyxwiz::AddOnCardStatus::Damaged && unticked.status_text == "Damaged" &&
              unticked.can_select && !unticked.checked,
          "Unticked, the Installer's card reads Damaged");
    const auto ticked = MuJoCoCard(InstallerCards(records, {kMuJoCoPack}));
    Check(ticked.status_text == "Repair selected" &&
              ticked.tone == cyxwiz::InstallerPackPresentationTone::Accent && ticked.checked &&
              ticked.explanation ==
                  dll + " does not match its signed package, so the Engine does not load this add-on. "
                        "Review & repair puts the signed files back.",
          "Ticked, it reads Repair selected");
    Check(Resolve(records, kMuJoCoPlugin) == kMuJoCoPack,
          "The Engine's Repair add-on... ticks the installed pack (--select)");

    const auto selection = cyxwiz::ResolveBackendPackInstallerSelection(
        cyxwiz::BackendPackInstallChoice::Custom, records, {kMuJoCoPack});
    const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
        selection, records, cyxwiz::CyxWizInstallerMode::Maintenance);
    Check(plan.valid && plan.plugin_pack_ids == std::vector<std::string>{kMuJoCoPack} &&
              plan.repair_plugin_pack_ids == std::vector<std::string>{kMuJoCoPack},
          "A ticked damaged add-on is repaired by the plan: " + plan.message);
    const auto lines = cyxwiz::PlannedAddOnLines(plan, records);
    Check(lines == std::vector<std::string>{"Simulation (MuJoCo), repaired"},
          "The summary says it is repaired");
}

}  // namespace

int main() {
    TestPluginManifestBecomesAnAddOnRecord();
    TestProfilesNeverIncludeAnAddOn();
    TestNotInstalledCard();
    TestTickedCardAndPlan();
    TestInstalledCard();
    TestUpdateAvailableCard();
    TestBuiltForAnotherEngineCard();
    TestNotPublishedAndUnavailableCards();
    TestUnknownAddOnGetsAGenericCard();
    TestAddOnWordsComeFromThePack();
    TestAddOnInstalledForAnotherBase();
    TestResolvePackToSelect();
    TestBaseUpdateReinstallsAddOn();
    TestBaseUpdateLosesAddOn();
    TestBaseUpdateLocksAddOnNotInstalled();
    TestEngineManagerWording();
    TestNoBaseUpdateStatesWithoutBaseUpdate();
    TestPlannedAddOnLines();
    TestDamagedCard();
    std::cout << "test_addon_pack_presentation: all checks passed\n";
    return 0;
}
