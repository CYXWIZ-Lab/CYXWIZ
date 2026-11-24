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
    const auto cards = cyxwiz::BuildAddOnCards(records, {});
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
    const auto card = MuJoCoCard(cyxwiz::BuildAddOnCards(records, ticked));
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
    const auto card = MuJoCoCard(cyxwiz::BuildAddOnCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::Installed && card.status_text == "Installed" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Success && card.checked &&
              !card.can_select && card.can_uninstall && !card.can_install &&
              card.installed_pack_id == kMuJoCoPack,
          "An installed add-on offers Uninstall add-on");
    Check(card.explanation ==
              "Loaded by the Engine at start. Turn it off for a while in Apps > Plugin Manager.",
          "Installed wording");
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
    const auto unlisted_card = MuJoCoCard(cyxwiz::BuildAddOnCards(unlisted, {}));
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
    const auto card = MuJoCoCard(cyxwiz::BuildAddOnCards(records, {}));
    Check(card.status == cyxwiz::AddOnCardStatus::UpdateAvailable &&
              card.status_text == "Update available" &&
              card.tone == cyxwiz::InstallerPackPresentationTone::Warning && card.can_select &&
              !card.checked && !card.can_uninstall &&
              card.action_hint == "Select this add-on to review and install its signed update." &&
              card.facts.rfind("Version 2", 0) == 0,
          "Update available asks to select the update");
    const auto ticked = MuJoCoCard(cyxwiz::BuildAddOnCards(records, {kNext}));
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
    const auto blocked_card = MuJoCoCard(cyxwiz::BuildAddOnCards(blocked, {}));
    Check(blocked_card.status == cyxwiz::AddOnCardStatus::UpdateBlocked &&
              !blocked_card.can_select && blocked_card.can_uninstall,
          "An older catalog package never replaces the installed add-on");
}

void TestBuiltForAnotherEngineCard() {
    auto snapshot = Catalog();
    snapshot.records[1] = MuJoCoRecord(kMuJoCoPack, "1", "cyxwiz-base-0.2.0-2-win64-x86_64");
    const auto records = cyxwiz::BuildBackendPackCatalogRecords(snapshot, ActiveBase());
    const auto card = MuJoCoCard(cyxwiz::BuildAddOnCards(records, {}));
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
        MuJoCoCard(cyxwiz::BuildAddOnCards(cyxwiz::BuildBackendPackCatalogRecords(base_only, {}), {}));
    Check(missing.status == cyxwiz::AddOnCardStatus::NotPublished &&
              missing.status_text == "Not published" && !missing.can_select &&
              missing.pack_id.empty() &&
              missing.explanation ==
                  "No signed MuJoCo add-on is published for this platform and release.",
          "No pack for this platform: Not published");

    auto other_platform = Catalog();
    other_platform.records[1].manifest->platform =
        cyxwiz::runtime::CurrentBackendPackPlatformId() == std::string_view("win64") ? "linux64"
                                                                                      : "win64";
    Check(MuJoCoCard(cyxwiz::BuildAddOnCards(
                         cyxwiz::BuildBackendPackCatalogRecords(other_platform, {}), {}))
                  .status == cyxwiz::AddOnCardStatus::NotPublished,
          "Another platform's pack does not count as published here");

    auto diagnostic = Catalog();
    diagnostic.records[1].catalog_entry.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Diagnostic;
    diagnostic.records[1].manifest->compatibility.support_status =
        cyxwiz::runtime::BackendPackSupportStatus::Diagnostic;
    const auto diagnostic_records = cyxwiz::BuildBackendPackCatalogRecords(diagnostic, {});
    const auto diagnostic_card = MuJoCoCard(cyxwiz::BuildAddOnCards(diagnostic_records, {}));
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
        cyxwiz::BuildAddOnCards(cyxwiz::BuildBackendPackCatalogRecords(snapshot, {}), {});
    Check(cards.size() == 2 && cards[0].key == kMuJoCoPlugin && cards[1].key == "com.example.other" &&
              cards[1].title == "com.example.other" &&
              cards[1].licence == "Its licence files come with the add-on." &&
              cards[1].status == cyxwiz::AddOnCardStatus::Optional,
          "Known add-ons come first; another plugin pack still gets a card");
    Check(cyxwiz::AddOnTitle(kMuJoCoPlugin) == "Simulation (MuJoCo)" &&
              cyxwiz::KnownAddOnPluginIds() == std::vector<std::string>{kMuJoCoPlugin},
          "Shared add-on names");
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
    const auto lost_card = MuJoCoCard(cyxwiz::BuildAddOnCards(lost, {}));
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
    Check(cyxwiz::ResolvePackToSelect(lost, kMuJoCoPlugin).empty(),
          "Nothing is ticked for it");

    const auto offered = cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), old_list);
    const auto offered_card = MuJoCoCard(cyxwiz::BuildAddOnCards(offered, {}));
    Check(offered_card.status == cyxwiz::AddOnCardStatus::NeedsReinstall &&
              offered_card.status_text == "Needs reinstall" && offered_card.can_select &&
              offered_card.can_install && offered_card.can_uninstall &&
              offered_card.pack_id == kMuJoCoPack && offered_card.installed_pack_id == kOldPack &&
              Detail(offered_card, "Installed pack") == kOldPack &&
              Detail(offered_card, "Installed for CPU Engine") == kOldBase,
          "With a pack for this base the card offers to reinstall it");
    const auto ticked = MuJoCoCard(cyxwiz::BuildAddOnCards(offered, {kMuJoCoPack}));
    Check(ticked.checked && ticked.status_text == "Reinstall selected" &&
              ticked.tone == cyxwiz::InstallerPackPresentationTone::Accent,
          "A ticked reinstall says so");
    Check(cyxwiz::ResolvePackToSelect(offered, kMuJoCoPlugin) == kMuJoCoPack,
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
    Check(cyxwiz::ResolvePackToSelect(fresh, kMuJoCoPlugin) == kMuJoCoPack &&
              cyxwiz::ResolvePackToSelect(fresh, kMuJoCoPack) == kMuJoCoPack,
          "An add-on not installed is ticked by plugin id or pack id");
    const auto installed =
        cyxwiz::BuildBackendPackCatalogRecords(Catalog(), ActiveBase(), ActiveMuJoCo());
    Check(cyxwiz::ResolvePackToSelect(installed, kMuJoCoPlugin).empty(),
          "An installed add-on is not ticked again");

    // Two packs of the add-on: the card's pack (the one this base takes) is
    // ticked, never just the first one listed.
    auto two = Catalog();
    two.records.insert(two.records.begin() + 1,
                       MuJoCoRecord("cyxwiz-plugin-mujoco-1.0.0-0-win64-x86_64", "1",
                                    "cyxwiz-base-0.1.0-1-win64-x86_64"));
    const auto two_records = cyxwiz::BuildBackendPackCatalogRecords(two, ActiveBase());
    Check(MuJoCoCard(cyxwiz::BuildAddOnCards(two_records, {})).pack_id == kMuJoCoPack &&
              cyxwiz::ResolvePackToSelect(two_records, kMuJoCoPlugin) == kMuJoCoPack,
          "The add-on is ticked through its card's pack");
    Check(cyxwiz::ResolvePackToSelect(two_records, kBase).empty() &&
              cyxwiz::ResolvePackToSelect(two_records, "missing").empty() &&
              cyxwiz::ResolvePackToSelect(two_records, "").empty(),
          "The CPU base and unknown ids are never ticked");
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
    TestAddOnInstalledForAnotherBase();
    TestResolvePackToSelect();
    std::cout << "test_addon_pack_presentation: all checks passed\n";
    return 0;
}
