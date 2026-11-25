#include "addon_pack_presentation.h"

#include <algorithm>
#include <string_view>

namespace cyxwiz {
namespace {

// Third-party components named in a pack's signed licence list, in words.
// The pack's own licence and notice files stay in Details.
struct LicencePhrase {
    std::string_view component;
    std::string_view phrase;
};

constexpr LicencePhrase kLicencePhrases[] = {
    {"mujoco", "MuJoCo by Google DeepMind (Apache-2.0 licence)"},
    {"gymnasium", "Gymnasium task models (MIT licence)"},
};

std::string Join(const std::vector<std::string>& values,
                 std::string_view separator) {
    std::string result;
    for (const auto& value : values) {
        if (!result.empty()) result += separator;
        result += value;
    }
    return result;
}

std::string LicenceLine(const std::vector<std::string>& licences) {
    std::vector<std::string> phrases;
    for (const auto& phrase : kLicencePhrases) {
        if (std::find(licences.begin(), licences.end(), phrase.component) !=
            licences.end()) {
            phrases.emplace_back(phrase.phrase);
        }
    }
    if (!phrases.empty()) return "Includes " + Join(phrases, " and ") + ".";
    return licences.empty() ? std::string{}
                            : "Its licence files come with the add-on.";
}

bool IsAddOn(const BackendPackManagerRecord& record) {
    return record.pack_kind == BackendPackRecordKind::Plugin;
}

// The record a card shows when the catalog lists several packs of one add-on:
// the update, else the installed one, else the first one this installation
// can take, else the first deliverable one.
const BackendPackManagerRecord* PickRecord(
    const std::vector<const BackendPackManagerRecord*>& candidates) {
    if (candidates.empty()) return nullptr;
    for (const auto* record : candidates) {
        if (record->update_available) return record;
    }
    for (const auto* record : candidates) {
        if (record->installed && record->active) return record;
    }
    for (const auto* record : candidates) {
        if (IsBackendPackSelectableForInstaller(*record)) return record;
    }
    for (const auto* record : candidates) {
        if (record->delivery_metadata_available) return record;
    }
    return candidates.front();
}

void AddDetail(std::vector<AddOnDetail>& details, std::string key,
               std::string value) {
    if (!value.empty()) details.emplace_back(std::move(key), std::move(value));
}

// Version, size and the Engine it needs: this release, or the CPU Engine
// update the plan installs with it.
std::string Facts(const BackendPackManagerRecord& record, bool for_base_update) {
    std::string facts = "Version " + (record.package_version.empty()
                                          ? std::string("unavailable")
                                          : record.package_version);
    if (record.download_size_bytes > 0) {
        facts += "  |  " + FormatBackendPackByteSize(record.download_size_bytes);
    }
    if (!record.companion_base_id.empty()) {
        facts += for_base_update ? "  |  Requires: the CPU Engine update"
                                 : "  |  Requires: CPU Engine (this release)";
    }
    return facts;
}

bool BuiltForAnotherEngine(const BackendPackManagerRecord& record) {
    return record.compatibility &&
           record.compatibility->eligibility ==
               runtime::BackendPackEligibility::Incompatible &&
           (record.compatibility->rule ==
                runtime::BackendPackCompatibilityRule::CompanionBase ||
            record.compatibility->rule ==
                runtime::BackendPackCompatibilityRule::RuntimeSet);
}

// The installed add-on against the catalog's pack, in add-on words (the
// update policy's own messages speak of backend packs).
std::string AddOnUpdateWords(const BackendPackManagerRecord& record) {
    if (!record.update_decision) return {};
    using Disposition = runtime::BackendPackUpdateDisposition;
    switch (record.update_decision->disposition) {
        case Disposition::SamePackage:
            return "This version is installed.";
        case Disposition::Upgrade:
            return "A newer signed version is available.";
        case Disposition::Downgrade:
            return "The catalog offers an older version, so the installed "
                   "version stays.";
        case Disposition::Ambiguous:
        case Disposition::Unknown:
            break;
    }
    if (BuiltForAnotherEngine(record)) {
        return "The catalog's version is built for a different CyxWiz Engine "
               "release, so the installed version stays until the CPU Engine "
               "is updated.";
    }
    return "The catalog's version cannot be compared safely with the "
           "installed one, so the installed version stays.";
}

constexpr const char* kInstalledReason =
    "Installed. Use Uninstall add-on to remove it.";

void Status(AddOnCardView& card, AddOnCardStatus status, std::string text,
            InstallerPackPresentationTone tone) {
    card.status = status;
    card.status_text = std::move(text);
    card.tone = tone;
}

// An add-on installed for another CPU base: the Engine no longer loads it.
// The card offers this base's pack when the catalog has a deliverable one,
// and always Uninstall add-on for the old pack.
void BuildReinstallCard(AddOnCardView& card,
                        const BackendPackManagerRecord& record,
                        const BackendPackManagerRecord& stale,
                        const std::set<std::string>& selected_pack_ids) {
    card.can_uninstall = true;
    card.checked = false;
    card.can_select = false;
    card.can_install = false;
    card.action_hint.clear();
    const bool own_record = &record == &stale;
    if (!own_record) {
        AddDetail(card.details, "Installed for CPU Engine",
                  stale.companion_base_id);
    }
    const bool reinstallable = !own_record &&
                               IsBackendPackSelectableForInstaller(record) &&
                               !BuiltForAnotherEngine(record);
    if (!reinstallable) {
        Status(card, AddOnCardStatus::NeedsReinstall, "Needs reinstall",
               InstallerPackPresentationTone::Warning);
        card.pack_id.clear();
        card.facts.clear();
        card.explanation =
            "Installed for another CyxWiz Engine release, so the Engine no "
            "longer loads it. No version for this release is published yet; "
            "Uninstall add-on removes its files.";
        return;
    }
    const bool picked = selected_pack_ids.contains(record.pack_id);
    Status(card, AddOnCardStatus::NeedsReinstall,
           picked ? "Reinstall selected" : "Needs reinstall",
           picked ? InstallerPackPresentationTone::Accent
                  : InstallerPackPresentationTone::Warning);
    card.explanation =
        picked ? "Installed for another CyxWiz Engine release, so the Engine "
                 "no longer loads it. The version for this release installs "
                 "with the other changes."
               : "Installed for another CyxWiz Engine release, so the Engine "
                 "no longer loads it. Select it to install the version for "
                 "this release.";
    card.checked = picked;
    card.can_select = true;
    card.can_install = true;
}

// An installed add-on the CPU base update reinstalls (the catalog has its
// pack for the new base). In the Installer it goes with the plan's update,
// nothing to tick; the Engine only says the update is available.
void BuildBaseUpdateCard(AddOnCardView& card, AddOnCardContext context) {
    if (context == AddOnCardContext::Installer) {
        Status(card, AddOnCardStatus::WithEngineUpdate,
               "Comes with the Engine update",
               InstallerPackPresentationTone::Accent);
        card.explanation =
            "Reinstalled for the new CPU Engine with this update.";
    } else {
        Status(card, AddOnCardStatus::WithEngineUpdate,
               "CPU Engine update available",
               InstallerPackPresentationTone::Accent);
        card.explanation =
            "A CPU Engine update is available. The CyxWiz Installer "
            "reinstalls this add-on with it.";
    }
    card.checked = true;
    card.can_select = false;
    card.can_install = false;
    card.can_uninstall = true;
}

// An add-on not installed while a CPU base update is pending: the plan
// installs add-ons for the new base only after the update, so the card is
// locked. published: the catalog has its pack for the new base.
void BuildAfterBaseUpdateCard(AddOnCardView& card, AddOnCardContext context,
                              bool published) {
    Status(card, AddOnCardStatus::AfterEngineUpdate, "After the Engine update",
           InstallerPackPresentationTone::Neutral);
    if (!published) card.facts.clear();
    if (context == AddOnCardContext::Installer) {
        card.explanation =
            published ? "This add-on can be added after the CPU Engine "
                        "update. Install the update first, then open the "
                        "installer again to add it."
                      : "This add-on can be added after the CPU Engine "
                        "update, once a version for the new Engine is "
                        "published.";
    } else {
        card.explanation =
            published ? "A CPU Engine update is available. This add-on can "
                        "be added after it: update the CPU Engine in the "
                        "CyxWiz Installer first."
                      : "A CPU Engine update is available. This add-on can "
                        "be added after it, once a version for the new "
                        "Engine is published.";
    }
    card.checked = false;
    card.can_select = false;
    card.can_install = false;
}

void BuildInstalledCard(AddOnCardView& card) {
    Status(card, AddOnCardStatus::Installed, "Installed",
           InstallerPackPresentationTone::Success);
    card.explanation =
        "Loaded by the Engine at start. Turn it off for a while in "
        "Apps > Plugin Manager.";
    card.checked = true;
    card.select_disabled_reason = kInstalledReason;
    card.can_uninstall = true;
}

// base_update: the CPU base update the plan applies (the Installer) or that
// is available (the Engine), or nullptr.
AddOnCardView BuildCard(const std::string& plugin_id,
                        const BackendPackManagerRecord* record,
                        const BackendPackManagerRecord* stale,
                        const BackendPackManagerRecord* base_update,
                        const std::vector<BackendPackManagerRecord>& records,
                        const std::set<std::string>& selected_pack_ids,
                        AddOnCardContext context) {
    AddOnCardView card;
    card.key = plugin_id;
    const auto* identity = FindAddOnIdentity(plugin_id);
    card.title = AddOnTitle(plugin_id);
    card.description = identity
        ? std::string(identity->description)
        : "An optional add-on from the signed catalog.";
    AddDetail(card.details, "Plugin", plugin_id);

    if (!record) record = stale;
    if (!record) {
        Status(card, AddOnCardStatus::NotPublished, "Not published",
               InstallerPackPresentationTone::Neutral);
        card.explanation =
            "No signed " +
            (identity ? std::string(identity->short_name) : card.title) +
            " add-on is published for this platform and release.";
        return card;
    }

    // The installed pack, and the add-on's pack for the new base of a CPU
    // base update: the card then shows that one.
    const BackendPackManagerRecord* installed =
        stale ? stale : (record->installed ? record : nullptr);
    const BackendPackManagerRecord* next =
        base_update
            ? FindAddOnPackForBase(records, plugin_id, base_update->pack_id)
            : nullptr;
    if (next) record = next;

    card.pack_id = record->pack_id;
    if (installed) card.installed_pack_id = installed->installed_pack_id;
    card.facts = Facts(*record, next != nullptr);
    card.licence = LicenceLine(record->licenses);
    card.error = record->delivery_metadata_error;
    AddDetail(card.details, "Pack", record->pack_id);
    if (installed && installed->installed_pack_id != record->pack_id) {
        AddDetail(card.details, "Installed pack", installed->installed_pack_id);
    }
    AddDetail(card.details, "Version", record->package_version);
    if (record->download_size_bytes > 0) {
        AddDetail(card.details, "Download size",
                  FormatBackendPackByteSize(record->download_size_bytes));
    }
    AddDetail(card.details, "Built for CPU Engine", record->companion_base_id);
    AddDetail(card.details, "Signed catalog",
              BackendPackCatalogSupportName(record->catalog_support));
    AddDetail(card.details, "Licence files", Join(record->licenses, ", "));
    if (!next && record->installed) {
        AddDetail(card.details, "Update", AddOnUpdateWords(*record));
    }
    AddDetail(card.details, "Signed metadata", record->delivery_metadata_error);

    if (next && installed) {
        BuildBaseUpdateCard(card, context);
        return card;
    }
    // The plan refuses an add-on for the old base during a base update.
    if (base_update && !installed) {
        BuildAfterBaseUpdateCard(card, context, next != nullptr);
        return card;
    }
    if (stale) {
        BuildReinstallCard(card, *record, *stale, selected_pack_ids);
        return card;
    }
    // The base update has no pack of this add-on for the new base.
    if (base_update && installed) {
        BuildInstalledCard(card);
        card.action_hint =
            context == AddOnCardContext::Installer
                ? "The CPU Engine update stops loading this add-on until a "
                  "version for the new Engine is published. You can "
                  "uninstall it."
                : "A CPU Engine update is available. Once it is installed, "
                  "the Engine stops loading this add-on until a version for "
                  "the new Engine is published. You can uninstall it.";
        return card;
    }

    const bool selectable = IsBackendPackSelectableForInstaller(*record);
    const bool picked = selected_pack_ids.contains(record->pack_id);

    if (record->installed && record->update_decision &&
        record->update_decision->disposition !=
            runtime::BackendPackUpdateDisposition::Upgrade &&
        record->update_decision->disposition !=
            runtime::BackendPackUpdateDisposition::SamePackage) {
        Status(card, AddOnCardStatus::UpdateBlocked, "Update blocked",
               InstallerPackPresentationTone::Warning);
        card.explanation = AddOnUpdateWords(*record);
        card.checked = true;
        card.select_disabled_reason = kInstalledReason;
        card.can_uninstall = true;
        return card;
    }
    if (record->installed && record->update_available) {
        card.checked = picked;
        card.can_select = selectable;
        card.can_install = selectable;
        if (picked) {
            Status(card, AddOnCardStatus::UpdateAvailable, "Update selected",
                   InstallerPackPresentationTone::Accent);
            card.explanation =
                "Its signed update installs with the other changes; the "
                "Engine loads it at its next start.";
        } else {
            Status(card, AddOnCardStatus::UpdateAvailable, "Update available",
                   InstallerPackPresentationTone::Warning);
            card.action_hint =
                "Select this add-on to review and install its signed update.";
        }
        return card;
    }
    if (record->installed) {
        BuildInstalledCard(card);
        return card;
    }
    if (BuiltForAnotherEngine(*record)) {
        Status(card, AddOnCardStatus::NeedsEngineUpdate, "Needs Engine update",
               InstallerPackPresentationTone::Warning);
        card.facts.clear();
        card.explanation =
            "This add-on is built for a different CyxWiz Engine release. "
            "Update the CPU Engine first, then add it.";
        return card;
    }
    if (!selectable) {
        Status(card, AddOnCardStatus::Unavailable, "Unavailable",
               InstallerPackPresentationTone::Danger);
        if (!record->delivery_metadata_available) {
            card.explanation =
                "Its signed details could not be verified, so it cannot be "
                "installed.";
        } else if (record->catalog_support ==
                   BackendPackCatalogSupport::Diagnostic) {
            card.explanation =
                "This add-on is published for testing only, not for general "
                "use.";
        } else {
            card.explanation =
                "The current signed release does not offer this add-on.";
        }
        return card;
    }
    if (picked) {
        Status(card, AddOnCardStatus::Selected, "Selected",
               InstallerPackPresentationTone::Accent);
    } else {
        Status(card, AddOnCardStatus::Optional, "Optional",
               InstallerPackPresentationTone::Neutral);
    }
    card.explanation =
        "Ready to use when the Engine next starts; no device check is needed.";
    card.checked = picked;
    card.can_select = true;
    card.can_install = true;
    return card;
}

}  // namespace

std::vector<AddOnCardView> BuildAddOnCards(
    const std::vector<BackendPackManagerRecord>& records,
    const std::set<std::string>& selected_pack_ids, AddOnCardContext context,
    CyxWizInstallerMode mode) {
    std::vector<std::string> plugin_ids = KnownAddOnPluginIds();
    for (const auto& record : records) {
        if (IsAddOn(record) && !record.plugin_id.empty() &&
            std::find(plugin_ids.begin(), plugin_ids.end(), record.plugin_id) ==
                plugin_ids.end()) {
            plugin_ids.push_back(record.plugin_id);
        }
    }
    const auto* base_update = FindBaseUpdate(records, mode);
    std::vector<AddOnCardView> cards;
    cards.reserve(plugin_ids.size());
    for (const auto& plugin_id : plugin_ids) {
        // A pack installed for another base is kept apart: the card shows
        // this base's pack and offers to uninstall the old one.
        std::vector<const BackendPackManagerRecord*> candidates;
        const BackendPackManagerRecord* stale = nullptr;
        for (const auto& record : records) {
            if (!IsAddOn(record) || record.plugin_id != plugin_id) continue;
            if (record.installed_for_other_base) {
                stale = &record;
            } else {
                candidates.push_back(&record);
            }
        }
        cards.push_back(BuildCard(plugin_id, PickRecord(candidates), stale,
                                  base_update, records, selected_pack_ids,
                                  context));
    }
    return cards;
}

std::vector<std::string> PlannedAddOnLines(
    const BackendPackInstallerPlan& plan,
    const std::vector<BackendPackManagerRecord>& records) {
    std::vector<std::string> lines;
    for (const auto& pack_id : plan.plugin_pack_ids) {
        const auto record = std::find_if(
            records.begin(), records.end(), [&](const auto& candidate) {
                return IsAddOn(candidate) && candidate.pack_id == pack_id;
            });
        if (record == records.end()) {
            lines.push_back(pack_id);
            continue;
        }
        std::string line = AddOnTitle(record->plugin_id);
        const auto installed = std::find_if(
            records.begin(), records.end(), [&](const auto& candidate) {
                return IsAddOn(candidate) &&
                       candidate.plugin_id == record->plugin_id &&
                       candidate.installed;
            });
        if (installed != records.end()) {
            line += plan.update_base ? ", reinstalled for the new CPU Engine"
                    : installed->installed_for_other_base
                        ? ", reinstalled for this CPU Engine"
                        : ", updated";
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

std::vector<std::string> SummaryAddOnLines(
    const BackendPackInstallerPlan& plan,
    const std::vector<std::string>& selected_pack_ids,
    const std::vector<BackendPackManagerRecord>& records) {
    if (plan.valid) return PlannedAddOnLines(plan, records);
    std::vector<std::string> lines;
    for (const auto& pack_id : selected_pack_ids) {
        const auto record = std::find_if(
            records.begin(), records.end(), [&](const auto& candidate) {
                return IsAddOn(candidate) && candidate.pack_id == pack_id;
            });
        if (record != records.end()) {
            lines.push_back(AddOnTitle(record->plugin_id));
        }
    }
    return lines;
}

std::string ResolvePackToSelect(
    const std::vector<BackendPackManagerRecord>& records,
    const std::string& requested, CyxWizInstallerMode mode) {
    if (requested.empty()) return {};
    for (const auto& card : BuildAddOnCards(
             records, {}, AddOnCardContext::Installer, mode)) {
        if (card.key == requested ||
            (!card.pack_id.empty() && card.pack_id == requested)) {
            return card.can_select ? card.pack_id : std::string{};
        }
    }
    const auto record = std::find_if(
        records.begin(), records.end(), [&](const auto& candidate) {
            return !IsAddOn(candidate) && candidate.pack_id == requested;
        });
    if (record == records.end() ||
        !IsBackendPackSelectableForInstaller(*record) ||
        (record->installed && !record->update_available)) {
        return {};
    }
    return record->pack_id;
}

PackRemovalWording BuildPackRemovalWording(
    const std::vector<BackendPackManagerRecord>& records,
    const std::string& pack_id) {
    const auto add_on = std::find_if(
        records.begin(), records.end(), [&](const auto& record) {
            return IsAddOn(record) && (record.installed_pack_id == pack_id ||
                                       record.pack_id == pack_id);
        });
    if (add_on != records.end()) {
        return {"Uninstall add-on",
                "CyxWiz removes this add-on's files and its downloaded "
                "package. The Engine stops loading it at its next start; your "
                "projects are not changed.",
                "Add-on: " + AddOnTitle(add_on->plugin_id),
                "Uninstall add-on", "Keep add-on"};
    }
    return {"Uninstall backend component",
            "CyxWiz will deactivate this compute route, remove its versioned "
            "package files, and clear its downloaded package cache.",
            "Component: " + pack_id, "Uninstall", "Keep component"};
}

}  // namespace cyxwiz
