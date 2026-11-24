#pragma once

// One presentation of add-ons (plugin packs: optional features such as
// Simulation (MuJoCo) that need no device check) for every screen that
// offers them: CyxWiz Installer > Individual components and Engine >
// Preferences > Installed packs. Pure data in, data out: no ImGui, no global
// state, no backend DLL calls (the installer compiles this file too).

#include "backend_pack_manager_model.h"
#include "installer_pack_presentation.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class AddOnCardStatus {
    Optional,           // offered, not ticked
    Selected,           // ticked for installation
    Installed,
    UpdateAvailable,
    UpdateBlocked,
    NeedsEngineUpdate,  // built for another CyxWiz Engine release
    NeedsReinstall,     // installed for another CPU base: no longer loaded
    Unavailable,        // not offered by the signed catalog, or unverifiable
    NotPublished        // a known add-on with no pack for this platform
};

using AddOnDetail = std::pair<std::string, std::string>;

struct AddOnCardView {
    std::string key;                 // plugin id: stable card identity
    std::string pack_id;             // catalog pack to install; empty when not published
    std::string installed_pack_id;   // pack to uninstall; empty when not installed
    std::string title;               // "Simulation (MuJoCo)"
    std::string description;
    std::string facts;               // "Version 1  |  3.2 MiB  |  Requires: ..."
    std::string explanation;         // what happens next, in words
    std::string licence;             // third-party licences in the pack
    std::string action_hint;         // a warning line (an update to select)
    std::string error;               // why signed metadata could not be read
    AddOnCardStatus status = AddOnCardStatus::Optional;
    std::string status_text;
    InstallerPackPresentationTone tone = InstallerPackPresentationTone::Neutral;
    bool checked = false;            // the card's checkbox
    bool can_select = false;         // the checkbox can change
    std::string select_disabled_reason;
    bool can_install = false;        // Engine: Install add-on... (opens the installer)
    bool can_uninstall = false;      // Uninstall add-on
    std::vector<AddOnDetail> details;  // every fact, empty values skipped
};

// The add-ons this release knows by plugin id, in card order. A known
// add-on with no catalog pack for this platform still gets a Not published
// card, as the compute backends do.
const std::vector<std::string>& KnownAddOnPluginIds();

// Display name of an add-on: its own wording for a known plugin id, else the
// plugin id itself.
std::string AddOnTitle(const std::string& plugin_id);

// Cards for the known add-ons (in order), then for any other plugin pack the
// catalog or the installation holds. selected_pack_ids are the packs ticked
// for installation.
std::vector<AddOnCardView> BuildAddOnCards(
    const std::vector<BackendPackManagerRecord>& records,
    const std::set<std::string>& selected_pack_ids);

// The pack the installer ticks for `--select <id>`: an add-on named by its
// plugin id or pack id follows its card (the pack the card offers, when its
// checkbox can change); any other id names a compute pack that can be
// installed or updated. Empty when nothing can be ticked.
std::string ResolvePackToSelect(
    const std::vector<BackendPackManagerRecord>& records,
    const std::string& requested);

// The words of the uninstall confirmation for one pack: an add-on's own
// wording, or the compute component's.
struct PackRemovalWording {
    std::string title;
    std::string body;
    std::string subject;   // "Add-on: Simulation (MuJoCo)" or "Component: <pack id>"
    std::string confirm;
    std::string keep;
};

PackRemovalWording BuildPackRemovalWording(
    const std::vector<BackendPackManagerRecord>& records,
    const std::string& pack_id);

}  // namespace cyxwiz
