#pragma once

// The installed add-ons' files against their signed inventory, for the
// add-on cards (Damaged, Repair) of the CyxWiz Installer and the Engine's
// Installed packs: the same check the Engine runs before it loads an add-on
// (redist/bootstrapper/backend_pack_installed_files.h), with the same trust
// file (the installation's runtime/trust/trusted-keys.json), so a card and
// the Engine's load always agree.

#include "backend_pack_manager_model.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {

// For every installed add-on of the active base: its signed installed
// metadata and every file of its folder. Fills the records' files_* fields
// (and integrity_verified when they match); records of the same add-on share
// one check. When the trust file cannot be read the add-ons are not called
// damaged: their files_check_note says why they were not checked.
// Reads and hashes every installed add-on: run it off the UI thread.
void CheckInstalledAddOnFiles(std::vector<BackendPackManagerRecord>& records,
                              const std::filesystem::path& runtime_root,
                              const std::string& client_version);

// The installed add-ons whose files CheckInstalledAddOnFiles will check (their
// cards say "Checking..." meanwhile).
void MarkAddOnFileChecksPending(std::vector<BackendPackManagerRecord>& records);

// Copies the file checks of `checked` (a copy CheckInstalledAddOnFiles filled)
// into the matching records of `records`.
void MergeAddOnFileChecks(std::vector<BackendPackManagerRecord>& records,
                          const std::vector<BackendPackManagerRecord>& checked);

}  // namespace cyxwiz
