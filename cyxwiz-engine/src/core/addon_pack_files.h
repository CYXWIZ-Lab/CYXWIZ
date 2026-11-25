#pragma once

// The installed add-ons' files against their signed inventory, for the
// add-on cards (Damaged, Repair) of the CyxWiz Installer and the Engine's
// Installed packs: the same check the Engine runs before it loads an add-on
// (redist/bootstrapper/backend_pack_installed_files.h).

#include "backend_pack_manager_model.h"

#include <filesystem>
#include <vector>

namespace cyxwiz::runtime {
class BackendPackMetadataVerifier;
}

namespace cyxwiz {

// For every installed add-on of the active base: its signed installed
// metadata and every file of its folder. Fills the records' files_* fields
// (and integrity_verified when they match); records of the same add-on share
// one check.
void CheckInstalledAddOnFiles(std::vector<BackendPackManagerRecord>& records,
                              const std::filesystem::path& runtime_root,
                              const runtime::BackendPackMetadataVerifier& verifier);

}  // namespace cyxwiz
