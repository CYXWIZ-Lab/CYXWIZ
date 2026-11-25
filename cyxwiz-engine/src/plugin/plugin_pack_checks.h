#pragma once

// The pack service's file check (redist backend_pack_installed_files.h) in the
// plugin host's words (plugin_pack.h): one translation for the Engine's load
// check (plugin_pack_trust.cpp) and the add-on cards (core/addon_pack_files.cpp).
// Header-only: the CyxWiz Installer uses it too.

#include "plugin_pack.h"

#include "backend_pack_installed_files.h"

#include <vector>

namespace cyxwiz::plugin {

inline PackFileProblem PackFileProblemOf(runtime::PackFileFault fault) {
    switch (fault) {
        case runtime::PackFileFault::Missing: return PackFileProblem::Missing;
        case runtime::PackFileFault::SizeDiffers: return PackFileProblem::SizeDiffers;
        case runtime::PackFileFault::Unreadable: return PackFileProblem::Unreadable;
        case runtime::PackFileFault::NotListed: return PackFileProblem::NotListed;
        case runtime::PackFileFault::ContentDiffers:
        case runtime::PackFileFault::None: break;
    }
    return PackFileProblem::ContentDiffers;
}

// The check's findings as file issues, and its error when it found no file
// to name (a folder it could not list, a link).
inline PackFolderCheck PackFolderCheckOf(const runtime::PackFilesCheck& check) {
    PackFolderCheck result;
    result.files_checked = check.files_checked;
    result.bytes_checked = check.bytes_checked;
    for (const auto& finding : check.findings)
        result.issues.push_back({finding.relative_path, PackFileProblemOf(finding.fault)});
    if (result.issues.empty()) result.error = check.error;
    return result;
}

}  // namespace cyxwiz::plugin
