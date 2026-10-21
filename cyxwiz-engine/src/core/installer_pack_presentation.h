#pragma once

#include "backend_pack_manager_model.h"

#include <string>

namespace cyxwiz {

enum class InstallerPackPresentationTone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger
};

struct InstallerPackPresentation {
    std::string status;
    std::string explanation;
    std::string action;
    InstallerPackPresentationTone tone =
        InstallerPackPresentationTone::Neutral;
};

InstallerPackPresentation BuildInstallerPackPresentation(
    const BackendPackManagerRecord& record);

}  // namespace cyxwiz
