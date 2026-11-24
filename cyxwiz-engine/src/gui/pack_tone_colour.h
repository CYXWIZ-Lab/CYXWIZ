#pragma once

// The colour of a pack card's status, compute or add-on, from the theme
// tokens: one mapping for the CyxWiz Installer and the Engine's Backend
// Manager, so every card on both screens agrees.

#include "../core/installer_pack_presentation.h"
#include "ui_tokens.h"

namespace cyxwiz::ui {

inline ImVec4 PackToneColour(InstallerPackPresentationTone tone,
                              const Tokens& tokens = CurrentTokens()) {
    switch (tone) {
        case InstallerPackPresentationTone::Success: return tokens.success;
        case InstallerPackPresentationTone::Warning: return tokens.warning;
        case InstallerPackPresentationTone::Danger: return tokens.error;
        case InstallerPackPresentationTone::Accent: return tokens.accent_text;
        case InstallerPackPresentationTone::Neutral: return tokens.text_dim;
    }
    return tokens.text_dim;
}

}  // namespace cyxwiz::ui
