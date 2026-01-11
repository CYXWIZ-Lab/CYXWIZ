#include "trust_badge.h"
#include "../icons.h"
#include <imgui.h>

namespace gui {

// ============================================================================
// TrustBadge
// ============================================================================

ImVec4 TrustBadge::GetColor(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self:
            return ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
        case network::TrustLevel::Signed:
            return ImVec4(0.2f, 0.6f, 0.9f, 1.0f);  // Blue
        case network::TrustLevel::Verified:
            return ImVec4(0.4f, 0.8f, 0.4f, 1.0f);  // Light green
        case network::TrustLevel::Attested:
            return ImVec4(0.9f, 0.7f, 0.2f, 1.0f);  // Gold
        case network::TrustLevel::Untrusted:
            return ImVec4(0.9f, 0.3f, 0.3f, 1.0f);  // Red
        default:
            return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray
    }
}

const char* TrustBadge::GetIcon(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self:
            return ICON_FA_USER_CHECK;
        case network::TrustLevel::Signed:
            return ICON_FA_SIGNATURE;
        case network::TrustLevel::Verified:
            return ICON_FA_CIRCLE_CHECK;
        case network::TrustLevel::Attested:
            return ICON_FA_SHIELD_HALVED;
        case network::TrustLevel::Untrusted:
            return ICON_FA_TRIANGLE_EXCLAMATION;
        default:
            return ICON_FA_QUESTION;
    }
}

const char* TrustBadge::GetName(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self: return "Self";
        case network::TrustLevel::Signed: return "Signed";
        case network::TrustLevel::Verified: return "Verified";
        case network::TrustLevel::Attested: return "Attested";
        case network::TrustLevel::Untrusted: return "Untrusted";
        default: return "Unknown";
    }
}

const char* TrustBadge::GetDescription(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self:
            return "Your own upload - fully trusted";
        case network::TrustLevel::Signed:
            return "Cryptographically signed by a trusted party";
        case network::TrustLevel::Verified:
            return "Hash verified against a known good source";
        case network::TrustLevel::Attested:
            return "Hardware attestation (TEE/SGX)";
        case network::TrustLevel::Untrusted:
            return "Unknown source - verification recommended";
        default:
            return "Unknown trust level";
    }
}

void TrustBadge::Render(network::TrustLevel level, Style style) {
    ImVec4 color = GetColor(level);
    const char* icon = GetIcon(level);
    const char* name = GetName(level);

    ImGui::PushStyleColor(ImGuiCol_Text, color);

    switch (style) {
        case Style::Full:
            ImGui::Text("%s %s", icon, name);
            break;
        case Style::Compact:
            ImGui::Text("%s %s", icon, name);
            break;
        case Style::IconOnly:
            ImGui::Text("%s", icon);
            break;
        case Style::TextOnly:
            ImGui::Text("%s", name);
            break;
    }

    ImGui::PopStyleColor();

    // Show tooltip on hover
    if (ImGui::IsItemHovered()) {
        RenderTooltip(level);
    }
}

void TrustBadge::RenderSized(network::TrustLevel level, float size, Style style) {
    // Scale font for the badge
    float old_scale = ImGui::GetFont()->Scale;
    ImGui::GetFont()->Scale *= (size / ImGui::GetFontSize());
    ImGui::PushFont(ImGui::GetFont());

    Render(level, style);

    ImGui::GetFont()->Scale = old_scale;
    ImGui::PopFont();
}

void TrustBadge::RenderTooltip(network::TrustLevel level) {
    ImGui::BeginTooltip();

    // Header with icon and name
    ImVec4 color = GetColor(level);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s %s", GetIcon(level), GetName(level));
    ImGui::PopStyleColor();

    ImGui::Separator();

    // Description
    ImGui::TextWrapped("%s", GetDescription(level));

    // Trust hierarchy visualization
    ImGui::Separator();
    ImGui::Text("Trust Hierarchy:");

    const network::TrustLevel levels[] = {
        network::TrustLevel::Self,
        network::TrustLevel::Signed,
        network::TrustLevel::Verified,
        network::TrustLevel::Attested,
        network::TrustLevel::Untrusted
    };

    for (auto l : levels) {
        ImVec4 c = (l <= level) ? GetColor(l) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, c);
        ImGui::Text("%s %s %s", GetIcon(l), GetName(l),
                    (l == level) ? " <--" : "");
        ImGui::PopStyleColor();
    }

    ImGui::EndTooltip();
}

bool TrustBadge::RenderSelector(const char* label, network::TrustLevel& current_level,
                                 bool include_any) {
    const char* current_name = GetName(current_level);
    bool changed = false;

    if (ImGui::BeginCombo(label, current_name)) {
        const network::TrustLevel levels[] = {
            network::TrustLevel::Self,
            network::TrustLevel::Signed,
            network::TrustLevel::Verified,
            network::TrustLevel::Attested,
            network::TrustLevel::Untrusted
        };

        for (auto level : levels) {
            if (!include_any && level == network::TrustLevel::Untrusted) continue;

            bool is_selected = (level == current_level);
            ImVec4 color = GetColor(level);
            ImGui::PushStyleColor(ImGuiCol_Text, color);

            char item_text[64];
            snprintf(item_text, sizeof(item_text), "%s %s", GetIcon(level), GetName(level));

            if (ImGui::Selectable(item_text, is_selected)) {
                current_level = level;
                changed = true;
            }

            ImGui::PopStyleColor();

            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return changed;
}

void TrustBadge::RenderVerificationProgress(network::TrustLevel target_level,
                                             network::TrustLevel current_level,
                                             float verification_progress) {
    ImGui::BeginGroup();

    // Current level
    ImGui::Text("Current: ");
    ImGui::SameLine();
    Render(current_level, Style::Compact);

    // Progress bar
    ImGui::Text("Verifying: ");
    ImGui::SameLine();
    ImGui::ProgressBar(verification_progress, ImVec2(100, 0));

    // Target level
    ImGui::Text("Target: ");
    ImGui::SameLine();
    Render(target_level, Style::Compact);

    // Status message
    if (verification_progress >= 1.0f) {
        if (current_level <= target_level) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                ICON_FA_CHECK " Verification successful");
        } else {
            ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                ICON_FA_XMARK " Verification failed");
        }
    } else if (verification_progress > 0) {
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
            ICON_FA_SPINNER " Verification in progress...");
    }

    ImGui::EndGroup();
}

// ============================================================================
// TrustLevelFilter
// ============================================================================

TrustLevelFilter::TrustLevelFilter() {
    EnableAll();
}

void TrustLevelFilter::Render() {
    if (minimum_mode_) {
        // Minimum level mode
        ImGui::Text("Minimum Trust Level:");
        TrustBadge::RenderSelector("##min_trust", minimum_level_, true);
    } else {
        // Multi-select mode
        ImGui::Text("Trust Levels:");

        const network::TrustLevel levels[] = {
            network::TrustLevel::Self,
            network::TrustLevel::Signed,
            network::TrustLevel::Verified,
            network::TrustLevel::Attested,
            network::TrustLevel::Untrusted
        };

        for (size_t i = 0; i < 5; i++) {
            auto level = levels[i];
            ImVec4 color = TrustBadge::GetColor(level);

            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::PushStyleColor(ImGuiCol_CheckMark, color);

            char label[64];
            snprintf(label, sizeof(label), "%s %s",
                     TrustBadge::GetIcon(level), TrustBadge::GetName(level));

            ImGui::Checkbox(label, &enabled_[i]);

            ImGui::PopStyleColor(2);

            if (i < 4) ImGui::SameLine();
        }

        // Quick actions
        if (ImGui::SmallButton("All")) EnableAll();
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) DisableAll();
        ImGui::SameLine();
        if (ImGui::SmallButton("Trusted Only")) {
            DisableAll();
            enabled_[0] = true;  // Self
            enabled_[1] = true;  // Signed
            enabled_[2] = true;  // Verified
        }
    }
}

bool TrustLevelFilter::Passes(network::TrustLevel level) const {
    if (minimum_mode_) {
        return level <= minimum_level_;
    }

    int index = static_cast<int>(level);
    if (index < 0 || index >= 5) return false;
    return enabled_[index];
}

bool TrustLevelFilter::IsEnabled(network::TrustLevel level) const {
    int index = static_cast<int>(level);
    if (index < 0 || index >= 5) return false;
    return enabled_[index];
}

void TrustLevelFilter::SetEnabled(network::TrustLevel level, bool enabled) {
    int index = static_cast<int>(level);
    if (index >= 0 && index < 5) {
        enabled_[index] = enabled;
    }
}

void TrustLevelFilter::EnableAll() {
    for (int i = 0; i < 5; i++) enabled_[i] = true;
}

void TrustLevelFilter::DisableAll() {
    for (int i = 0; i < 5; i++) enabled_[i] = false;
}

// ============================================================================
// TrustRequirementBanner
// ============================================================================

bool TrustRequirementBanner::RenderIfNeeded(network::TrustLevel actual_level,
                                             network::TrustLevel required_level) {
    if (actual_level <= required_level) {
        return false;  // Requirement met
    }

    // Show warning banner
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.5f, 0.2f, 0.1f, 0.3f));

    if (ImGui::BeginChild("##trust_warning", ImVec2(0, 60), true)) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " Trust Level Warning");

        ImGui::Text("This dataset is ");
        ImGui::SameLine();
        TrustBadge::Render(actual_level, TrustBadge::Style::Compact);
        ImGui::SameLine();
        ImGui::Text(" but minimum ");
        ImGui::SameLine();
        TrustBadge::Render(required_level, TrustBadge::Style::Compact);
        ImGui::SameLine();
        ImGui::Text(" is required.");

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);
        if (ImGui::SmallButton("Verify Now")) {
            // TODO: Trigger verification
        }
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();

    return true;
}

void TrustRequirementBanner::RenderRequirementInfo(network::TrustLevel required_level) {
    ImGui::BeginGroup();

    ImGui::Text("Required Trust Level: ");
    ImGui::SameLine();
    TrustBadge::Render(required_level, TrustBadge::Style::Full);

    ImGui::TextWrapped("Datasets must meet this trust level or better to be used for training.");

    // Show what's acceptable
    ImGui::Separator();
    ImGui::Text("Acceptable levels:");

    const network::TrustLevel levels[] = {
        network::TrustLevel::Self,
        network::TrustLevel::Signed,
        network::TrustLevel::Verified,
        network::TrustLevel::Attested,
        network::TrustLevel::Untrusted
    };

    for (auto level : levels) {
        bool acceptable = (level <= required_level);
        ImGui::Text("  ");
        ImGui::SameLine();

        if (acceptable) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_CHECK);
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_XMARK);
        }

        ImGui::SameLine();
        TrustBadge::Render(level, TrustBadge::Style::Compact);
    }

    ImGui::EndGroup();
}

} // namespace gui
