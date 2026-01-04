#pragma once

#include "../../network/datastream_client.h"
#include <imgui.h>
#include <string>

namespace gui {

/**
 * Trust Badge Widget
 *
 * Reusable UI component for displaying dataset trust levels.
 * Provides consistent visualization across the application.
 */
class TrustBadge {
public:
    // Badge display styles
    enum class Style {
        Full,      // Icon + text + color
        Compact,   // Icon + text
        IconOnly,  // Just icon
        TextOnly   // Just text
    };

    // Render a trust badge
    static void Render(network::TrustLevel level, Style style = Style::Full);

    // Render with custom size
    static void RenderSized(network::TrustLevel level, float size, Style style = Style::Full);

    // Render trust level selector dropdown
    // Returns true if selection changed
    static bool RenderSelector(const char* label, network::TrustLevel& current_level,
                               bool include_any = true);

    // Render trust level progress bar (for verification progress)
    static void RenderVerificationProgress(network::TrustLevel target_level,
                                            network::TrustLevel current_level,
                                            float verification_progress = 0.0f);

    // Get display properties
    static ImVec4 GetColor(network::TrustLevel level);
    static const char* GetIcon(network::TrustLevel level);
    static const char* GetName(network::TrustLevel level);
    static const char* GetDescription(network::TrustLevel level);

private:
    static void RenderTooltip(network::TrustLevel level);
};

/**
 * Trust Level Filter Widget
 *
 * Multi-select filter for trust levels.
 */
class TrustLevelFilter {
public:
    TrustLevelFilter();

    // Render the filter widget
    void Render();

    // Check if a trust level passes the filter
    bool Passes(network::TrustLevel level) const;

    // Get/Set filter state
    bool IsEnabled(network::TrustLevel level) const;
    void SetEnabled(network::TrustLevel level, bool enabled);
    void EnableAll();
    void DisableAll();

    // Minimum trust level mode (alternative to multi-select)
    void SetMinimumMode(bool enabled) { minimum_mode_ = enabled; }
    bool IsMinimumMode() const { return minimum_mode_; }
    void SetMinimumLevel(network::TrustLevel level) { minimum_level_ = level; }
    network::TrustLevel GetMinimumLevel() const { return minimum_level_; }

private:
    bool enabled_[5] = {true, true, true, true, true};  // One per trust level
    bool minimum_mode_ = false;
    network::TrustLevel minimum_level_ = network::TrustLevel::Verified;
};

/**
 * Trust Requirement Banner
 *
 * Shows a warning banner if data doesn't meet trust requirements.
 */
class TrustRequirementBanner {
public:
    // Render warning banner if trust level is insufficient
    // Returns true if banner was shown
    static bool RenderIfNeeded(network::TrustLevel actual_level,
                                network::TrustLevel required_level);

    // Render info about trust requirements
    static void RenderRequirementInfo(network::TrustLevel required_level);
};

} // namespace gui
