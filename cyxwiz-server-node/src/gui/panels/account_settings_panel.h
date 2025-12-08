// account_settings_panel.h - Account settings and user info panel
#pragma once

#include "gui/server_panel.h"
#include <string>

namespace cyxwiz::servernode::gui {

class AccountSettingsPanel : public ServerPanel {
public:
    AccountSettingsPanel() : ServerPanel("Account Settings") {}
    void Render() override;

private:
    void RenderProfileSection();
    void RenderAccountDetails();
    void RenderWalletSection();
    void RenderSecuritySection();
    void RenderPreferencesSection();

    // Copy to clipboard helper
    void CopyToClipboard(const std::string& text, const char* label);

    // UI state
    bool show_copy_notification_ = false;
    float copy_notification_timer_ = 0.0f;
    std::string copied_item_;
};

} // namespace cyxwiz::servernode::gui
