// account_settings_panel.h - Account settings and user info panel
#pragma once

#include "gui/server_panel.h"
#include "auth/auth_manager.h"
#include <string>
#include <future>

namespace cyxwiz::servernode::gui {

class AccountSettingsPanel : public ServerPanel {
public:
    AccountSettingsPanel() : ServerPanel("Account Settings") {}
    void Render() override;
    void Update();  // Call each frame to check async operations

private:
    void RenderLoginForm();
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

    // Login form state
    char login_email_[256] = "";
    char login_password_[256] = "";
    bool show_password_ = false;
    bool is_logging_in_ = false;
    std::string login_error_;
    std::string login_success_;
    std::future<auth::AuthResult> login_future_;
};

} // namespace cyxwiz::servernode::gui
