// login_panel.h - Login overlay panel for user authentication
#pragma once

#include "gui/server_panel.h"
#include "auth/auth_manager.h"
#include <string>
#include <future>

namespace cyxwiz::servernode::gui {

class LoginPanel : public ServerPanel {
public:
    LoginPanel();
    void Render() override;
    void Update() override;

    // Check if user is logged in
    bool IsLoggedIn() const;

    // Get user display name
    std::string GetUserDisplayName() const;

private:
    // Main render sections
    void RenderLoginForm();
    void RenderLoggedInState();

    // Login form components
    void RenderHeader();
    void RenderMessages();
    void RenderFormFields();
    void RenderLoginButton();
    void RenderRegisterSection();
    void RenderAlternativeOptions();

    // Legacy methods (now handled in RenderAlternativeOptions)
    void RenderWalletLoginSection();
    void RenderOfflineModeSection();

    // Actions
    void DoLogin();
    void DoLogout();

    // Form state
    char email_[256] = "";
    char password_[256] = "";
    bool show_password_ = false;

    // Login state
    bool is_logging_in_ = false;
    std::string error_message_;
    std::string success_message_;

    // Async login result
    std::future<auth::AuthResult> login_future_;

    // Async node registration result
    std::future<auth::NodeRegistrationResult> node_registration_future_;
    bool is_registering_node_ = false;

    // Offline mode
    bool offline_mode_ = false;
    
    // Deferred session restoration
    bool session_restore_pending_ = false;

    // Heartbeat timer
    std::chrono::steady_clock::time_point last_heartbeat_time_;
    static constexpr int kHeartbeatIntervalSeconds = 30;
};

} // namespace cyxwiz::servernode::gui
