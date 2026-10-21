// login_panel.h - the sign-in overlay (TOFIX139). A thin renderer: every word
// and state comes from core/sign_in_presentation; this file owns the form
// buffers, the async requests and the ImGui drawing.
#pragma once

#include "auth/auth_manager.h"
#include "core/sign_in_presentation.h"
#include "gui/server_panel.h"
#include "gui/texture.h"

#include <chrono>
#include <future>
#include <string>

namespace cyxwiz::servernode::gui {

class LoginPanel : public ServerPanel {
public:
    LoginPanel();
    void Render() override;
    void Update() override;

    // Signed in, or working offline: the dashboard may show.
    bool IsLoggedIn() const;

    // Once this session registered the machine, the after-sign-in card asks
    // the main window to open the Allocation page ("Choose what to lend").
    bool TakeOpenAllocationRequest();

private:
    using Clock = std::chrono::steady_clock;

    // Drawing
    void RenderCard();
    void RenderMessage(const core::SignInMessage& message);
    void RenderProgressCard();
    void RenderServiceDialog();
    void RenderOfflineConfirm();
    core::SignInInputs CurrentInputs() const;
    core::ProgressInputs CurrentProgressInputs() const;

    // Actions
    void StartSignIn();
    void StartNodeRegistration();
    void UseService(const std::string& api_url);
    void OpenCreateAccount();
    void ClearMessage();
    void ShowProblem(core::SignInProblem problem, int http_status, const std::string& detail);

    // Polling (Update)
    void PollSessionRestore();
    void PollSignIn();
    void PollRegistration();
    void PollDaemonStatus();
    void PollHeartbeat();
    void PollServiceProbe();
    void PollProgressCard();

    static bool CapsLockOn();

    // Form
    char email_[256] = "";
    char password_[256] = "";
    bool show_password_ = false;
    bool focus_pending_ = true;       // put the cursor in the right field once
    bool email_prefilled_ = false;    // from the remembered email, once the restore finished
    float card_height_ = 460.0f;      // last measured, to centre the card
    Texture logo_;                    // faint CyxWiz logo behind the card (P7)
    bool logo_tried_ = false;

    // Sign-in state
    core::SignInPhase phase_ = core::SignInPhase::Form;
    core::SignInProblem problem_ = core::SignInProblem::None;
    int problem_status_ = 0;
    std::string problem_detail_;
    std::future<auth::AuthResult> login_future_;

    // Session restore (once, off the UI thread)
    bool restore_started_ = false;
    std::future<bool> restore_future_;

    // After sign-in
    bool progress_visible_ = false;
    bool registered_this_session_ = false;
    core::StepState register_state_ = core::StepState::Pending;
    std::string register_error_;
    bool register_unreachable_ = false;
    std::string node_name_;
    Clock::time_point progress_hide_at_{};
    bool open_allocation_request_ = false;
    std::future<auth::NodeRegistrationResult> node_registration_future_;
    Clock::time_point last_register_attempt_{};
    static constexpr int kRegisterRetrySeconds = 60;
    static constexpr int kProgressCardSeconds = 4;

    // Change the sign-in service
    bool open_service_dialog_ = false;
    char service_buffer_[512] = "";
    std::string probed_address_;
    core::ServiceReach service_reach_ = core::ServiceReach::Unknown;
    std::future<bool> probe_future_;
    Clock::time_point service_edited_at_{};
    bool service_dirty_ = false;

    // Work offline
    bool open_offline_confirm_ = false;
    bool offline_mode_ = false;

    // Heartbeat (off the UI thread) and daemon status
    std::future<bool> heartbeat_future_;
    Clock::time_point last_heartbeat_time_{};
    static constexpr int kHeartbeatIntervalSeconds = 30;
    Clock::time_point last_status_check_time_{};
    static constexpr int kStatusCheckIntervalMs = 2000;
    bool cached_connected_to_central_ = false;
    bool cached_auth_required_ = false;
    std::string daemon_version_;
};

}  // namespace cyxwiz::servernode::gui
