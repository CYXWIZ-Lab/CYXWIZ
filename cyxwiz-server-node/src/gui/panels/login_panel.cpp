// login_panel.cpp - the sign-in overlay (TOFIX139). Words and states come
// from core/sign_in_presentation; colours from the theme through gui/ui_widgets.
#include "gui/panels/login_panel.h"

#include "core/backend_manager.h"
#include "core/config_manager.h"
#include "gui/icons.h"
#include "gui/theme.h"
#include "gui/ui_widgets.h"
#include "gui/window_frame.h"
#include "ipc/daemon_client.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>  // gethostname
#endif

namespace cyxwiz::servernode::gui {

namespace {

constexpr float kCardWidth = 420.0f;
constexpr float kDialogWidth = 440.0f;
constexpr float kCardRounding = 10.0f;
constexpr float kLogoAlpha = 0.28f;  // the logo stays a backdrop: the card and text keep their contrast
const ImVec2 kCardPadding(24.0f, 24.0f);

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, 1.0f);
}

ImVec4 Surface() { return ImGui::GetStyle().Colors[ImGuiCol_WindowBg]; }
// A darker shade of the theme's title-bar tone, so the card stands out.
ImVec4 Backdrop() {
    const ImVec4 c = ImGui::GetStyle().Colors[ImGuiCol_TitleBg];
    return ImVec4(c.x * 0.55f, c.y * 0.55f, c.z * 0.55f, 1.0f);
}

// Centred text on its own line.
void CenteredText(const char* text, const ImVec4* color = nullptr) {
    const float width = ImGui::GetContentRegionAvail().x;
    const float text_width = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (width - text_width) * 0.5f));
    if (color) {
        ImGui::TextColored(*color, "%s", text);
    } else {
        ImGui::TextUnformatted(text);
    }
}

std::string MachineName() {
    char hostname[256] = {0};
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    if (GetComputerNameA(hostname, &size)) return std::string("CyxWiz-") + hostname;
#else
    if (gethostname(hostname, sizeof(hostname)) == 0) return std::string("CyxWiz-") + hostname;
#endif
    return "CyxWiz-Node";
}

}  // namespace

LoginPanel::LoginPanel() : ServerPanel("Login", true) {}

// ---------------------------------------------------------------------------
// Update: async results and background work

void LoginPanel::Update() {
    PollSessionRestore();
    PollSignIn();
    PollRegistration();
    PollDaemonStatus();
    PollHeartbeat();
    PollServiceProbe();
    PollProgressCard();
}

void LoginPanel::PollSessionRestore() {
    if (!restore_started_) {
        // Off the UI thread: the restore talks to the web API.
        restore_started_ = true;
        restore_future_ = std::async(std::launch::async, []() {
            auto& auth = auth::AuthManager::Instance();
            const bool restored = auth.LoadSavedSession();
            if (restored) {
                spdlog::info("Restored saved login session");
                if (auth.IsNodeRegistered()) {
                    auth.SendHeartbeatToApi();
                    spdlog::info("Sent initial heartbeat for restored node: {}", auth.GetNodeId());
                }
            }
            return restored;
        });
        return;
    }
    if (!restore_future_.valid() || restore_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }
    const bool restored = restore_future_.get();
    auto& auth = auth::AuthManager::Instance();
    if (restored) last_heartbeat_time_ = Clock::now();
    if (!email_prefilled_) {
        email_prefilled_ = true;
        const std::string remembered = auth.GetRememberedEmail();
        if (email_[0] == '\0' && !remembered.empty()) {
            std::strncpy(email_, remembered.c_str(), sizeof(email_) - 1);
        }
        focus_pending_ = true;
    }
    if (!restored && auth.LastSessionExpired()) {
        ShowProblem(core::SignInProblem::Expired, 0, "saved session no longer accepted");
    }
}

void LoginPanel::PollSignIn() {
    if (!login_future_.valid() || login_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }
    const auto result = login_future_.get();
    phase_ = core::SignInPhase::Form;
    auto& auth = auth::AuthManager::Instance();
    if (result.success) {
        ClearMessage();
        cached_auth_required_ = false;
        spdlog::info("Login successful: {}", result.user_info.email);
        progress_visible_ = true;
        if (auth.IsNodeRegistered()) {
            register_state_ = core::StepState::Done;
            node_name_.clear();
            progress_hide_at_ = Clock::now() + std::chrono::seconds(kProgressCardSeconds);
            spdlog::info("Node already registered: {}", auth.GetNodeId());
        } else {
            StartNodeRegistration();
        }
        return;
    }
    ShowProblem(core::ClassifySignInError(result.http_status, result.error), result.http_status, result.error);
    focus_pending_ = true;
}

void LoginPanel::PollRegistration() {
    auto& auth = auth::AuthManager::Instance();
    if (node_registration_future_.valid() &&
        node_registration_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const auto result = node_registration_future_.get();
        if (result.success) {
            register_state_ = core::StepState::Done;
            registered_this_session_ = true;
            register_error_.clear();
            progress_hide_at_ = Clock::now() + std::chrono::seconds(kProgressCardSeconds);
            spdlog::info("Node registered successfully: {}", result.node_id);
            last_heartbeat_time_ = Clock::now();
            heartbeat_future_ = std::async(std::launch::async, []() {
                return auth::AuthManager::Instance().SendHeartbeatToApi();
            });
        } else {
            register_state_ = core::StepState::Failed;
            register_error_ = result.error;
            register_unreachable_ = result.unreachable;
            spdlog::error("Node registration failed: {}", result.error);
        }
    }

    // A signed-in node the web API no longer knows registers again (the
    // heartbeat cleared the stale registration), at most once a minute.
    const auto now = Clock::now();
    if (!auth.IsNodeRegistered() && !offline_mode_ && auth.IsAuthenticated() && !node_registration_future_.valid() &&
        !progress_visible_ &&
        std::chrono::duration_cast<std::chrono::seconds>(now - last_register_attempt_).count() >= kRegisterRetrySeconds) {
        StartNodeRegistration();
    }
}

void LoginPanel::PollDaemonStatus() {
    const auto now = Clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_check_time_).count() < kStatusCheckIntervalMs) {
        return;
    }
    last_status_check_time_ = now;
    auto& auth = auth::AuthManager::Instance();
    if (IsDaemonConnected() && GetDaemonClient()) {
        ipc::DaemonStatus status;
        if (GetDaemonClient()->GetStatus(status)) {  // 3 s deadline inside
            cached_connected_to_central_ = status.connected_to_central;
            daemon_version_ = status.version;
            if (status.auth_required && !cached_auth_required_) {
                cached_auth_required_ = true;
                spdlog::warn("Auth required detected - prompting user to re-login");
                auth.Logout();
                auth.RememberEmailOnly();
                ShowProblem(core::SignInProblem::Expired, 0, "the daemon reports the token was refused");
                focus_pending_ = true;
            } else if (!status.auth_required) {
                cached_auth_required_ = false;
            }
            return;
        }
    }
    cached_connected_to_central_ = false;
}

void LoginPanel::PollHeartbeat() {
    if (heartbeat_future_.valid()) {
        if (heartbeat_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        heartbeat_future_.get();
    }
    auto& auth = auth::AuthManager::Instance();
    if (!auth.IsNodeRegistered() || offline_mode_ || !cached_connected_to_central_) return;
    const auto now = Clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_time_).count() < kHeartbeatIntervalSeconds) {
        return;
    }
    last_heartbeat_time_ = now;
    heartbeat_future_ = std::async(std::launch::async, []() { return auth::AuthManager::Instance().SendHeartbeatToApi(); });
}

void LoginPanel::PollServiceProbe() {
    if (probe_future_.valid() && probe_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        service_reach_ = probe_future_.get() ? core::ServiceReach::Reachable : core::ServiceReach::Unreachable;
    }
    if (!open_service_dialog_ || !service_dirty_ || probe_future_.valid()) return;
    // Probe 600 ms after the last keystroke, and only a well-formed address.
    if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - service_edited_at_).count() < 600) return;
    service_dirty_ = false;
    const auto check = core::CheckServiceAddress(service_buffer_);
    if (!check.ok || check.normalized == probed_address_) return;
    probed_address_ = check.normalized;
    service_reach_ = core::ServiceReach::Checking;
    probe_future_ = auth::AuthManager::ProbeService(check.normalized);
}

void LoginPanel::PollProgressCard() {
    if (!progress_visible_ || register_state_ != core::StepState::Done) return;
    if (Clock::now() < progress_hide_at_) return;
    progress_visible_ = false;
    if (registered_this_session_) {
        open_allocation_request_ = true;
        registered_this_session_ = false;
    }
}

// ---------------------------------------------------------------------------
// State

bool LoginPanel::IsLoggedIn() const { return auth::AuthManager::Instance().IsAuthenticated() || offline_mode_; }

bool LoginPanel::TakeOpenAllocationRequest() {
    const bool request = open_allocation_request_;
    open_allocation_request_ = false;
    return request;
}

core::SignInInputs LoginPanel::CurrentInputs() const {
    core::SignInInputs in;
    in.phase = phase_;
    in.service_url = auth::AuthManager::Instance().GetApiBaseUrl();
    in.email = email_;
    in.password_filled = password_[0] != '\0';
    in.caps_lock = CapsLockOn();
    in.problem = problem_;
    in.http_status = problem_status_;
    in.problem_detail = problem_detail_;
    in.daemon_connected = IsDaemonConnected();
    in.version = daemon_version_;
    return in;
}

core::ProgressInputs LoginPanel::CurrentProgressInputs() const {
    core::ProgressInputs in;
    const auto user = auth::AuthManager::Instance().GetUserInfo();
    in.user_name = !user.name.empty() ? user.name : user.username;
    in.user_email = user.email;
    in.registering = register_state_;
    in.node_name = node_name_;
    in.register_error = register_error_;
    in.register_unreachable = register_unreachable_;
    return in;
}

bool LoginPanel::CapsLockOn() {
#ifdef _WIN32
    return (GetKeyState(VK_CAPITAL) & 1) != 0;
#else
    return false;
#endif
}

void LoginPanel::ClearMessage() {
    problem_ = core::SignInProblem::None;
    problem_status_ = 0;
    problem_detail_.clear();
}

void LoginPanel::ShowProblem(core::SignInProblem problem, int http_status, const std::string& detail) {
    problem_ = problem;
    problem_status_ = http_status;
    problem_detail_ = detail;
    const auto card = core::BuildSignInCard(CurrentInputs());
    if (!card.log_line.empty()) spdlog::warn("{}", card.log_line);
}

// ---------------------------------------------------------------------------
// Actions

void LoginPanel::StartSignIn() {
    if (phase_ == core::SignInPhase::SigningIn) return;
    const auto card = core::BuildSignInCard(CurrentInputs());
    if (!card.button_enabled) return;
    phase_ = core::SignInPhase::SigningIn;
    ClearMessage();
    login_future_ = auth::AuthManager::Instance().LoginWithEmail(email_, password_);
    // The password leaves memory as soon as the request is on its way.
    std::memset(password_, 0, sizeof(password_));
}

void LoginPanel::StartNodeRegistration() {
    if (node_registration_future_.valid()) return;
    last_register_attempt_ = Clock::now();
    register_state_ = core::StepState::Running;
    register_error_.clear();
    register_unreachable_ = false;
    node_name_ = MachineName();
    spdlog::info("Registering this machine with the web API as {}", node_name_);
    node_registration_future_ = auth::AuthManager::Instance().RegisterNodeWithApi(node_name_, "server");
}

void LoginPanel::UseService(const std::string& api_url) {
    auth::AuthManager::Instance().SetApiBaseUrl(api_url);
    // Saved in the node config: the daemon reads the same key to renew the sign-in.
    auto& backend = core::BackendManager::Instance();
    if (backend.IsInitialized()) {
        auto config = backend.GetConfig();
        config.auth_api_url = api_url;
        backend.UpdateConfig(config);
    } else {
        core::ConfigManager manager;
        const std::string path = core::ConfigManager::FindConfigFile();
        manager.Load(path);
        auto config = manager.GetConfig();
        config.auth_api_url = api_url;
        manager.SetConfig(config);
        if (!manager.Save()) spdlog::warn("Sign-in service {} set for this run only (config not saved)", api_url);
    }
    spdlog::info("Sign-in service set to {}", api_url);
    ClearMessage();
}

void LoginPanel::OpenCreateAccount() {
    const std::string website = core::WebsiteOf(auth::AuthManager::Instance().GetApiBaseUrl());
    auth::AuthManager::OpenWebsite(website + "/register");
}

// ---------------------------------------------------------------------------
// Render

void LoginPanel::Render() {
    if (IsLoggedIn() && !progress_visible_) {
        // The dashboard: the frame follows the theme's title tone.
        SetWindowFrameColor(ImGui::GetStyle().Colors[ImGuiCol_TitleBg]);
        return;
    }
    if (!logo_tried_) {
        logo_tried_ = true;
        const std::string path = FindResourceFile("cyxwiz.png");
        if (!path.empty()) LoadTexture(path, logo_);
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoDocking;
    // The after-sign-in card shows over the dashboard: keep the overlay in front
    // while it is visible (the dashboard windows appear once signed in).
    if (progress_visible_) ImGui::SetNextWindowFocus();
    // Read the theme's surfaces before the backdrop replaces the window background.
    const ImVec4 surface = Surface();
    // With the logo drawn at kLogoAlpha over the backdrop, the visible tone is
    // the mix; the OS frame takes the same tone (owner: "the window frame
    // should match it").
    const ImVec4 backdrop = logo_.Loaded() ? Mix(Backdrop(), logo_.edge_color, kLogoAlpha) : Backdrop();
    SetWindowFrameColor(backdrop);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Backdrop());
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("##SignInOverlay", nullptr, flags);

    // The CyxWiz logo, faint, covering the window (cover-fit, centred).
    if (logo_.Loaded()) {
        const ImVec2 min = ImGui::GetWindowPos();
        const ImVec2 size = ImGui::GetWindowSize();
        const float scale = std::max(size.x / static_cast<float>(logo_.width), size.y / static_cast<float>(logo_.height));
        const ImVec2 drawn(logo_.width * scale, logo_.height * scale);
        const ImVec2 p0(min.x + (size.x - drawn.x) * 0.5f, min.y + (size.y - drawn.y) * 0.5f);
        const ImVec2 p1(p0.x + drawn.x, p0.y + drawn.y);
        ImGui::GetWindowDrawList()->AddImage(logo_.ImGuiId(), p0, p1, ImVec2(0, 0), ImVec2(1, 1),
                                             ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, kLogoAlpha)));
    }

    // Esc clears the message (when no dialog is open).
    if (!open_service_dialog_ && !open_offline_confirm_ && ImGui::IsKeyPressed(ImGuiKey_Escape)) ClearMessage();

    // The card, centred; its height is the last frame's measurement.
    const ImVec2 area = ImGui::GetWindowSize();
    ImGui::SetCursorPos(ImVec2((area.x - kCardWidth) * 0.5f, std::max(16.0f, (area.y - card_height_) * 0.5f)));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, surface);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, kCardRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, kCardPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
    if (ImGui::BeginChild("##card", ImVec2(kCardWidth, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        if (progress_visible_) {
            RenderProgressCard();
        } else {
            RenderCard();
        }
    }
    card_height_ = ImGui::GetWindowSize().y;
    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor();

    // Footer line at the bottom of the backdrop.
    if (!progress_visible_) {
        const auto card = core::BuildSignInCard(CurrentInputs());
        const ImVec4 muted = ui::Muted();
        ImGui::SetCursorPos(ImVec2(0.0f, area.y - ImGui::GetTextLineHeight() - 14.0f));
        CenteredText(card.footer.c_str(), &muted);
    }

    RenderServiceDialog();
    RenderOfflineConfirm();

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void LoginPanel::RenderCard() {
    const core::SignInCard card = core::BuildSignInCard(CurrentInputs());
    const ImVec4 muted = ui::Muted();
    const ImVec4 accent = ui::Accent();
    const bool busy = card.button_busy;

    // Header
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    CenteredText(ICON_FA_MICROCHIP, &accent);
    CenteredText(card.title.c_str());
    ImGui::PopFont();
    CenteredText(card.subtitle.c_str(), &muted);
    ImGui::Spacing();

    // Service line: "Signing in to 127.0.0.1:3002  Change"
    {
        const float total = ImGui::CalcTextSize(card.service_prefix.c_str()).x + 6.0f +
                            ImGui::CalcTextSize(card.service_host.c_str()).x + 10.0f +
                            ImGui::CalcTextSize(card.change_label.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - total) * 0.5f));
        ImGui::TextColored(muted, "%s", card.service_prefix.c_str());
        ImGui::SameLine(0.0f, 6.0f);
        ImGui::PushFont(GetSafeFont(FONT_MONO));
        ImGui::TextUnformatted(card.service_host.c_str());
        ImGui::PopFont();
        ImGui::SameLine(0.0f, 10.0f);
        if (ui::LinkButton(card.change_label.c_str(), card.change_enabled)) {
            std::strncpy(service_buffer_, auth::AuthManager::Instance().GetApiBaseUrl().c_str(), sizeof(service_buffer_) - 1);
            probed_address_.clear();
            service_reach_ = core::ServiceReach::Unknown;
            service_dirty_ = true;
            service_edited_at_ = Clock::now();
            open_service_dialog_ = true;
        }
    }

    // One message at a time, above the fields.
    if (card.message.shown) RenderMessage(card.message);

    // Fields
    bool submit = false;
    ImGui::TextUnformatted(card.email_label.c_str());
    if (focus_pending_ && card.focus == core::SignInFocus::Email) {
        ImGui::SetKeyboardFocusHere();
        focus_pending_ = false;
    }
    submit |= ui::TextField("##email", email_, sizeof(email_), card.email_placeholder.c_str(),
                            ImGuiInputTextFlags_EnterReturnsTrue, card.fields_enabled);
    if (ImGui::IsItemEdited()) ClearMessage();

    ImGui::TextUnformatted(card.password_label.c_str());
    if (focus_pending_ && card.focus == core::SignInFocus::Password) {
        ImGui::SetKeyboardFocusHere();
        focus_pending_ = false;
    }
    {
        const float eye_width = ImGui::GetFrameHeight() + 12.0f;
        ImGuiInputTextFlags pwd_flags = ImGuiInputTextFlags_EnterReturnsTrue;
        if (!show_password_) pwd_flags |= ImGuiInputTextFlags_Password;
        submit |= ui::TextField("##password", password_, sizeof(password_), "", pwd_flags, card.fields_enabled,
                                ImGui::GetContentRegionAvail().x - eye_width - 4.0f);
        if (ImGui::IsItemEdited()) ClearMessage();
        ImGui::SameLine(0.0f, 4.0f);
        const ImVec2 eye_padding = ui::FieldPadding();
        if (ui::SecondaryButton(show_password_ ? ICON_FA_EYE_SLASH : ICON_FA_EYE, eye_width, card.fields_enabled,
                                &eye_padding)) {
            show_password_ = !show_password_;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(show_password_ ? "Hide password" : "Show password");
    }
    if (submit && !busy) StartSignIn();

    // The one primary action
    ImGui::Spacing();
    if (ui::PrimaryButton(card.button_label.c_str(), -1.0f, card.button_enabled,
                          "Enter your email and password first", busy)) {
        StartSignIn();
    }
    if (!card.busy_note.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", card.busy_note.c_str());
        ImGui::PopTextWrapPos();
    }

    // Create an account
    {
        const float total = ImGui::CalcTextSize(card.create_prefix.c_str()).x + 6.0f +
                            ImGui::CalcTextSize(card.create_link.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - total) * 0.5f));
        ImGui::TextColored(muted, "%s", card.create_prefix.c_str());
        ImGui::SameLine(0.0f, 6.0f);
        if (ui::LinkButton(card.create_link.c_str(), card.create_enabled)) OpenCreateAccount();
    }

    // Work offline, demoted to a link with its meaning
    ImGui::Spacing();
    {
        const float link_width = ImGui::CalcTextSize(card.offline_link.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                             std::max(0.0f, (ImGui::GetContentRegionAvail().x - link_width) * 0.5f));
        if (ui::LinkButton(card.offline_link.c_str(), card.offline_enabled)) open_offline_confirm_ = true;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", card.offline_note.c_str());
        ImGui::PopTextWrapPos();
    }
}

void LoginPanel::RenderMessage(const core::SignInMessage& message) {
    const auto& semantic = ::gui::GetTheme().Semantic();
    const bool error = message.tone == core::MessageTone::Error;
    const ImVec4 strong = error ? semantic.error : ImGui::GetStyle().Colors[ImGuiCol_Text];
    const ImVec4 tint = error ? ui::Tinted(semantic.error, 0.18f) : ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
    const ImVec4 muted = ui::Muted();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, tint);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    if (ImGui::BeginChild("##message", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        ImGui::TextColored(strong, "%s", error ? ICON_FA_CIRCLE_EXCLAMATION : ICON_FA_CLOCK);
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(strong, "%s", message.title.c_str());
        if (!message.link_word.empty() && message.body.find(message.link_word) != std::string::npos) {
            const auto at = message.body.find(message.link_word);
            const std::string before = message.body.substr(0, at);
            const std::string after = message.body.substr(at + message.link_word.size());
            ImGui::TextColored(muted, "%s", before.c_str());
            ImGui::SameLine(0.0f, 0.0f);
            if (ui::LinkButton(message.link_word.c_str())) OpenCreateAccount();
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(muted, "%s", after.c_str());
        } else {
            ImGui::TextColored(muted, "%s", message.body.c_str());
        }
        ImGui::PopTextWrapPos();
        if (message.caps_lock_hint) {
            ImGui::TextColored(semantic.warning, ICON_FA_ARROW_UP " Caps Lock is on");
        }
        if (!message.actions.empty()) {
            ImGui::Spacing();
            bool first = true;
            for (const auto action : message.actions) {
                if (!first) ImGui::SameLine(0.0f, 8.0f);
                first = false;
                switch (action) {
                    case core::SignInAction::TryAgain:
                        if (ui::SecondaryButton("Try again")) {
                            if (core::BuildSignInCard(CurrentInputs()).button_enabled) {
                                StartSignIn();
                            } else {
                                ClearMessage();
                                focus_pending_ = true;
                            }
                        }
                        break;
                    case core::SignInAction::ChangeService:
                        if (ui::LinkButton("Change service")) {
                            std::strncpy(service_buffer_, auth::AuthManager::Instance().GetApiBaseUrl().c_str(),
                                         sizeof(service_buffer_) - 1);
                            probed_address_.clear();
                            service_reach_ = core::ServiceReach::Unknown;
                            service_dirty_ = true;
                            service_edited_at_ = Clock::now();
                            open_service_dialog_ = true;
                        }
                        break;
                    case core::SignInAction::CreateAccount:
                        break;  // the link inside the body
                }
            }
        }
        ImGui::EndGroup();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void LoginPanel::RenderProgressCard() {
    const core::ProgressCard card = core::BuildProgressCard(CurrentProgressInputs());
    const auto& semantic = ::gui::GetTheme().Semantic();
    const ImVec4 muted = ui::Muted();
    const ImVec4 accent = ui::Accent();

    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    CenteredText(ICON_FA_MICROCHIP, &accent);
    CenteredText(card.title.c_str());
    ImGui::PopFont();
    CenteredText(card.subtitle.c_str(), &muted);
    ImGui::Spacing();

    for (size_t i = 0; i < card.steps.size(); ++i) {
        const auto& step = card.steps[i];
        ImGui::PushID(static_cast<int>(i));
        const float line = ImGui::GetTextLineHeight();
        switch (step.state) {
            case core::StepState::Done: ImGui::TextColored(semantic.success, ICON_FA_CIRCLE_CHECK); break;
            case core::StepState::Failed: ImGui::TextColored(semantic.error, ICON_FA_CIRCLE_EXCLAMATION); break;
            case core::StepState::Running: ui::Spinner("##step", line * 0.42f, 2.5f); break;
            case core::StepState::Pending: ImGui::TextColored(muted, ICON_FA_CIRCLE); break;
        }
        ImGui::SameLine(0.0f, 12.0f);
        ImGui::BeginGroup();
        if (step.state == core::StepState::Pending) ImGui::BeginDisabled();
        ImGui::TextUnformatted(step.title.c_str());
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", step.detail.c_str());
        ImGui::PopTextWrapPos();
        if (step.state == core::StepState::Pending) ImGui::EndDisabled();
        ImGui::EndGroup();
        ImGui::PopID();
    }

    if (card.failed) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Tinted(semantic.error, 0.18f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
        if (ImGui::BeginChild("##failure", ImVec2(0.0f, 0.0f),
                              ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
            ImGui::TextColored(semantic.error, "%s", card.failure_title.c_str());
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(muted, "%s", card.failure_detail.c_str());
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            if (ui::SecondaryButton(card.retry_label.c_str())) StartNodeRegistration();
            ImGui::SameLine(0.0f, 8.0f);
            if (ui::LinkButton(card.continue_label.c_str())) progress_visible_ = false;
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(muted, "%s", card.failure_note.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    } else if (!card.footnote.empty()) {
        ImGui::Spacing();
        CenteredText(card.footnote.c_str(), &muted);
    }
}

void LoginPanel::RenderServiceDialog() {
    if (open_service_dialog_ && !ImGui::IsPopupOpen("##service")) ImGui::OpenPopup("##service");
    if (!open_service_dialog_) return;

    const core::ServiceDialog dialog = core::BuildServiceDialog(service_buffer_, service_reach_);
    const auto& semantic = ::gui::GetTheme().Semantic();
    const ImVec4 muted = ui::Muted();

    ImGui::SetNextWindowSize(ImVec2(kDialogWidth, 0.0f));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, kCardRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, kCardPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
    if (ImGui::BeginPopupModal("##service", nullptr,
                               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextUnformatted(dialog.title.c_str());
        ImGui::PopFont();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", dialog.explanation.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();

        ImGui::TextUnformatted(dialog.address_label.c_str());
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::PushFont(GetSafeFont(FONT_MONO));
        const bool entered = ui::TextField("##address", service_buffer_, sizeof(service_buffer_),
                                           "http://host:port/api", ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopFont();
        if (ImGui::IsItemEdited()) {
            service_dirty_ = true;
            service_edited_at_ = Clock::now();
            service_reach_ = core::ServiceReach::Unknown;
        }
        if (!dialog.address_problem.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(semantic.error, "%s", dialog.address_problem.c_str());
            ImGui::PopTextWrapPos();
        }

        // Examples: clicking one fills the field.
        if (ImGui::BeginTable("##examples", 2, ImGuiTableFlags_SizingStretchProp)) {
            for (const auto& [what, address] : dialog.examples) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(muted, "%s", what.c_str());
                ImGui::TableNextColumn();
                ImGui::PushFont(GetSafeFont(FONT_MONO));
                if (ui::LinkButton(address.c_str())) {
                    std::strncpy(service_buffer_, address.c_str(), sizeof(service_buffer_) - 1);
                    service_buffer_[sizeof(service_buffer_) - 1] = '\0';
                    service_dirty_ = true;
                    service_edited_at_ = Clock::now() - std::chrono::seconds(1);
                    service_reach_ = core::ServiceReach::Unknown;
                }
                ImGui::PopFont();
            }
            ImGui::EndTable();
        }

        if (!dialog.reach_line.empty()) {
            switch (dialog.reach) {
                case core::ServiceReach::Reachable:
                    ImGui::TextColored(semantic.success, ICON_FA_CIRCLE_CHECK " %s", dialog.reach_line.c_str());
                    break;
                case core::ServiceReach::Unreachable:
                    ImGui::TextColored(semantic.error, ICON_FA_CIRCLE_EXCLAMATION " %s", dialog.reach_line.c_str());
                    break;
                case core::ServiceReach::Checking:
                    ui::Spinner("##reach", ImGui::GetTextLineHeight() * 0.4f, 2.0f);
                    ImGui::SameLine(0.0f, 8.0f);
                    ImGui::TextColored(muted, "%s", dialog.reach_line.c_str());
                    break;
                case core::ServiceReach::Unknown: break;
            }
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(muted, "%s", dialog.note.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();

        // Buttons, right-aligned
        const float use_width = ImGui::CalcTextSize(dialog.use_label.c_str()).x + 32.0f;
        const float cancel_width = ImGui::CalcTextSize(dialog.cancel_label.c_str()).x + 28.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                             std::max(0.0f, ImGui::GetContentRegionAvail().x - use_width - cancel_width - 8.0f));
        if (ui::SecondaryButton(dialog.cancel_label.c_str(), cancel_width) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            open_service_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine(0.0f, 8.0f);
        if ((ui::PrimaryButton(dialog.use_label.c_str(), use_width, dialog.use_enabled, dialog.address_problem.c_str()) ||
             (entered && dialog.use_enabled))) {
            UseService(core::CheckServiceAddress(service_buffer_).normalized);
            open_service_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar(4);
}

void LoginPanel::RenderOfflineConfirm() {
    if (open_offline_confirm_ && !ImGui::IsPopupOpen("##offline")) ImGui::OpenPopup("##offline");
    if (!open_offline_confirm_) return;

    const core::OfflineConfirm confirm = core::BuildOfflineConfirm();
    const ImVec4 muted = ui::Muted();
    ImGui::SetNextWindowSize(ImVec2(kDialogWidth, 0.0f));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, kCardRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, kCardPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 8.0f));
    if (ImGui::BeginPopupModal("##offline", nullptr,
                               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextUnformatted(confirm.title.c_str());
        ImGui::PopFont();
        ImGui::PushTextWrapPos(0.0f);
        for (size_t i = 0; i < confirm.lines.size(); ++i) {
            if (i + 1 == confirm.lines.size()) {
                ImGui::TextColored(muted, "%s", confirm.lines[i].c_str());
            } else {
                ImGui::TextUnformatted(confirm.lines[i].c_str());
            }
        }
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        const float confirm_width = ImGui::CalcTextSize(confirm.confirm_label.c_str()).x + 32.0f;
        const float back_width = ImGui::CalcTextSize(confirm.back_label.c_str()).x + 28.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                             std::max(0.0f, ImGui::GetContentRegionAvail().x - confirm_width - back_width - 8.0f));
        if (ui::SecondaryButton(confirm.back_label.c_str(), back_width) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            open_offline_confirm_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine(0.0f, 8.0f);
        if (ui::SecondaryButton(confirm.confirm_label.c_str(), confirm_width)) {
            offline_mode_ = true;
            ClearMessage();
            spdlog::info("Entering offline mode");
            open_offline_confirm_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar(4);
}

}  // namespace cyxwiz::servernode::gui
