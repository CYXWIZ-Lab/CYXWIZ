// login_panel.cpp - Professional login overlay panel implementation
#include "gui/panels/login_panel.h"
#include "gui/icons.h"
#include "ipc/daemon_client.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>  // for gethostname on Unix/macOS
#endif

namespace cyxwiz::servernode::gui {

namespace {
    // Professional color palette
    constexpr ImVec4 kPrimaryColor = ImVec4(0.25f, 0.56f, 0.96f, 1.0f);      // Blue
    constexpr ImVec4 kPrimaryHover = ImVec4(0.35f, 0.65f, 1.0f, 1.0f);
    constexpr ImVec4 kPrimaryActive = ImVec4(0.20f, 0.45f, 0.85f, 1.0f);
    constexpr ImVec4 kSecondaryColor = ImVec4(0.55f, 0.35f, 0.75f, 1.0f);    // Purple
    constexpr ImVec4 kSecondaryHover = ImVec4(0.65f, 0.45f, 0.85f, 1.0f);
    constexpr ImVec4 kSuccessColor = ImVec4(0.20f, 0.75f, 0.40f, 1.0f);      // Green
    constexpr ImVec4 kErrorColor = ImVec4(0.90f, 0.30f, 0.30f, 1.0f);        // Red
    constexpr ImVec4 kWarningColor = ImVec4(0.95f, 0.70f, 0.20f, 1.0f);      // Yellow
    constexpr ImVec4 kTextMuted = ImVec4(0.55f, 0.55f, 0.60f, 1.0f);
    constexpr ImVec4 kTextLight = ImVec4(0.85f, 0.85f, 0.90f, 1.0f);
    constexpr ImVec4 kInputBg = ImVec4(0.08f, 0.08f, 0.10f, 1.0f);
    constexpr ImVec4 kInputBorder = ImVec4(0.25f, 0.25f, 0.30f, 1.0f);
    constexpr ImVec4 kInputBorderFocus = ImVec4(0.25f, 0.56f, 0.96f, 0.8f);
    constexpr ImVec4 kCardBg = ImVec4(0.12f, 0.12f, 0.14f, 0.98f);
    constexpr ImVec4 kOverlayBg = ImVec4(0.05f, 0.05f, 0.07f, 0.95f);

    // Dimensions
    constexpr float kCardWidth = 420.0f;
    constexpr float kCardPadding = 20.0f;
    constexpr float kInputHeight = 36.0f;
    constexpr float kButtonHeight = 36.0f;
    constexpr float kSpacingSmall = 4.0f;
    constexpr float kSpacingMedium = 8.0f;
    constexpr float kSpacingLarge = 12.0f;
    constexpr float kBorderRadius = 8.0f;

    // Helper to draw a styled input field
    bool StyledInputText(const char* label, const char* icon, char* buf, size_t buf_size,
                         ImGuiInputTextFlags flags = 0, bool* focused = nullptr) {
        ImGui::PushID(label);

        // Label with icon
        ImGui::TextColored(kTextLight, "%s %s", icon, label);
        ImGui::Spacing();

        // Input field styling
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f, 12.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, kInputBorder);

        ImGui::SetNextItemWidth(-1);
        bool result = ImGui::InputText("##input", buf, buf_size, flags);

        if (focused) {
            *focused = ImGui::IsItemActive();
        }

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
        ImGui::PopID();

        return result;
    }

    // Helper to draw a styled button
    bool StyledButton(const char* label, const ImVec2& size, const ImVec4& color,
                      const ImVec4& hover, const ImVec4& active, bool disabled = false) {
        if (disabled) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        }

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);

        bool result = ImGui::Button(label, size);

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);

        return result && !disabled;
    }

    // Helper to draw a text link
    bool TextLink(const char* label, const ImVec4& color = kPrimaryColor) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        bool result = ImGui::SmallButton(label);

        // Underline on hover
        if (ImGui::IsItemHovered()) {
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(min.x, max.y - 1), ImVec2(max.x, max.y - 1),
                ImGui::GetColorU32(color), 1.0f);
        }

        ImGui::PopStyleColor(4);
        return result;
    }

    // Helper to draw a horizontal separator with text
    void SeparatorWithText(const char* text) {
        float width = ImGui::GetContentRegionAvail().x;
        float text_width = ImGui::CalcTextSize(text).x;
        float line_width = (width - text_width - 20.0f) / 2.0f;

        ImVec2 pos = ImGui::GetCursorScreenPos();
        float y = pos.y + ImGui::GetTextLineHeight() / 2.0f;

        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(pos.x, y), ImVec2(pos.x + line_width, y),
            ImGui::GetColorU32(kTextMuted), 1.0f);

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + line_width + 10.0f);
        ImGui::TextColored(kTextMuted, "%s", text);
        ImGui::SameLine();

        pos = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(pos.x, y), ImVec2(pos.x + line_width, y),
            ImGui::GetColorU32(kTextMuted), 1.0f);

        ImGui::NewLine();
    }

    // Draw animated spinner
    void Spinner(const char* label, float radius = 10.0f, float thickness = 3.0f) {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems) return;

        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;
        const ImGuiID id = window->GetID(label);

        ImVec2 pos = window->DC.CursorPos;
        ImVec2 size(radius * 2, radius * 2);

        const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        ImGui::ItemSize(bb, style.FramePadding.y);
        if (!ImGui::ItemAdd(bb, id)) return;

        // Render
        window->DrawList->PathClear();
        float t = (float)g.Time;
        int num_segments = 24;
        int start = (int)std::abs(std::sin(t * 1.8f) * (num_segments - 5));

        const float a_min = IM_PI * 2.0f * ((float)start) / (float)num_segments;
        const float a_max = IM_PI * 2.0f * ((float)num_segments - 3) / (float)num_segments;

        const ImVec2 centre = ImVec2(pos.x + radius, pos.y + radius);

        for (int i = 0; i < num_segments; i++) {
            const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
            window->DrawList->PathLineTo(ImVec2(
                centre.x + std::cos(a + t * 4.0f) * radius,
                centre.y + std::sin(a + t * 4.0f) * radius));
        }

        window->DrawList->PathStroke(ImGui::GetColorU32(kPrimaryColor), false, thickness);
    }
}

LoginPanel::LoginPanel()
    : ServerPanel("Login", true) {
    // Session restoration will happen asynchronously on first Update()
    // to avoid blocking the UI during startup
    session_restore_pending_ = true;
}

void LoginPanel::Render() {
    auto& auth = auth::AuthManager::Instance();
    auto state = auth.GetState();

    // Don't render if in offline mode or connected
    if (offline_mode_ && state == auth::AuthState::Offline) {
        return;
    }

    // Full screen overlay when not logged in
    if (state < auth::AuthState::Authenticated && !offline_mode_) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus |
                                 ImGuiWindowFlags_NoNavFocus |
                                 ImGuiWindowFlags_NoScrollbar;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, kOverlayBg);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

        ImGui::Begin("##LoginOverlay", nullptr, flags);

        RenderLoginForm();

        ImGui::End();

        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }
}

void LoginPanel::Update() {
    auto& auth = auth::AuthManager::Instance();

    // Deferred session restoration (runs once on first Update to avoid blocking startup)
    if (session_restore_pending_) {
        session_restore_pending_ = false;
        // Launch session restore in background thread
        std::thread([this]() {
            auto& auth = auth::AuthManager::Instance();
            if (auth.LoadSavedSession()) {
                spdlog::info("Restored saved login session");
                if (auth.IsNodeRegistered()) {
                    auth.SendHeartbeatToApi();
                    last_heartbeat_time_ = std::chrono::steady_clock::now();
                    spdlog::info("Sent initial heartbeat for restored node: {}", auth.GetNodeId());
                }
            }
        }).detach();
    }

    // Check if async login completed
    if (login_future_.valid()) {
        auto status = login_future_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            auto result = login_future_.get();
            is_logging_in_ = false;

            if (result.success) {
                error_message_.clear();
                spdlog::info("Login successful: {}", result.user_info.email);

                // After successful login, check if node needs to be registered
                if (!auth.IsNodeRegistered()) {
                    success_message_ = "Login successful! Registering node...";
                    is_registering_node_ = true;

                    // Generate node name from hostname or username
                    std::string node_name = "CyxWiz-Node";
                    char hostname[256];
#ifdef _WIN32
                    DWORD size = sizeof(hostname);
                    if (GetComputerNameA(hostname, &size)) {
                        node_name = std::string("CyxWiz-") + hostname;
                    }
#else
                    if (gethostname(hostname, sizeof(hostname)) == 0) {
                        node_name = std::string("CyxWiz-") + hostname;
                    }
#endif
                    node_registration_future_ = auth.RegisterNodeWithApi(node_name, "server");
                } else {
                    success_message_ = "Login successful! Node already registered.";
                    spdlog::info("Node already registered: {}", auth.GetNodeId());
                }
            } else {
                error_message_ = result.error;
                success_message_.clear();
                spdlog::error("Login failed: {}", result.error);
            }
        }
    }

    // Check if async node registration completed
    if (node_registration_future_.valid()) {
        auto status = node_registration_future_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            auto result = node_registration_future_.get();
            is_registering_node_ = false;

            if (result.success) {
                success_message_ = "Node registered! Connected to network.";
                spdlog::info("Node registered successfully: {}", result.node_id);
                // Send first heartbeat immediately
                auth.SendHeartbeatToApi();
                last_heartbeat_time_ = std::chrono::steady_clock::now();
            } else {
                error_message_ = "Node registration failed: " + result.error;
                success_message_.clear();
                spdlog::error("Node registration failed: {}", result.error);
            }
        }
    }

    // Send periodic heartbeats while node is registered AND connected to Central Server
    bool connected_to_central = false;
    if (IsDaemonConnected() && GetDaemonClient()) {
        ipc::DaemonStatus status;
        if (GetDaemonClient()->GetStatus(status)) {
            connected_to_central = status.connected_to_central;
        }
    }

    bool node_registered = auth.IsNodeRegistered();

    // Debug: log conditions every 10 seconds
    static auto last_debug_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto debug_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_debug_time).count();
    if (debug_elapsed >= 10) {
        spdlog::info("Heartbeat check: node_registered={}, offline={}, central={}",
                     node_registered, offline_mode_, connected_to_central);
        last_debug_time = now;
    }

    if (node_registered && !offline_mode_ && connected_to_central) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_time_).count();

        if (elapsed >= kHeartbeatIntervalSeconds) {
            spdlog::info("Sending Web API heartbeat (node_registered={}, offline={}, central={})",
                         node_registered, offline_mode_, connected_to_central);
            auth.SendHeartbeatToApi();
            last_heartbeat_time_ = now;
        }
    }
}

bool LoginPanel::IsLoggedIn() const {
    return auth::AuthManager::Instance().IsAuthenticated() || offline_mode_;
}

std::string LoginPanel::GetUserDisplayName() const {
    if (offline_mode_) {
        return "Offline Mode";
    }

    auto user = auth::AuthManager::Instance().GetUserInfo();
    if (!user.name.empty()) {
        return user.name;
    }
    if (!user.username.empty()) {
        return user.username;
    }
    return user.email;
}

void LoginPanel::RenderLoginForm() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Calculate card position (centered)
    float card_height = 460.0f;
    ImVec2 card_pos(
        (viewport->WorkSize.x - kCardWidth) / 2.0f,
        (viewport->WorkSize.y - card_height) / 2.0f
    );

    ImGui::SetCursorPos(card_pos);

    // Card background with rounded corners
    ImVec2 card_screen_pos = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(
        card_screen_pos,
        ImVec2(card_screen_pos.x + kCardWidth, card_screen_pos.y + card_height),
        ImGui::GetColorU32(kCardBg),
        kBorderRadius
    );

    // Subtle border
    ImGui::GetWindowDrawList()->AddRect(
        card_screen_pos,
        ImVec2(card_screen_pos.x + kCardWidth, card_screen_pos.y + card_height),
        ImGui::GetColorU32(ImVec4(0.2f, 0.2f, 0.25f, 0.5f)),
        kBorderRadius,
        0,
        1.0f
    );

    // Card content
    ImGui::BeginChild("LoginCard", ImVec2(kCardWidth, card_height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);

    ImGui::SetCursorPos(ImVec2(kCardPadding, kCardPadding));

    // ===== Header Section =====
    RenderHeader();

    ImGui::SetCursorPosX(kCardPadding);
    ImGui::Dummy(ImVec2(0, kSpacingLarge));

    // ===== Messages Section =====
    ImGui::SetCursorPosX(kCardPadding);
    RenderMessages();

    // ===== Form Section =====
    ImGui::SetCursorPosX(kCardPadding);
    RenderFormFields();

    ImGui::SetCursorPosX(kCardPadding);
    ImGui::Dummy(ImVec2(0, kSpacingLarge));

    // ===== Login Button =====
    ImGui::SetCursorPosX(kCardPadding);
    RenderLoginButton();

    ImGui::SetCursorPosX(kCardPadding);
    ImGui::Dummy(ImVec2(0, kSpacingMedium));

    // ===== Register Link =====
    ImGui::SetCursorPosX(kCardPadding);
    RenderRegisterSection();

    ImGui::SetCursorPosX(kCardPadding);
    ImGui::Dummy(ImVec2(0, kSpacingMedium));

    // ===== Divider =====
    ImGui::SetCursorPosX(kCardPadding);
    ImGui::BeginGroup();
    ImGui::PushItemWidth(kCardWidth - kCardPadding * 2);
    SeparatorWithText("or");
    ImGui::PopItemWidth();
    ImGui::EndGroup();

    ImGui::SetCursorPosX(kCardPadding);
    ImGui::Dummy(ImVec2(0, kSpacingMedium));

    // ===== Alternative Options =====
    ImGui::SetCursorPosX(kCardPadding);
    RenderAlternativeOptions();

    ImGui::EndChild();
}

void LoginPanel::RenderHeader() {
    float content_width = kCardWidth - kCardPadding * 2;

    // Logo icon
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    float icon_width = ImGui::CalcTextSize(ICON_FA_MICROCHIP).x;
    ImGui::SetCursorPosX(kCardPadding + (content_width - icon_width) / 2.0f);
    ImGui::TextColored(kPrimaryColor, ICON_FA_MICROCHIP);
    ImGui::PopFont();

    ImGui::Dummy(ImVec2(0, kSpacingSmall));

    // Title
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    const char* title = "CyxWiz Server Node";
    float title_width = ImGui::CalcTextSize(title).x;
    ImGui::SetCursorPosX(kCardPadding + (content_width - title_width) / 2.0f);
    ImGui::TextColored(kTextLight, "%s", title);
    ImGui::PopFont();

    ImGui::Dummy(ImVec2(0, kSpacingSmall));

    // Subtitle
    const char* subtitle = "Sign in to connect to the network";
    float subtitle_width = ImGui::CalcTextSize(subtitle).x;
    ImGui::SetCursorPosX(kCardPadding + (content_width - subtitle_width) / 2.0f);
    ImGui::TextColored(kTextMuted, "%s", subtitle);
}

void LoginPanel::RenderMessages() {
    float content_width = kCardWidth - kCardPadding * 2;

    // Error message - compact inline display
    if (!error_message_.empty()) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float msg_height = 36.0f;

        // Draw background
        ImGui::GetWindowDrawList()->AddRectFilled(
            pos,
            ImVec2(pos.x + content_width, pos.y + msg_height),
            ImGui::GetColorU32(ImVec4(0.9f, 0.2f, 0.2f, 0.15f)),
            4.0f
        );

        // Draw content
        ImGui::SetCursorPosX(kCardPadding + 12.0f);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f);
        ImGui::TextColored(kErrorColor, "%s", ICON_FA_CIRCLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::TextColored(kErrorColor, "%s", error_message_.c_str());

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 12.0f);
        ImGui::Dummy(ImVec2(0, kSpacingSmall));
        ImGui::SetCursorPosX(kCardPadding);
    }

    // Success message - compact inline display
    if (!success_message_.empty()) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float msg_height = 36.0f;

        // Draw background
        ImGui::GetWindowDrawList()->AddRectFilled(
            pos,
            ImVec2(pos.x + content_width, pos.y + msg_height),
            ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.4f, 0.15f)),
            4.0f
        );

        // Draw content
        ImGui::SetCursorPosX(kCardPadding + 12.0f);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f);
        ImGui::TextColored(kSuccessColor, "%s", ICON_FA_CIRCLE_CHECK);
        ImGui::SameLine();
        ImGui::TextColored(kSuccessColor, "%s", success_message_.c_str());

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 12.0f);
        ImGui::Dummy(ImVec2(0, kSpacingSmall));
        ImGui::SetCursorPosX(kCardPadding);
    }
}

void LoginPanel::RenderFormFields() {
    float content_width = kCardWidth - kCardPadding * 2;

    ImGui::BeginGroup();
    ImGui::PushItemWidth(content_width);

    // Email field
    bool email_enter = StyledInputText("Email", ICON_FA_ENVELOPE, email_, sizeof(email_),
                                       ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::Dummy(ImVec2(0, kSpacingMedium));
    ImGui::SetCursorPosX(kCardPadding);

    // Password field
    ImGuiInputTextFlags pwd_flags = ImGuiInputTextFlags_EnterReturnsTrue;
    if (!show_password_) {
        pwd_flags |= ImGuiInputTextFlags_Password;
    }

    ImGui::TextColored(kTextLight, "%s Password", ICON_FA_LOCK);
    ImGui::Spacing();

    // Password input with show/hide button
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f, 12.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kInputBg);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, kInputBorder);

    ImGui::SetNextItemWidth(content_width - 45.0f);
    bool pwd_enter = ImGui::InputText("##password", password_, sizeof(password_), pwd_flags);

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);

    ImGui::SameLine();

    // Eye toggle button
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.15f, 0.2f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);

    if (ImGui::Button(show_password_ ? ICON_FA_EYE_SLASH : ICON_FA_EYE, ImVec2(38, 42))) {
        show_password_ = !show_password_;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(show_password_ ? "Hide password" : "Show password");
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    ImGui::PopItemWidth();
    ImGui::EndGroup();

    // Handle enter key
    bool can_login = strlen(email_) > 0 && strlen(password_) > 0 && !is_logging_in_;
    if (can_login && (email_enter || pwd_enter)) {
        DoLogin();
    }
}

void LoginPanel::RenderLoginButton() {
    float content_width = kCardWidth - kCardPadding * 2;
    bool can_login = strlen(email_) > 0 && strlen(password_) > 0 && !is_logging_in_;

    if (is_logging_in_) {
        // Loading state
        ImGui::PushStyleColor(ImGuiCol_Button, kPrimaryColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kPrimaryColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kPrimaryColor);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);

        ImGui::Button("##loading", ImVec2(content_width, kButtonHeight));

        // Center spinner and text
        ImVec2 button_pos = ImGui::GetItemRectMin();
        ImVec2 button_size = ImGui::GetItemRectSize();

        float spinner_size = 16.0f;
        float text_width = ImGui::CalcTextSize("Signing in...").x;
        float total_width = spinner_size + 10.0f + text_width;
        float start_x = button_pos.x + (button_size.x - total_width) / 2.0f;
        float center_y = button_pos.y + (button_size.y - spinner_size) / 2.0f;

        ImGui::SetCursorScreenPos(ImVec2(start_x, center_y));
        Spinner("##login_spinner", spinner_size / 2.0f, 2.5f);
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
        ImGui::Text("Signing in...");

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
    } else {
        if (StyledButton("Sign In", ImVec2(content_width, kButtonHeight),
                         kPrimaryColor, kPrimaryHover, kPrimaryActive, !can_login)) {
            DoLogin();
        }
    }
}

void LoginPanel::RenderRegisterSection() {
    float content_width = kCardWidth - kCardPadding * 2;

    const char* text1 = "Don't have an account?";
    const char* text2 = "Create one";
    float text1_width = ImGui::CalcTextSize(text1).x;
    float text2_width = ImGui::CalcTextSize(text2).x;
    float total_width = text1_width + 8.0f + text2_width;

    ImGui::SetCursorPosX(kCardPadding + (content_width - total_width) / 2.0f);
    ImGui::TextColored(kTextMuted, "%s", text1);
    ImGui::SameLine();
    if (TextLink(text2)) {
        auth::AuthManager::OpenRegistrationPage();
    }
}

void LoginPanel::RenderAlternativeOptions() {
    float content_width = kCardWidth - kCardPadding * 2;
    float button_width = (content_width - kSpacingMedium) / 2.0f;

    // Wallet login button
    if (StyledButton(ICON_FA_WALLET " Wallet", ImVec2(button_width, 36),
                     kSecondaryColor, kSecondaryHover, ImVec4(0.45f, 0.25f, 0.65f, 1.0f))) {
        error_message_ = "Wallet login coming soon";
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Connect with Solana wallet");
    }

    ImGui::SameLine(0, kSpacingMedium);

    // Offline mode button - use amber/orange color to stand out
    ImVec4 offline_color = ImVec4(0.85f, 0.55f, 0.15f, 1.0f);  // Amber/orange
    ImVec4 offline_hover = ImVec4(0.95f, 0.65f, 0.25f, 1.0f);  // Lighter amber
    ImVec4 offline_active = ImVec4(0.75f, 0.45f, 0.10f, 1.0f); // Darker amber

    if (StyledButton(ICON_FA_DESKTOP " Work Offline", ImVec2(button_width, 36),
                     offline_color, offline_hover, offline_active)) {
        offline_mode_ = true;
        error_message_.clear();
        success_message_.clear();
        spdlog::info("Entering offline mode");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Continue without network - local training only");
    }

}

void LoginPanel::RenderLoggedInState() {
    auto& auth = auth::AuthManager::Instance();
    auto user = auth.GetUserInfo();

    ImGui::Text("%s Logged in as:", ICON_FA_USER);
    ImGui::SameLine();
    ImGui::TextColored(kSuccessColor, "%s", GetUserDisplayName().c_str());

    if (!user.wallet_address.empty()) {
        ImGui::Text("%s Wallet:", ICON_FA_WALLET);
        ImGui::SameLine();
        std::string short_addr = user.wallet_address.substr(0, 6) + "..." +
                                 user.wallet_address.substr(user.wallet_address.length() - 4);
        ImGui::TextColored(kTextMuted, "%s", short_addr.c_str());
    }

    ImGui::Spacing();

    if (StyledButton("Logout", ImVec2(100, 32), kErrorColor,
                     ImVec4(1.0f, 0.4f, 0.4f, 1.0f), ImVec4(0.8f, 0.2f, 0.2f, 1.0f))) {
        DoLogout();
    }
}

void LoginPanel::RenderWalletLoginSection() {
    // Handled in RenderAlternativeOptions now
}

void LoginPanel::RenderOfflineModeSection() {
    // Handled in RenderAlternativeOptions now
}

void LoginPanel::DoLogin() {
    if (is_logging_in_) return;

    is_logging_in_ = true;
    error_message_.clear();
    success_message_.clear();

    auto& auth = auth::AuthManager::Instance();
    login_future_ = auth.LoginWithEmail(email_, password_);

    // Clear password from memory after starting login
    std::memset(password_, 0, sizeof(password_));
}

void LoginPanel::DoLogout() {
    auto& auth = auth::AuthManager::Instance();
    auth.Logout();

    offline_mode_ = false;
    error_message_.clear();
    success_message_.clear();

    // Clear form
    std::memset(email_, 0, sizeof(email_));
    std::memset(password_, 0, sizeof(password_));
}

} // namespace cyxwiz::servernode::gui
