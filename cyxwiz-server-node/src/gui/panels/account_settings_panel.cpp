// account_settings_panel.cpp
#include "gui/panels/account_settings_panel.h"
#include "gui/icons.h"
#include "auth/auth_manager.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>

namespace cyxwiz::servernode::gui {

void AccountSettingsPanel::Update() {
    // Check if async login completed
    if (login_future_.valid()) {
        auto status = login_future_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            auto result = login_future_.get();
            is_logging_in_ = false;

            if (result.success) {
                login_error_.clear();
                login_success_ = "Login successful!";
                std::memset(login_password_, 0, sizeof(login_password_));
                spdlog::info("Login successful: {}", result.user_info.email);
            } else {
                login_error_ = result.error;
                login_success_.clear();
                spdlog::error("Login failed: {}", result.error);
            }
        }
    }
}

void AccountSettingsPanel::Render() {
    auto& auth = auth::AuthManager::Instance();

    // Update async operations
    Update();

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text("%s Account Settings", ICON_FA_USER_GEAR);
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Handle copy notification fade out
    if (show_copy_notification_) {
        copy_notification_timer_ -= ImGui::GetIO().DeltaTime;
        if (copy_notification_timer_ <= 0.0f) {
            show_copy_notification_ = false;
        }
    }

    if (!auth.IsAuthenticated()) {
        // Not logged in - show login form
        RenderLoginForm();
        return;
    }

    // User is logged in - show account info
    auto user_info = auth.GetUserInfo();

    if (ImGui::BeginTabBar("AccountTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_USER " Profile")) {
            RenderProfileSection();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_ID_CARD " Account Details")) {
            RenderAccountDetails();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_WALLET " Wallet")) {
            RenderWalletSection();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_SHIELD_HALVED " Security")) {
            RenderSecuritySection();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_SLIDERS " Preferences")) {
            RenderPreferencesSection();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void AccountSettingsPanel::RenderLoginForm() {
    ImGui::Spacing();

    // Header
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), ICON_FA_RIGHT_TO_BRACKET);
    ImGui::SameLine();
    ImGui::Text("Sign In to CyxWiz");
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "Sign in to access your account settings and connect to the network.");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Error message
    if (!login_error_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::Text("%s %s", ICON_FA_CIRCLE_EXCLAMATION, login_error_.c_str());
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // Success message
    if (!login_success_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.9f, 0.4f, 1.0f));
        ImGui::Text("%s %s", ICON_FA_CIRCLE_CHECK, login_success_.c_str());
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    float input_width = 300.0f;
    float button_size = 28.0f;

    // Email field
    ImGui::Text("Email");
    ImGui::SetNextItemWidth(input_width - button_size - 4.0f);
    ImGui::InputText("##login_email", login_email_, sizeof(login_email_));

    // Paste button for email
    ImGui::SameLine(0, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
    if (ImGui::Button(ICON_FA_PASTE "##paste_email", ImVec2(button_size, 0))) {
        if (ImGui::GetClipboardText()) {
            strncpy(login_email_, ImGui::GetClipboardText(), sizeof(login_email_) - 1);
            login_email_[sizeof(login_email_) - 1] = '\0';
        }
    }
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Paste from clipboard");
    }

    ImGui::Spacing();

    // Password field
    ImGui::Text("Password");
    float password_field_width = input_width - (button_size * 2) - 8.0f;
    ImGui::SetNextItemWidth(password_field_width);

    ImGuiInputTextFlags password_flags = ImGuiInputTextFlags_EnterReturnsTrue;
    if (!show_password_) {
        password_flags |= ImGuiInputTextFlags_Password;
    }
    bool enter_pressed = ImGui::InputText("##login_password", login_password_, sizeof(login_password_), password_flags);

    // Show/Hide password toggle
    ImGui::SameLine(0, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
    if (ImGui::Button(show_password_ ? ICON_FA_EYE_SLASH "##toggle_pw" : ICON_FA_EYE "##toggle_pw", ImVec2(button_size, 0))) {
        show_password_ = !show_password_;
    }
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(show_password_ ? "Hide password" : "Show password");
    }

    // Paste button for password
    ImGui::SameLine(0, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
    if (ImGui::Button(ICON_FA_PASTE "##paste_password", ImVec2(button_size, 0))) {
        if (ImGui::GetClipboardText()) {
            strncpy(login_password_, ImGui::GetClipboardText(), sizeof(login_password_) - 1);
            login_password_[sizeof(login_password_) - 1] = '\0';
        }
    }
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Paste from clipboard");
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Login button
    bool can_login = strlen(login_email_) > 0 && strlen(login_password_) > 0 && !is_logging_in_;

    if (is_logging_in_) {
        ImGui::BeginDisabled();
        ImGui::Button("Signing in...", ImVec2(input_width, 36));
        ImGui::EndDisabled();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.45f, 0.85f, 1.0f));

        if (!can_login) {
            ImGui::BeginDisabled();
        }

        if ((ImGui::Button(ICON_FA_RIGHT_TO_BRACKET " Sign In", ImVec2(input_width, 36)) || enter_pressed) && can_login) {
            // Start async login
            is_logging_in_ = true;
            login_error_.clear();
            login_success_.clear();
            auto& auth = auth::AuthManager::Instance();
            login_future_ = auth.LoginWithEmail(login_email_, login_password_);
            spdlog::info("Starting login for: {}", login_email_);
        }

        if (!can_login) {
            ImGui::EndDisabled();
        }

        ImGui::PopStyleColor(3);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Create account link
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Don't have an account?");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    if (ImGui::SmallButton("Create one")) {
        auth::AuthManager::OpenRegistrationPage();
    }
    ImGui::PopStyleColor(3);
}

void AccountSettingsPanel::RenderProfileSection() {
    auto& auth = auth::AuthManager::Instance();
    auto user_info = auth.GetUserInfo();

    ImGui::Spacing();

    // Avatar/User icon
    ImGui::BeginGroup();
    {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
        ImGui::Text(ICON_FA_CIRCLE_USER);
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::SameLine();

        // Display name
        if (!user_info.name.empty()) {
            ImGui::Text("%s", user_info.name.c_str());
        } else if (!user_info.username.empty()) {
            ImGui::Text("@%s", user_info.username.c_str());
        } else {
            ImGui::Text("%s", user_info.email.c_str());
        }
    }
    ImGui::EndGroup();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Role badge
    ImGui::Text("Account Type:");
    ImGui::SameLine();
    ImVec4 role_color;
    const char* role_icon;
    if (user_info.role == "admin") {
        role_color = ImVec4(0.9f, 0.4f, 0.4f, 1.0f);
        role_icon = ICON_FA_CROWN;
    } else if (user_info.role == "pro") {
        role_color = ImVec4(0.9f, 0.8f, 0.2f, 1.0f);
        role_icon = ICON_FA_STAR;
    } else {
        role_color = ImVec4(0.4f, 0.7f, 1.0f, 1.0f);
        role_icon = ICON_FA_USER;
    }
    ImGui::TextColored(role_color, "%s %s",
        role_icon, user_info.role.empty() ? "User" : user_info.role.c_str());

    ImGui::Spacing();

    // Quick info cards
    if (ImGui::BeginTable("ProfileInfo", 2, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        if (!user_info.username.empty()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Username");
            ImGui::TableNextColumn();
            ImGui::Text("@%s", user_info.username.c_str());
        }

        if (!user_info.email.empty()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Email");
            ImGui::TableNextColumn();
            ImGui::Text("%s", user_info.email.c_str());
        }

        if (!user_info.name.empty()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Display Name");
            ImGui::TableNextColumn();
            ImGui::Text("%s", user_info.name.c_str());
        }

        ImGui::EndTable();
    }

    // Copy notification
    if (show_copy_notification_) {
        ImGui::Spacing();
        float alpha = std::min(1.0f, copy_notification_timer_ / 0.5f);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, alpha),
            ICON_FA_CHECK " Copied %s to clipboard", copied_item_.c_str());
    }
}

void AccountSettingsPanel::RenderAccountDetails() {
    auto& auth = auth::AuthManager::Instance();
    auto user_info = auth.GetUserInfo();

    ImGui::Spacing();
    ImGui::Text("%s Account Information", ICON_FA_ID_CARD);
    ImGui::Separator();
    ImGui::Spacing();

    // Account ID
    ImGui::Text("Account ID");
    ImGui::BeginGroup();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    if (!user_info.id.empty()) {
        ImGui::TextWrapped("%s", user_info.id.c_str());
    } else {
        ImGui::TextDisabled("Not available");
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_COPY "##CopyID")) {
        CopyToClipboard(user_info.id, "Account ID");
    }
    ImGui::EndGroup();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Email
    ImGui::Text("Email Address");
    ImGui::BeginGroup();
    ImGui::Text("%s", user_info.email.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_COPY "##CopyEmail")) {
        CopyToClipboard(user_info.email, "Email");
    }
    ImGui::EndGroup();

    ImGui::Spacing();

    // Username
    ImGui::Text("Username");
    if (!user_info.username.empty()) {
        ImGui::BeginGroup();
        ImGui::Text("@%s", user_info.username.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_COPY "##CopyUsername")) {
            CopyToClipboard(user_info.username, "Username");
        }
        ImGui::EndGroup();
    } else {
        ImGui::TextDisabled("Not set");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Authentication state
    ImGui::Text("Authentication Status");
    auto state = auth.GetState();
    ImVec4 state_color;
    const char* state_text;
    const char* state_icon;

    switch (state) {
        case auth::AuthState::Connected:
            state_color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);
            state_text = "Connected";
            state_icon = ICON_FA_CIRCLE_CHECK;
            break;
        case auth::AuthState::Authenticated:
            state_color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f);
            state_text = "Authenticated (Not Connected)";
            state_icon = ICON_FA_KEY;
            break;
        case auth::AuthState::Connecting:
            state_color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f);
            state_text = "Connecting...";
            state_icon = ICON_FA_SPINNER;
            break;
        default:
            state_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
            state_text = "Unknown";
            state_icon = ICON_FA_QUESTION;
            break;
    }
    ImGui::TextColored(state_color, "%s %s", state_icon, state_text);

    // Copy notification
    if (show_copy_notification_) {
        ImGui::Spacing();
        float alpha = std::min(1.0f, copy_notification_timer_ / 0.5f);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, alpha),
            ICON_FA_CHECK " Copied %s to clipboard", copied_item_.c_str());
    }
}

void AccountSettingsPanel::RenderWalletSection() {
    auto& auth = auth::AuthManager::Instance();
    auto user_info = auth.GetUserInfo();

    ImGui::Spacing();
    ImGui::Text("%s Linked Wallet", ICON_FA_WALLET);
    ImGui::Separator();
    ImGui::Spacing();

    if (!user_info.wallet_address.empty()) {
        // Wallet is linked
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f),
            ICON_FA_LINK " Wallet Connected");
        ImGui::Spacing();

        ImGui::Text("Wallet Address (Solana)");
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::BeginChild("WalletAddressBox", ImVec2(-1, 50), true);
        {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "%s", user_info.wallet_address.c_str());
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        // Action buttons
        if (ImGui::Button(ICON_FA_COPY " Copy Address")) {
            CopyToClipboard(user_info.wallet_address, "Wallet Address");
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " View on Solscan")) {
            // Open Solscan URL in browser
            std::string url = "https://solscan.io/account/" + user_info.wallet_address;
#ifdef _WIN32
            std::string cmd = "start " + url;
#elif __APPLE__
            std::string cmd = "open " + url;
#else
            std::string cmd = "xdg-open " + url;
#endif
            std::system(cmd.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "This wallet is linked to your CyxWiz account and will be used for:");
        ImGui::BulletText("Receiving training job earnings");
        ImGui::BulletText("Marketplace transactions");
        ImGui::BulletText("Pool mining rewards");

        ImGui::Spacing();
        ImGui::TextDisabled("To change your wallet, visit the CyxWiz web dashboard.");
    } else {
        // No wallet linked
        ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.2f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " No Wallet Linked");
        ImGui::Spacing();

        ImGui::TextWrapped(
            "You haven't linked a Solana wallet to your account yet. "
            "A wallet is required to receive earnings from training jobs "
            "and participate in the marketplace.");

        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Link Wallet on Web Dashboard")) {
            // Open web dashboard
#ifdef _WIN32
            std::system("start https://cyxwiz.com/dashboard/wallet");
#elif __APPLE__
            std::system("open https://cyxwiz.com/dashboard/wallet");
#else
            std::system("xdg-open https://cyxwiz.com/dashboard/wallet");
#endif
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Supported wallets: Phantom, Solflare, Backpack");
    }

    // Copy notification
    if (show_copy_notification_) {
        ImGui::Spacing();
        float alpha = std::min(1.0f, copy_notification_timer_ / 0.5f);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, alpha),
            ICON_FA_CHECK " Copied %s to clipboard", copied_item_.c_str());
    }
}

void AccountSettingsPanel::RenderSecuritySection() {
    auto& auth = auth::AuthManager::Instance();

    ImGui::Spacing();
    ImGui::Text("%s Security Settings", ICON_FA_SHIELD_HALVED);
    ImGui::Separator();
    ImGui::Spacing();

    // Session info
    ImGui::Text("Current Session");
    bool has_token = !auth.GetJwtToken().empty();
    if (has_token) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f),
            ICON_FA_CIRCLE_CHECK " Active session");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            ICON_FA_CIRCLE_XMARK " No active session");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Refresh token
    ImGui::Text("Token Management");
    if (ImGui::Button(ICON_FA_ROTATE " Refresh Token")) {
        auth.RefreshJwtToken();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Extend your session");

    ImGui::Spacing();

    // Logout button
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
    if (ImGui::Button(ICON_FA_RIGHT_FROM_BRACKET " Sign Out", ImVec2(150, 35))) {
        auth.Logout();
    }
    ImGui::PopStyleColor(2);
    ImGui::SameLine();
    ImGui::TextDisabled("End current session");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Password change link
    ImGui::Text("Password");
    if (ImGui::Button(ICON_FA_KEY " Change Password")) {
#ifdef _WIN32
        std::system("start https://cyxwiz.com/dashboard/security");
#elif __APPLE__
        std::system("open https://cyxwiz.com/dashboard/security");
#else
        std::system("xdg-open https://cyxwiz.com/dashboard/security");
#endif
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Opens in web browser");

    ImGui::Spacing();

    // Two-factor authentication
    ImGui::Text("Two-Factor Authentication");
    if (ImGui::Button(ICON_FA_MOBILE_SCREEN " Manage 2FA")) {
#ifdef _WIN32
        std::system("start https://cyxwiz.com/dashboard/security/2fa");
#elif __APPLE__
        std::system("open https://cyxwiz.com/dashboard/security/2fa");
#else
        std::system("xdg-open https://cyxwiz.com/dashboard/security/2fa");
#endif
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Opens in web browser");
}

void AccountSettingsPanel::RenderPreferencesSection() {
    ImGui::Spacing();
    ImGui::Text("%s Preferences", ICON_FA_SLIDERS);
    ImGui::Separator();
    ImGui::Spacing();

    // Notification preferences
    ImGui::Text("Notifications");
    static bool email_notifications = true;
    static bool job_completion_alerts = true;
    static bool earnings_alerts = true;

    ImGui::Checkbox("Email notifications", &email_notifications);
    ImGui::Checkbox("Job completion alerts", &job_completion_alerts);
    ImGui::Checkbox("Earnings alerts", &earnings_alerts);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Display preferences
    ImGui::Text("Display");
    static int currency_display = 0;
    const char* currencies[] = { "CYXWIZ", "USD", "EUR" };
    ImGui::Combo("Earnings Display", &currency_display, currencies, IM_ARRAYSIZE(currencies));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Privacy
    ImGui::Text("Privacy");
    static bool show_online_status = true;
    static bool show_earnings_public = false;

    ImGui::Checkbox("Show online status", &show_online_status);
    ImGui::Checkbox("Show earnings publicly", &show_earnings_public);

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Preferences")) {
        // TODO: Save preferences to server
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Some settings may require refresh to take effect.");
}

void AccountSettingsPanel::CopyToClipboard(const std::string& text, const char* label) {
    if (!text.empty()) {
        ImGui::SetClipboardText(text.c_str());
        show_copy_notification_ = true;
        copy_notification_timer_ = 2.0f;
        copied_item_ = label;
    }
}

} // namespace cyxwiz::servernode::gui
