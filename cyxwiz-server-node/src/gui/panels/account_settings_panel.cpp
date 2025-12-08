// account_settings_panel.cpp
#include "gui/panels/account_settings_panel.h"
#include "gui/icons.h"
#include "auth/auth_manager.h"
#include <imgui.h>

namespace cyxwiz::servernode::gui {

void AccountSettingsPanel::Render() {
    auto& auth = auth::AuthManager::Instance();

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
        // Not logged in
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "You are not logged in.");
        ImGui::Spacing();
        ImGui::TextWrapped("Please sign in to view and manage your account settings.");
        ImGui::Spacing();
        ImGui::TextDisabled("Go to the login panel to sign in with your CyxWiz account.");
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
