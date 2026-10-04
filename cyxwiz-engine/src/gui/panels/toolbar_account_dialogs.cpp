// Toolbar account modal rendering.

#include "toolbar.h"
#include "../icons.h"
#include "../../auth/auth_client.h"
#include "../../core/engine_config.h"

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {
void ToolbarPanel::RenderAccountDialogs() {
    // Account Settings dialog
    if (show_account_settings_dialog_) {
        ImGui::OpenPopup("##AccountSettings");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        // Professional styling
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 24));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;

        if (ImGui::BeginPopupModal("##AccountSettings", &show_account_settings_dialog_, flags)) {

            if (!is_logged_in_) {
                // ========== Sign In View ==========

                // Logo/Brand area
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);  // Use default font
                float window_width = ImGui::GetWindowWidth();

                // Center the title
                const char* title = "CyxWiz";
                float title_width = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX((window_width - title_width) * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                ImGui::Text("%s", title);
                ImGui::PopStyleColor();
                ImGui::PopFont();

                ImGui::Spacing();

                // Subtitle
                const char* subtitle = "Sign in to your account";
                float subtitle_width = ImGui::CalcTextSize(subtitle).x;
                ImGui::SetCursorPosX((window_width - subtitle_width) * 0.5f);
                ImGui::TextDisabled("%s", subtitle);

                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();

                // Input fields with consistent width
                float input_width = 320.0f;
                float start_x = (window_width - input_width) * 0.5f;

                // Email or Phone field
                ImGui::SetCursorPosX(start_x);
                ImGui::Text("Email or Phone");
                ImGui::SetCursorPosX(start_x);

                float button_size = 28.0f;
                float field_width = input_width - button_size - 4.0f;

                ImGui::SetNextItemWidth(field_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                ImGui::InputText("##identifier", login_identifier_, sizeof(login_identifier_));
                ImGui::PopStyleColor(2);

                // Paste button for email
                ImGui::SameLine(0, 4.0f);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
                if (ImGui::Button(ICON_FA_PASTE "##paste_email", ImVec2(button_size, 0))) {
                    if (ImGui::GetClipboardText()) {
                        strncpy(login_identifier_, ImGui::GetClipboardText(), sizeof(login_identifier_) - 1);
                        login_identifier_[sizeof(login_identifier_) - 1] = '\0';
                    }
                }
                ImGui::PopStyleColor(2);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Paste from clipboard");
                }

                ImGui::Spacing();

                // Password field
                ImGui::SetCursorPosX(start_x);
                ImGui::Text("Password");
                ImGui::SetCursorPosX(start_x);

                float password_field_width = input_width - (button_size * 2) - 8.0f;
                ImGui::SetNextItemWidth(password_field_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                ImGuiInputTextFlags password_flags = ImGuiInputTextFlags_EnterReturnsTrue;
                if (!show_password_) {
                    password_flags |= ImGuiInputTextFlags_Password;
                }
                bool enter_pressed = ImGui::InputText("##password", login_password_, sizeof(login_password_), password_flags);
                ImGui::PopStyleColor(2);

                // Show/Hide password toggle button
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

                // Error message
                if (!login_error_message_.empty()) {
                    ImGui::Spacing();
                    ImGui::SetCursorPosX(start_x);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui::Text("%s", login_error_message_.c_str());
                    ImGui::PopStyleColor();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                // Sign In button
                ImGui::SetCursorPosX(start_x);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.45f, 0.85f, 1.0f));

                bool can_login = strlen(login_identifier_) > 0 && strlen(login_password_) > 0 && !is_logging_in_;

                if (is_logging_in_) {
                    // Show loading state
                    ImGui::BeginDisabled();
                    ImGui::Button("Signing in...", ImVec2(input_width, 38));
                    ImGui::EndDisabled();
                } else if ((ImGui::Button("Sign In", ImVec2(input_width, 38)) || enter_pressed) && can_login) {
                    // Validate input
                    std::string identifier = login_identifier_;
                    std::string password = login_password_;

                    if (identifier.empty()) {
                        login_error_message_ = "Please enter your email or phone number";
                    } else if (password.empty()) {
                        login_error_message_ = "Please enter your password";
                    } else {
                        // Auto-detect if email or phone
                        bool is_email = identifier.find('@') != std::string::npos;

                        if (is_email) {
                            // Start async login
                            is_logging_in_ = true;
                            login_error_message_.clear();
                            login_success_message_.clear();
                            auto& auth = auth::AuthClient::Instance();
                            login_future_ = auth.LoginWithEmail(identifier, password);
                            spdlog::info("Starting login for: {}", identifier);
                        } else {
                            login_error_message_ = "Please enter a valid email address";
                        }
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();
                ImGui::Spacing();

                // Links row
                ImGui::SetCursorPosX(start_x);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));

                ImGui::Text("Forgot password?");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                    ImGui::SetTooltip("Reset your password");
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemClicked()) {
                    spdlog::info("Forgot password clicked");
                }

                ImGui::SameLine();
                ImGui::SetCursorPosX(start_x + input_width - ImGui::CalcTextSize("Create account").x);

                ImGui::Text("Create account");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                    ImGui::SetTooltip("Sign up for CyxWiz");
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemClicked()) {
                    auth::AuthClient::OpenRegistrationPage();
                }

                ImGui::PopStyleColor();

            } else {
                // ========== Logged In View ==========
                auto& auth = auth::AuthClient::Instance();
                auto user = auth.GetUserInfo();

                // Get initials from name
                std::string initials;
                if (!user.name.empty()) {
                    std::istringstream iss(user.name);
                    std::string word;
                    while (iss >> word && initials.length() < 2) {
                        if (!word.empty()) {
                            initials += static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
                        }
                    }
                }
                if (initials.empty() && !user.email.empty()) {
                    initials = std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(user.email[0]))));
                }
                if (initials.empty()) initials = "U";

                // Header with user avatar placeholder
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                ImGui::Text(ICON_FA_USER "  Account Settings");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                // User card with better info
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##UserCard", ImVec2(340, 90), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 12));

                // Avatar circle with initials
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float avatar_radius = 28.0f;
                ImVec2 avatar_center(pos.x + avatar_radius, pos.y + avatar_radius);
                draw_list->AddCircleFilled(avatar_center, avatar_radius, IM_COL32(40, 80, 160, 255));

                // Draw initials centered
                float font_scale = 1.4f;
                ImVec2 text_size = ImGui::CalcTextSize(initials.c_str());
                ImVec2 text_pos(avatar_center.x - text_size.x * font_scale * 0.5f,
                              avatar_center.y - text_size.y * font_scale * 0.5f);
                draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize() * font_scale,
                                  text_pos, IM_COL32(255, 255, 255, 255), initials.c_str());

                // User info next to avatar
                ImGui::SetCursorPos(ImVec2(72, 12));

                // Name (or username if no name)
                std::string display_name = user.name.empty() ? user.username : user.name;
                if (display_name.empty()) display_name = "User";
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                ImGui::Text("%s", display_name.c_str());
                ImGui::PopStyleColor();

                // Email
                ImGui::SetCursorPos(ImVec2(72, 32));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
                ImGui::Text("%s", user.email.c_str());
                ImGui::PopStyleColor();

                // Role badge
                ImGui::SetCursorPos(ImVec2(72, 54));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
                ImGui::Text(ICON_FA_CIRCLE_CHECK);
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
                std::string role_display = user.role.empty() ? "User" : user.role;
                role_display[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(role_display[0])));
                ImGui::Text("%s", role_display.c_str());
                ImGui::PopStyleColor();

                ImGui::EndChild();
                ImGui::PopStyleColor();

                ImGui::Spacing();
                ImGui::Spacing();

                // Server Section
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
                ImGui::Text(ICON_FA_SERVER "  SERVER");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##ServerCard", ImVec2(340, 65), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 10));
                ImGui::Text("Default Server");
                ImGui::SetCursorPos(ImVec2(12, 32));

                static char server_address[256] = "";
                static bool server_address_initialized = false;
                if (!server_address_initialized) {
                    std::strncpy(server_address, core::EngineConfig::Instance().GetCentralServerAddress().c_str(), sizeof(server_address) - 1);
                    server_address[sizeof(server_address) - 1] = '\0';
                    server_address_initialized = true;
                }
                ImGui::SetNextItemWidth(316);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.10f, 1.0f));
                ImGui::InputText("##server", server_address, sizeof(server_address));
                ImGui::PopStyleColor();

                ImGui::EndChild();
                ImGui::PopStyleColor();

                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();

                // Sign Out button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.18f, 0.18f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.15f, 0.15f, 1.0f));
                if (ImGui::Button("Sign Out", ImVec2(340, 36))) {
                    auth.Logout();
                    is_logged_in_ = false;
                    logged_in_user_.clear();
                    memset(login_identifier_, 0, sizeof(login_identifier_));
                    memset(login_password_, 0, sizeof(login_password_));
                    login_error_message_.clear();
                    login_success_message_.clear();
                    spdlog::info("User signed out");
                    // Notify application of logout
                    if (on_logout_callback_) {
                        on_logout_callback_();
                    }
                }
                ImGui::PopStyleColor(3);
            }

            ImGui::Spacing();

            // Close button (subtle, bottom right)
            float close_width = 70;
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - close_width - 24);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            if (ImGui::Button("Close", ImVec2(close_width, 28))) {
                show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    auto& auth = auth::AuthClient::Instance();
                    on_login_success_callback_(auth.GetJwtToken());
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(5);
    }
}

} // namespace cyxwiz
