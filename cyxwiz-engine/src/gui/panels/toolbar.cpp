#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include "../../auth/auth_client.h"
#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/cyxwiz.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <chrono>
#include <initializer_list>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../../core/route_qualification_service.h"
#include "../icons.h"

namespace cyxwiz {

ToolbarPanel::ToolbarPanel()
    : Panel("Toolbar", true)
    , show_about_dialog_(false)
    , route_qualification_service_(
          std::make_shared<RouteQualificationService>())
{

}

void ToolbarPanel::SetEditorFontScale(float scale) {
    // Convert editor scale to the native atlas font size shown in Preferences
    // 1.0 -> 14px, 1.3 -> 16px, 1.6 -> 20px, 2.0 -> 24px
    if (scale <= 1.15f) editor_font_size_ = 14;
    else if (scale <= 1.45f) editor_font_size_ = 16;
    else if (scale <= 1.8f) editor_font_size_ = 20;
    else editor_font_size_ = 24;
}

void ToolbarPanel::Render() {
    if (!visible_) return;

    // Check for session restore (runs once on first render)
    if (session_restore_pending_) {
        session_restore_pending_ = false;
        auto& auth = auth::AuthClient::Instance();
        if (auth.LoadSavedSession()) {
            is_logged_in_ = true;
            auto user = auth.GetUserInfo();
            logged_in_user_ = user.email.empty() ? user.username : user.email;
            spdlog::info("Restored saved session for: {}", logged_in_user_);
            // Notify application of restored session with JWT token
            if (on_login_success_callback_) {
                on_login_success_callback_(auth.GetJwtToken());
            }
        }
    }

    // Check if async login completed
    if (login_future_.valid()) {
        auto status = login_future_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            auto result = login_future_.get();
            is_logging_in_ = false;

            if (result.success) {
                is_logged_in_ = true;
                login_error_message_.clear();
                auto user = result.user_info;
                logged_in_user_ = user.email.empty() ? user.username : user.email;
                login_success_message_ = "Login successful!";
                spdlog::info("Login successful: {}", logged_in_user_);
                memset(login_password_, 0, sizeof(login_password_));
                // Close login dialog on success
                // Notify application of successful login with JWT token
                show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    auto& auth = auth::AuthClient::Instance();
                    on_login_success_callback_(auth.GetJwtToken());
                }
            } else {
                login_error_message_ = result.error;
                login_success_message_.clear();
                spdlog::error("Login failed: {}", result.error);
            }
        }
    }

    // Main menu bar, drawn from the menu presentation model (TOFIX129).
    RenderMenuBar();

    RenderProjectDialogs();

    HandleAutoSaveTimer();

    RenderEditorDialogs();

    // Render command palette overlay
    RenderCommandPalette();
}
} // namespace cyxwiz
