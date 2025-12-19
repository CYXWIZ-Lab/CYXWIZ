#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include "../tutorial/tutorial_system.h"
#include "../../auth/auth_client.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <sstream>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../icons.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

void ToolbarPanel::RenderNodesMenu() {
    if (ImGui::BeginMenu("Nodes")) {
        if (ImGui::BeginMenu("Add Layer")) {
            if (ImGui::MenuItem("Dense/Linear")) {
                // TODO: Add dense layer
            }
            if (ImGui::MenuItem("Convolutional")) {
                // TODO: Add conv layer
            }
            if (ImGui::MenuItem("Pooling")) {
                // TODO: Add pooling layer
            }
            if (ImGui::MenuItem("Dropout")) {
                // TODO: Add dropout
            }
            if (ImGui::MenuItem("Batch Normalization")) {
                // TODO: Add batch norm
            }
            if (ImGui::MenuItem("Attention")) {
                // TODO: Add attention
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Group Selected", "Ctrl+G")) {
            // TODO: Group nodes
        }

        if (ImGui::MenuItem("Ungroup", "Ctrl+Shift+G")) {
            // TODO: Ungroup nodes
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
            // TODO: Duplicate selected
        }

        if (ImGui::MenuItem("Delete Selected", "Delete")) {
            // TODO: Delete selected nodes
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_WAND_MAGIC_SPARKLES " Custom Node Editor...")) {
            // Signal to open Custom Node Editor panel
            // This is handled via callback in MainWindow
            if (open_custom_node_editor_callback_) {
                open_custom_node_editor_callback_();
            }
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderTrainMenu() {
    if (ImGui::BeginMenu("Train")) {
        if (ImGui::MenuItem("Start Training", "F5")) {
            // TODO: Start training
        }

        if (ImGui::MenuItem("Pause", "F6")) {
            // TODO: Pause training
        }

        if (ImGui::MenuItem("Stop", "Shift+F5")) {
            // TODO: Stop training
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Training Settings...")) {
            // TODO: Open training settings
        }

        if (ImGui::MenuItem("Optimizer Settings...")) {
            // TODO: Open optimizer settings
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDatasetMenu() {
    if (ImGui::BeginMenu("Dataset")) {
        if (ImGui::MenuItem("Import Dataset...")) {
            if (import_dataset_callback_) {
                import_dataset_callback_();
            }
        }

        if (ImGui::MenuItem("Create Custom Dataset...")) {
            // TODO: Create dataset
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Preprocess...")) {
            // TODO: Preprocess dataset
        }

        if (ImGui::MenuItem("Tokenize...")) {
            // TODO: Tokenize dataset
        }

        if (ImGui::MenuItem("Augment...")) {
            // TODO: Data augmentation
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Dataset Statistics")) {
            // TODO: Show statistics
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderScriptMenu() {
    if (ImGui::BeginMenu("Script")) {
        if (ImGui::MenuItem("Open Python Console", "F12")) {
            // TODO: Show Python console
        }

        ImGui::Separator();

        if (ImGui::MenuItem("New Script...")) {
            // TODO: Create new script
        }

        if (ImGui::MenuItem("Run Script...", "Ctrl+R")) {
            // TODO: Run script
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Script Editor...")) {
            // TODO: Open script editor
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderPlotsMenu() {
    if (ImGui::BeginMenu("Plots")) {
        // Test Control - Interactive option
        if (ImGui::MenuItem("Test Control")) {
            if (toggle_plot_test_control_callback_) {
                toggle_plot_test_control_callback_();
            }
        }

        ImGui::Separator();
        ImGui::TextDisabled("Available Plot Types (View Only)");
        ImGui::Separator();

        // Basic 2D Plots (from cheatsheet)
        if (ImGui::BeginMenu("Basic 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot() - Line plot");
        ImGui::TextDisabled("scatter() - Scatter plot");
        ImGui::TextDisabled("bar() / barh() - Bar chart");
        ImGui::TextDisabled("imshow() - Image display");
        ImGui::TextDisabled("contour() / contourf() - Contour plot");
        ImGui::TextDisabled("pcolormesh() - Pseudocolor plot");
        ImGui::TextDisabled("quiver() - Vector field");
        ImGui::TextDisabled("pie() - Pie chart");
        ImGui::TextDisabled("fill_between() - Filled area");
        ImGui::Unindent();

        ImGui::Separator();

        // Advanced 2D Plots
        if (ImGui::BeginMenu("Advanced 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("step() - Step plot");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("errorbar() - Error bar plot");
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("barbs() - Barbs plot");
        ImGui::TextDisabled("eventplot() - Event plot");
        ImGui::TextDisabled("hexbin() - Hexagonal binning");
        ImGui::Unindent();

        ImGui::Separator();

        // 3D Plots
        if (ImGui::BeginMenu("3D Plots", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot3D() - 3D line plot");
        ImGui::TextDisabled("scatter3D() - 3D scatter");
        ImGui::TextDisabled("plot_surface() - Surface plot");
        ImGui::TextDisabled("plot_wireframe() - Wireframe");
        ImGui::TextDisabled("contour3D() - 3D contour");
        ImGui::Unindent();

        ImGui::Separator();

        // Polar Plots
        if (ImGui::BeginMenu("Polar", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("polar() - Polar plot");
        ImGui::Unindent();

        ImGui::Separator();

        // Statistical Plots
        if (ImGui::BeginMenu("Statistical", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("kde plot - Density estimation");
        ImGui::Unindent();

        ImGui::Separator();

        // Specialized Plots
        if (ImGui::BeginMenu("Specialized", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("heatmap - Heat map");
        ImGui::TextDisabled("streamplot() - Stream plot");
        ImGui::TextDisabled("specgram() - Spectrogram");
        ImGui::TextDisabled("spy() - Sparse matrix viz");
        ImGui::Unindent();

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDeployMenu() {
    if (ImGui::BeginMenu("Deploy")) {
        if (ImGui::MenuItem("Connect to Server...")) {
            if (!is_logged_in_) {
                // User not logged in - show warning popup
                show_login_required_popup_ = true;
                login_required_action_ = "connect to the server";
            } else if (connect_to_server_callback_) {
                connect_to_server_callback_();
            }
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Export Model")) {
            if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " CyxWiz Model (.cyxmodel)")) {
                if (export_model_callback_) {
                    export_model_callback_(0);  // 0 = CyxModel
                }
            }
            if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Safetensors (.safetensors)")) {
                if (export_model_callback_) {
                    export_model_callback_(1);  // 1 = Safetensors
                }
            }
            if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " ONNX Format (.onnx)")) {
                if (export_model_callback_) {
                    export_model_callback_(2);  // 2 = ONNX
                }
            }
            if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " GGUF Format (.gguf)")) {
                if (export_model_callback_) {
                    export_model_callback_(3);  // 3 = GGUF
                }
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Quantize")) {
            if (ImGui::MenuItem("INT8")) {
                // TODO: Quantize INT8
            }
            if (ImGui::MenuItem("INT4")) {
                // TODO: Quantize INT4
            }
            if (ImGui::MenuItem("FP16")) {
                // TODO: Quantize FP16
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_ROCKET " Deploy to Server Node...")) {
            if (!is_logged_in_) {
                // User not logged in - show warning popup
                show_login_required_popup_ = true;
                login_required_action_ = "deploy to a server node";
            } else if (deploy_to_server_callback_) {
                deploy_to_server_callback_();
            }
        }

        if (ImGui::MenuItem("Publish to Marketplace...")) {
            // TODO: Publish model
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderHelpMenu() {
    if (ImGui::BeginMenu("Help")) {
        // ========== Interactive Tutorials ==========
        // Push solid opaque style for better readability
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.12f, 0.18f, 1.0f));  // Fully opaque dark background
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));        // Bright white text
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));      // Blue border
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 2.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 6.0f);

        if (ImGui::BeginMenu(ICON_FA_LIGHTBULB " Interactive Tutorials")) {
            auto& tutorial_system = TutorialSystem::Instance();
            const auto& tutorials = tutorial_system.GetAvailableTutorials();

            for (const auto& tutorial : tutorials) {
                bool completed = tutorial_system.IsTutorialComplete(tutorial.id);
                std::string label = tutorial.name;
                if (completed) {
                    label += " " ICON_FA_CHECK;
                }

                if (ImGui::MenuItem(label.c_str())) {
                    tutorial_system.StartTutorial(tutorial.id);
                    spdlog::info("Started tutorial: {}", tutorial.name);
                }

                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.15f, 0.15f, 0.22f, 1.0f));
                    ImGui::TextUnformatted(tutorial.description.c_str());
                    ImGui::PopStyleColor();
                    ImGui::EndTooltip();
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem(ICON_FA_LIST_CHECK " Browse All Tutorials...")) {
                tutorial_system.OpenTutorialBrowser();
            }

            ImGui::EndMenu();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_BOOKMARK " Documentation", "F1")) {
            // TODO: Open docs
        }

        if (ImGui::MenuItem(ICON_FA_KEYBOARD " Keyboard Shortcuts")) {
            // TODO: Show shortcuts
        }

        if (ImGui::MenuItem(ICON_FA_CODE " API Reference")) {
            // TODO: Open API docs
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_BUG " Report Issue...")) {
            // TODO: Open issue tracker
        }

        if (ImGui::MenuItem(ICON_FA_DOWNLOAD " Check for Updates...")) {
            // TODO: Check updates
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " About CyxWiz")) {
            show_about_dialog_ = true;
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type) {
    // Create new plot window with auto-generated data
    auto plot_window = std::make_shared<PlotWindow>(title, type, true);
    plot_windows_.push_back(plot_window);
    spdlog::info("Created new plot window: {}", title);
}

void ToolbarPanel::RenderUserAvatar() {
    // Calculate button/avatar size
    float avatar_size = 20.0f;
    const char* login_text = ICON_FA_USER " Login";
    ImVec2 login_button_size = ImGui::CalcTextSize(login_text);
    login_button_size.x += 16.0f;  // Padding for button

    // Get available width and calculate right-aligned position
    float content_region_width = ImGui::GetContentRegionAvail().x;
    float item_width = is_logged_in_ ? avatar_size : login_button_size.x;

    // Add spacing and move cursor to right-align
    ImGui::SameLine(ImGui::GetCursorPosX() + content_region_width - item_width - 8.0f);

    if (is_logged_in_) {
        auto& auth_client = auth::AuthClient::Instance();
        auto user = auth_client.GetUserInfo();

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

        // Draw avatar button
        ImVec2 screen_pos = ImGui::GetCursorScreenPos();
        ImVec2 center(screen_pos.x + avatar_size / 2, screen_pos.y + avatar_size / 2);

        ImGui::InvisibleButton("##UserAvatar", ImVec2(avatar_size, avatar_size));
        bool hovered = ImGui::IsItemHovered();
        bool clicked = ImGui::IsItemClicked();

        // Draw avatar circle
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImU32 bg_color = hovered ? IM_COL32(80, 120, 200, 255) : IM_COL32(50, 90, 170, 255);
        draw_list->AddCircleFilled(center, avatar_size / 2, bg_color);

        // Draw initials
        ImVec2 text_size = ImGui::CalcTextSize(initials.c_str());
        ImVec2 text_pos(center.x - text_size.x / 2, center.y - text_size.y / 2);
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), initials.c_str());

        if (clicked) {
            show_user_profile_popup_ = !show_user_profile_popup_;
            avatar_popup_x_ = screen_pos.x;
            if (show_user_profile_popup_) {
                popup_open_frames_ = 0;  // Reset frame counter when opening
            }
        }

        if (hovered) {
            ImGui::SetTooltip("%s", user.name.empty() ? user.email.c_str() : user.name.c_str());
        }
    } else {
        // Show login button when not logged in
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.7f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 2));
        if (ImGui::Button(login_text)) {
            show_account_settings_dialog_ = true;
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }
}

void ToolbarPanel::RenderUserProfilePopup() {
    auto& auth_client = auth::AuthClient::Instance();
    auto user = auth_client.GetUserInfo();

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

    float popup_width = 320.0f;
    float popup_height = 240.0f;

    // Position below avatar on the right side
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetIO().DisplaySize.x - popup_width - 10, 28),
        ImGuiCond_Always
    );
    ImGui::SetNextWindowSize(ImVec2(popup_width, popup_height));

    // Window styling
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.25f, 0.25f, 0.3f, 1.0f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##UserProfileWindow", &show_user_profile_popup_, flags)) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 window_pos = ImGui::GetWindowPos();

        // Header background
        draw_list->AddRectFilled(
            window_pos,
            ImVec2(window_pos.x + popup_width, window_pos.y + 36),
            IM_COL32(20, 20, 25, 255),
            8.0f, ImDrawFlags_RoundCornersTop
        );

        // Header: CyxWiz Account with icon
        ImGui::SetCursorPos(ImVec2(12, 8));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
        ImGui::Text(ICON_FA_MICROCHIP);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.95f, 1.0f));
        ImGui::Text("CyxWiz Account");
        ImGui::PopStyleColor();

        // Main content area
        ImGui::SetCursorPos(ImVec2(16, 48));

        // Large avatar circle
        float avatar_size = 56.0f;
        ImVec2 avatar_pos = ImGui::GetCursorScreenPos();
        ImVec2 avatar_center(avatar_pos.x + avatar_size * 0.5f, avatar_pos.y + avatar_size * 0.5f);

        // Avatar background
        draw_list->AddCircleFilled(avatar_center, avatar_size * 0.5f, IM_COL32(40, 80, 160, 255));

        // Draw initials
        float font_scale = 1.6f;
        ImVec2 text_size = ImGui::CalcTextSize(initials.c_str());
        ImVec2 text_pos(avatar_center.x - text_size.x * font_scale * 0.5f,
                       avatar_center.y - text_size.y * font_scale * 0.5f);
        draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize() * font_scale,
                          text_pos, IM_COL32(255, 255, 255, 255), initials.c_str());

        ImGui::Dummy(ImVec2(avatar_size, avatar_size));
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);

        // User info next to avatar
        ImGui::BeginGroup();

        // Name (bold white)
        std::string display_name = user.name.empty() ? user.username : user.name;
        if (display_name.empty()) display_name = "User";
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::Text("%s", display_name.c_str());
        ImGui::PopStyleColor();

        // Email (gray)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.55f, 0.6f, 1.0f));
        ImGui::Text("%s", user.email.c_str());
        ImGui::PopStyleColor();

        // Status with green checkmark
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 0.4f, 1.0f));
        ImGui::Text(ICON_FA_CIRCLE_CHECK);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
        ImGui::Text("Connected");
        ImGui::PopStyleColor();

        ImGui::EndGroup();

        // Wallet address (if available)
        if (!user.wallet_address.empty()) {
            ImGui::SetCursorPos(ImVec2(16, 112));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
            ImGui::Text(ICON_FA_WALLET);
            ImGui::PopStyleColor();
            ImGui::SameLine();

            // Truncate wallet address for display
            std::string wallet_display = user.wallet_address;
            if (wallet_display.length() > 16) {
                wallet_display = wallet_display.substr(0, 6) + "..." + wallet_display.substr(wallet_display.length() - 4);
            }
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
            ImGui::Text("%s", wallet_display.c_str());
            ImGui::PopStyleColor();
        }

        // Account settings link
        ImGui::SetCursorPos(ImVec2(16, 138));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.6f, 1.0f, 1.0f));
        if (ImGui::Selectable("Account settings", false, 0, ImVec2(popup_width - 32, 20))) {
            show_account_settings_dialog_ = true;
            show_user_profile_popup_ = false;
        }
        ImGui::PopStyleColor();

        // Separator line
        ImGui::SetCursorPos(ImVec2(0, 168));
        draw_list->AddLine(
            ImVec2(window_pos.x, window_pos.y + 168),
            ImVec2(window_pos.x + popup_width, window_pos.y + 168),
            IM_COL32(50, 50, 55, 255)
        );

        // Sign out option
        ImGui::SetCursorPos(ImVec2(16, 180));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
        if (ImGui::Selectable(ICON_FA_RIGHT_FROM_BRACKET "  Sign out", false, 0, ImVec2(popup_width - 32, 24))) {
            auth_client.Logout();
            is_logged_in_ = false;
            logged_in_user_.clear();
            show_user_profile_popup_ = false;
            spdlog::info("User signed out");
        }
        ImGui::PopStyleColor();

        // Increment frame counter each frame popup is open
        popup_open_frames_++;

        // Close if clicked outside (but wait a few frames to avoid closing from the same click that opened it)
        if (popup_open_frames_ > 3 &&
            !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            show_user_profile_popup_ = false;
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

} // namespace cyxwiz
