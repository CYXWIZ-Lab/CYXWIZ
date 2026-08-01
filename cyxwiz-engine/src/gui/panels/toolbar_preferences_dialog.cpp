// Toolbar preferences modal rendering.

#include "toolbar.h"
#include "../icons.h"

#include <cstring>
#include <exception>
#include <string>

#include <cyxwiz/cyxwiz.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {
void ToolbarPanel::RenderPreferencesDialog() {
    // ========== Preferences Dialog ==========
    if (show_preferences_dialog_) {
        // Note: shortcuts_ is initialized in RenderEditMenu() when Preferences is clicked

        // Python settings have been moved to PythonSettingsPanel
        // This initialization code is no longer needed

        ImGui::OpenPopup("Preferences");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(650, 500), ImGuiCond_Appearing);

        if (ImGui::BeginPopupModal("Preferences", &show_preferences_dialog_)) {
            // Tab bar for different preference sections
            if (ImGui::BeginTabBar("PreferenceTabs")) {

                // ========== General Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_GEAR " General")) {
                    preferences_tab_ = 0;
                    ImGui::Spacing();

                    ImGui::Text("Startup");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Restore last session on startup", &general_restore_last_session_);
                    ImGui::Checkbox("Check for updates on startup", &general_check_updates_);

                    ImGui::Spacing();
                    ImGui::Text("Recent Files Limit:");
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##recent_limit", &general_recent_files_limit_);
                    if (general_recent_files_limit_ < 1) general_recent_files_limit_ = 1;
                    if (general_recent_files_limit_ > 50) general_recent_files_limit_ = 50;

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Exit Behavior");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Confirm before exit with unsaved changes", &general_confirm_on_exit_);

                    ImGui::EndTabItem();
                }

                // ========== Editor Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_PEN " Editor")) {
                    preferences_tab_ = 1;
                    ImGui::Spacing();

                    ImGui::Text("Theme & Colors");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Editor Theme:");
                    ImGui::SetNextItemWidth(200);
                    const char* theme_items[] = { "Dark", "Light", "Retro Blue", "Monokai", "Dracula", "One Dark", "GitHub" };
                    int prev_theme = editor_theme_;
                    if (ImGui::Combo("##editor_theme", &editor_theme_, theme_items, IM_ARRAYSIZE(theme_items))) {
                        if (editor_theme_callback_ && editor_theme_ != prev_theme) {
                            editor_theme_callback_(editor_theme_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Font & Display");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Font Size:");
                    ImGui::SetNextItemWidth(200);
                    const char* font_size_items[] = { "Small (14 px)", "Medium (16 px)", "Large (20 px)", "Extra Large (24 px)" };
                    int font_size_index = 1;  // Default to Medium
                    if (editor_font_size_ <= 14) font_size_index = 0;
                    else if (editor_font_size_ <= 16) font_size_index = 1;
                    else if (editor_font_size_ <= 20) font_size_index = 2;
                    else font_size_index = 3;

                    if (ImGui::Combo("##font_size", &font_size_index, font_size_items, IM_ARRAYSIZE(font_size_items))) {
                        float scales[] = { 1.0f, 1.3f, 1.6f, 2.0f };
                        int sizes[] = { 14, 16, 20, 24 };
                        editor_font_size_ = sizes[font_size_index];
                        if (editor_font_scale_callback_) {
                            editor_font_scale_callback_(scales[font_size_index]);
                        }
                    }

                    ImGui::Spacing();

                    ImGui::Text("Tab Size:");
                    ImGui::SetNextItemWidth(200);
                    const char* tab_size_items[] = { "2 Spaces", "4 Spaces", "8 Spaces" };
                    int tab_size_index = (editor_tab_size_ == 2) ? 0 : (editor_tab_size_ == 8) ? 2 : 1;
                    int prev_tab_index = tab_size_index;
                    if (ImGui::Combo("##tab_size", &tab_size_index, tab_size_items, IM_ARRAYSIZE(tab_size_items))) {
                        int sizes[] = { 2, 4, 8 };
                        editor_tab_size_ = sizes[tab_size_index];
                        if (editor_tab_size_callback_ && tab_size_index != prev_tab_index) {
                            editor_tab_size_callback_(editor_tab_size_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Editor Features");
                    ImGui::Separator();
                    ImGui::Spacing();

                    bool prev_show_whitespace = editor_show_whitespace_;
                    if (ImGui::Checkbox("Show Whitespace Characters", &editor_show_whitespace_)) {
                        if (editor_show_whitespace_callback_ && editor_show_whitespace_ != prev_show_whitespace) {
                            editor_show_whitespace_callback_(editor_show_whitespace_);
                        }
                    }

                    bool prev_word_wrap = editor_word_wrap_;
                    if (ImGui::Checkbox("Word Wrap", &editor_word_wrap_)) {
                        if (editor_word_wrap_callback_ && editor_word_wrap_ != prev_word_wrap) {
                            editor_word_wrap_callback_(editor_word_wrap_);
                        }
                    }

                    bool prev_auto_indent = editor_auto_indent_;
                    if (ImGui::Checkbox("Auto Indent", &editor_auto_indent_)) {
                        if (editor_auto_indent_callback_ && editor_auto_indent_ != prev_auto_indent) {
                            editor_auto_indent_callback_(editor_auto_indent_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::TextDisabled("Line numbers are always shown. Current line is highlighted.");
                    ImGui::TextDisabled("These settings will be saved with your project.");

                    ImGui::EndTabItem();
                }

                // ========== Appearance Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_PALETTE " Appearance")) {
                    preferences_tab_ = 2;
                    ImGui::Spacing();

                    ImGui::Text("User Interface");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("UI Scale:");
                    ImGui::SetNextItemWidth(200);
                    ImGui::SliderFloat("##ui_scale", &appearance_ui_scale_, 0.8f, 2.0f, "%.1fx");
                    ImGui::SameLine();
                    if (ImGui::Button("Reset##scale")) {
                        appearance_ui_scale_ = 1.0f;
                    }

                    ImGui::Spacing();
                    ImGui::Checkbox("Smooth Scrolling", &appearance_smooth_scrolling_);

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Layout");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Sidebar Position:");
                    ImGui::RadioButton("Left", &appearance_sidebar_position_, 0);
                    ImGui::SameLine();
                    ImGui::RadioButton("Right", &appearance_sidebar_position_, 1);

                    ImGui::Spacing();
                    ImGui::TextDisabled("Note: Editor theme can be changed in the Editor tab.");

                    ImGui::EndTabItem();
                }

                // ========== Files Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_FILE " Files")) {
                    preferences_tab_ = 3;
                    ImGui::Spacing();

                    ImGui::Text("File Encoding");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Default Encoding:");
                    const char* encodings[] = { "UTF-8", "UTF-16", "ASCII" };
                    ImGui::SetNextItemWidth(150);
                    ImGui::Combo("##encoding", &files_default_encoding_, encodings, IM_ARRAYSIZE(encodings));

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Line Endings");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Default Line Ending:");
                    const char* line_endings[] = { "Auto (OS default)", "LF (Unix/macOS)", "CRLF (Windows)" };
                    ImGui::SetNextItemWidth(200);
                    ImGui::Combo("##line_ending", &files_line_ending_, line_endings, IM_ARRAYSIZE(line_endings));

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Save Options");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Trim trailing whitespace on save", &files_trim_trailing_whitespace_);
                    ImGui::Checkbox("Insert final newline on save", &files_insert_final_newline_);

                    ImGui::EndTabItem();
                }

                // ========== Python/Scripting Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_CODE " Python")) {
                    preferences_tab_ = 4;
                    ImGui::Spacing();

                    python_settings_panel_.Render();

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Startup Script
                    ImGui::Text("Startup Script (run on launch):");
                    ImGui::SetNextItemWidth(-100);
                    ImGui::InputText("##startup_script", python_startup_script_, sizeof(python_startup_script_));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##startup")) {
                        std::string path = OpenFileDialog("Python Scripts (*.py)\0*.py\0CyxWiz Scripts (*.cyx)\0*.cyx\0All Files (*.*)\0*.*\0", "Select Startup Script");
                        if (!path.empty()) {
                            strncpy(python_startup_script_, path.c_str(), sizeof(python_startup_script_) - 1);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Auto-import options
                    ImGui::Text("Auto-Import Libraries:");
                    ImGui::Checkbox("Import NumPy as 'np'", &python_auto_import_numpy_);
                    ImGui::Checkbox("Import CyxWiz module", &python_auto_import_cyxwiz_);

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Output limit
                    ImGui::Text("Console Output Limit (lines):");
                    ImGui::SetNextItemWidth(150);
                    ImGui::InputInt("##output_limit", &python_output_limit_);
                    if (python_output_limit_ < 100) python_output_limit_ = 100;
                    if (python_output_limit_ > 10000) python_output_limit_ = 10000;
                    ImGui::TextDisabled("Range: 100 - 10000 lines");

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    if (ImGui::Button("Show Runtime Details")) {
                        if (python_diagnostics_callback_) {
                            python_diagnostics_text_ = python_diagnostics_callback_();
                        } else {
                            python_diagnostics_text_ = "Python diagnostics are not available.";
                        }
                        show_python_diagnostics_popup_ = true;
                        ImGui::OpenPopup("Python Runtime Details");
                    }

                    if (ImGui::BeginPopupModal("Python Runtime Details", &show_python_diagnostics_popup_, ImGuiWindowFlags_AlwaysAutoResize)) {
                        ImGui::BeginChild("##python_diag_text", ImVec2(560, 300), true, ImGuiWindowFlags_HorizontalScrollbar);
                        ImGui::TextUnformatted(python_diagnostics_text_.c_str());
                        ImGui::EndChild();

                        if (ImGui::Button("Copy")) {
                            ImGui::SetClipboardText(python_diagnostics_text_.c_str());
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Close")) {
                            show_python_diagnostics_popup_ = false;
                            ImGui::CloseCurrentPopup();
                        }

                        ImGui::EndPopup();
                    }

                    ImGui::EndTabItem();
                }

                // ========== Keyboard Shortcuts Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_KEYBOARD " Shortcuts")) {
                    preferences_tab_ = 5;
                    ImGui::Spacing();

                    ImGui::TextDisabled("Double-click a shortcut to edit. Some shortcuts are system-level and cannot be changed.");
                    ImGui::Spacing();

                    // Table of shortcuts with category grouping
                    if (ImGui::BeginTable("ShortcutsTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 320))) {
                        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 180);
                        ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 150);
                        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        std::string current_category = "";
                        for (int i = 0; i < static_cast<int>(shortcuts_.size()); ++i) {
                            auto& shortcut = shortcuts_[i];

                            // Check if we're entering a new category
                            if (shortcut.category != current_category) {
                                current_category = shortcut.category;

                                // Render category header row
                                ImGui::TableNextRow();
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TableHeaderBg));

                                ImGui::TableNextColumn();
                                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                                ImGui::TextUnformatted(ICON_FA_FOLDER);
                                ImGui::SameLine();
                                ImGui::Text("%s", current_category.c_str());
                                ImGui::PopStyleColor();

                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("---");

                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("---");
                            }

                            ImGui::TableNextRow();

                            // Action column (indented to show hierarchy)
                            ImGui::TableNextColumn();
                            ImGui::Text("  %s", shortcut.action.c_str());

                            // Shortcut column
                            ImGui::TableNextColumn();
                            if (editing_shortcut_index_ == i) {
                                // Edit mode
                                ImGui::SetNextItemWidth(-1);
                                if (ImGui::InputText("##edit_shortcut", shortcut_edit_buffer_, sizeof(shortcut_edit_buffer_), ImGuiInputTextFlags_EnterReturnsTrue)) {
                                    shortcut.shortcut = shortcut_edit_buffer_;
                                    editing_shortcut_index_ = -1;
                                }
                                if (ImGui::IsItemDeactivated() && !ImGui::IsItemActive()) {
                                    editing_shortcut_index_ = -1;
                                }
                            } else {
                                // Display mode
                                if (shortcut.editable) {
                                    if (ImGui::Selectable(shortcut.shortcut.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                                        editing_shortcut_index_ = i;
                                        strncpy(shortcut_edit_buffer_, shortcut.shortcut.c_str(), sizeof(shortcut_edit_buffer_) - 1);
                                    }
                                } else {
                                    ImGui::TextDisabled("%s", shortcut.shortcut.c_str());
                                }
                            }

                            // Description column
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("%s", shortcut.description.c_str());
                        }

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Reset to Defaults")) {
                        shortcuts_.clear();  // Will be re-initialized on next open
                    }

                    ImGui::EndTabItem();
                }

                // ========== Devices Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_MICROCHIP " Devices")) {
                    preferences_tab_ = 6;
                    ImGui::Spacing();

                    // Show currently active backend
                    ImGui::Text("Currently Active Backend:");
                    ImGui::SameLine();
                    try {
                        auto* current_device = cyxwiz::Device::GetCurrentDevice();
                        if (current_device) {
                            const char* backend_name = "Unknown";
                            ImVec4 backend_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                            switch (current_device->GetType()) {
                                case cyxwiz::DeviceType::CPU:
                                    backend_name = "CPU";
                                    backend_color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::CUDA:
                                    backend_name = "CUDA (NVIDIA)";
                                    backend_color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::OPENCL:
                                    backend_name = "OpenCL";
                                    backend_color = ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::METAL:
                                    backend_name = "Metal (Apple)";
                                    backend_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                                    break;
                                default:
                                    break;
                            }
                            ImGui::TextColored(backend_color, "%s %s", ICON_FA_CIRCLE_CHECK, backend_name);
                        } else {
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "%s Not Set (using default)", ICON_FA_CIRCLE_EXCLAMATION);
                        }
                    } catch (...) {
                        ImGui::TextDisabled("Unable to query");
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Available Compute Devices");
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextDisabled("Select which device to use for training. CUDA devices are fastest for NVIDIA GPUs.");
                    ImGui::Spacing();

                    // Initialize device list if needed
                    if (!devices_initialized_) {
                        cached_devices_.clear();
                        try {
                            auto devices = cyxwiz::Device::GetAvailableDevices();
                            for (const auto& dev : devices) {
                                CachedDevice cached;
                                cached.type = static_cast<int>(dev.type);
                                cached.device_id = dev.device_id;
                                cached.name = dev.name;
                                cached.memory_total = dev.memory_total;
                                cached.memory_available = dev.memory_available;
                                cached_devices_.push_back(cached);
                            }

                            // Default to first CUDA device if available, otherwise keep CPU (index 0)
                            selected_device_index_ = 0;
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                if (cached_devices_[i].type == 1) {  // 1 = CUDA
                                    selected_device_index_ = static_cast<int>(i);
                                    // Set CUDA as the active device
                                    cyxwiz::Device cuda_device(cyxwiz::DeviceType::CUDA, cached_devices_[i].device_id);
                                    cuda_device.SetActive();
                                    spdlog::info("Default device set to CUDA: {}", cached_devices_[i].name);
                                    break;
                                }
                            }
                        } catch (...) {
                            spdlog::warn("Failed to enumerate devices");
                        }
                        devices_initialized_ = true;
                    }

                    // Device selection list
                    if (cached_devices_.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No compute devices found!");
                    } else {
                        for (size_t i = 0; i < cached_devices_.size(); ++i) {
                            const auto& dev = cached_devices_[i];

                            const char* type_icon = ICON_FA_MICROCHIP;
                            const char* type_name = "Unknown";
                            ImVec4 type_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);

                            switch (dev.type) {
                                case 0: // CPU
                                    type_icon = ICON_FA_MICROCHIP;
                                    type_name = "CPU";
                                    type_color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                                    break;
                                case 1: // CUDA
                                    type_icon = ICON_FA_BOLT;
                                    type_name = "CUDA";
                                    type_color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                                    break;
                                case 2: // OpenCL
                                    type_icon = ICON_FA_DESKTOP;
                                    type_name = "OpenCL";
                                    type_color = ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                                    break;
                                case 3: // Metal
                                    type_icon = ICON_FA_MICROCHIP;
                                    type_name = "Metal";
                                    type_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                                    break;
                            }

                            ImGui::PushID(static_cast<int>(i));

                            // Radio button for selection
                            bool is_selected = (selected_device_index_ == static_cast<int>(i));
                            if (ImGui::RadioButton("##device_select", is_selected)) {
                                selected_device_index_ = static_cast<int>(i);
                                // Apply device selection
                                try {
                                    cyxwiz::Device device(static_cast<cyxwiz::DeviceType>(dev.type), dev.device_id);
                                    device.SetActive();
                                    spdlog::info("Selected device: {} [{}]", dev.name, type_name);
                                } catch (const std::exception& e) {
                                    spdlog::error("Failed to set device: {}", e.what());
                                }
                            }
                            ImGui::SameLine();

                            // Device info
                            ImGui::TextColored(type_color, "%s", type_icon);
                            ImGui::SameLine();
                            ImGui::Text("%s", dev.name.c_str());
                            ImGui::SameLine();
                            ImGui::TextDisabled("[%s]", type_name);

                            // Memory info on same line if available
                            if (dev.memory_total > 0) {
                                ImGui::SameLine();
                                double mem_gb = dev.memory_total / (1024.0 * 1024.0 * 1024.0);
                                ImGui::TextDisabled("(%.1f GB)", mem_gb);
                            }

                            // Show "Active" badge if this is the current device
                            auto* current_device = cyxwiz::Device::GetCurrentDevice();
                            if (current_device &&
                                current_device->GetType() == static_cast<cyxwiz::DeviceType>(dev.type) &&
                                current_device->GetDeviceId() == dev.device_id) {
                                ImGui::SameLine();
                                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%s Active", ICON_FA_CIRCLE_CHECK);
                            }

                            ImGui::PopID();
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();

                    // Refresh button
                    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh Devices")) {
                        devices_initialized_ = false;
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Current device info
                    if (selected_device_index_ >= 0 && selected_device_index_ < static_cast<int>(cached_devices_.size())) {
                        const auto& dev = cached_devices_[selected_device_index_];
                        ImGui::Text("Selected Device Details:");
                        ImGui::Indent();
                        ImGui::BulletText("Name: %s", dev.name.c_str());
                        if (dev.memory_total > 0) {
                            double total_gb = dev.memory_total / (1024.0 * 1024.0 * 1024.0);
                            double avail_gb = dev.memory_available / (1024.0 * 1024.0 * 1024.0);
                            ImGui::BulletText("Memory: %.2f GB total, %.2f GB available", total_gb, avail_gb);
                        }
                        ImGui::Unindent();
                    }

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Dialog buttons
            float button_width = 100.0f;
            float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total_width) * 0.5f);

            if (ImGui::Button("OK", ImVec2(button_width, 0))) {
                // Python settings are now in PythonSettingsPanel
                // Save preferences to project if one is open
                if (save_project_settings_callback_) {
                    save_project_settings_callback_();
                }
                show_preferences_dialog_ = false;
                spdlog::info("Preferences saved");
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
                show_preferences_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }
}

} // namespace cyxwiz