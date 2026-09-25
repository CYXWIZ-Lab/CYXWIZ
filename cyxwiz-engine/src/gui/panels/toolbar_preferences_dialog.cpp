// Toolbar preferences modal rendering.

#include "toolbar.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../../core/compute_runtime_config.h"
#include "../../core/compute_runtime_paths.h"
#include "../../core/async_task_manager.h"
#include "../../core/execution_device_preferences.h"
#include "../../core/route_qualification_service.h"
#include "../../core/route_recommendation.h"
#include "../../core/training_manager.h"
#include "../../core/training_trace_collector.h"
#include "../../core/window_manager.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <cyxwiz/cyxwiz.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {
namespace {

const char* ArrayFireBackendDisplayName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "ArrayFire CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        default: return "Unsupported";
    }
}

const char* ArrayFireBackendIcon(DeviceType type) {
    switch (type) {
        case DeviceType::CUDA: return ICON_FA_BOLT;
        case DeviceType::OPENCL: return ICON_FA_DESKTOP;
        case DeviceType::CPU:
        case DeviceType::ONEAPI:
        default: return ICON_FA_MICROCHIP;
    }
}

ImVec4 ArrayFireBackendColor(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
        case DeviceType::CUDA: return ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
        case DeviceType::OPENCL: return ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
        case DeviceType::ONEAPI: return ImVec4(0.8f, 0.6f, 1.0f, 1.0f);
        default: return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
    }
}

} // namespace

void ToolbarPanel::RenderPreferencesDialog() {
    // ========== Preferences Dialog ==========
    if (show_preferences_dialog_) {
        // Note: shortcuts_ is initialized in RenderEditMenu() when Preferences is clicked

        // Python settings have been moved to PythonSettingsPanel
        // This initialization code is no longer needed

        ImGui::OpenPopup("Preferences");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        // Fit the Engine window: a fixed 1000x760 overflowed smaller windows and
        // clipped the OK / Cancel footer.
        const ImVec2 viewport = ImGui::GetMainViewport()->WorkSize;
        const ImVec2 limit(viewport.x * 0.92f, viewport.y * 0.92f);
        ImGui::SetNextWindowSize(
            ImVec2((std::min)(1000.0f, limit.x), (std::min)(760.0f, limit.y)),
            ImGuiCond_Appearing);
        ImGui::SetNextWindowSizeConstraints(
            ImVec2((std::min)(560.0f, limit.x), (std::min)(360.0f, limit.y)), limit);

        if (ImGui::BeginPopupModal("Preferences", &show_preferences_dialog_)) {
            // Tabs scroll inside their own region so OK / Cancel stay visible
            // at the bottom however long a tab is.
            const float footer_height = ImGui::GetTextLineHeight() + 16.0f +
                ImGui::GetStyle().ItemSpacing.y * 4.0f + 6.0f;
            ImGui::BeginChild("PreferencesBody", ImVec2(0.0f, -footer_height));
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

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Materialization Memory");
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::Text("Hard limit (MB):");
                    ImGui::SetNextItemWidth(160.0f);
                    ImGui::InputInt("##materialization_memory_limit_mb",
                                    &materialization_memory_limit_mb_,
                                    256,
                                    1024);
                    materialization_memory_limit_mb_ = std::clamp(
                        materialization_memory_limit_mb_, 0, 2000000);
                    ImGui::TextDisabled(
                        "0 uses the automatic available-RAM safety budget.");

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

                    bool backend_manager_verify_requested = false;

                    const bool training_active =
                        cyxwiz::TrainingManager::Instance().IsTrainingActive();
                    const auto active_run_trace =
                        cyxwiz::TrainingTraceCollector::LatestTrace();
                    const bool has_run_bound_device =
                        active_run_trace.available &&
                        !active_run_trace.effective_backend.empty();

                    // Show training backend truth. ArrayFire backend state can
                    // be thread-local, so the GUI thread's current device is
                    // not authoritative after a training-thread selection.
                    ImGui::Text(
                        has_run_bound_device
                            ? (training_active
                                   ? "Current Training Backend:"
                                   : "Last Training Backend:")
                            : "Currently Active Backend:");
                    ImGui::SameLine();
                    if (has_run_bound_device) {
                        ImVec4 backend_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                        if (active_run_trace.effective_backend == "arrayfire_cpu") {
                            backend_color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                        } else if (active_run_trace.effective_backend == "arrayfire_cuda") {
                            backend_color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                        } else if (active_run_trace.effective_backend == "arrayfire_opencl") {
                            backend_color = ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                        } else if (active_run_trace.effective_backend == "arrayfire_oneapi") {
                            backend_color = ImVec4(0.8f, 0.6f, 1.0f, 1.0f);
                        }
                        ImGui::TextColored(
                            backend_color,
                            "%s %s device %d%s%s",
                            ICON_FA_CIRCLE_CHECK,
                            active_run_trace.effective_backend.c_str(),
                            active_run_trace.effective_device_id,
                            active_run_trace.effective_device_name.empty()
                                ? ""
                                : " - ",
                            active_run_trace.effective_device_name.c_str());
                        ImGui::TextDisabled(
                            "Requested: %s device %d",
                            active_run_trace.requested_backend.empty()
                                ? "Not recorded"
                                : active_run_trace.requested_backend.c_str(),
                            active_run_trace.requested_device_id);
                        ImGui::TextDisabled(
                            "Execution preflight: %s%s%s",
                            active_run_trace.execution_validated
                                ? "Validated"
                                : "Not validated",
                            active_run_trace.preflight_stage.empty()
                                ? ""
                                : " - ",
                            active_run_trace.preflight_stage.c_str());
                        ImGui::TextDisabled(
                            "Requested verification: %s | evidence %s",
                            !active_run_trace
                                 .requested_qualification_evidence_available
                                ? "No evidence"
                                : (active_run_trace.requested_route_qualified
                                       ? "Passed"
                                       : "Failed"),
                            cyxwiz::RouteQualificationEvidenceLabel(
                                active_run_trace
                                    .requested_qualification_matrix_id));
                        ImGui::TextDisabled(
                            "Effective verification: %s | evidence %s",
                            !active_run_trace
                                 .effective_qualification_evidence_available
                                ? "No evidence"
                                : (active_run_trace.effective_route_qualified
                                       ? "Passed"
                                       : "Failed"),
                            cyxwiz::RouteQualificationEvidenceLabel(
                                active_run_trace
                                    .effective_qualification_matrix_id));
                        if (active_run_trace.selection_fallback_applied) {
                            ImGui::TextDisabled(
                                "Selection fallback: Applied to ArrayFire CPU");
                        }
                        ImGui::TextDisabled(
                            "Placement: %s (%llu entries)",
                            active_run_trace.placement_fingerprint.empty()
                                ? "Not recorded"
                                : active_run_trace.placement_fingerprint.c_str(),
                            static_cast<unsigned long long>(
                                active_run_trace.placement_entry_count));
                    } else {
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
                                case cyxwiz::DeviceType::ONEAPI:
                                    backend_name = "oneAPI";
                                    backend_color = ImVec4(0.8f, 0.6f, 1.0f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::METAL:
                                    backend_name = "Metal (unsupported)";
                                    backend_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::VULKAN:
                                    backend_name = "Vulkan (unsupported)";
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
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Execution Policy");
                    ImGui::Separator();
                    ImGui::Spacing();

                    if (has_run_bound_device &&
                        !active_run_trace.fallback_policy.empty()) {
                        const bool run_is_strict =
                            active_run_trace.fallback_policy ==
                            "forbid_native_cpu_fallback";
                        ImGui::TextDisabled(
                            training_active ? "Current run" : "Last run");
                        ImGui::SameLine(130.0f);
                        ImGui::TextColored(
                            run_is_strict
                                ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
                                : ImVec4(1.0f, 0.82f, 0.35f, 1.0f),
                            "%s",
                            run_is_strict
                                ? "Strict ArrayFire residency"
                                : "Compatibility with recorded fallback");
                    }

                    const auto next_run_policy =
                        cyxwiz::GetNextRunExecutionPolicy();
                    const auto selected_policy =
                        next_run_policy.value_or(
                            cyxwiz::ArrayFireFallbackPolicy::
                                AllowNativeCpuFallback);
                    ImGui::TextDisabled("Next runs");
                    ImGui::SameLine(130.0f);
                    if (training_active) {
                        ImGui::BeginDisabled();
                    }
                    const bool compatibility_clicked = ImGui::RadioButton(
                        "Compatibility##execution_policy",
                        selected_policy ==
                            cyxwiz::ArrayFireFallbackPolicy::
                                AllowNativeCpuFallback);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Allow declared native CPU compatibility paths and record observed fallback");
                    }
                    ImGui::SameLine();
                    const bool strict_clicked = ImGui::RadioButton(
                        "Strict residency##execution_policy",
                        selected_policy ==
                            cyxwiz::ArrayFireFallbackPolicy::
                                ForbidNativeCpuFallback);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Reject non-ArrayFire placement before training and terminate on unexpected fallback");
                    }
                    if (training_active) {
                        ImGui::EndDisabled();
                    }
                    const auto persist_policy = [&](auto policy,
                                                    const char* description) {
                        std::string persistence_error;
                        if (!cyxwiz::UpdateDefaultFallbackPolicyAtomic(
                                cyxwiz::GetComputeRuntimeConfigPath(),
                                policy,
                                persistence_error)) {
                            device_selection_error_ =
                                "stage=commit reason=" + persistence_error;
                            spdlog::error(
                                "Compute execution policy was not saved: {}",
                                persistence_error);
                            return;
                        }
                        cyxwiz::SetNextRunExecutionPolicy(policy);
                        device_selection_error_.clear();
                        spdlog::info("Saved {} as the machine default",
                                     description);
                    };
                    if (!training_active && compatibility_clicked) {
                        persist_policy(
                            cyxwiz::ArrayFireFallbackPolicy::
                                AllowNativeCpuFallback,
                            "compatibility execution policy");
                    } else if (!training_active && strict_clicked) {
                        persist_policy(
                            cyxwiz::ArrayFireFallbackPolicy::
                                ForbidNativeCpuFallback,
                            "strict ArrayFire residency policy");
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    backend_manager_verify_requested =
                        RenderBackendManagerSection(training_active);

                    ImGui::Text("Compute devices");
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextDisabled(
                        "Choose a candidate ArrayFire route. OK validates and commits it.");
                    if (training_active) {
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                            "%s Training is active; device changes are disabled until the run finishes.",
                            ICON_FA_TRIANGLE_EXCLAMATION);
                    } else {
                        ImGui::TextDisabled(
                            "The candidate remains local until you click OK.");
                    }
                    ImGui::TextDisabled("Operator fallbacks are reported as native CPU fallback events.");
                    ImGui::Spacing();

                    // Initialize device list if needed
                    if (!devices_initialized_ &&
                        !HasActiveExecutionDeviceContext()) {
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
                                cached.kind = static_cast<int>(dev.kind);
                                cached.identity_confidence =
                                    static_cast<int>(
                                        dev.identity_confidence);
                                cached.provider = dev.provider;
                                cached.driver_version = dev.driver_version;
                                cached.physical_fingerprint =
                                    dev.physical_fingerprint;
                                cached.hardware_vendor_id =
                                    dev.hardware_vendor_id;
                                cached.hardware_vendor_id_known =
                                    dev.hardware_vendor_id_known;
                                cached.provider_known = dev.provider_known;
                                cached.driver_version_known =
                                    dev.driver_version_known;
                                cached.pci_location_known =
                                    dev.pci_location_known;
                                cached.pci_domain = dev.pci_domain;
                                cached.pci_bus = dev.pci_bus;
                                cached.pci_device = dev.pci_device;
                                cached.pci_function = dev.pci_function;
                                cached.physical_fingerprint_known =
                                    dev.physical_fingerprint_known;
                                cached.metadata_status =
                                    static_cast<int>(dev.metadata_status);
                                cached.device_selectable =
                                    dev.device_selectable;
                                cached.execution_validated =
                                    dev.execution_validated;
                                cached.name_is_fallback =
                                    dev.name_is_fallback;
                                cached.memory_total_known =
                                    dev.memory_total_known;
                                cached.memory_available_known =
                                    dev.memory_available_known;
                                const auto qualification =
                                    cyxwiz::EvaluateRouteQualification(dev);
                                if (cached.name_is_fallback &&
                                    qualification.display_name_available) {
                                    cached.name = qualification.display_name;
                                    cached.name_from_qualification = true;
                                }
                                if (cached.kind == static_cast<int>(
                                        cyxwiz::DeviceKind::Unknown) &&
                                    qualification.device_kind_known) {
                                    cached.kind = static_cast<int>(
                                        qualification.device_kind);
                                }
                                cached.identity_source =
                                    qualification.identity_source;
                                cached.qualification_evidence_available =
                                    qualification.evidence_available;
                                cached.matrix_qualified =
                                    qualification.qualified;
                                const auto authorization =
                                    cyxwiz::EvaluateRouteTrainingAuthorization(
                                        dev, qualification);
                                cached.training_authorized =
                                    authorization.authorized;
                                cached.training_authorization_status =
                                    static_cast<int>(authorization.status);
                                cached.qualification_matrix_id =
                                    qualification.matrix_id;
                                cached.qualification_message =
                                    qualification.message;
                                cached.training_authorization_message =
                                    authorization.message;
                                cached.failure_category =
                                    cyxwiz::RouteFailureCategoryName(
                                        authorization.failure.category);
                                cached.failed_operation =
                                    authorization.failure.operation;
                                cached.observed_failure =
                                    authorization.failure.observed_fact;
                                cached.failure_interpretation =
                                    authorization.failure
                                        .bounded_interpretation;
                                cached.recommended_action =
                                    authorization.failure.recommended_action;
                                cached_devices_.push_back(cached);
                            }

                            // Device enumeration is read-only. Opening Preferences must
                            // never change the runtime backend.
                            selected_device_index_ = -1;
                            selected_backend_type_ = -1;
                        } catch (...) {
                            spdlog::warn("Failed to enumerate devices");
                        }
                        devices_initialized_ = true;
                    }

                    // Device selection list
                    if (cached_devices_.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No compute devices found!");
                    } else {
                        auto* active_device = cyxwiz::Device::GetCurrentDevice();
                        const auto pending_selection =
                            cyxwiz::GetPendingExecutionDeviceSelection();
                        const auto saved_selection =
                            cyxwiz::GetSavedExecutionDeviceSelection();
                        auto matches_active_run =
                            [&](const CachedDevice& candidate) {
                                if (!has_run_bound_device ||
                                    active_run_trace.effective_device_id !=
                                        candidate.device_id) {
                                    return false;
                                }
                                switch (static_cast<cyxwiz::DeviceType>(
                                    candidate.type)) {
                                    case cyxwiz::DeviceType::CPU:
                                        return active_run_trace.effective_backend ==
                                            "arrayfire_cpu";
                                    case cyxwiz::DeviceType::CUDA:
                                        return active_run_trace.effective_backend ==
                                            "arrayfire_cuda";
                                    case cyxwiz::DeviceType::OPENCL:
                                        return active_run_trace.effective_backend ==
                                            "arrayfire_opencl";
                                    case cyxwiz::DeviceType::ONEAPI:
                                        return active_run_trace.effective_backend ==
                                            "arrayfire_oneapi";
                                    default:
                                        return false;
                                }
                            };
                        if (!device_selection_dirty_) {
                            selected_device_index_ = -1;
                        if (pending_selection.has_value()) {
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                const auto& candidate = cached_devices_[i];
                                if (pending_selection->type ==
                                        static_cast<cyxwiz::DeviceType>(candidate.type) &&
                                    pending_selection->device_id == candidate.device_id) {
                                    selected_device_index_ = static_cast<int>(i);
                                    break;
                                }
                            }
                        } else if (saved_selection.has_value()) {
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                const auto& candidate = cached_devices_[i];
                                if (saved_selection->type ==
                                        static_cast<cyxwiz::DeviceType>(candidate.type) &&
                                    saved_selection->device_id ==
                                        candidate.device_id) {
                                    selected_device_index_ = static_cast<int>(i);
                                    break;
                                }
                            }
                        } else if (has_run_bound_device) {
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                if (matches_active_run(cached_devices_[i])) {
                                    selected_device_index_ = static_cast<int>(i);
                                    break;
                                }
                            }
                        } else if (active_device) {
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                const auto& candidate = cached_devices_[i];
                                if (active_device->GetType() ==
                                        static_cast<cyxwiz::DeviceType>(candidate.type) &&
                                    active_device->GetDeviceId() == candidate.device_id) {
                                    selected_device_index_ = static_cast<int>(i);
                                    break;
                                }
                            }
                        }

                        if (selected_device_index_ < 0 && !cached_devices_.empty()) {
                            selected_device_index_ = 0;
                        }
                        if (selected_device_index_ >= 0 &&
                            selected_backend_type_ < 0) {
                            selected_backend_type_ =
                                cached_devices_[selected_device_index_].type;
                        }
                        }

                        const auto select_candidate = [&](size_t index) {
                            const auto& selected = cached_devices_[index];
                            selected_device_index_ = static_cast<int>(index);
                            selected_backend_type_ = selected.type;
                            device_selection_dirty_ = true;
                            spdlog::info(
                                "Selected ArrayFire candidate in Preferences: "
                                "backend={} device={} id={}",
                                ArrayFireBackendDisplayName(
                                    static_cast<cyxwiz::DeviceType>(selected.type)),
                                selected.name,
                                selected.device_id);
                        };

                        const auto request_device = [&](size_t index) {
                            const auto requested_type =
                                static_cast<cyxwiz::DeviceType>(
                                    cached_devices_[index].type);
                            if (requested_type == cyxwiz::DeviceType::ONEAPI &&
                                !cached_devices_[index].training_authorized) {
                                pending_oneapi_device_index_ =
                                    static_cast<int>(index);
                                show_oneapi_training_warning_ = true;
                                return;
                            }
                            select_candidate(index);
                        };

                        const auto cards_task =
                            route_qualification_task_id_ == 0
                                ? std::shared_ptr<cyxwiz::AsyncTask>{}
                                : cyxwiz::AsyncTaskManager::Instance().GetTask(
                                      route_qualification_task_id_);
                        ComputeDeviceCardsContext cards_context;
                        cards_context.training_active = training_active;
                        cards_context.verification_running =
                            cards_task &&
                            (cards_task->GetState() == cyxwiz::TaskState::Pending ||
                             cards_task->GetState() == cyxwiz::TaskState::Running);
                        cards_context.is_active = [&](size_t index) {
                            const auto& dev = cached_devices_[index];
                            const auto type =
                                static_cast<cyxwiz::DeviceType>(dev.type);
                            return has_run_bound_device
                                ? matches_active_run(dev)
                                : (active_device && active_device->GetType() == type &&
                                   active_device->GetDeviceId() == dev.device_id);
                        };
                        cards_context.is_pending = [&](size_t index) {
                            const auto& dev = cached_devices_[index];
                            return pending_selection.has_value() &&
                                pending_selection->type ==
                                    static_cast<cyxwiz::DeviceType>(dev.type) &&
                                pending_selection->device_id == dev.device_id;
                        };
                        cards_context.is_saved = [&](size_t index) {
                            const auto& dev = cached_devices_[index];
                            return saved_selection.has_value() &&
                                saved_selection->type ==
                                    static_cast<cyxwiz::DeviceType>(dev.type) &&
                                saved_selection->device_id == dev.device_id;
                        };
                        cards_context.request_device = [&](size_t index) {
                            try {
                                request_device(index);
                            } catch (const std::exception& e) {
                                spdlog::error(
                                    "Failed to queue backend/device: {}", e.what());
                            }
                        };
                        RenderComputeDeviceCards(cards_context);

                        if (show_oneapi_training_warning_) {
                            ImGui::OpenPopup("Compute Route Warning");
                            show_oneapi_training_warning_ = false;
                        }
                        const ImVec2 warning_center =
                            ImGui::GetMainViewport()->GetCenter();
                        ImGui::SetNextWindowPos(
                            warning_center,
                            ImGuiCond_Appearing,
                            ImVec2(0.5f, 0.5f));
                        ImGui::SetNextWindowSizeConstraints(
                            ImVec2(460.0f, 0.0f), ImVec2(620.0f, 500.0f));
                        if (ImGui::IsPopupOpen("Compute Route Warning")) {
                            ImGui::SetNextWindowFocus();
                        }
                        if (ImGui::BeginPopupModal(
                                "Compute Route Warning",
                                nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                                "%s This compute route is not authorized for normal training.",
                                ICON_FA_TRIANGLE_EXCLAMATION);
                            ImGui::Separator();
                            ImGui::Text("Compatibility result");
                            ImGui::BulletText(
                                "Evidence scope: Backend and device route");
                            ImGui::BulletText("Backend pack: Installed and loadable");
                            ImGui::BulletText("Device enumeration: Passed");
                            ImGui::BulletText(
                                "Bounded execution: Loaded from isolated probes");
                            if (pending_oneapi_device_index_ >= 0 &&
                                pending_oneapi_device_index_ <
                                    static_cast<int>(cached_devices_.size())) {
                                const auto& candidate = cached_devices_[
                                    pending_oneapi_device_index_];
                                ImGui::BulletText("Device: %s (backend ID %d)",
                                                  candidate.name.c_str(),
                                                  candidate.device_id);
                                ImGui::BulletText(
                                    "Metadata: %s",
                                    cyxwiz::DeviceMetadataStatusName(
                                        static_cast<
                                            cyxwiz::DeviceMetadataStatus>(
                                            candidate.metadata_status)));
                                ImGui::BulletText(
                                    "Qualification evidence: %s",
                                    candidate.qualification_evidence_available
                                        ? "Loaded"
                                        : "Not available");
                                ImGui::TextWrapped(
                                    "Evidence: %s",
                                    candidate.qualification_message.c_str());
                                ImGui::TextWrapped(
                                    "Authorization: %s",
                                    candidate.training_authorization_message.c_str());
                                if (!candidate.observed_failure.empty()) {
                                    ImGui::TextWrapped(
                                        "Observed: %s",
                                        candidate.observed_failure.c_str());
                                }
                                if (!candidate.recommended_action.empty()) {
                                    ImGui::TextWrapped(
                                        "Action: %s",
                                        candidate.recommended_action.c_str());
                                }
                            }
                            ImGui::Spacing();
                            ImGui::TextWrapped(
                                pending_oneapi_device_index_ >= 0 &&
                                        pending_oneapi_device_index_ <
                                            static_cast<int>(
                                                cached_devices_.size()) &&
                                        cached_devices_[pending_oneapi_device_index_]
                                            .matrix_qualified
                                    ? "This exact route passed its retained isolated "
                                      "verification, but an explicit current policy still "
                                      "blocks production selection. The previous "
                                      "selection remains unchanged."
                                    : "This exact route failed its retained isolated "
                                      "verification. Strict residency rejects it before model "
                                      "construction. The current selection remains "
                                      "unchanged.");
                            ImGui::Spacing();
                            if (ui::PrimaryButton("Select ArrayFire CPU")) {
                                for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                    if (cached_devices_[i].type ==
                                        static_cast<int>(cyxwiz::DeviceType::CPU)) {
                                        try {
                                            select_candidate(i);
                                        } catch (const std::exception& e) {
                                            spdlog::error(
                                                "Failed to queue ArrayFire CPU: {}",
                                                e.what());
                                        }
                                        break;
                                    }
                                }
                                pending_oneapi_device_index_ = -1;
                                ImGui::CloseCurrentPopup();
                            }
                            ImGui::SameLine();
                            if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) {
                                pending_oneapi_device_index_ = -1;
                                ImGui::CloseCurrentPopup();
                            }
                            ImGui::EndPopup();
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();

                    const auto qualification_task =
                        route_qualification_task_id_ == 0
                            ? std::shared_ptr<cyxwiz::AsyncTask>{}
                            : cyxwiz::AsyncTaskManager::Instance().GetTask(
                                  route_qualification_task_id_);
                    const bool qualification_running =
                        qualification_task &&
                        (qualification_task->GetState() ==
                             cyxwiz::TaskState::Pending ||
                         qualification_task->GetState() ==
                             cyxwiz::TaskState::Running);
                    if (qualification_task && !qualification_running &&
                        !route_qualification_task_refreshed_) {
                        devices_initialized_ = false;
                        route_qualification_task_refreshed_ = true;
                    }

                    // selected_only: Verify Selected (merge into existing
                    // evidence, no benchmark); otherwise Verify All (replace).
                    const auto begin_verification =
                        [&](std::vector<cyxwiz::DeviceInfo> routes,
                            bool selected_only) {
                            if (routes.empty() || qualification_running ||
                                training_active) {
                                return;
                            }
                            std::filesystem::path probe_path =
                                cyxwiz::core::WindowManager::GetExecutablePath();
#ifdef _WIN32
                            probe_path.replace_filename(
                                "cyxwiz-route-probe.exe");
#else
                            probe_path.replace_filename("cyxwiz-route-probe");
#endif
                            if (!std::filesystem::is_regular_file(probe_path)) {
                                device_selection_error_ =
                                    "Qualification helper is missing: " +
                                    probe_path.string();
                                return;
                            }
                            cyxwiz::RouteQualificationOptions options;
                            options.probe_executable = probe_path;
                            options.cache_path =
                                cyxwiz::GetRouteQualificationCachePath();
                            options.matrix_id =
                                cyxwiz::kRouteQualificationMatrixId;
                            std::string runtime_identity_error;
                            options.runtime_identity =
                                cyxwiz::ReadActiveRuntimeQualificationIdentity(
                                    runtime_identity_error);
                            if (!runtime_identity_error.empty()) {
                                device_selection_error_ =
                                    runtime_identity_error;
                                return;
                            }
                            if (!options.runtime_identity.has_value()) {
                                options.pack_id =
                                    "local-arrayfire-installation";
                            }
                            const bool verify_all = !selected_only;
                            options.benchmark_verified_routes = verify_all;
                            // One isolated process per route (backend loads once).
                            options.batch_operations = true;
                            const bool with_recovery =
                                selected_only && routes.size() > 1;
                            const auto service =
                                route_qualification_service_;
                            route_qualification_task_refreshed_ = false;
                            device_selection_error_.clear();
                            route_qualification_task_id_ =
                                cyxwiz::AsyncTaskManager::Instance().RunAsync(
                                    verify_all
                                        ? "Verify all compute routes"
                                        : with_recovery
                                            ? "Verify compute route and CPU recovery"
                                            : "Verify compute route",
                                    [service,
                                     routes = std::move(routes),
                                     options = std::move(options),
                                     verify_all](cyxwiz::LambdaTask& task) {
                                        const auto result = verify_all
                                            ? service->VerifyAll(
                                                  routes, options,
                                                  [&](const auto& progress) {
                                                      const float total =
                                                          static_cast<float>(
                                                              progress.route_count *
                                                              progress.operation_count);
                                                      const float completed =
                                                          static_cast<float>(
                                                              progress.route_index *
                                                                  progress.operation_count +
                                                              progress.operation_index);
                                                      task.ReportProgress(
                                                          total > 0.0f
                                                              ? completed / total
                                                              : 0.0f,
                                                          progress.backend + ":" +
                                                              std::to_string(
                                                                  progress.device_id) +
                                                              " " + progress.operation);
                                                  })
                                            : service->VerifyRoutes(
                                                  routes, options,
                                                  [&](const auto& progress) {
                                                      const float total =
                                                          static_cast<float>(
                                                              progress.route_count *
                                                              progress.operation_count);
                                                      task.ReportProgress(
                                                          total > 0.0f
                                                              ? static_cast<float>(
                                                                    progress.route_index *
                                                                        progress.operation_count +
                                                                    progress.operation_index) /
                                                                    total
                                                              : 0.0f,
                                                          progress.backend + ":" +
                                                              std::to_string(
                                                                  progress.device_id) +
                                                              " " + progress.operation);
                                                  });
                                        if (result.status ==
                                            cyxwiz::RouteQualificationRunStatus::
                                                Cancelled) {
                                            return;
                                        }
                                        if (!result.published) {
                                            throw std::runtime_error(
                                                result.message);
                                        }
                                    });
                        };

                    // A route's Verify checks only that route; a pack's
                    // "Verify this pack's routes" checks that backend. Both merge
                    // into the saved results. Only Verify All replaces them.
                    if (pending_route_verify_index_ >= 0 ||
                        (backend_manager_verify_requested &&
                         !backend_pack_verify_backend_.empty())) {
                        const int route_index = pending_route_verify_index_;
                        const std::string pack_backend = backend_pack_verify_backend_;
                        pending_route_verify_index_ = -1;
                        backend_pack_verify_backend_.clear();
                        auto inventory = cyxwiz::Device::GetAvailableDevices();
                        inventory.erase(
                            std::remove_if(
                                inventory.begin(), inventory.end(),
                                [&](const auto& route) {
                                    if (route_index >= 0 &&
                                        route_index <
                                            static_cast<int>(cached_devices_.size())) {
                                        const auto& wanted = cached_devices_[route_index];
                                        return !(static_cast<int>(route.type) ==
                                                     wanted.type &&
                                                 route.device_id == wanted.device_id);
                                    }
                                    const char* key =
                                        route.type == cyxwiz::DeviceType::CPU ? "cpu"
                                        : route.type == cyxwiz::DeviceType::CUDA ? "cuda"
                                        : route.type == cyxwiz::DeviceType::OPENCL ? "opencl"
                                        : route.type == cyxwiz::DeviceType::ONEAPI ? "oneapi"
                                        : "";
                                    return pack_backend != key;
                                }),
                            inventory.end());
                        begin_verification(std::move(inventory), true);
                    }

                    if (qualification_running) {
                        const auto progress =
                            route_qualification_service_->GetProgress();
                        const size_t total = progress.route_count *
                                             progress.operation_count;
                        const size_t completed = progress.route_index *
                                                     progress.operation_count +
                                                 progress.operation_index;
                        ImGui::ProgressBar(
                            total == 0
                                ? 0.0f
                                : static_cast<float>(completed) /
                                      static_cast<float>(total),
                            ImVec2(-1.0f, 0.0f));
                        ImGui::TextWrapped(
                            "Verifying %s device %d: %s (route %zu of %zu, operation %zu of %zu)",
                            progress.backend.c_str(), progress.device_id,
                            progress.operation.c_str(), progress.route_index + 1,
                            progress.route_count, progress.operation_index + 1,
                            progress.operation_count);
                        if (ui::SecondaryButton(ICON_FA_STOP " Cancel verification")) {
                            route_qualification_service_->Cancel();
                            cyxwiz::AsyncTaskManager::Instance().Cancel(
                                route_qualification_task_id_);
                        }
                    } else {
                        if (training_active) ImGui::BeginDisabled();
                        if (ui::PrimaryButton(ICON_FA_SHIELD_HALVED " Verify all")) {
                            begin_verification(
                                cyxwiz::Device::GetAvailableDevices(), false);
                        }
                        if (training_active) ImGui::EndDisabled();
                        // Failures used to reach only the task log; show them here.
                        const bool task_failed =
                            qualification_task &&
                            qualification_task->GetState() == cyxwiz::TaskState::Failed;
                        if (task_failed || !device_selection_error_.empty()) {
                            ImGui::PushTextWrapPos(0.0f);
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.48f, 0.45f, 1.0f),
                                "%s Verification did not complete: %s",
                                ICON_FA_TRIANGLE_EXCLAMATION,
                                task_failed
                                    ? qualification_task->GetErrorMessage().c_str()
                                    : device_selection_error_.c_str());
                            ImGui::TextDisabled(
                                "Saved results were not changed. Fix the cause and verify again.");
                            ImGui::PopTextWrapPos();
                        }
                    }

                    if (!qualification_running) {
                        const auto snapshot =
                            cyxwiz::GetRouteQualificationSnapshot();
                        if (snapshot.has_value()) {
                            ImGui::Spacing();
                            ImGui::SeparatorText("Latest verification result");
                            const auto fastest =
                                cyxwiz::RecommendFastestVerifiedRoute(snapshot);
                            if (fastest.has_value()) {
                                int fastest_index = -1;
                                for (size_t index = 0;
                                     index < cached_devices_.size(); ++index) {
                                    const auto& candidate =
                                        cached_devices_[index];
                                    if (candidate.type == static_cast<int>(
                                            fastest->type) &&
                                        candidate.device_id ==
                                            fastest->device_id) {
                                        fastest_index =
                                            static_cast<int>(index);
                                        break;
                                    }
                                }
                                const char* name =
                                    fastest_index >= 0
                                        ? cached_devices_[fastest_index]
                                              .name.c_str()
                                        : (fastest->display_name.empty()
                                               ? "Verified route"
                                               : fastest->display_name.c_str());
                                std::string fastest_label = name;
                                std::replace(fastest_label.begin(),
                                             fastest_label.end(), '_', ' ');
                                ImGui::TextColored(
                                    ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                                    "%s Fastest verified route for the dense benchmark",
                                    ICON_FA_GAUGE_HIGH);
                                ImGui::BulletText(
                                    "%s / %s (backend ID %d)",
                                    ArrayFireBackendDisplayName(fastest->type),
                                    fastest_label.c_str(), fastest->device_id);
                                ImGui::BulletText(
                                    "Median: %.3f ms/iteration (%d samples, %d iterations each)",
                                    fastest->median_iteration_ms,
                                    fastest->sample_count,
                                    fastest->iterations_per_sample);
                                ImGui::TextDisabled(
                                    "Fixed 512x512 dense forward/backward compute; actual model performance may differ.");
                                if (fastest_index >= 0) {
                                    if (training_active) ImGui::BeginDisabled();
                                    if (ui::SecondaryButton(ICON_FA_WAND_MAGIC_SPARKLES " Use fastest route")) {
                                        selected_device_index_ = fastest_index;
                                        selected_backend_type_ =
                                            cached_devices_[fastest_index].type;
                                        device_selection_dirty_ = true;
                                    }
                                    if (training_active) ImGui::EndDisabled();
                                }
                            } else {
                                ImGui::TextDisabled(
                                    "No performance recommendation is available. Run Verify All to benchmark every verified route.");
                            }

                        }
                    }

                    ImGui::Spacing();

                    // Refresh button
                    if (ui::SecondaryButton(ICON_FA_ARROWS_ROTATE " Refresh devices")) {
                        devices_initialized_ = false;
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
            ImGui::EndChild();

            ImGui::Separator();
            ImGui::Spacing();

            // Dialog buttons
            float button_width = 100.0f;
            float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total_width) * 0.5f);

            if (ui::PrimaryButton("OK", true, nullptr, ui::ButtonSize::Regular, button_width)) {
                bool selection_committed = true;
                if (device_selection_dirty_) {
                    if (selected_device_index_ < 0 ||
                        selected_device_index_ >=
                            static_cast<int>(cached_devices_.size())) {
                        selection_committed = false;
                        device_selection_error_ =
                            "stage=inventory reason=Selected candidate is unavailable";
                    } else {
                        const auto& selected =
                            cached_devices_[selected_device_index_];
                        const auto type = static_cast<cyxwiz::DeviceType>(
                            selected.type);
                        if (compute_device_changed_callback_) {
                            selection_committed =
                                compute_device_changed_callback_(
                                    type,
                                    selected.device_id,
                                    selected.physical_fingerprint_known
                                        ? selected.physical_fingerprint
                                        : std::string{},
                                    device_selection_error_);
                        } else {
                            const auto result =
                                cyxwiz::CommitExecutionDeviceSelection(
                                    {type,
                                     selected.device_id,
                                     selected.physical_fingerprint_known
                                         ? selected.physical_fingerprint
                                         : std::string{}});
                            selection_committed = result.committed;
                            if (!selection_committed) {
                                device_selection_error_ =
                                    "stage=" +
                                    std::string(
                                        cyxwiz::DeviceSelectionTransactionStageName(
                                            result.stage)) +
                                    " reason=" + result.message;
                            }
                        }
                    }
                }
                if (selection_committed) {
                    if (save_project_settings_callback_) {
                        save_project_settings_callback_();
                    }
                    device_selection_dirty_ = false;
                    device_selection_recommendation_indices_.clear();
                    devices_initialized_ = false;
                    show_preferences_dialog_ = false;
                    spdlog::info("Preferences saved");
                } else {
                    device_selection_recommendation_indices_.clear();
                    if (selected_device_index_ >= 0 &&
                        selected_device_index_ <
                            static_cast<int>(cached_devices_.size())) {
                        std::vector<cyxwiz::DeviceInfo> inventory;
                        inventory.reserve(cached_devices_.size());
                        for (const auto& cached : cached_devices_) {
                            cyxwiz::DeviceInfo info;
                            info.type = static_cast<cyxwiz::DeviceType>(
                                cached.type);
                            info.device_id = cached.device_id;
                            info.name = cached.name;
                            info.kind = static_cast<cyxwiz::DeviceKind>(
                                cached.kind);
                            info.identity_confidence = static_cast<
                                cyxwiz::DeviceIdentityConfidence>(
                                cached.identity_confidence);
                            info.provider = cached.provider;
                            info.driver_version = cached.driver_version;
                            info.physical_fingerprint =
                                cached.physical_fingerprint;
                            info.hardware_vendor_id =
                                cached.hardware_vendor_id;
                            info.provider_known = cached.provider_known;
                            info.driver_version_known =
                                cached.driver_version_known;
                            info.physical_fingerprint_known =
                                cached.physical_fingerprint_known;
                            info.hardware_vendor_id_known =
                                cached.hardware_vendor_id_known;
                            inventory.push_back(std::move(info));
                        }
                        const auto recommendations =
                            cyxwiz::RecommendExecutionRoutes(
                                inventory[static_cast<size_t>(
                                    selected_device_index_)],
                                inventory,
                                cyxwiz::GetRouteQualificationSnapshot());
                        for (const auto& recommendation :
                             recommendations.recommendations) {
                            for (size_t index = 0;
                                 index < cached_devices_.size(); ++index) {
                                if (cached_devices_[index].type ==
                                        static_cast<int>(
                                            recommendation.route.type) &&
                                    cached_devices_[index].device_id ==
                                        recommendation.route.device_id) {
                                    device_selection_recommendation_indices_
                                        .push_back(static_cast<int>(index));
                                    break;
                                }
                            }
                        }
                    }
                    show_device_selection_error_ = true;
                }
            }
            ImGui::SameLine();
            if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular, button_width)) {
                device_selection_dirty_ = false;
                devices_initialized_ = false;
                selected_device_index_ = -1;
                selected_backend_type_ = -1;
                pending_oneapi_device_index_ = -1;
                device_selection_recommendation_indices_.clear();
                show_preferences_dialog_ = false;
            }

            if (show_device_selection_error_) {
                ImGui::OpenPopup("Device Selection Failed");
                show_device_selection_error_ = false;
            }
            ImGui::SetNextWindowPos(
                ImGui::GetMainViewport()->GetCenter(),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSizeConstraints(
                ImVec2(440.0f, 0.0f), ImVec2(640.0f, 420.0f));
            if (ImGui::BeginPopupModal(
                    "Device Selection Failed",
                    nullptr,
                    ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextColored(
                    ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                    "%s Selection was not changed",
                    ICON_FA_TRIANGLE_EXCLAMATION);
                ImGui::Separator();
                ImGui::TextWrapped("%s", device_selection_error_.c_str());
                if (!device_selection_recommendation_indices_.empty()) {
                    ImGui::Spacing();
                    ImGui::Text("Matrix-passed alternatives");
                    bool alternative_selected = false;
                    for (const int index :
                         device_selection_recommendation_indices_) {
                        if (index < 0 || index >=
                            static_cast<int>(cached_devices_.size())) {
                            continue;
                        }
                        const auto& alternative = cached_devices_[index];
                        const std::string label =
                            std::string("Select ") +
                            ArrayFireBackendDisplayName(
                                static_cast<cyxwiz::DeviceType>(
                                    alternative.type)) +
                            " / " + alternative.name + "##recommendation" +
                            std::to_string(index);
                        if (ui::SecondaryButton(label.c_str(), true, nullptr, ui::ButtonSize::Regular)) {
                            selected_device_index_ = index;
                            selected_backend_type_ = alternative.type;
                            device_selection_dirty_ = true;
                            alternative_selected = true;
                            ImGui::CloseCurrentPopup();
                            break;
                        }
                    }
                    if (alternative_selected) {
                        device_selection_recommendation_indices_.clear();
                    }
                    ImGui::TextDisabled(
                        "The alternative remains a candidate until OK is clicked.");
                }
                ImGui::Spacing();
                if (ui::SecondaryButton("Close", true, nullptr, ui::ButtonSize::Regular)) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            ImGui::EndPopup();
        }
        if (!show_preferences_dialog_) {
            device_selection_dirty_ = false;
            devices_initialized_ = false;
            selected_device_index_ = -1;
            selected_backend_type_ = -1;
            pending_oneapi_device_index_ = -1;
            device_selection_recommendation_indices_.clear();
        }
    }
}

} // namespace cyxwiz
