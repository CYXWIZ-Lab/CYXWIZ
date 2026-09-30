// Main menu bar drawn from the menu presentation model (TOFIX129 step 1.3).
//
// The model (core/menu_presentation) decides what every menu shows, in what
// state and with which chord. This file only builds the state snapshot,
// draws the model and maps action ids to the handlers the rest of the Engine
// already registers on ToolbarPanel.
#include "toolbar.h"

#include "../appearance_settings.h"
#include "../dock_style.h"
#include "../icons.h"
#include "../theme.h"
#include "../tutorial/tutorial_system.h"
#include "../../auth/auth_client.h"
#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/window_manager.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/registries/plugin_panel_registry.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace cyxwiz {

namespace {

// Sidebar panels grouped as View > Panels shows them. Names not listed go
// under "Other" in registration order.
const std::vector<std::pair<const char*, const char*>>& PanelGroups() {
    static const std::vector<std::pair<const char*, const char*>> groups = {
        {"CyxWiz Studio", "Workspace"}, {"Properties", "Workspace"}, {"Node Browser", "Workspace"},
        {"Node Info", "Workspace"}, {"Patterns", "Workspace"}, {"Asset Browser", "Workspace"},
        {"Script Editor", "Workspace"}, {"Console", "Workspace"}, {"Viewport", "Workspace"},
        {"Data Studio", "Data"}, {"Table Viewer", "Data"}, {"Data Explorer", "Data"},
        {"Annotation Editor", "Data"}, {"Visualizer", "Data"}, {"Query Console", "Data"},
        {"Variable Explorer", "Data"}, {"Plot Output", "Data"},
        {"Training", "Training"}, {"Studio Debugger", "Training"}, {"Tasks", "Training"},
        {"Jobs", "Network"}, {"P2P Training", "Network"}, {"Cloud Browser", "Network"},
        {"Cloud Manager", "Network"}, {"Wallet", "Network"},
        {"Plugin Manager", "Extensions"}};
    return groups;
}

// One FontAwesome icon per meaning. Actions without an entry draw no icon.
const char* IconFor(const std::string& id) {
    static const std::unordered_map<std::string, const char*> icons = {
        {"file.new_project", ICON_FA_FOLDER_PLUS}, {"file.open_project", ICON_FA_FOLDER_OPEN},
        {"file.recent", ICON_FA_CLOCK}, {"file.close_project", ICON_FA_FOLDER_MINUS},
        {"file.new_window", ICON_FA_WINDOW_RESTORE}, {"file.new_window_recent", ICON_FA_WINDOW_RESTORE},
        {"script.new", ICON_FA_FILE_CODE}, {"script.open", ICON_FA_FILE_IMPORT},
        {"file.save", ICON_FA_FLOPPY_DISK}, {"file.save_as", ICON_FA_FILE_EXPORT}, {"file.save_all", ICON_FA_COPY},
        {"file.auto_save", ICON_FA_CLOCK_ROTATE_LEFT}, {"file.import_model", ICON_FA_FILE_IMPORT},
        {"file.export", ICON_FA_FILE_EXPORT}, {"deploy.export", ICON_FA_FILE_EXPORT},
        {"file.preferences", ICON_FA_GEAR}, {"file.restart", ICON_FA_ROTATE_RIGHT}, {"file.exit", ICON_FA_XMARK},
        {"edit.undo", ICON_FA_ROTATE_LEFT}, {"edit.redo", ICON_FA_ROTATE_RIGHT}, {"edit.cut", ICON_FA_SCISSORS},
        {"edit.copy", ICON_FA_COPY}, {"edit.paste", ICON_FA_PASTE}, {"edit.delete", ICON_FA_TRASH},
        {"edit.select_all", ICON_FA_OBJECT_GROUP}, {"edit.find", ICON_FA_MAGNIFYING_GLASS},
        {"edit.replace", ICON_FA_RIGHT_LEFT}, {"edit.find_in_files", ICON_FA_FOLDER_TREE},
        {"edit.replace_in_files", ICON_FA_FOLDER_TREE}, {"edit.go_to_line", ICON_FA_HASHTAG},
        {"edit.lines", ICON_FA_BARS}, {"edit.transform", ICON_FA_FONT},
        {"edit.toggle_line_comment", ICON_FA_COMMENT}, {"edit.toggle_block_comment", ICON_FA_COMMENT},
        {"view.command_palette", ICON_FA_MAGNIFYING_GLASS}, {"view.panels", ICON_FA_TABLE_COLUMNS},
        {"view.layout", ICON_FA_GRIP_VERTICAL}, {"view.themes", ICON_FA_PALETTE},
        {"view.theme_editor", ICON_FA_BRUSH}, {"view.icon_packs", ICON_FA_GRIP}, {"view.minimaps", ICON_FA_EYE},
        {"view.fullscreen", ICON_FA_EXPAND},
        {"nodes.add_layer", ICON_FA_PLUS}, {"nodes.group", ICON_FA_OBJECT_GROUP},
        {"nodes.ungroup", ICON_FA_OBJECT_UNGROUP}, {"nodes.duplicate", ICON_FA_COPY},
        {"nodes.delete", ICON_FA_TRASH}, {"nodes.custom_node_editor", ICON_FA_CUBE},
        {"train.compile", ICON_FA_WRENCH}, {"train.local_debug", ICON_FA_BUG}, {"train.start", ICON_FA_PLAY},
        {"train.pause", ICON_FA_PAUSE}, {"train.resume", ICON_FA_PLAY}, {"train.stop", ICON_FA_STOP},
        {"train.test", ICON_FA_FLASK}, {"train.dashboard", ICON_FA_CHART_LINE},
        {"train.optimizer_settings", ICON_FA_SLIDERS}, {"train.hyperparam_search", ICON_FA_MAGNIFYING_GLASS_CHART},
        {"data.import", ICON_FA_FILE_IMPORT}, {"data.create_custom", ICON_FA_FOLDER_PLUS},
        {"data.statistics", ICON_FA_CHART_SIMPLE}, {"data.explore", ICON_FA_MAGNIFYING_GLASS_CHART},
        {"data.transform", ICON_FA_WAND_MAGIC_SPARKLES}, {"data.augment", ICON_FA_LAYER_GROUP},
        {"tools.model", ICON_FA_CUBES}, {"tools.evaluation", ICON_FA_CHECK_DOUBLE},
        {"tools.clustering", ICON_FA_CIRCLE_NODES}, {"tools.statistics", ICON_FA_CHART_BAR},
        {"tools.linear_algebra", ICON_FA_TABLE_CELLS}, {"tools.signal", ICON_FA_WAVE_SQUARE},
        {"tools.optimization", ICON_FA_ARROW_TREND_DOWN}, {"tools.time_series", ICON_FA_CHART_LINE},
        {"tools.text", ICON_FA_FONT}, {"tools.utilities", ICON_FA_TOOLBOX}, {"tools.monitoring", ICON_FA_GAUGE_HIGH},
        {"tools.diagnostics", ICON_FA_STETHOSCOPE},
        {"script.run", ICON_FA_PLAY}, {"script.stop", ICON_FA_STOP}, {"script.python_console", ICON_FA_TERMINAL},
        {"deploy.connect", ICON_FA_PLUG}, {"deploy.deploy", ICON_FA_CLOUD_ARROW_UP}, {"deploy.serving", ICON_FA_SERVER},
        {"deploy.convert", ICON_FA_RIGHT_LEFT}, {"deploy.quantize", ICON_FA_COMPRESS},
        {"deploy.marketplace", ICON_FA_TAG},
        {"sim.run", ICON_FA_PLAY}, {"sim.pause", ICON_FA_PAUSE}, {"sim.stop", ICON_FA_STOP},
        {"sim.step", ICON_FA_FORWARD_STEP}, {"sim.viewport", ICON_FA_DESKTOP}, {"sim.env_library", ICON_FA_MICROCHIP},
        {"sim.load_mjcf", ICON_FA_FILE_IMPORT},
        {"apps.plugin_manager", ICON_FA_PLUG},
        {"help.tutorials", ICON_FA_LIGHTBULB}, {"help.shortcuts", ICON_FA_KEYBOARD},
        {"help.documentation", ICON_FA_BOOK}, {"help.api_reference", ICON_FA_CODE}, {"help.report_issue", ICON_FA_BUG},
        {"help.check_updates", ICON_FA_DOWNLOAD}, {"help.about", ICON_FA_CIRCLE_INFO},
        {"account.settings", ICON_FA_USER}, {"account.wallet", ICON_FA_WALLET},
        {"account.sign_out", ICON_FA_RIGHT_FROM_BRACKET}, {"account.sign_in", ICON_FA_RIGHT_TO_BRACKET}};
    auto it = icons.find(id);
    return it == icons.end() ? "" : it->second;
}

void OpenUrl(const char* url) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    std::system((std::string("open ") + url).c_str());
#else
    std::system((std::string("xdg-open ") + url).c_str());
#endif
}

void ShowSidebarPanel(const std::string& name, bool toggle) {
    for (const auto& panel : gui::GetDockStyle().GetPanels()) {
        if (panel.name != name) continue;
        if (panel.visible_ptr) *panel.visible_ptr = toggle ? !*panel.visible_ptr : true;
        if (panel.on_toggle) panel.on_toggle();
        return;
    }
    spdlog::warn("Menu: no panel named '{}'", name);
}

}  // namespace

// ---------------------------------------------------------------------------
// State snapshot

menu::MenuInputs ToolbarPanel::BuildMenuInputs() const {
    menu::MenuInputs in;
    if (menu_state_provider_) {
        const MenuStateSnapshot s = menu_state_provider_();
        in.focus = s.focus;
        in.selected_nodes = s.selected_nodes;
        in.training = s.training;
        in.script_open = s.script_open;
        in.script_running = s.script_running;
    }
    auto& pm = ProjectManager::Instance();
    in.has_project = pm.HasActiveProject();
    in.signed_in = is_logged_in_;
    in.user_display_name = logged_in_user_;
#ifdef CYXWIZ_HAS_ONNX_EXPORT
    in.onnx_export_available = true;
#endif
    auto& panel_reg = plugin::PluginPanelRegistry::Instance();
    in.mujoco_loaded = panel_reg.HasPanel("mujoco_viewport");
    in.mujoco_viewport_visible = in.mujoco_loaded && panel_reg.IsPanelVisible("mujoco_viewport");
    in.mujoco_env_visible = in.mujoco_loaded && panel_reg.IsPanelVisible("mujoco_env_browser");
    in.auto_save = auto_save_enabled_;
    in.studio_minimap = node_editor_minimap_ptr_ && *node_editor_minimap_ptr_;
    in.script_minimap = script_editor_minimap_ptr_ && *script_editor_minimap_ptr_;
    in.idle_log = idle_log_ptr_ && *idle_log_ptr_;
    in.verbose_python = verbose_python_log_ptr_ && *verbose_python_log_ptr_;

    // Panels: sidebar registry, grouped and ordered as View > Panels shows them.
    {
        const auto& groups = PanelGroups();
        const auto& registered = gui::GetDockStyle().GetPanels();
        std::vector<menu::PanelEntry> listed;
        for (const auto& [name, group] : groups) {
            for (const auto& p : registered) {
                if (p.name != name) continue;
                listed.push_back({p.name, group, p.visible_ptr && *p.visible_ptr, ""});
            }
        }
        for (const auto& p : registered) {
            if (p.name == "Command Palette") continue;  // an action, not a panel
            bool known = false;
            for (const auto& [name, group] : groups) if (p.name == name) known = true;
            if (!known) listed.push_back({p.name, "Other", p.visible_ptr && *p.visible_ptr, ""});
        }
        in.panels = std::move(listed);
    }

    for (const auto& rp : pm.GetRecentProjects()) in.recent_projects.push_back({rp.path, rp.name, false, ""});

    {
        const auto current = gui::GetTheme().GetCurrentPreset();
        for (auto preset : gui::Theme::GetAvailablePresets()) {
            in.themes.push_back({std::to_string(static_cast<int>(preset)), gui::Theme::GetPresetName(preset),
                                 preset == current, gui::Theme::GetPresetGroup(preset)});
        }
    }
    {
        const int current = get_icon_pack_callback_ ? get_icon_pack_callback_() : 0;
        const char* packs[] = {"FontAwesome (Default)", "Tabler Icons", "Remix Icon",
                               "Lucide Icons", "Iconoir", "Phosphor Icons"};
        for (int i = 0; i < 6; ++i) in.icon_packs.push_back({std::to_string(i), packs[i], current == i, ""});
    }
    {
        auto& tutorials = TutorialSystem::Instance();
        for (const auto& t : tutorials.GetAvailableTutorials())
            in.tutorials.push_back({t.id, t.name, tutorials.IsTutorialComplete(t.id), ""});
    }
    {
        std::map<std::string, std::vector<plugin::PluginPanelInfo>> by_category;
        for (const auto& p : panel_reg.GetAllPanels())
            by_category[p.category.empty() ? "Other" : p.category].push_back(p);
        for (const auto& [category, panels] : by_category)
            for (const auto& p : panels)
                in.plugin_panels.push_back({p.panel_id, p.title, panel_reg.IsPanelVisible(p.panel_id), category});
    }
    return in;
}

// ---------------------------------------------------------------------------
// Drawing

void ToolbarPanel::RenderMenuItems(const std::vector<menu::MenuItem>& items) {
    using Kind = menu::MenuItem::Kind;
    for (const auto& item : items) {
        switch (item.kind) {
            case Kind::Separator:
                ImGui::Separator();
                break;
            case Kind::Header:
                ImGui::TextDisabled("%s", item.label.c_str());
                break;
            case Kind::Submenu: {
                const std::string label = std::string(IconFor(item.id)) + (IconFor(item.id)[0] ? " " : "") + item.label;
                const bool open = ImGui::BeginMenu(label.c_str(), item.enabled);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    status_hint_ = item.enabled ? item.hint : "Not available: " + item.disabled_reason;
                    if (!item.enabled) ImGui::SetTooltip("%s", item.disabled_reason.c_str());
                }
                if (open) {
                    RenderMenuItems(item.children);
                    ImGui::EndMenu();
                }
                break;
            }
            case Kind::Action: {
                std::string label = std::string(IconFor(item.id)) + (IconFor(item.id)[0] ? " " : "") + item.label;
                if (item.planned) label += "  (planned)";
                ImGui::PushID(item.argument.empty() ? item.id.c_str() : (item.id + "#" + item.argument).c_str());
                const bool clicked = ImGui::MenuItem(label.c_str(), item.shortcut.empty() ? nullptr : item.shortcut.c_str(),
                                                     item.checked, item.enabled);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    if (item.planned) {
                        status_hint_ = "Planned, not available yet. " + item.hint;
                        ImGui::SetTooltip("Planned: not available yet.");
                    } else if (!item.enabled) {
                        status_hint_ = "Not available: " + item.disabled_reason;
                        ImGui::SetTooltip("%s", item.disabled_reason.c_str());
                    } else {
                        status_hint_ = item.hint;
                    }
                }
                ImGui::PopID();
                if (clicked) Dispatch(item.id, item.argument);
                break;
            }
        }
    }
}

void ToolbarPanel::RenderMenuBar() {
    status_hint_.clear();
    const menu::MenuModel model = menu::BuildMenuModel(BuildMenuInputs());

    if (!ImGui::BeginMainMenuBar()) return;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 6.0f));
    for (const auto& m : model.menus) {
        if (ImGui::BeginMenu(m.label.c_str())) {
            RenderMenuItems(m.items);
            ImGui::EndMenu();
        }
    }
    ImGui::PopStyleVar();

    // Right side: project name, then the account menu.
    auto& pm = ProjectManager::Instance();
    const std::string project = pm.HasActiveProject() ? pm.GetProjectName() : "";
    const std::string account_label = is_logged_in_ ? std::string(ICON_FA_USER) : std::string("Sign in");
    const float right_width = ImGui::CalcTextSize(account_label.c_str()).x +
                              (project.empty() ? 0.0f : ImGui::CalcTextSize(project.c_str()).x + 40.0f) + 40.0f;
    const float x = ImGui::GetWindowContentRegionMax().x - right_width;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    if (!project.empty()) {
        ImGui::TextDisabled("%s %s", ICON_FA_FOLDER, project.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pm.GetProjectFilePath().c_str());
    }
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 6.0f));
    if (ImGui::BeginMenu(account_label.c_str())) {
        RenderMenuItems(model.account.items);
        ImGui::EndMenu();
    }
    if (ImGui::IsItemHovered() && is_logged_in_) ImGui::SetTooltip("%s", logged_in_user_.c_str());
    ImGui::PopStyleVar();
    ImGui::EndMainMenuBar();
}

// ---------------------------------------------------------------------------
// Dispatch

void ToolbarPanel::Dispatch(const std::string& id, const std::string& argument) {
    if (action_handlers_.empty()) BuildActionHandlers();
    auto it = action_handlers_.find(id);
    if (it == action_handlers_.end()) {
        spdlog::warn("Menu action '{}' has no handler", id);
        return;
    }
    it->second(argument);
}

void ToolbarPanel::OpenScriptFromDialog() {
    auto result = FileDialogs::OpenScript();
    if (result) {
        if (open_script_in_editor_callback_) {
            open_script_in_editor_callback_(*result);
        }
        spdlog::info("Opening script: {}", *result);
    }
}

void ToolbarPanel::BuildActionHandlers() {
    auto& h = action_handlers_;
    auto call = [&h](const char* id, std::function<void()>& fn) {
        h[id] = [&fn](const std::string&) { if (fn) fn(); };
    };
    auto login_gated = [this, &h](const char* id, std::function<void()>& fn, const char* what) {
        h[id] = [this, &fn, what](const std::string&) {
            if (!is_logged_in_) {
                show_login_required_popup_ = true;
                login_required_action_ = what;
            } else if (fn) {
                fn();
            }
        };
    };

    // File
    h["file.new_project"] = [this](const std::string&) { show_new_project_dialog_ = true; };
    h["file.open_project"] = [](const std::string&) {
        auto result = FileDialogs::OpenProject();
        if (!result) return;
        if (ProjectManager::Instance().OpenProject(*result)) spdlog::info("Project opened: {}", *result);
        else spdlog::error("Failed to open project: {}", *result);
    };
    h["file.open_recent"] = [](const std::string& path) {
        if (ProjectManager::Instance().OpenProject(path)) spdlog::info("Opened recent project: {}", path);
        else spdlog::error("Failed to open recent project: {}", path);
    };
    h["file.clear_recent"] = [](const std::string&) { ProjectManager::Instance().ClearRecentProjects(); };
    h["file.close_project"] = [this](const std::string&) {
        if (save_project_settings_callback_) save_project_settings_callback_();
        ProjectManager::Instance().CloseProject();
    };
    h["file.new_window"] = [](const std::string&) {
        if (core::WindowManager::LaunchWindowWithDialog()) spdlog::info("Launched new window");
        else spdlog::error("Failed to launch new window");
    };
    h["file.open_in_new_window"] = [](const std::string& path) {
        if (core::WindowManager::LaunchWindow(path)) spdlog::info("Launched new window with project: {}", path);
        else spdlog::error("Failed to launch new window with project: {}", path);
    };
    h["script.new"] = [this](const std::string&) {
        show_new_script_dialog_ = true;
        std::memset(new_script_name_, 0, sizeof(new_script_name_));
        new_script_type_ = 0;
    };
    h["script.open"] = [this](const std::string&) { OpenScriptFromDialog(); };
    h["file.save"] = [this](const std::string&) {
        if (save_project_settings_callback_) save_project_settings_callback_();
        else if (ProjectManager::Instance().SaveProject()) spdlog::info("Project saved");
    };
    h["file.save_as"] = [this](const std::string&) {
        if (save_project_settings_callback_) save_project_settings_callback_();
        show_save_as_dialog_ = true;
        auto& pm = ProjectManager::Instance();
        std::strncpy(save_as_name_buffer_, (pm.GetProjectName() + "_copy").c_str(), sizeof(save_as_name_buffer_) - 1);
        save_as_name_buffer_[sizeof(save_as_name_buffer_) - 1] = '\0';
        const std::string parent = std::filesystem::path(pm.GetProjectRoot()).parent_path().string();
        std::strncpy(save_as_path_buffer_, parent.c_str(), sizeof(save_as_path_buffer_) - 1);
        save_as_path_buffer_[sizeof(save_as_path_buffer_) - 1] = '\0';
    };
    call("file.save_all", save_all_callback_);
    h["file.auto_save"] = [this](const std::string&) {
        auto_save_enabled_ = !auto_save_enabled_;
        spdlog::info("Auto-save {}", auto_save_enabled_ ? "enabled" : "disabled");
    };
    call("file.import_model", import_model_callback_);
    h["export.cyxmodel"] = [this](const std::string&) { if (export_model_callback_) export_model_callback_(0); };
    h["export.safetensors"] = [this](const std::string&) { if (export_model_callback_) export_model_callback_(1); };
    h["export.onnx"] = [this](const std::string&) { if (export_model_callback_) export_model_callback_(2); };
    h["file.preferences"] = [this](const std::string&) { OpenPreferences(); };
    h["file.restart"] = [this](const std::string&) {
        auto& pm = ProjectManager::Instance();
        const std::string project_path = pm.GetProjectFilePath();
        if (save_project_settings_callback_) save_project_settings_callback_();
        pm.SaveProject();
        if (project_path.empty()) return;
        spdlog::info("Restarting engine with project: {}", project_path);
        if (core::WindowManager::RestartEngine(project_path)) {
            if (exit_callback_) exit_callback_();
        } else {
            spdlog::error("Failed to restart engine");
        }
    };
    h["file.exit"] = [this](const std::string&) {
        const bool has_unsaved = has_unsaved_changes_callback_ && has_unsaved_changes_callback_();
        if (has_unsaved) show_exit_confirmation_dialog_ = true;
        else if (exit_callback_) exit_callback_();
    };

    // Edit
    call("edit.undo", undo_callback_);
    call("edit.redo", redo_callback_);
    call("edit.cut", cut_callback_);
    call("edit.copy", copy_callback_);
    call("edit.paste", paste_callback_);
    call("edit.delete", delete_callback_);
    call("edit.select_all", select_all_callback_);
    h["edit.find"] = [this](const std::string&) { OpenFindDialog(); };
    h["edit.replace"] = [this](const std::string&) { OpenReplaceDialog(); };
    h["edit.find_in_files"] = [this](const std::string&) { OpenFindInFilesDialog(); };
    h["edit.replace_in_files"] = [this](const std::string&) { OpenReplaceInFilesDialog(); };
    h["edit.go_to_line"] = [this](const std::string&) { OpenGoToLineDialog(); };
    call("edit.duplicate_line", duplicate_line_callback_);
    call("edit.move_line_up", move_line_up_callback_);
    call("edit.move_line_down", move_line_down_callback_);
    call("edit.indent", indent_callback_);
    call("edit.outdent", outdent_callback_);
    call("edit.join_lines", join_lines_callback_);
    call("edit.sort_asc", sort_lines_asc_callback_);
    call("edit.sort_desc", sort_lines_desc_callback_);
    call("edit.uppercase", transform_uppercase_callback_);
    call("edit.lowercase", transform_lowercase_callback_);
    call("edit.titlecase", transform_titlecase_callback_);
    call("edit.toggle_line_comment", toggle_line_comment_callback_);
    call("edit.toggle_block_comment", toggle_block_comment_callback_);

    // View
    h["view.command_palette"] = [this](const std::string&) { show_command_palette_ = true; };
    h["view.panel"] = [](const std::string& name) { ShowSidebarPanel(name, true); };
    call("view.save_layout", save_layout_callback_);
    call("view.reset_layout", reset_layout_callback_);
    h["view.theme"] = [this](const std::string& preset_id) {
        const auto preset = static_cast<gui::ThemePreset>(std::stoi(preset_id));
        gui::SetThemePreset(preset);
        spdlog::info("Theme changed to: {}", gui::Theme::GetPresetName(preset));
        if (app_theme_changed_callback_) app_theme_changed_callback_(static_cast<int>(preset));
    };
    call("view.theme_editor", open_theme_editor_callback_);
    h["view.icon_pack"] = [this](const std::string& pack) {
        if (set_icon_pack_callback_) set_icon_pack_callback_(std::stoi(pack));
    };
    h["view.studio_minimap"] = [this](const std::string&) {
        if (node_editor_minimap_ptr_) *node_editor_minimap_ptr_ = !*node_editor_minimap_ptr_;
    };
    h["view.script_minimap"] = [this](const std::string&) {
        if (script_editor_minimap_ptr_) *script_editor_minimap_ptr_ = !*script_editor_minimap_ptr_;
    };

    // Nodes
    call("nodes.add_dense", add_dense_node_callback_);
    call("nodes.add_conv", add_conv_node_callback_);
    call("nodes.add_pooling", add_pooling_node_callback_);
    call("nodes.add_dropout", add_dropout_node_callback_);
    call("nodes.add_batchnorm", add_batchnorm_node_callback_);
    call("nodes.add_attention", add_attention_node_callback_);
    call("nodes.group", group_nodes_callback_);
    call("nodes.ungroup", ungroup_nodes_callback_);
    call("nodes.duplicate", duplicate_nodes_callback_);
    call("nodes.delete", delete_nodes_callback_);
    call("nodes.custom_node_editor", open_custom_node_editor_callback_);

    // Train
    call("train.compile", compile_graph_callback_);
    call("train.local_debug", local_debug_callback_);
    call("train.start", start_training_callback_);
    call("train.pause", pause_training_callback_);
    call("train.resume", resume_training_callback_);
    call("train.stop", stop_training_callback_);
    call("train.run_test", run_test_callback_);
    call("train.quick_test", run_quick_test_callback_);
    call("train.view_test_results", view_test_results_callback_);
    call("train.compare_test_results", compare_test_results_callback_);
    call("train.export_test_report", export_test_report_callback_);
    call("train.load_checkpoint", load_checkpoint_callback_);
    call("train.dashboard", training_settings_callback_);
    call("train.optimizer_settings", optimizer_settings_callback_);
    call("train.hyperparam_search", open_hyperparam_search_callback_);

    // Data
    call("data.import", import_dataset_callback_);
    call("data.create_custom", create_custom_dataset_callback_);
    call("data.statistics", dataset_statistics_callback_);
    call("data.profiler", open_data_profiler_callback_);
    call("data.missing_values", open_missing_value_callback_);
    call("data.outliers", open_outlier_detection_callback_);
    call("data.correlation", open_correlation_matrix_callback_);
    call("data.normalization", open_normalization_callback_);
    call("data.standardization", open_standardization_callback_);
    call("data.log_transform", open_log_transform_callback_);
    call("data.boxcox", open_boxcox_callback_);
    call("data.feature_scaling", open_feature_scaling_callback_);

    // Tools
    call("tools.model_summary", open_model_summary_callback_);
    call("tools.flops", open_model_summary_callback_);
    call("tools.architecture_diagram", open_architecture_diagram_callback_);
    call("tools.lr_finder", open_lr_finder_callback_);
    call("tools.gradcam", open_gradcam_callback_);
    call("tools.saliency", open_gradcam_callback_);
    call("tools.nas", open_nas_callback_);
    call("tools.architecture_suggestions", open_nas_callback_);
    call("tools.dnn_inference", open_dnn_inference_callback_);
    call("tools.cross_validation", open_cross_validation_callback_);
    call("tools.confusion_matrix", open_confusion_matrix_callback_);
    call("tools.roc_auc", open_roc_auc_callback_);
    call("tools.pr_curve", open_pr_curve_callback_);
    call("tools.learning_curves", open_learning_curves_callback_);
    call("tools.feature_importance", open_feature_importance_callback_);
    call("tools.kmeans", open_kmeans_callback_);
    call("tools.dbscan", open_dbscan_callback_);
    call("tools.hierarchical", open_hierarchical_callback_);
    call("tools.gmm", open_gmm_callback_);
    call("tools.cluster_eval", open_cluster_eval_callback_);
    call("tools.dim_reduction", open_dim_reduction_callback_);
    call("tools.descriptive_stats", open_descriptive_stats_callback_);
    call("tools.hypothesis_test", open_hypothesis_test_callback_);
    call("tools.regression", open_regression_callback_);
    call("tools.distribution_fitter", open_distribution_fitter_callback_);
    call("tools.matrix_calculator", open_matrix_calculator_callback_);
    call("tools.eigen", open_eigen_decomp_callback_);
    call("tools.svd", open_svd_callback_);
    call("tools.qr", open_qr_callback_);
    call("tools.cholesky", open_cholesky_callback_);
    call("tools.fft", open_fft_callback_);
    call("tools.spectrogram", open_spectrogram_callback_);
    call("tools.filter_designer", open_filter_designer_callback_);
    call("tools.convolution", open_convolution_callback_);
    call("tools.wavelet", open_wavelet_callback_);
    call("tools.gradient_descent", open_gradient_descent_callback_);
    call("tools.convexity", open_convexity_callback_);
    call("tools.lp", open_lp_callback_);
    call("tools.qp", open_qp_callback_);
    call("tools.differentiation", open_differentiation_callback_);
    call("tools.integration", open_integration_callback_);
    call("tools.decomposition", open_decomposition_callback_);
    call("tools.acf_pacf", open_acf_pacf_callback_);
    call("tools.stationarity", open_stationarity_callback_);
    call("tools.seasonality", open_seasonality_callback_);
    call("tools.forecasting", open_forecasting_callback_);
    call("tools.tokenization", open_tokenization_callback_);
    call("tools.word_frequency", open_word_frequency_callback_);
    call("tools.tfidf", open_tfidf_callback_);
    call("tools.embeddings", open_embeddings_callback_);
    call("tools.sentiment", open_sentiment_callback_);
    call("tools.language_model", open_language_model_generation_callback_);
    call("tools.calculator", open_calculator_callback_);
    call("tools.unit_converter", open_unit_converter_callback_);
    call("tools.random_generator", open_random_generator_callback_);
    call("tools.hash_generator", open_hash_generator_callback_);
    call("tools.json_viewer", open_json_viewer_callback_);
    call("tools.regex_tester", open_regex_tester_callback_);
    call("tools.profiler", open_profiler_callback_);
    call("tools.memory_panel", open_memory_panel_callback_);
    call("tools.system_monitor", open_memory_monitor_callback_);
    call("tools.clear_cache", clear_cache_callback_);
    call("tools.gc", run_gc_callback_);
    h["tools.idle_log"] = [this](const std::string&) {
        if (!idle_log_ptr_) return;
        *idle_log_ptr_ = !*idle_log_ptr_;
        spdlog::info("Idle mode logging {}", *idle_log_ptr_ ? "ENABLED" : "DISABLED");
    };
    h["tools.verbose_python"] = [this](const std::string&) {
        if (!verbose_python_log_ptr_) return;
        *verbose_python_log_ptr_ = !*verbose_python_log_ptr_;
        spdlog::info("Verbose Python logging {}", *verbose_python_log_ptr_ ? "ENABLED" : "DISABLED");
    };

    // Script
    call("script.run", run_script_callback_);
    call("script.stop", stop_script_callback_);
    call("script.python_console", open_python_console_callback_);

    // Deploy
    login_gated("deploy.connect", connect_to_server_callback_, "connect to the server");
    login_gated("deploy.deploy", deploy_to_server_callback_, "deploy to a server node");
    call("deploy.serving", open_serving_callback_);
    call("deploy.convert_bin_to_dir", convert_binary_to_dir_callback_);
    call("deploy.convert_dir_to_bin", convert_dir_to_binary_callback_);

    // Simulation
    h["sim.viewport"] = [](const std::string&) {
        plugin::PluginPanelRegistry::Instance().TogglePanelVisible("mujoco_viewport");
    };
    h["sim.env_library"] = [](const std::string&) {
        plugin::PluginPanelRegistry::Instance().TogglePanelVisible("mujoco_env_browser");
    };

    // Apps
    h["apps.panel"] = [](const std::string& panel_id) {
        plugin::PluginPanelRegistry::Instance().TogglePanelVisible(panel_id);
    };
    h["apps.plugin_manager"] = [](const std::string&) { ShowSidebarPanel("Plugin Manager", false); };

    // Help
    h["help.tutorial"] = [](const std::string& id) {
        TutorialSystem::Instance().StartTutorial(id);
        spdlog::info("Started tutorial: {}", id);
    };
    h["help.browse_tutorials"] = [](const std::string&) { TutorialSystem::Instance().OpenTutorialBrowser(); };
    h["help.shortcuts"] = [this](const std::string&) { OpenPreferences("Shortcuts"); };
    h["help.report_issue"] = [](const std::string&) { OpenUrl("https://github.com/CYXWIZ-Lab/CYXWIZ/issues"); };
    h["help.about"] = [this](const std::string&) { show_about_dialog_ = true; };

    // Account
    h["account.settings"] = [this](const std::string&) { show_account_settings_dialog_ = true; };
    h["account.sign_in"] = [this](const std::string&) { show_account_settings_dialog_ = true; };
    h["account.wallet"] = [](const std::string&) { ShowSidebarPanel("Wallet", false); };
    h["account.sign_out"] = [this](const std::string&) {
        auth::AuthClient::Instance().Logout();
        is_logged_in_ = false;
        logged_in_user_.clear();
        spdlog::info("User signed out");
        if (on_logout_callback_) on_logout_callback_();
    };
}

}  // namespace cyxwiz
