// Main menu bar drawn from the menu presentation model (TOFIX129 step 1.3).
//
// The model (core/menu_presentation) decides what every menu shows, in what
// state and with which chord. This file only builds the state snapshot,
// draws the model and maps action ids to the handlers the rest of the Engine
// already registers on ToolbarPanel.
#include "../../core/engine_config.h"
#include "toolbar.h"

#include "../appearance_settings.h"
#include "../dock_style.h"
#include "../icons.h"
#include "../theme.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../tutorial/tutorial_system.h"
#include "../../auth/auth_client.h"
#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/window_manager.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/plugin_session.h"
#include "../../plugin/registries/plugin_panel_registry.h"
#include "../../plugin/registries/plugin_command_registry.h"
#include "../separate_windows.h"
#include "../python_bundle_ui.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
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

// One FontAwesome icon per meaning. Actions without an entry draw no icon.
const char* ToolbarPanel::IconForAction(const std::string& id) {
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
        {"view.layout", ICON_FA_GRIP_VERTICAL}, {"view.window", ICON_FA_WINDOW_RESTORE}, {"view.themes", ICON_FA_PALETTE},
        {"view.theme_editor", ICON_FA_BRUSH}, {"view.icon_packs", ICON_FA_GRIP}, {"view.minimaps", ICON_FA_EYE},
        {"view.fullscreen", ICON_FA_EXPAND},
        {"nodes.add_layer", ICON_FA_PLUS}, {"nodes.group", ICON_FA_OBJECT_GROUP},
        {"nodes.ungroup", ICON_FA_OBJECT_UNGROUP}, {"nodes.duplicate", ICON_FA_COPY},
        {"nodes.delete", ICON_FA_TRASH}, {"nodes.custom_node_editor", ICON_FA_CUBE},
        {"nodes.freeze", ICON_FA_SNOWFLAKE},
        {"train.compile", ICON_FA_WRENCH}, {"train.local_debug", ICON_FA_BUG}, {"train.start", ICON_FA_PLAY},
        {"train.pause", ICON_FA_PAUSE}, {"train.resume", ICON_FA_PLAY}, {"train.stop", ICON_FA_STOP},
        {"train.test", ICON_FA_FLASK}, {"train.evaluate", ICON_FA_ROTATE}, {"train.dashboard", ICON_FA_CHART_LINE},
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
        {"apps.plugin_manager", ICON_FA_PLUG},
        {"apps.plugin", ICON_FA_PLUG},
        {"help.tutorials", ICON_FA_LIGHTBULB}, {"help.shortcuts", ICON_FA_KEYBOARD},
        {"help.documentation", ICON_FA_BOOK}, {"help.api_reference", ICON_FA_CODE}, {"help.report_issue", ICON_FA_BUG},
        {"help.check_updates", ICON_FA_DOWNLOAD}, {"help.about", ICON_FA_CIRCLE_INFO},
        {"help.whats_new", ICON_FA_GIFT}, {"help.known_issues", ICON_FA_TRIANGLE_EXCLAMATION},
        {"help.open_logs", ICON_FA_FILE_LINES}, {"view.fullscreen", ICON_FA_EXPAND},
        {"file.show_project_folder", ICON_FA_FOLDER_OPEN}, {"file.copy_project_path", ICON_FA_COPY},
        {"account.servers", ICON_FA_SERVER},
        {"account.settings", ICON_FA_USER}, {"account.wallet", ICON_FA_WALLET},
        {"account.sign_out", ICON_FA_RIGHT_FROM_BRACKET}, {"account.sign_in", ICON_FA_RIGHT_TO_BRACKET}};
    auto it = icons.find(id);
    return it == icons.end() ? "" : it->second;
}

namespace {

// Help > Documentation, API Reference, What's New: the guides online (only
// KNOWN_ISSUES.md ships beside the Engine).
constexpr const char* kDocsUrl = "https://github.com/CYXWIZ-Lab/CYXWIZ/blob/HEAD/docs/usage/README.md";
constexpr const char* kApiUrl = "https://github.com/CYXWIZ-Lab/CYXWIZ/blob/HEAD/docs/extending/python-api-reference.md";
constexpr const char* kReleaseNotesUrl = "https://github.com/CYXWIZ-Lab/CYXWIZ/blob/HEAD/docs/RELEASE_NOTES.md";

// KNOWN_ISSUES.md beside the Engine (a release), or up its folders (a
// development build); empty when there is none. Looked up once.
const std::string& KnownIssuesPath() {
    static const std::string path = [] {
        std::error_code ec;
        std::filesystem::path dir = core::ExecutableDirectory();
        for (int up = 0; up < 5 && !dir.empty(); ++up) {
            const auto candidate = dir / "KNOWN_ISSUES.md";
            if (std::filesystem::is_regular_file(candidate, ec)) return candidate.string();
            if (dir.parent_path() == dir) break;
            dir = dir.parent_path();
        }
        return std::string();
    }();
    return path;
}

// View > Full Screen: the main window without its frame over the whole
// monitor it is on (not exclusive full screen, which minimises itself when a
// panel's own window takes the focus). F11 again puts it back as it was.
struct WindowedPlacement {
    bool fullscreen = false;
    bool maximized = false;
    int x = 100, y = 100, width = 1280, height = 720;
};
WindowedPlacement& Placement() {
    static WindowedPlacement placement;
    return placement;
}
GLFWwindow* MainGlfwWindow() { return static_cast<GLFWwindow*>(ImGui::GetMainViewport()->PlatformHandle); }
bool IsFullscreen() { return Placement().fullscreen; }
void ToggleFullscreen() {
    GLFWwindow* window = MainGlfwWindow();
    if (!window) return;
    WindowedPlacement& p = Placement();
    if (p.fullscreen) {
        glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);
        glfwSetWindowMonitor(window, nullptr, p.x, p.y, p.width, p.height, GLFW_DONT_CARE);
        if (p.maximized) glfwMaximizeWindow(window);
        p.fullscreen = false;
        spdlog::info("Full screen off");
        return;
    }
    p.maximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE;
    if (p.maximized) glfwRestoreWindow(window);
    glfwGetWindowPos(window, &p.x, &p.y);
    glfwGetWindowSize(window, &p.width, &p.height);
    // The monitor under the window's centre.
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    GLFWmonitor* target = glfwGetPrimaryMonitor();
    const int cx = p.x + p.width / 2, cy = p.y + p.height / 2;
    for (int i = 0; i < count; ++i) {
        int mx = 0, my = 0;
        glfwGetMonitorPos(monitors[i], &mx, &my);
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        if (mode && cx >= mx && cx < mx + mode->width && cy >= my && cy < my + mode->height) target = monitors[i];
    }
    const GLFWvidmode* mode = target ? glfwGetVideoMode(target) : nullptr;
    if (!mode) return;
    int mx = 0, my = 0;
    glfwGetMonitorPos(target, &mx, &my);
    glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
    glfwSetWindowPos(window, mx, my);
    glfwSetWindowSize(window, mode->width, mode->height);
    p.fullscreen = true;
    spdlog::info("Full screen on ({} x {})", mode->width, mode->height);
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
        in.simulation_graph = s.simulation_graph;
        in.simulation_graph_name = s.simulation_graph_name;
        in.simulation_commands = s.simulation_commands;
    }
    auto& pm = ProjectManager::Instance();
    in.has_project = pm.HasActiveProject();
    in.signed_in = is_logged_in_;
    if (is_logged_in_) {
        const auth::UserInfo user = auth::AuthClient::Instance().GetUserInfo();
        in.user_name = user.name;
        in.user_username = user.username;
        in.user_email = user.email.empty() ? logged_in_user_ : user.email;
        in.user_role = user.role;
    }
    in.fullscreen = IsFullscreen();
    in.known_issues_found = !KnownIssuesPath().empty();
#ifdef CYXWIZ_HAS_ONNX_EXPORT
    in.onnx_export_available = true;
#endif
    auto& panel_reg = plugin::PluginPanelRegistry::Instance();
    in.auto_save = auto_save_enabled_;
    in.focused_window = ::gui::FocusedWindowName();
    in.focused_window_may_leave = ::gui::WindowMayLeave(in.focused_window.c_str());
    in.focused_window_outside = ::gui::WindowIsOutside(in.focused_window.c_str());
    in.studio_minimap = node_editor_minimap_ptr_ && *node_editor_minimap_ptr_;
    in.script_minimap = script_editor_minimap_ptr_ && *script_editor_minimap_ptr_;
    in.idle_log = idle_log_ptr_ && *idle_log_ptr_;
    in.verbose_python = verbose_python_log_ptr_ && *verbose_python_log_ptr_;
    in.frame_overlay = frame_overlay_ptr_ && *frame_overlay_ptr_;

    // Panels: the sidebar registry, already in group order (main_window.cpp).
    for (const auto& p : gui::GetDockStyle().GetPanels()) {
        if (!p.visible_ptr) continue;  // actions such as the Command Palette
        in.panels.push_back({p.name, p.group.empty() ? "Other" : p.group, *p.visible_ptr, p.shortcut});
        if (p.name == "Simulation Monitor") in.simulation_monitor_visible = *p.visible_ptr;
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
    for (const auto& c : plugin::PluginCommandRegistry::Instance().Snapshot()) {
        // The key is "<plugin id>:<command id>"
        const auto* owner = plugin::PluginManager::Instance().GetLoadedPlugin(c.key.substr(0, c.key.rfind(':')));
        in.plugin_commands.push_back({c.key, c.info.menu, c.info.group, c.info.label, c.info.hint, c.state.enabled,
                                      c.state.disabled_reason, c.info.checkable, c.state.checked,
                                      owner ? owner->manifest.name : std::string(), c.info.replaced_by_graph_run});
    }
    for (const auto* p : plugin::PluginManager::Instance().GetAllPlugins()) {
        if (!p) continue;
        const auto adds = plugin::DescribeContribution(p->manifest.id);
        in.loaded_plugins.push_back({p->manifest.id, p->manifest.name, cyxwiz::PluginMenuDetail(adds),
                                     adds.trainable_nodes + adds.simulation_nodes > 0});
    }
    for (const auto& b : python_ui::MenuEntries()) in.python_bundles.push_back({b.id, b.name, b.detail, true});
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
                const std::string label = std::string(IconForAction(item.id)) + (IconForAction(item.id)[0] ? " " : "") + item.label;
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
                std::string label = std::string(IconForAction(item.id)) + (IconForAction(item.id)[0] ? " " : "") + item.label;
                if (item.planned) label += "  (planned)";
                ImGui::PushID(item.argument.empty() ? item.id.c_str() : (item.id + "#" + item.argument).c_str());
                // The right column holds the chord, or a note such as "2 nodes".
                const std::string& right = item.shortcut.empty() ? item.detail : item.shortcut;
                const bool clicked = ImGui::MenuItem(label.c_str(), right.empty() ? nullptr : right.c_str(),
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

    // Right side (TOFIX129 A3b): the project chip, then the account in the
    // top-right corner, flush with the window's right edge.
    const ui::Tokens& t = ui::CurrentTokens();
    auto& pm = ProjectManager::Instance();
    const std::string project = pm.HasActiveProject() ? pm.GetProjectName() : "";
    const menu::AccountChip& chip = model.account_chip;
    const float pad = 10.0f, gap = 6.0f;
    const float bar_top = ImGui::GetWindowPos().y;
    const float bar_h = ImGui::GetWindowHeight();
    const float avatar = std::round(ImGui::GetFontSize() * 1.25f);
    const float badge_pad = 5.0f;
    const char* chevron = ICON_FA_CHEVRON_DOWN;
    const float chevron_w = ImGui::CalcTextSize(chevron).x;
    const std::string sign_in = std::string(ICON_FA_USER) + "  Sign in";
    const float account_w =
        is_logged_in_
            ? pad + avatar + gap + ImGui::CalcTextSize(chip.short_name.c_str()).x +
                  (chip.plan.empty() ? 0.0f : gap + ImGui::CalcTextSize(chip.plan.c_str()).x + 2.0f * badge_pad) + gap +
                  chevron_w + pad
            : 2.0f * pad + ImGui::CalcTextSize(sign_in.c_str()).x;
    const std::string project_text = std::string(ICON_FA_FOLDER) + "  " + project;
    const float project_w = project.empty() ? 0.0f : 2.0f * pad + ImGui::CalcTextSize(project_text.c_str()).x + gap + chevron_w;
    const float right_edge = ImGui::GetWindowPos().x + ImGui::GetWindowWidth();
    float x = std::max(right_edge - account_w - project_w, ImGui::GetCursorScreenPos().x + 12.0f);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const auto text_y = [&](float text_h) { return bar_top + (bar_h - text_h) * 0.5f; };
    // A chip: the whole bar's height, the menu hover colour while hovered or open.
    const auto chip_button = [&](const char* id, const char* popup, float width, ImVec2& min, ImVec2& max) {
        min = ImVec2(x, bar_top);
        max = ImVec2(x + width, bar_top + bar_h);
        ImGui::SetCursorScreenPos(min);
        const bool clicked = ImGui::InvisibleButton(id, ImVec2(width, bar_h));
        if (ImGui::IsItemHovered() || ImGui::IsPopupOpen(popup))
            draw->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_HeaderHovered));
        x += width;
        return clicked;
    };
    const auto badge = [&](const ImVec2& at, const std::string& label) {
        const ImVec2 size = ImGui::CalcTextSize(label.c_str());
        draw->AddRectFilled(ImVec2(at.x, at.y - 1.0f), ImVec2(at.x + size.x + 2.0f * badge_pad, at.y + size.y + 1.0f),
                            ImGui::GetColorU32(ui::WithAlpha(t.accent_text, 0.18f)), 3.0f);
        draw->AddText(ImVec2(at.x + badge_pad, at.y), ImGui::GetColorU32(t.accent_text), label.c_str());
        return size.x + 2.0f * badge_pad;
    };
    const auto avatar_circle = [&](const ImVec2& centre, float diameter) {
        draw->AddCircleFilled(centre, diameter * 0.5f, ImGui::GetColorU32(t.accent), 24);
        const ImVec2 size = ImGui::CalcTextSize(chip.initials.c_str());
        draw->AddText(ImVec2(centre.x - size.x * 0.5f, centre.y - size.y * 0.5f), ImGui::GetColorU32(ui::ReadableOn(t, t.accent)),
                      chip.initials.c_str());
    };

    if (!project.empty()) {
        ImVec2 min, max;
        if (chip_button("##project_chip", "##project_menu", project_w, min, max)) ImGui::OpenPopup("##project_menu");
        if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("##project_menu"))
            ImGui::SetTooltip("%s", pm.GetProjectFilePath().c_str());
        const float ty = text_y(ImGui::GetTextLineHeight());
        draw->AddText(ImVec2(min.x + pad, ty), ImGui::GetColorU32(t.text_dim), project_text.c_str());
        draw->AddText(ImVec2(max.x - pad - chevron_w, ty), ImGui::GetColorU32(t.text_faint), chevron);
        ImGui::SetNextWindowPos(ImVec2(max.x, max.y), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        if (ImGui::BeginPopup("##project_menu")) {
            RenderMenuItems(model.project.items);
            ImGui::EndPopup();
        }
    }
    {
        ImVec2 min, max;
        if (chip_button("##account_chip", "##account_menu", account_w, min, max)) ImGui::OpenPopup("##account_menu");
        if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("##account_menu"))
            ImGui::SetTooltip("%s", is_logged_in_ ? chip.email.c_str() : "Sign in to CyxWiz");
        const float ty = text_y(ImGui::GetTextLineHeight());
        if (is_logged_in_) {
            float cx = min.x + pad;
            avatar_circle(ImVec2(cx + avatar * 0.5f, bar_top + bar_h * 0.5f), avatar);
            cx += avatar + gap;
            draw->AddText(ImVec2(cx, ty), ImGui::GetColorU32(t.text), chip.short_name.c_str());
            cx += ImGui::CalcTextSize(chip.short_name.c_str()).x + gap;
            if (!chip.plan.empty()) cx += badge(ImVec2(cx, ty), chip.plan) + gap;
            draw->AddText(ImVec2(cx, ty), ImGui::GetColorU32(t.text_faint), chevron);
        } else {
            draw->AddText(ImVec2(min.x + pad, ty), ImGui::GetColorU32(t.text), sign_in.c_str());
        }
        ImGui::SetNextWindowPos(ImVec2(max.x, max.y), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        if (ImGui::BeginPopup("##account_menu")) {
            if (is_logged_in_) {
                // Who you are: initials, name and plan, email.
                const float big = std::round(avatar * 1.6f);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                draw = ImGui::GetWindowDrawList();
                avatar_circle(ImVec2(at.x + big * 0.5f, at.y + big * 0.5f), big);
                ImGui::Dummy(ImVec2(big, big));
                ImGui::SameLine(0.0f, 10.0f);
                ImGui::BeginGroup();
                ImGui::TextColored(t.text_bright, "%s", chip.full_name.c_str());
                if (!chip.plan.empty()) {
                    ImGui::SameLine(0.0f, gap);
                    const float w = badge(ImGui::GetCursorScreenPos(), chip.plan);
                    ImGui::Dummy(ImVec2(w, ImGui::GetTextLineHeight()));
                }
                ImGui::TextColored(t.text_dim, "%s", chip.email.c_str());
                ImGui::EndGroup();
                ImGui::Separator();
            }
            RenderMenuItems(model.account.items);
            ImGui::EndPopup();
        }
    }
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
    h["file.new_project"] = [this](const std::string&) { create_project_dialog_.Open(0); };
    h["file.open_project"] = [](const std::string&) {
        if (!ProjectManager::Instance().MayCloseProject("open another project")) return;
        auto result = FileDialogs::OpenProject();
        if (!result) return;
        if (ProjectManager::Instance().OpenProject(*result)) spdlog::info("Project opened: {}", *result);
        else spdlog::error("Failed to open project: {}", *result);
    };
    h["file.open_recent"] = [](const std::string& path) {
        if (!ProjectManager::Instance().MayCloseProject("open another project")) return;
        if (ProjectManager::Instance().OpenProject(path)) spdlog::info("Opened recent project: {}", path);
        else spdlog::error("Failed to open recent project: {}", path);
    };
    h["file.clear_recent"] = [](const std::string&) { ProjectManager::Instance().ClearRecentProjects(); };
    h["file.close_project"] = [this](const std::string&) {
        if (!ProjectManager::Instance().MayCloseProject("close the project")) return;
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
    h["script.new"] = [this](const std::string&) { new_script_dialog_.Open(NewFileDialog::Mode::Script); };
    h["script.open"] = [this](const std::string&) { OpenScriptFromDialog(); };
    h["file.save"] = [this](const std::string&) {
        if (save_project_callback_) save_project_callback_();
        else if (save_project_settings_callback_) save_project_settings_callback_();
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
    // The application lists unsaved work in its one close dialog.
    h["file.exit"] = [this](const std::string&) {
        if (exit_callback_) exit_callback_();
    };

    // Edit
    call("edit.undo", undo_callback_);
    call("edit.redo", redo_callback_);
    call("edit.cut", cut_callback_);
    call("edit.copy", copy_callback_);
    call("edit.paste", paste_callback_);
    call("edit.delete", delete_callback_);
    call("edit.select_all", select_all_callback_);
    h["edit.find"] = [this](const std::string&) { if (open_find_callback_) open_find_callback_(false); };
    h["edit.replace"] = [this](const std::string&) { if (open_find_callback_) open_find_callback_(true); };
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
    h["view.command_palette"] = [this](const std::string&) { OpenCommandPalette(); };
    h["view.panel"] = [](const std::string& name) { ShowSidebarPanel(name, true); };
    call("view.save_layout", save_layout_callback_);
    call("view.reset_layout", reset_layout_callback_);
    h["view.window_out"] = [](const std::string& name) { ::gui::MoveWindowOut(name.c_str()); };
    h["view.window_back"] = [](const std::string& name) { ::gui::MoveWindowBack(name.c_str()); };
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
    for (const auto& [id, command] : {std::pair<const char*, const char*>{"nodes.freeze_selected", "selected"},
                                      {"nodes.unfreeze_selected", "unselected"},
                                      {"nodes.freeze_loaded", "loaded"},
                                      {"nodes.freeze_but_head", "but_head"},
                                      {"nodes.freeze_but_last", "but_last"},
                                      {"nodes.unfreeze_all", "unfreeze_all"}}) {
        h[id] = [this, command](const std::string&) {
            if (freeze_nodes_callback_) freeze_nodes_callback_(command);
        };
    }
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
    call("train.cross_validate", cross_validate_callback_);
    call("train.learning_curve", learning_curve_callback_);
    call("train.compare_test_results", compare_test_results_callback_);
    call("train.export_test_report", export_test_report_callback_);
    call("train.load_checkpoint", load_checkpoint_callback_);
    call("train.dashboard", training_settings_callback_);
    call("train.optimizer_settings", optimizer_settings_callback_);
    call("train.hyperparam_search", open_hyperparam_search_callback_);
    call("tools.nas", open_nas_callback_);

    // Data
    call("data.import", import_dataset_callback_);
    call("data.create_custom", create_custom_dataset_callback_);
    call("data.statistics", dataset_statistics_callback_);
    call("data.missing_values", open_missing_value_callback_);
    call("data.outliers", open_outlier_detection_callback_);
    call("data.correlation", open_correlation_matrix_callback_);
    call("data.normalization", open_normalization_callback_);
    call("data.standardization", open_standardization_callback_);
    call("data.log_transform", open_log_transform_callback_);
    call("data.boxcox", open_boxcox_callback_);
    call("data.feature_scaling", open_feature_scaling_callback_);

    // Tools
    for (const auto& [id, tab] : {std::pair<const char*, const char*>{"tools.model_summary", "summary"},
                                  {"tools.flops", "summary"}, {"tools.lr_finder", "learning_rate"},
                                  {"tools.gradcam", "explain"}, {"tools.saliency", "explain"},
                                  {"tools.model_health", "health"}, {"tools.projector", "projector"}})
        h[id] = [this, tab = std::string(tab)](const std::string&) {
            if (open_model_callback_) open_model_callback_(tab);
        };
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
    for (const auto& [id, operation] : {std::pair<const char*, const char*>{"tools.matrix_calculator", "multiply"},
                                        {"tools.eigen", "eigen"}, {"tools.svd", "svd"}, {"tools.qr", "qr"}, {"tools.cholesky", "cholesky"}})
        h[id] = [this, operation = std::string(operation)](const std::string&) {
            if (open_linear_algebra_callback_) open_linear_algebra_callback_(operation);
        };
    call("tools.fft", open_fft_callback_);
    call("tools.spectrogram", open_spectrogram_callback_);
    call("tools.filter_designer", open_filter_designer_callback_);
    call("tools.convolution", open_convolution_callback_);
    call("tools.wavelet", open_wavelet_callback_);
    for (const auto& [id, operation] : {std::pair<const char*, const char*>{"tools.gradient_descent", "optimizers"},
                                        {"tools.convexity", "curvature"}, {"tools.lp", "linear"}, {"tools.qp", "quadratic"},
                                        {"tools.differentiation", "derivative"}, {"tools.integration", "integral"}})
        h[id] = [this, operation = std::string(operation)](const std::string&) {
            if (open_optimization_callback_) open_optimization_callback_(operation);
        };
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
    for (const auto& [id, tab] : {std::pair<const char*, const char*>{"tools.calculator", "calculator"},
                                  {"tools.unit_converter", "units"}, {"tools.random_generator", "random"},
                                  {"tools.hash_generator", "hash"}, {"tools.json_viewer", "json"},
                                  {"tools.regex_tester", "regex"}})
        h[id] = [this, tab = std::string(tab)](const std::string&) {
            if (open_utilities_callback_) open_utilities_callback_(tab);
        };
    call("tools.profiler", open_profiler_callback_);
    for (const auto& [id, tab] : {std::pair<const char*, const char*>{"tools.monitor_system", "system"},
                                  {"tools.monitor_memory", "memory"}, {"tools.monitor_loaded_data", "loaded_data"},
                                  {"tools.monitor_frames", "frames"}})
        h[id] = [this, tab = std::string(tab)](const std::string&) {
            if (open_monitor_callback_) open_monitor_callback_(tab);
        };
    call("tools.clear_cache", clear_cache_callback_);
    call("tools.gc", run_gc_callback_);
    h["tools.idle_log"] = [this](const std::string&) {
        if (!idle_log_ptr_) return;
        *idle_log_ptr_ = !*idle_log_ptr_;
        spdlog::info("Frame-rate changes are {} logged", *idle_log_ptr_ ? "now" : "no longer");
        SaveDiagnostics();
    };
    h["tools.frame_overlay"] = [this](const std::string&) {
        if (!frame_overlay_ptr_) return;
        *frame_overlay_ptr_ = !*frame_overlay_ptr_;
        SaveDiagnostics();
    };
    h["tools.verbose_python"] = [this](const std::string&) {
        if (!verbose_python_log_ptr_) return;
        const bool on = !verbose_python_log_ptr_->load();
        verbose_python_log_ptr_->store(on);
        spdlog::info("Python commands are {} logged", on ? "now" : "no longer");
        SaveDiagnostics();
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

    // Simulation: the open graph's run, then the Monitor window
    for (const char* id : {"simulation.run", "simulation.pause", "simulation.step", "simulation.stop",
                           "simulation.clear"}) {
        h[id] = [this, id](const std::string&) {
            if (simulation_command_callback_) simulation_command_callback_(id);
        };
    }
    h["simulation.monitor"] = [](const std::string&) { ShowSidebarPanel("Simulation Monitor", true); };

    // Commands plugins add to the menus (Simulation and others)
    h["plugin.command"] = [](const std::string& key) {
        if (!plugin::PluginCommandRegistry::Instance().Run(key)) spdlog::warn("Menu: plugin command {} did not run", key);
    };

    // Apps
    h["apps.panel"] = [](const std::string& panel_id) {
        plugin::PluginPanelRegistry::Instance().TogglePanelVisible(panel_id);
    };
    h["apps.plugin"] = [this](const std::string& plugin_id) {
        if (show_plugin_nodes_callback_) show_plugin_nodes_callback_(plugin_id);
    };
    h["apps.plugin_manager"] = [](const std::string&) { ShowSidebarPanel("Plugin Manager", false); };

    // Help
    h["help.tutorial"] = [](const std::string& id) {
        TutorialSystem::Instance().StartTutorial(id);
        spdlog::info("Started tutorial: {}", id);
    };
    h["help.browse_tutorials"] = [](const std::string&) { TutorialSystem::Instance().OpenTutorialBrowser(); };
    h["help.shortcuts"] = [this](const std::string&) { OpenPreferences("Shortcuts"); };
    h["help.report_issue"] = [](const std::string&) { ui::OpenUrl("https://github.com/CYXWIZ-Lab/CYXWIZ/issues"); };
    h["help.documentation"] = [](const std::string&) { ui::OpenUrl(kDocsUrl); };
    h["help.api_reference"] = [](const std::string&) { ui::OpenUrl(kApiUrl); };
    h["help.whats_new"] = [](const std::string&) { ui::OpenUrl(kReleaseNotesUrl); };
    h["help.known_issues"] = [](const std::string&) {
        const std::string& path = KnownIssuesPath();
        if (path.empty()) return;
        // No program opens .md files on some machines: then show the file.
        if (!ui::OpenUrl(path)) ui::ShowInFileManager(path);
    };
    h["help.open_logs"] = [](const std::string&) {
        const auto log = core::ExecutableDirectory() / "engine_log.txt";
        std::error_code ec;
        ui::ShowInFileManager(std::filesystem::exists(log, ec) ? log.string() : log.parent_path().string());
    };
    h["view.fullscreen"] = [](const std::string&) { ToggleFullscreen(); };
    h["file.show_project_folder"] = [](const std::string&) {
        auto& projects = ProjectManager::Instance();
        if (projects.HasActiveProject()) ui::ShowInFileManager(projects.GetProjectRoot());
    };
    h["file.copy_project_path"] = [](const std::string&) {
        auto& projects = ProjectManager::Instance();
        if (!projects.HasActiveProject()) return;
        ImGui::SetClipboardText(projects.GetProjectFilePath().c_str());
        spdlog::info("Copied the project's path: {}", projects.GetProjectFilePath());
    };
    h["help.about"] = [this](const std::string&) { about_dialog_.Open(); };

    // Account
    h["account.settings"] = [this](const std::string&) { show_account_settings_dialog_ = true; };
    h["account.sign_in"] = [this](const std::string&) { show_account_settings_dialog_ = true; };
    h["account.wallet"] = [](const std::string&) { ShowSidebarPanel("Wallet", false); };
    h["account.servers"] = [this](const std::string&) { OpenPreferences("Servers"); };
    h["account.sign_out"] = [this](const std::string&) {
        auth::AuthClient::Instance().Logout();
        is_logged_in_ = false;
        logged_in_user_.clear();
        spdlog::info("User signed out");
        if (on_logout_callback_) on_logout_callback_();
    };
}

void ToolbarPanel::ApplySavedDiagnostics() {
    const core::MonitorConfig saved = core::EngineConfig::Instance().GetMonitor();
    if (idle_log_ptr_) *idle_log_ptr_ = saved.log_frame_rate;
    if (verbose_python_log_ptr_) *verbose_python_log_ptr_ = saved.log_python_commands;
    if (frame_overlay_ptr_) *frame_overlay_ptr_ = saved.frame_overlay;
}

void ToolbarPanel::SaveDiagnostics() {
    auto& config = core::EngineConfig::Instance();
    core::MonitorConfig saved = config.GetMonitor();
    if (idle_log_ptr_) saved.log_frame_rate = *idle_log_ptr_;
    if (verbose_python_log_ptr_) saved.log_python_commands = *verbose_python_log_ptr_;
    if (frame_overlay_ptr_) saved.frame_overlay = *frame_overlay_ptr_;
    config.SetMonitor(saved);
    if (!config.Save()) spdlog::warn("The diagnostics switches could not be saved to the Engine settings");
}

}  // namespace cyxwiz
