#include "ui_tokens.h"
#include "../core/appearance_options.h"
#include <array>
#include "theme.h"
#include "dock_style.h"
#include <imgui.h>
#include <imnodes.h>

namespace gui {

// Global theme instance
static Theme g_theme;

Theme& GetTheme() {
    return g_theme;
}

Theme::Theme() {
    // Don't apply preset here - contexts don't exist yet during static init
    // The application will call ApplyPreset() after ImGui/ImNodes contexts are created
}

const char* Theme::GetPresetName(ThemePreset preset) {
    switch (preset) {
        // CyxWiz branded
        case ThemePreset::CyxWizDark:      return "CyxWiz Dark";
        case ThemePreset::CyxWizLight:     return "CyxWiz Light";
        case ThemePreset::CyxWizCream:     return "CyxWiz Cream";
        case ThemePreset::CyxWizLaunch:    return "CyxWiz Launch";
        // Classic IDE
        case ThemePreset::VSCodeDark:      return "VS Code Dark";
        case ThemePreset::UnrealEngine:    return "Unreal Engine";
        case ThemePreset::ModernDark:      return "Modern Dark";
        case ThemePreset::HighContrast:    return "High Contrast";
        // Vibrant themes
        case ThemePreset::Dracula:         return "Dracula";
        case ThemePreset::OneDarkPro:      return "One Dark Pro";
        case ThemePreset::Nord:            return "Nord";
        case ThemePreset::CatppuccinMocha: return "Catppuccin Mocha";
        // CyxOS Platform themes
        case ThemePreset::CyxOSAqua:       return "CyxOS Aqua";
        case ThemePreset::CyxOSFluent:     return "CyxOS Fluent";
        case ThemePreset::CyxOSCoder:      return "CyxOS Coder";
        case ThemePreset::CyxOSOffice:     return "CyxOS Office";
        // CyxOS Retro TUI themes
        case ThemePreset::CyxOSTuiClassic: return "CyxOS TUI Classic";
        case ThemePreset::CyxOSTuiMatrix:  return "CyxOS TUI Matrix";
        case ThemePreset::CyxOSTuiAmber:   return "CyxOS TUI Amber";
        default:                           return "Unknown";
    }
}

std::vector<ThemePreset> Theme::GetAvailablePresets() {
    return {
        // CyxWiz branded
        ThemePreset::CyxWizDark,
        ThemePreset::CyxWizLight,
        ThemePreset::CyxWizCream,
        ThemePreset::CyxWizLaunch,
        // Classic IDE
        ThemePreset::VSCodeDark,
        ThemePreset::UnrealEngine,
        ThemePreset::ModernDark,
        ThemePreset::HighContrast,
        // Vibrant themes
        ThemePreset::Dracula,
        ThemePreset::OneDarkPro,
        ThemePreset::Nord,
        ThemePreset::CatppuccinMocha,
        // CyxOS Platform themes
        ThemePreset::CyxOSAqua,
        ThemePreset::CyxOSFluent,
        ThemePreset::CyxOSCoder,
        ThemePreset::CyxOSOffice,
        // CyxOS Retro TUI themes
        ThemePreset::CyxOSTuiClassic,
        ThemePreset::CyxOSTuiMatrix,
        ThemePreset::CyxOSTuiAmber
    };
}

const char* Theme::GetPresetGroup(ThemePreset preset) {
    switch (preset) {
        case ThemePreset::CyxWizDark:
        case ThemePreset::CyxWizLight:
        case ThemePreset::CyxWizCream:
        case ThemePreset::CyxWizLaunch:    return "CyxWiz";
        case ThemePreset::VSCodeDark:
        case ThemePreset::UnrealEngine:
        case ThemePreset::ModernDark:
        case ThemePreset::HighContrast:    return "IDE";
        case ThemePreset::Dracula:
        case ThemePreset::OneDarkPro:
        case ThemePreset::Nord:
        case ThemePreset::CatppuccinMocha: return "Vibrant";
        case ThemePreset::CyxOSAqua:
        case ThemePreset::CyxOSFluent:
        case ThemePreset::CyxOSCoder:
        case ThemePreset::CyxOSOffice:     return "CyxOS";
        case ThemePreset::CyxOSTuiClassic:
        case ThemePreset::CyxOSTuiMatrix:
        case ThemePreset::CyxOSTuiAmber:   return "Retro terminal";
        default:                           return "Other";
    }
}

Theme::PresetSwatch Theme::GetPresetSwatch(ThemePreset preset) {
    // Read from the presets themselves (no second copy of their colours):
    // each preset runs once on scratch styles, everything restored after.
    static std::array<PresetSwatch, static_cast<size_t>(ThemePreset::COUNT)> swatches{};
    static bool built = false;
    if (!built) {
        built = true;
        Theme& theme = GetTheme();
        const ImGuiStyle saved_style = ImGui::GetStyle();
        const ImNodesStyle saved_nodes = ImNodes::GetStyle();
        const ThemePreset saved_preset = theme.current_preset_;
        const ThemeConfig saved_config = theme.config_;
        const ImVec4 saved_accent = theme.accent_color_;
        for (size_t i = 0; i < swatches.size(); ++i) {
            theme.RunPresetFunction(static_cast<ThemePreset>(i));
            const ImVec4* c = ImGui::GetStyle().Colors;
            swatches[i] = {c[ImGuiCol_WindowBg], c[ImGuiCol_FrameBg], theme.accent_color_};
        }
        ImGui::GetStyle() = saved_style;
        ImNodes::GetStyle() = saved_nodes;
        theme.current_preset_ = saved_preset;
        theme.config_ = saved_config;
        theme.accent_color_ = saved_accent;
    }
    const auto i = static_cast<size_t>(preset);
    return i < swatches.size() ? swatches[i] : swatches[0];
}

void Theme::ApplyPreset(ThemePreset preset) {
    current_preset_ = preset;
    // Start from the defaults: a preset that does not set its own padding
    // or spacing must not inherit the previous preset's (TOFIX129 0.4).
    config_ = ThemeConfig{};
    ImGui::GetStyle().TabBarOverlineSize = 2.0f;

    RunPresetFunction(preset);

    ApplyStyleConfig();
    ApplyImNodesStyle();  // Apply matching node editor styling
    ApplyDockStyle();     // Apply matching dock tab styling
    HarmonizeSurfaces();  // the dock style set its own tab colours
    // The rest of the theme's own colours: the preset's accent and the
    // shipped families / data / status colours for its lightness.
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    engine_ = cyxwiz::ui::DefaultEngineColours(cyxwiz::appearance::IsLightBackground(bg.x, bg.y, bg.z));
    engine_.accent = accent_color_;
    ApplyEngineColours();  // last
}

void Theme::RunPresetFunction(ThemePreset preset) {
    switch (preset) {
        // CyxWiz branded
        case ThemePreset::CyxWizDark:      ApplyCyxWizDark(); break;
        case ThemePreset::CyxWizLight:     ApplyCyxWizLight(); break;
        case ThemePreset::CyxWizCream:     ApplyCyxWizCream(); break;
        case ThemePreset::CyxWizLaunch:    ApplyCyxWizLaunch(); break;
        // Classic IDE
        case ThemePreset::VSCodeDark:      ApplyVSCodeDark(); break;
        case ThemePreset::UnrealEngine:    ApplyUnrealEngine(); break;
        case ThemePreset::ModernDark:      ApplyModernDark(); break;
        case ThemePreset::HighContrast:    ApplyHighContrast(); break;
        // Vibrant themes
        case ThemePreset::Dracula:         ApplyDracula(); break;
        case ThemePreset::OneDarkPro:      ApplyOneDarkPro(); break;
        case ThemePreset::Nord:            ApplyNord(); break;
        case ThemePreset::CatppuccinMocha: ApplyCatppuccinMocha(); break;
        // CyxOS Platform themes
        case ThemePreset::CyxOSAqua:       ApplyCyxOSAqua(); break;
        case ThemePreset::CyxOSFluent:     ApplyCyxOSFluent(); break;
        case ThemePreset::CyxOSCoder:      ApplyCyxOSCoder(); break;
        case ThemePreset::CyxOSOffice:     ApplyCyxOSOffice(); break;
        // CyxOS Retro TUI themes
        case ThemePreset::CyxOSTuiClassic: ApplyCyxOSTuiClassic(); break;
        case ThemePreset::CyxOSTuiMatrix:  ApplyCyxOSTuiMatrix(); break;
        case ThemePreset::CyxOSTuiAmber:   ApplyCyxOSTuiAmber(); break;
        default:                           ApplyCyxWizDark(); break;
    }
}

void Theme::SetEngineColours(const cyxwiz::ui::EngineColours& colours) {
    engine_ = colours;
    ApplyEngineColours();
}

// Owner 2026-11-14: one colour system. What marks a choice or the focus
// (check marks, slider grabs, hovered and pressed headers, text selection,
// the keyboard cursor, drag targets, the selected tab's line, the canvas
// box selector) is the accent; presets gave some of them a second colour
// (Unreal Engine: orange check marks and headers with a blue accent).
void Theme::ApplyEngineColours() {
    ImVec4* c = ImGui::GetStyle().Colors;
    const ImVec4 a(engine_.accent.x, engine_.accent.y, engine_.accent.z, 1.0f);
    const auto alpha = [&a](float w) { return ImVec4(a.x, a.y, a.z, w); };
    c[ImGuiCol_CheckMark] = a;
    c[ImGuiCol_SliderGrab] = a;
    c[ImGuiCol_SliderGrabActive] = cyxwiz::ui::Mix(a, ImVec4(1, 1, 1, 1), 0.15f);
    c[ImGuiCol_HeaderHovered] = alpha(0.30f);
    c[ImGuiCol_HeaderActive] = alpha(0.45f);
    c[ImGuiCol_TextSelectedBg] = alpha(0.35f);
    c[ImGuiCol_NavHighlight] = a;
    c[ImGuiCol_ResizeGripHovered] = alpha(0.60f);
    c[ImGuiCol_ResizeGripActive] = a;
    c[ImGuiCol_DragDropTarget] = a;
    c[ImGuiCol_DockingPreview] = alpha(0.60f);
    ImNodesStyle& nodes = ImNodes::GetStyle();
    nodes.Colors[ImNodesCol_BoxSelector] = ImGui::ColorConvertFloat4ToU32(alpha(0.25f));
    nodes.Colors[ImNodesCol_BoxSelectorOutline] = ImGui::ColorConvertFloat4ToU32(a);
    // Subgraph frames (subgraph_presentation.cpp) read the title bar colours.
    nodes.Colors[ImNodesCol_TitleBar] = ImGui::ColorConvertFloat4ToU32(a);
    nodes.Colors[ImNodesCol_TitleBarHovered] = ImGui::ColorConvertFloat4ToU32(cyxwiz::ui::Mix(a, ImVec4(1, 1, 1, 1), 0.10f));
    nodes.Colors[ImNodesCol_TitleBarSelected] = ImGui::ColorConvertFloat4ToU32(cyxwiz::ui::Mix(a, ImVec4(1, 1, 1, 1), 0.20f));
    DockStyle& dock = GetDockStyle();
    DockTabStyle tabs = dock.GetStyle();
    tabs.active_indicator_color = a;
    dock.SetStyle(tabs);  // writes the ImGui tab colours
    c[ImGuiCol_TabSelectedOverline] = a;
    c[ImGuiCol_TabDimmedSelectedOverline] = alpha(0.50f);
    cyxwiz::ui::SetEngineColours(engine_);
}

// Owner 2026-10-02: every surface follows the window background. Presets
// (and the dock style after them) gave table headers, scrollbars, title
// bars, dock tabs and the empty dock area their own darker or lighter greys
// (Unreal Engine: window 41, table header 31, title bar 20/31, tabs 31, an
// orange selected tab, scrollbar track 20 with a 77 thumb), so tables, docked
// windows and title bars looked pasted on. These are derived from the window
// colour now, in every theme; each preset keeps its own window, text and
// accent colours, and the selected tab keeps the preset's accent overline.
void Theme::HarmonizeSurfaces() {
    ImVec4* c = ImGui::GetStyle().Colors;
    const ImVec4 bg = c[ImGuiCol_WindowBg];
    const ImVec4 text = c[ImGuiCol_Text];
    auto toward_text = [&](float amount) {
        return ImVec4(bg.x + (text.x - bg.x) * amount, bg.y + (text.y - bg.y) * amount, bg.z + (text.z - bg.z) * amount, 1.0f);
    };
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TableHeaderBg] = ImVec4(bg.x, bg.y, bg.z, 1.0f);
    c[ImGuiCol_TableBorderStrong] = toward_text(0.10f);
    c[ImGuiCol_TableBorderLight] = toward_text(0.06f);
    c[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(text.x, text.y, text.z, 0.025f);

    // Scrollbar thumbs: a step toward the text, not a separate grey.
    c[ImGuiCol_ScrollbarGrab] = toward_text(0.14f);
    c[ImGuiCol_ScrollbarGrabHovered] = toward_text(0.22f);
    c[ImGuiCol_ScrollbarGrabActive] = toward_text(0.30f);

    // Lines (window border, separators) in the surface tone.
    c[ImGuiCol_Border] = toward_text(0.10f);
    c[ImGuiCol_Separator] = toward_text(0.10f);

    // Owner 2026-10-21 ("the whole engine" is the window tone; only inputs
    // are darker - compare the Properties panel): menus, popups, tooltips
    // and every area drawn in the panel token (Tokens::bg_panel: SQL
    // Workbench panels, notebook toolbars, plot legends, completion lists)
    // are a step lighter than the window, never the preset's near-black
    // (Unreal Engine: 20 on a 41 window, darker than its inputs).
    c[ImGuiCol_PopupBg] = toward_text(0.035f);

    // Buttons a step off the window, in every preset (several filled every
    // button with their accent, so Save looked like Train). Styled buttons
    // read these; the primary button is the accent.
    cyxwiz::ui::ButtonFills(bg, &c[ImGuiCol_Button], &c[ImGuiCol_ButtonHovered], &c[ImGuiCol_ButtonActive]);
    // One field colour everywhere (owner 2026-11-14): ImGui's fields and every
    // screen that reads Tokens::bg_input.
    cyxwiz::ui::FieldFills(bg, &c[ImGuiCol_FrameBg], &c[ImGuiCol_FrameBgHovered], &c[ImGuiCol_FrameBgActive]);

    // The node canvas and its minimap: the window colour with grid lines a
    // step toward the text (presets gave the canvas its own near-black).
    // Nodes keep the preset's colours.
    ImNodesStyle& nodes = ImNodes::GetStyle();
    const auto u32 = [](const ImVec4& v, float alpha) { return ImGui::ColorConvertFloat4ToU32(ImVec4(v.x, v.y, v.z, alpha)); };
    nodes.Colors[ImNodesCol_GridBackground] = u32(bg, 1.0f);
    nodes.Colors[ImNodesCol_GridLine] = u32(toward_text(0.05f), 1.0f);
    nodes.Colors[ImNodesCol_GridLinePrimary] = u32(toward_text(0.09f), 1.0f);
    nodes.Colors[ImNodesCol_MiniMapBackground] = u32(toward_text(0.035f), 0.92f);
    nodes.Colors[ImNodesCol_MiniMapBackgroundHovered] = u32(toward_text(0.06f), 0.95f);
    nodes.Colors[ImNodesCol_MiniMapOutline] = u32(toward_text(0.10f), 1.0f);
    nodes.Colors[ImNodesCol_MiniMapOutlineHovered] = u32(toward_text(0.16f), 1.0f);
    nodes.Colors[ImNodesCol_MiniMapCanvas] = u32(toward_text(0.08f), 0.5f);
    nodes.Colors[ImNodesCol_MiniMapCanvasOutline] = u32(toward_text(0.16f), 1.0f);

    // Window title bars, menu bars and the dock area: the window colour.
    const ImVec4 solid(bg.x, bg.y, bg.z, 1.0f);
    c[ImGuiCol_TitleBg] = solid;
    c[ImGuiCol_TitleBgActive] = solid;
    c[ImGuiCol_TitleBgCollapsed] = solid;
    c[ImGuiCol_DockingEmptyBg] = solid;
    c[ImGuiCol_MenuBarBg] = solid;  // the main menu and a window's own menu row

    // Dock and window tabs: the window colour; the selected tab is marked by
    // the preset's accent overline, a hovered one by a light wash.
    DockStyle& dock = GetDockStyle();
    DockTabStyle tabs = dock.GetStyle();
    tabs.tab_bg = solid;
    tabs.tab_bg_active = solid;
    tabs.tab_bg_unfocused = solid;
    tabs.tab_bg_hovered = toward_text(0.08f);
    tabs.tab_text = ImVec4(text.x + (bg.x - text.x) * 0.35f, text.y + (bg.y - text.y) * 0.35f, text.z + (bg.z - text.z) * 0.35f, 1.0f);
    tabs.tab_text_active = text;
    tabs.tab_separator_color = toward_text(0.10f);
    tabs.dock_bg = solid;
    tabs.dock_border = toward_text(0.10f);
    dock.SetStyle(tabs);  // writes the ImGui tab, dock and border colours
    const ImVec4 mark = tabs.active_indicator_color;
    c[ImGuiCol_TabSelectedOverline] = mark;
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(mark.x, mark.y, mark.z, mark.w * 0.5f);
}

void Theme::ApplyConfig(const ThemeConfig& config) {
    config_ = config;
    ApplyStyleConfig();
}

void Theme::ApplyStyleConfig() {
    ImGuiStyle& style = ImGui::GetStyle();

    // Rounding
    style.WindowRounding = config_.window_rounding;
    style.FrameRounding = config_.frame_rounding;
    style.PopupRounding = config_.popup_rounding;
    style.ScrollbarRounding = config_.scrollbar_rounding;
    style.GrabRounding = config_.grab_rounding;
    style.TabRounding = config_.tab_rounding;

    // Borders
    style.WindowBorderSize = config_.window_border_size;
    style.FrameBorderSize = config_.frame_border_size;
    style.PopupBorderSize = config_.popup_border_size;
    style.ChildBorderSize = config_.child_border_size;

    // Padding and spacing
    style.WindowPadding = config_.window_padding;
    style.FramePadding = config_.frame_padding;
    style.ItemSpacing = config_.item_spacing;
    style.ItemInnerSpacing = config_.item_inner_spacing;

    // Sizes
    style.ScrollbarSize = config_.scrollbar_size;
    style.GrabMinSize = config_.grab_min_size;
    style.IndentSpacing = config_.indent_spacing;
}

bool Theme::RenderThemeSelector() {
    bool changed = false;

    if (ImGui::BeginCombo("Theme", GetPresetName(current_preset_))) {
        for (auto preset : GetAvailablePresets()) {
            bool is_selected = (current_preset_ == preset);
            if (ImGui::Selectable(GetPresetName(preset), is_selected)) {
                ApplyPreset(preset);
                changed = true;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    return changed;
}

} // namespace gui
