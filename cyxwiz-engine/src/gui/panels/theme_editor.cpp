#include "../ui_buttons.h"
#include "../appearance_settings.h"
#include "theme_editor.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <imnodes.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace gui {

using json = nlohmann::json;
namespace fs = std::filesystem;

// ImGui color groups for organization
const std::vector<ThemeEditorPanel::ColorGroupDef> ThemeEditorPanel::kImGuiColorGroups = {
    {"Text", {
        {ImGuiCol_Text, "Text"},
        {ImGuiCol_TextDisabled, "Text Disabled"},
        {ImGuiCol_TextSelectedBg, "Text Selected Bg"}
    }},
    {"Window", {
        {ImGuiCol_WindowBg, "Window Bg"},
        {ImGuiCol_ChildBg, "Child Bg"},
        {ImGuiCol_PopupBg, "Popup Bg"},
        {ImGuiCol_Border, "Border"},
        {ImGuiCol_BorderShadow, "Border Shadow"}
    }},
    {"Frame", {
        {ImGuiCol_FrameBg, "Frame Bg"},
        {ImGuiCol_FrameBgHovered, "Frame Bg Hovered"},
        {ImGuiCol_FrameBgActive, "Frame Bg Active"}
    }},
    {"Title Bar", {
        {ImGuiCol_TitleBg, "Title Bg"},
        {ImGuiCol_TitleBgActive, "Title Bg Active"},
        {ImGuiCol_TitleBgCollapsed, "Title Bg Collapsed"},
        {ImGuiCol_MenuBarBg, "Menu Bar Bg"}
    }},
    {"Scrollbar", {
        {ImGuiCol_ScrollbarBg, "Scrollbar Bg"},
        {ImGuiCol_ScrollbarGrab, "Scrollbar Grab"},
        {ImGuiCol_ScrollbarGrabHovered, "Scrollbar Grab Hovered"},
        {ImGuiCol_ScrollbarGrabActive, "Scrollbar Grab Active"}
    }},
    {"Buttons", {
        {ImGuiCol_Button, "Button"},
        {ImGuiCol_ButtonHovered, "Button Hovered"},
        {ImGuiCol_ButtonActive, "Button Active"},
        {ImGuiCol_CheckMark, "Check Mark"}
    }},
    {"Slider/Drag", {
        {ImGuiCol_SliderGrab, "Slider Grab"},
        {ImGuiCol_SliderGrabActive, "Slider Grab Active"}
    }},
    {"Header", {
        {ImGuiCol_Header, "Header"},
        {ImGuiCol_HeaderHovered, "Header Hovered"},
        {ImGuiCol_HeaderActive, "Header Active"}
    }},
    {"Separator", {
        {ImGuiCol_Separator, "Separator"},
        {ImGuiCol_SeparatorHovered, "Separator Hovered"},
        {ImGuiCol_SeparatorActive, "Separator Active"}
    }},
    {"Resize Grip", {
        {ImGuiCol_ResizeGrip, "Resize Grip"},
        {ImGuiCol_ResizeGripHovered, "Resize Grip Hovered"},
        {ImGuiCol_ResizeGripActive, "Resize Grip Active"}
    }},
    {"Tab", {
        {ImGuiCol_Tab, "Tab"},
        {ImGuiCol_TabHovered, "Tab Hovered"},
        {ImGuiCol_TabSelected, "Tab Selected"},
        {ImGuiCol_TabSelectedOverline, "Tab Selected Overline"},
        {ImGuiCol_TabDimmed, "Tab Dimmed"},
        {ImGuiCol_TabDimmedSelected, "Tab Dimmed Selected"},
        {ImGuiCol_TabDimmedSelectedOverline, "Tab Dimmed Selected Overline"}
    }},
    {"Docking", {
        {ImGuiCol_DockingPreview, "Docking Preview"},
        {ImGuiCol_DockingEmptyBg, "Docking Empty Bg"}
    }},
    {"Plot", {
        {ImGuiCol_PlotLines, "Plot Lines"},
        {ImGuiCol_PlotLinesHovered, "Plot Lines Hovered"},
        {ImGuiCol_PlotHistogram, "Plot Histogram"},
        {ImGuiCol_PlotHistogramHovered, "Plot Histogram Hovered"}
    }},
    {"Table", {
        {ImGuiCol_TableHeaderBg, "Table Header Bg"},
        {ImGuiCol_TableBorderStrong, "Table Border Strong"},
        {ImGuiCol_TableBorderLight, "Table Border Light"},
        {ImGuiCol_TableRowBg, "Table Row Bg"},
        {ImGuiCol_TableRowBgAlt, "Table Row Bg Alt"}
    }},
    {"Navigation", {
        {ImGuiCol_NavHighlight, "Nav Highlight"},
        {ImGuiCol_NavWindowingHighlight, "Nav Windowing Highlight"},
        {ImGuiCol_NavWindowingDimBg, "Nav Windowing Dim Bg"}
    }},
    {"Modal", {
        {ImGuiCol_ModalWindowDimBg, "Modal Window Dim Bg"}
    }}
};

// ImNodes color definitions
const std::vector<ThemeEditorPanel::ImNodesColorDef> ThemeEditorPanel::kImNodesColors = {
    {ImNodesCol_NodeBackground, "Node Background"},
    {ImNodesCol_NodeBackgroundHovered, "Node Background Hovered"},
    {ImNodesCol_NodeBackgroundSelected, "Node Background Selected"},
    {ImNodesCol_NodeOutline, "Node Outline"},
    {ImNodesCol_TitleBar, "Title Bar"},
    {ImNodesCol_TitleBarHovered, "Title Bar Hovered"},
    {ImNodesCol_TitleBarSelected, "Title Bar Selected"},
    {ImNodesCol_Link, "Link"},
    {ImNodesCol_LinkHovered, "Link Hovered"},
    {ImNodesCol_LinkSelected, "Link Selected"},
    {ImNodesCol_Pin, "Pin"},
    {ImNodesCol_PinHovered, "Pin Hovered"},
    {ImNodesCol_BoxSelector, "Box Selector"},
    {ImNodesCol_BoxSelectorOutline, "Box Selector Outline"},
    {ImNodesCol_GridBackground, "Grid Background"},
    {ImNodesCol_GridLine, "Grid Line"},
    {ImNodesCol_GridLinePrimary, "Grid Line Primary"},
    {ImNodesCol_MiniMapBackground, "MiniMap Background"},
    {ImNodesCol_MiniMapBackgroundHovered, "MiniMap Background Hovered"},
    {ImNodesCol_MiniMapOutline, "MiniMap Outline"},
    {ImNodesCol_MiniMapOutlineHovered, "MiniMap Outline Hovered"},
    {ImNodesCol_MiniMapNodeBackground, "MiniMap Node Background"},
    {ImNodesCol_MiniMapNodeBackgroundHovered, "MiniMap Node Background Hovered"},
    {ImNodesCol_MiniMapNodeBackgroundSelected, "MiniMap Node Background Selected"},
    {ImNodesCol_MiniMapNodeOutline, "MiniMap Node Outline"},
    {ImNodesCol_MiniMapLink, "MiniMap Link"},
    {ImNodesCol_MiniMapLinkSelected, "MiniMap Link Selected"},
    {ImNodesCol_MiniMapCanvas, "MiniMap Canvas"},
    {ImNodesCol_MiniMapCanvasOutline, "MiniMap Canvas Outline"}
};

ThemeEditorPanel::ThemeEditorPanel()
    : Panel("Theme Editor", false)  // Hidden by default
{
    memset(theme_name_buffer_, 0, sizeof(theme_name_buffer_));
    memset(color_filter_, 0, sizeof(color_filter_));

    // Initialize collapsed state for all groups
    for (const auto& group : kImGuiColorGroups) {
        group_collapsed_[group.name] = true;  // Start collapsed
    }
}

void ThemeEditorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(560, 640), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Theme Editor", &visible_, ImGuiWindowFlags_MenuBar)) {
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save as custom theme...")) {
                    show_save_dialog_ = true;
                }
                if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Load theme file...")) {
                    LoadThemeFromFileDialog();
                }
                ImGui::Separator();
                if (ImGui::MenuItem(ICON_FA_ROTATE_LEFT " Discard changes", nullptr, false,
                                    has_unsaved_changes_)) {
                    ReapplySavedTheme();
                    has_unsaved_changes_ = false;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        RenderPresetSelector();
        ImGui::Separator();

        if (ImGui::BeginTabBar("ThemeEditorTabs")) {
            if (ImGui::BeginTabItem("Colors")) {
                current_tab_ = 0;
                RenderImGuiColorsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Node Editor")) {
                current_tab_ = 1;
                RenderImNodesColorsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Shape")) {
                current_tab_ = 2;
                RenderStyleTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Saved themes")) {
                current_tab_ = 3;
                RenderSaveLoadTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    if (show_save_dialog_) {
        ImGui::OpenPopup("Save Theme");
        show_save_dialog_ = false;
    }
    if (ImGui::BeginPopupModal("Save Theme", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Save the current colours and shape as a custom theme.");
        ImGui::TextDisabled("Stored in %s", CustomThemesDirectory().string().c_str());
        ImGui::SetNextItemWidth(320.0f);
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        const bool enter = ImGui::InputTextWithHint("##theme_name", "Theme name",
                                                    theme_name_buffer_, sizeof(theme_name_buffer_),
                                                    ImGuiInputTextFlags_EnterReturnsTrue);
        if (!status_message_.empty() && status_error_) {
            ImGui::TextColored(ImVec4(1.0f, 0.48f, 0.45f, 1.0f), "%s", status_message_.c_str());
        }
        ImGui::Spacing();
        const bool has_name = theme_name_buffer_[0] != '\0';
        if (cyxwiz::ui::PrimaryButton("Save", has_name, "Type a name first.",
                                      cyxwiz::ui::ButtonSize::Regular) ||
            (enter && has_name)) {
            if (SaveTheme(theme_name_buffer_)) ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (cyxwiz::ui::SecondaryButton("Cancel", true, nullptr, cyxwiz::ui::ButtonSize::Regular)) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void ThemeEditorPanel::RenderPresetSelector() {
    auto& theme = GetTheme();
    auto presets = Theme::GetAvailablePresets();

    // Base theme: the preset the edits start from. Picking one applies it
    // Engine-wide (the Appearance page shows every theme as cards).
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Base theme");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::BeginCombo("##PresetCombo", Theme::GetPresetName(theme.GetCurrentPreset()))) {
        for (const auto& preset : presets) {
            const bool is_selected = (preset == theme.GetCurrentPreset());
            if (ImGui::Selectable(Theme::GetPresetName(preset), is_selected)) {
                SetThemePreset(preset);  // applies and saves Engine-wide
                has_unsaved_changes_ = false;
                status_message_.clear();
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImVec4 accent = theme.GetAccentColor();
    if (ImGui::ColorEdit4("##AccentColor", &accent.x,
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) {
        theme.SetAccentColor(accent);
        has_unsaved_changes_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Accent colour");

    const std::string custom = ActiveCustomThemeName();
    if (!custom.empty()) {
        ImGui::TextDisabled("Custom theme in use: %s", custom.c_str());
    } else {
        ImGui::TextDisabled("Browse all themes in Edit > Preferences > Appearance.");
    }

    // Edits apply live but are only kept once saved.
    if (has_unsaved_changes_) {
        ImGui::TextColored(ImVec4(0.89f, 0.70f, 0.25f, 1.0f),
                           ICON_FA_CIRCLE_EXCLAMATION " Unsaved changes: save them to keep them "
                                                      "after a restart.");
        if (cyxwiz::ui::PrimaryButton(ICON_FA_FLOPPY_DISK " Save as custom theme...", true,
                                      nullptr, cyxwiz::ui::ButtonSize::Small)) {
            if (!custom.empty() && theme_name_buffer_[0] == '\0') {
                std::strncpy(theme_name_buffer_, custom.c_str(), sizeof(theme_name_buffer_) - 1);
                theme_name_buffer_[sizeof(theme_name_buffer_) - 1] = '\0';
            }
            status_message_.clear();
            show_save_dialog_ = true;
        }
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Discard changes")) {
            ReapplySavedTheme();
            has_unsaved_changes_ = false;
        }
    } else if (!status_message_.empty()) {
        ImGui::TextColored(status_error_ ? ImVec4(1.0f, 0.48f, 0.45f, 1.0f)
                                         : ImVec4(0.24f, 0.84f, 0.55f, 1.0f),
                           "%s", status_message_.c_str());
    }
}

void ThemeEditorPanel::RenderImGuiColorsTab() {
    // Filter input
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##ColorFilter", ICON_FA_MAGNIFYING_GLASS " Filter colors...", color_filter_, sizeof(color_filter_));

    ImGui::BeginChild("ImGuiColorList", ImVec2(0, 0), true);

    ImGuiStyle& style = ImGui::GetStyle();
    std::string filter_lower;
    for (char c : std::string(color_filter_)) {
        filter_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto& group : kImGuiColorGroups) {
        // Filter check for entire group
        bool group_has_match = filter_lower.empty();
        if (!group_has_match) {
            for (const auto& [color_id, name] : group.colors) {
                std::string name_lower;
                for (char c : std::string(name)) {
                    name_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
                if (name_lower.find(filter_lower) != std::string::npos) {
                    group_has_match = true;
                    break;
                }
            }
        }

        if (!group_has_match) continue;

        // Get collapsed state
        bool& collapsed = group_collapsed_[group.name];

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (filter_lower.empty() && collapsed) {
            flags = 0;
        }

        if (ImGui::CollapsingHeader(group.name, flags)) {
            collapsed = false;
            ImGui::Indent();

            for (const auto& [color_id, name] : group.colors) {
                // Filter individual colors
                if (!filter_lower.empty()) {
                    std::string name_lower;
                    for (char c : std::string(name)) {
                        name_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    }
                    if (name_lower.find(filter_lower) == std::string::npos) {
                        continue;
                    }
                }

                ImVec4 color = style.Colors[color_id];
                if (ImGui::ColorEdit4(name, &color.x, ImGuiColorEditFlags_AlphaPreviewHalf)) {
                    style.Colors[color_id] = color;
                    has_unsaved_changes_ = true;
                }
            }

            ImGui::Unindent();
        } else {
            collapsed = true;
        }
    }

    ImGui::EndChild();
}

void ThemeEditorPanel::RenderImNodesColorsTab() {
    ImGui::BeginChild("ImNodesColorList", ImVec2(0, 0), true);

    ImGui::Text("Node Editor Colors");
    ImGui::Separator();

    for (const auto& [color_id, name] : kImNodesColors) {
        ImU32 color_u32 = ImNodes::GetStyle().Colors[color_id];
        ImVec4 color = ImGui::ColorConvertU32ToFloat4(color_u32);

        if (ImGui::ColorEdit4(name, &color.x, ImGuiColorEditFlags_AlphaPreviewHalf)) {
            ImNodes::GetStyle().Colors[color_id] = ImGui::ColorConvertFloat4ToU32(color);
            has_unsaved_changes_ = true;
        }
    }

    ImGui::EndChild();
}

void ThemeEditorPanel::RenderStyleTab() {
    ImGui::BeginChild("StyleContent", ImVec2(0, 0), true);

    RenderRoundingSection();
    ImGui::Separator();
    RenderBorderSection();
    ImGui::Separator();
    RenderPaddingSection();
    ImGui::Separator();
    RenderSizeSection();

    ImGui::EndChild();
}

void ThemeEditorPanel::RenderRoundingSection() {
    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::CollapsingHeader("Rounding", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::SliderFloat("Window Rounding", &style.WindowRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Frame Rounding", &style.FrameRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Popup Rounding", &style.PopupRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Scrollbar Rounding", &style.ScrollbarRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Grab Rounding", &style.GrabRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Tab Rounding", &style.TabRounding, 0.0f, 12.0f)) {
            has_unsaved_changes_ = true;
        }

        ImGui::Unindent();
    }
}

void ThemeEditorPanel::RenderBorderSection() {
    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::CollapsingHeader("Borders", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::SliderFloat("Window Border", &style.WindowBorderSize, 0.0f, 3.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Frame Border", &style.FrameBorderSize, 0.0f, 3.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Popup Border", &style.PopupBorderSize, 0.0f, 3.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Child Border", &style.ChildBorderSize, 0.0f, 3.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Tab Border", &style.TabBorderSize, 0.0f, 3.0f)) {
            has_unsaved_changes_ = true;
        }

        ImGui::Unindent();
    }
}

void ThemeEditorPanel::RenderPaddingSection() {
    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::CollapsingHeader("Padding & Spacing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::SliderFloat2("Window Padding", &style.WindowPadding.x, 0.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat2("Frame Padding", &style.FramePadding.x, 0.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat2("Item Spacing", &style.ItemSpacing.x, 0.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat2("Item Inner Spacing", &style.ItemInnerSpacing.x, 0.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat2("Cell Padding", &style.CellPadding.x, 0.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }

        ImGui::Unindent();
    }
}

void ThemeEditorPanel::RenderSizeSection() {
    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::CollapsingHeader("Sizes", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::SliderFloat("Scrollbar Size", &style.ScrollbarSize, 8.0f, 24.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Grab Min Size", &style.GrabMinSize, 8.0f, 20.0f)) {
            has_unsaved_changes_ = true;
        }
        if (ImGui::SliderFloat("Indent Spacing", &style.IndentSpacing, 0.0f, 30.0f)) {
            has_unsaved_changes_ = true;
        }

        ImGui::Unindent();
    }
}

void ThemeEditorPanel::RenderSaveLoadTab() {
    ImGui::BeginChild("SaveLoadContent", ImVec2(0, 0), false);

    ImGui::TextUnformatted("Your custom themes");
    ImGui::TextDisabled("%s", CustomThemesDirectory().string().c_str());
    ImGui::Spacing();

    const auto themes = ListCustomThemes();
    const std::string active = ActiveCustomThemeName();
    if (themes.empty()) {
        ImGui::TextDisabled("No custom themes yet. Change colours or shape, then save.");
    } else if (ImGui::BeginTable("##custom_themes", 2, ImGuiTableFlags_SizingStretchProp |
                                                           ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthFixed,
                                cyxwiz::ui::ButtonWidth("Apply", cyxwiz::ui::ButtonSize::Small));
        for (const auto& path : themes) {
            const std::string name = path.stem().string();
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            if (name == active) {
                ImGui::Text(ICON_FA_CIRCLE_CHECK " %s", name.c_str());
            } else {
                ImGui::TextUnformatted(name.c_str());
            }
            ImGui::TableNextColumn();
            ImGui::PushID(name.c_str());
            if (name == active) {
                ImGui::TextDisabled("In use");
            } else if (cyxwiz::ui::SecondaryButton("Apply")) {
                std::string error;
                if (LoadCustomTheme(path, &error)) {
                    has_unsaved_changes_ = false;
                    status_message_ = "Applied " + name;
                    status_error_ = false;
                } else {
                    status_message_ = "Could not apply " + name + ": " + error;
                    status_error_ = true;
                }
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (cyxwiz::ui::SecondaryButton(ICON_FA_FLOPPY_DISK " Save current as...")) {
        status_message_.clear();
        show_save_dialog_ = true;
    }
    ImGui::SameLine();
    if (cyxwiz::ui::SecondaryButton(ICON_FA_FOLDER_OPEN " Load theme file...")) {
        LoadThemeFromFileDialog();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Edits apply live. A saved or loaded theme is re-applied at startup;");
    ImGui::TextDisabled("choosing a base theme replaces it.");
    ImGui::EndChild();
}

bool ThemeEditorPanel::SaveTheme(const std::string& name) {
    std::filesystem::path saved;
    std::string error;
    if (!SaveCustomTheme(name, &saved, &error)) {
        status_message_ = error;
        status_error_ = true;
        return false;
    }
    has_unsaved_changes_ = false;
    status_message_ = "Saved " + saved.stem().string();
    status_error_ = false;
    return true;
}

bool ThemeEditorPanel::LoadTheme(const std::string& path) {
    std::string error;
    if (!LoadCustomTheme(path, &error)) {
        status_message_ = "Could not load the theme: " + error;
        status_error_ = true;
        return false;
    }
    has_unsaved_changes_ = false;
    status_message_ = "Loaded " + std::filesystem::path(path).stem().string();
    status_error_ = false;
    return true;
}

void ThemeEditorPanel::LoadThemeFromFileDialog() {
    const auto dir = CustomThemesDirectory().string();
    if (const auto result = cyxwiz::FileDialogs::OpenTheme(dir.c_str())) LoadTheme(*result);
}

bool ThemeEditorPanel::ExportTheme(const std::string& path) {
    return SaveTheme(fs::path(path).stem().string());
}

void ThemeEditorPanel::BackupCurrentStyle() {
    backup_style_ = ImGui::GetStyle();
    has_backup_ = true;
}

void ThemeEditorPanel::RestoreBackupStyle() {
    if (has_backup_) {
        ImGui::GetStyle() = backup_style_;
        has_unsaved_changes_ = false;
    }
}

bool ThemeEditorPanel::StyleDiffersFromBackup() const {
    if (!has_backup_) return true;

    const ImGuiStyle& current = ImGui::GetStyle();
    // Compare a few key values
    return (current.WindowRounding != backup_style_.WindowRounding ||
            current.FrameRounding != backup_style_.FrameRounding ||
            current.Colors[ImGuiCol_WindowBg].x != backup_style_.Colors[ImGuiCol_WindowBg].x);
}

void ThemeEditorPanel::RenderColorGroup(const char* group_name, const std::vector<std::pair<ImGuiCol_, const char*>>& colors) {
    (void)group_name;
    (void)colors;
    // Implemented in RenderImGuiColorsTab
}

void ThemeEditorPanel::RenderImNodesColorGroup(const char* group_name, const std::vector<std::pair<int, const char*>>& colors) {
    (void)group_name;
    (void)colors;
    // Implemented in RenderImNodesColorsTab
}


} // namespace gui
