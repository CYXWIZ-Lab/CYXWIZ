#include "theme_editor.h"
#include "../icons.h"
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
    memset(theme_path_buffer_, 0, sizeof(theme_path_buffer_));
    memset(color_filter_, 0, sizeof(color_filter_));

    // Initialize collapsed state for all groups
    for (const auto& group : kImGuiColorGroups) {
        group_collapsed_[group.name] = true;  // Start collapsed
    }
}

void ThemeEditorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Theme Editor", &visible_, ImGuiWindowFlags_MenuBar)) {
        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Theme")) {
                    show_save_dialog_ = true;
                }
                if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Load Theme")) {
                    show_load_dialog_ = true;
                }
                ImGui::Separator();
                if (ImGui::MenuItem(ICON_FA_ROTATE_LEFT " Reset to Preset")) {
                    auto& theme = GetTheme();
                    theme.ApplyPreset(theme.GetCurrentPreset());
                    has_unsaved_changes_ = false;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Preset selector at the top
        RenderPresetSelector();

        ImGui::Separator();

        // Tab bar for different sections
        if (ImGui::BeginTabBar("ThemeEditorTabs")) {
            if (ImGui::BeginTabItem("ImGui Colors")) {
                current_tab_ = 0;
                RenderImGuiColorsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Node Editor")) {
                current_tab_ = 1;
                RenderImNodesColorsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Style")) {
                current_tab_ = 2;
                RenderStyleTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Save/Load")) {
                current_tab_ = 3;
                RenderSaveLoadTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        // Status indicator
        if (has_unsaved_changes_) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), ICON_FA_CIRCLE_EXCLAMATION " Unsaved changes");
        }
    }
    ImGui::End();

    // Save dialog
    if (show_save_dialog_) {
        ImGui::OpenPopup("Save Theme");
    }

    if (ImGui::BeginPopupModal("Save Theme", &show_save_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Save current theme:");
        ImGui::InputText("Theme Name", theme_name_buffer_, sizeof(theme_name_buffer_));

        ImGui::Separator();

        if (ImGui::Button("Save", ImVec2(120, 0))) {
            if (strlen(theme_name_buffer_) > 0) {
                SaveTheme(theme_name_buffer_);
                show_save_dialog_ = false;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_save_dialog_ = false;
        }
        ImGui::EndPopup();
    }
}

void ThemeEditorPanel::RenderPresetSelector() {
    auto& theme = GetTheme();
    auto presets = Theme::GetAvailablePresets();

    ImGui::Text(ICON_FA_PALETTE " Theme Preset:");
    ImGui::SameLine();

    int current_preset = static_cast<int>(theme.GetCurrentPreset());

    if (ImGui::BeginCombo("##PresetCombo", Theme::GetPresetName(theme.GetCurrentPreset()))) {
        for (const auto& preset : presets) {
            bool is_selected = (preset == theme.GetCurrentPreset());
            if (ImGui::Selectable(Theme::GetPresetName(preset), is_selected)) {
                theme.ApplyPreset(preset);
                has_unsaved_changes_ = false;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Accent color
    ImGui::SameLine();
    ImVec4 accent = theme.GetAccentColor();
    if (ImGui::ColorEdit4("Accent##AccentColor", &accent.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) {
        theme.SetAccentColor(accent);
        has_unsaved_changes_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Accent Color");
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
    ImGui::BeginChild("SaveLoadContent", ImVec2(0, 0), true);

    ImGui::Text(ICON_FA_FLOPPY_DISK " Save Current Theme");
    ImGui::Separator();

    ImGui::InputText("Theme Name", theme_name_buffer_, sizeof(theme_name_buffer_));

    if (ImGui::Button("Save Theme", ImVec2(-1, 0))) {
        if (strlen(theme_name_buffer_) > 0) {
            SaveTheme(theme_name_buffer_);
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_FOLDER_OPEN " Load Custom Theme");
    ImGui::Separator();

    ImGui::InputText("Theme File", theme_path_buffer_, sizeof(theme_path_buffer_));
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        // TODO: Open file dialog
    }

    if (ImGui::Button("Load Theme", ImVec2(-1, 0))) {
        if (strlen(theme_path_buffer_) > 0) {
            LoadTheme(theme_path_buffer_);
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_CIRCLE_INFO " Tips");
    ImGui::Separator();
    ImGui::BulletText("Changes are applied live");
    ImGui::BulletText("Use 'Reset to Preset' to undo all changes");
    ImGui::BulletText("Themes are saved as JSON files");
    ImGui::BulletText("Custom themes are stored in 'themes/' folder");

    ImGui::EndChild();
}

bool ThemeEditorPanel::SaveTheme(const std::string& name) {
    // Create themes directory if needed
    std::string themes_dir = "themes";
    if (!fs::exists(themes_dir)) {
        fs::create_directories(themes_dir);
    }

    std::string path = themes_dir + "/" + name + ".json";

    json j;
    j["name"] = name;
    j["version"] = "1.0";

    // Save ImGui colors
    ImGuiStyle& style = ImGui::GetStyle();
    j["imgui_colors"] = json::array();
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        json color;
        color["id"] = i;
        color["r"] = style.Colors[i].x;
        color["g"] = style.Colors[i].y;
        color["b"] = style.Colors[i].z;
        color["a"] = style.Colors[i].w;
        j["imgui_colors"].push_back(color);
    }

    // Save style settings
    j["style"] = {
        {"window_rounding", style.WindowRounding},
        {"frame_rounding", style.FrameRounding},
        {"popup_rounding", style.PopupRounding},
        {"scrollbar_rounding", style.ScrollbarRounding},
        {"grab_rounding", style.GrabRounding},
        {"tab_rounding", style.TabRounding},
        {"window_border", style.WindowBorderSize},
        {"frame_border", style.FrameBorderSize},
        {"popup_border", style.PopupBorderSize},
        {"window_padding", {style.WindowPadding.x, style.WindowPadding.y}},
        {"frame_padding", {style.FramePadding.x, style.FramePadding.y}},
        {"item_spacing", {style.ItemSpacing.x, style.ItemSpacing.y}},
        {"scrollbar_size", style.ScrollbarSize},
        {"grab_min_size", style.GrabMinSize},
        {"indent_spacing", style.IndentSpacing}
    };

    // Save ImNodes colors
    j["imnodes_colors"] = json::array();
    auto& imnodes_style = ImNodes::GetStyle();
    for (int i = 0; i < ImNodesCol_COUNT; ++i) {
        json color;
        color["id"] = i;
        ImVec4 c = ImGui::ColorConvertU32ToFloat4(imnodes_style.Colors[i]);
        color["r"] = c.x;
        color["g"] = c.y;
        color["b"] = c.z;
        color["a"] = c.w;
        j["imnodes_colors"].push_back(color);
    }

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to save theme: {}", path);
        return false;
    }

    file << j.dump(2);
    file.close();

    has_unsaved_changes_ = false;
    spdlog::info("Saved theme: {}", path);
    return true;
}

bool ThemeEditorPanel::LoadTheme(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open theme file: {}", path);
        return false;
    }

    try {
        json j;
        file >> j;

        // Load ImGui colors
        ImGuiStyle& style = ImGui::GetStyle();
        if (j.contains("imgui_colors")) {
            for (const auto& color : j["imgui_colors"]) {
                int id = color["id"].get<int>();
                if (id >= 0 && id < ImGuiCol_COUNT) {
                    style.Colors[id] = ImVec4(
                        color["r"].get<float>(),
                        color["g"].get<float>(),
                        color["b"].get<float>(),
                        color["a"].get<float>()
                    );
                }
            }
        }

        // Load style settings
        if (j.contains("style")) {
            auto& s = j["style"];
            if (s.contains("window_rounding")) style.WindowRounding = s["window_rounding"].get<float>();
            if (s.contains("frame_rounding")) style.FrameRounding = s["frame_rounding"].get<float>();
            if (s.contains("popup_rounding")) style.PopupRounding = s["popup_rounding"].get<float>();
            if (s.contains("scrollbar_rounding")) style.ScrollbarRounding = s["scrollbar_rounding"].get<float>();
            if (s.contains("grab_rounding")) style.GrabRounding = s["grab_rounding"].get<float>();
            if (s.contains("tab_rounding")) style.TabRounding = s["tab_rounding"].get<float>();
            if (s.contains("window_border")) style.WindowBorderSize = s["window_border"].get<float>();
            if (s.contains("frame_border")) style.FrameBorderSize = s["frame_border"].get<float>();
            if (s.contains("popup_border")) style.PopupBorderSize = s["popup_border"].get<float>();
            if (s.contains("window_padding")) {
                style.WindowPadding.x = s["window_padding"][0].get<float>();
                style.WindowPadding.y = s["window_padding"][1].get<float>();
            }
            if (s.contains("frame_padding")) {
                style.FramePadding.x = s["frame_padding"][0].get<float>();
                style.FramePadding.y = s["frame_padding"][1].get<float>();
            }
            if (s.contains("item_spacing")) {
                style.ItemSpacing.x = s["item_spacing"][0].get<float>();
                style.ItemSpacing.y = s["item_spacing"][1].get<float>();
            }
            if (s.contains("scrollbar_size")) style.ScrollbarSize = s["scrollbar_size"].get<float>();
            if (s.contains("grab_min_size")) style.GrabMinSize = s["grab_min_size"].get<float>();
            if (s.contains("indent_spacing")) style.IndentSpacing = s["indent_spacing"].get<float>();
        }

        // Load ImNodes colors
        if (j.contains("imnodes_colors")) {
            auto& imnodes_style = ImNodes::GetStyle();
            for (const auto& color : j["imnodes_colors"]) {
                int id = color["id"].get<int>();
                if (id >= 0 && id < ImNodesCol_COUNT) {
                    ImVec4 c(
                        color["r"].get<float>(),
                        color["g"].get<float>(),
                        color["b"].get<float>(),
                        color["a"].get<float>()
                    );
                    imnodes_style.Colors[id] = ImGui::ColorConvertFloat4ToU32(c);
                }
            }
        }

        has_unsaved_changes_ = false;
        spdlog::info("Loaded theme: {}", path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading theme: {}", e.what());
        return false;
    }
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
