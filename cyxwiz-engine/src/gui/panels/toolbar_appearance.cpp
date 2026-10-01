// Preferences > Appearance (tofix121): Engine-wide theme, interface and code
// text sizes, sidebar side, and a live preview. Every change applies at once
// and is saved through gui/appearance_settings.

#include "toolbar.h"
#include "../../core/appearance_options.h"
#include "../appearance_settings.h"
#include "../console_palette.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../theme.h"
#include "../ui_buttons.h"

#include <algorithm>
#include <array>
#include <imgui.h>
#include <string>

namespace cyxwiz {

namespace {

void SectionTitle(const char* title, const char* subtitle) {
    ImGui::PushFont(nullptr);
    ImGui::TextUnformatted(title);
    if (subtitle && *subtitle) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::TextDisabled("%s", subtitle);
    }
    ImGui::PopFont();
}

// One theme card: swatch strip and name; returns true when clicked.
bool ThemeCard(::gui::ThemePreset preset, bool selected, float width) {
    const auto swatch = ::gui::Theme::GetPresetSwatch(preset);
    const auto& palette = ::gui::CurrentConsolePalette();
    const char* name = ::gui::Theme::GetPresetName(preset);
    const float line = ImGui::GetTextLineHeight();
    const float height = 44.0f + line + 16.0f;
    ImGui::PushID(static_cast<int>(preset));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##card", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 max(pos.x + width, pos.y + height);
    dl->AddRectFilled(pos, max, ImGui::GetColorU32(palette.panel), 8.0f);
    const ImVec2 sw_min(pos.x + 6.0f, pos.y + 6.0f);
    const ImVec2 sw_max(max.x - 6.0f, pos.y + 50.0f);
    const float sw_w = sw_max.x - sw_min.x;
    dl->AddRectFilled(sw_min, sw_max, ImGui::GetColorU32(swatch.background), 5.0f);
    dl->AddRectFilled(ImVec2(sw_min.x + sw_w * 0.62f, sw_min.y),
                      ImVec2(sw_max.x - 14.0f, sw_max.y), ImGui::GetColorU32(swatch.panel));
    dl->AddRectFilled(ImVec2(sw_max.x - 14.0f, sw_min.y), sw_max,
                      ImGui::GetColorU32(swatch.accent), 5.0f, ImDrawFlags_RoundCornersRight);
    dl->AddText(ImVec2(pos.x + 8.0f, sw_max.y + 6.0f), ImGui::GetColorU32(palette.text), name);
    if (selected) {
        const char* mark = ICON_FA_CIRCLE_CHECK;
        const float mark_w = ImGui::CalcTextSize(mark).x;
        dl->AddText(ImVec2(max.x - 8.0f - mark_w, sw_max.y + 6.0f),
                    ImGui::GetColorU32(palette.accent_text), mark);
        dl->AddRect(pos, max, ImGui::GetColorU32(palette.accent), 8.0f, 0, 2.0f);
    } else {
        dl->AddRect(pos, max, ImGui::GetColorU32(hovered ? palette.border : palette.inner_border),
                    8.0f);
    }
    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("%s%s", name, selected ? " (current)" : " - click to apply");
    }
    ImGui::PopID();
    return clicked;
}

void RenderThemeGrid() {
    const auto presets = ::gui::Theme::GetAvailablePresets();
    const auto current = ::gui::CurrentThemePreset();
    const float avail = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(avail / 150.0f), 2, 4);
    const float gap = 10.0f;
    const float card_w = (avail - gap * static_cast<float>(columns - 1)) / static_cast<float>(columns);

    const char* group = nullptr;
    int column = 0;
    for (const auto preset : presets) {
        const char* preset_group = ::gui::Theme::GetPresetGroup(preset);
        if (!group || std::string(group) != preset_group) {
            if (group)
                ImGui::Dummy(ImVec2(0.0f, 2.0f));
            group = preset_group;
            ImGui::TextDisabled("%s", group);
            column = 0;
        }
        if (column > 0)
            ImGui::SameLine(0.0f, gap);
        if (ThemeCard(preset, preset == current, card_w))
            ::gui::SetThemePreset(preset);
        column = (column + 1) % columns;
    }
}

void RenderPreview() {
    const auto& palette = ::gui::CurrentConsolePalette();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_WindowBg]);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::BeginChild("##appearance_preview", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    if (ImGui::BeginTabBar("##preview_tabs")) {
        if (ImGui::BeginTabItem("Studio")) {
            if (ui::SecondaryButton("Compile")) {}
            ImGui::SameLine();
            if (ui::PrimaryButton("Train", true, nullptr, ui::ButtonSize::Small)) {}
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("Ready");
            static char search[64] = "";
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputTextWithHint("##preview_search", "Search nodes...", search, sizeof(search));
            ImGui::TextColored(palette.muted, "07:52:19");
            ImGui::SameLine();
            ImGui::TextColored(palette.warning, "Warn ");
            ImGui::SameLine();
            ImGui::TextUnformatted("Connection timed out");
            ImGui::TextColored(palette.muted, "07:52:20");
            ImGui::SameLine();
            ImGui::TextColored(palette.error, "Error");
            ImGui::SameLine();
            ImGui::TextUnformatted("Export failed");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Data Studio")) {
            ImGui::TextDisabled("Another panel");
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    // Code sample at the code text size.
    if (ImFont* code = cyxwiz::gui::GetCodeFont())
        ImGui::PushFont(code);
    ImGui::TextColored(palette.accent, ">>>");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.keyword, " for");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.text, " x ");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.keyword, "in");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.builtin, " range");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.text, "(");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.number, "3");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.text, "):");
    ImGui::TextColored(palette.continuation, "...");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.builtin, "     print");
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(palette.text, "(x)");
    if (cyxwiz::gui::GetCodeFont())
        ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::TextDisabled("The preview follows the theme and both text sizes.");
}

}  // namespace

void ToolbarPanel::RenderAppearancePreferences() {
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool side_by_side = avail >= 820.0f;
    const float preview_w = side_by_side ? 340.0f : 0.0f;

    ImGui::BeginChild("##appearance_settings",
                      ImVec2(side_by_side ? avail - preview_w - 20.0f : 0.0f, 0.0f),
                      side_by_side ? ImGuiChildFlags_None : ImGuiChildFlags_AutoResizeY);

    // Theme
    SectionTitle("Theme", "Applies to the whole Engine and every project");
    ImGui::SameLine();
    const char* editor_label = "Theme Editor...";
    const float editor_w = ImGui::CalcTextSize(editor_label).x + 8.0f;
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - editor_w;
    if (right > ImGui::GetCursorPosX())
        ImGui::SetCursorPosX(right);
    if (ui::LinkButton(editor_label) && open_theme_editor_callback_)
        open_theme_editor_callback_();
    ImGui::Spacing();
    RenderThemeGrid();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Text sizes
    SectionTitle("Text", nullptr);
    if (ImGui::BeginTable("##appearance_text", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed,
                                ImGui::CalcTextSize("Interface text size").x + 16.0f);
        ImGui::TableSetupColumn("control", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Interface text size");
        ImGui::TableNextColumn();
        {
            std::array<const char*, cyxwiz::appearance::kUiTextSizes.size()> labels{};
            for (size_t i = 0; i < labels.size(); ++i)
                labels[i] = cyxwiz::appearance::kUiTextSizes[i].label;
            int index = cyxwiz::appearance::UiTextIndex(::gui::UiTextPixels());
            if (ui::SegmentedControl("ui_text", labels.data(), static_cast<int>(labels.size()),
                                     &index)) {
                ::gui::SetUiTextPixels(cyxwiz::appearance::kUiTextSizes[index].pixels);
            }
            ImGui::TextDisabled("Menus, panels and dialogs");
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Code text size");
        ImGui::TableNextColumn();
        {
            std::array<const char*, cyxwiz::appearance::kCodeTextSizes.size()> labels{};
            for (size_t i = 0; i < labels.size(); ++i)
                labels[i] = cyxwiz::appearance::kCodeTextSizes[i].label;
            int index = cyxwiz::appearance::CodeScaleIndex(::gui::CodeTextScale());
            if (ui::SegmentedControl("code_text", labels.data(), static_cast<int>(labels.size()),
                                     &index)) {
                ::gui::SetCodeTextScale(cyxwiz::appearance::kCodeTextSizes[index].scale);
                editor_font_size_ = cyxwiz::appearance::kCodeTextSizes[index].pixels;
            }
            ImGui::TextDisabled("Script Editor, Python REPL, Logs, Commands and terminals");
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Changes apply immediately and are kept for every project.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Layout
    SectionTitle("Layout", nullptr);
    if (ImGui::BeginTable("##appearance_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed,
                                ImGui::CalcTextSize("Interface text size").x + 16.0f);
        ImGui::TableSetupColumn("control", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Tool sidebar");
        ImGui::TableNextColumn();
        const char* sides[] = {"Left", "Right"};
        int side = ::gui::SidebarOnLeft() ? 0 : 1;
        if (ui::SegmentedControl("sidebar", sides, 2, &side))
            ::gui::SetSidebarOnLeft(side == 0);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("The column of panel icons");
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ui::LinkButton("Reset appearance to defaults"))
        ::gui::ResetAppearance();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("CyxWiz Dark, 15 px interface text, 16 px code text, sidebar on the right");
    ImGui::EndChild();

    if (side_by_side)
        ImGui::SameLine(0.0f, 20.0f);
    ImGui::BeginGroup();
    if (side_by_side)
        ImGui::PushItemWidth(preview_w);
    SectionTitle("Preview", nullptr);
    ImGui::BeginChild("##appearance_preview_column", ImVec2(side_by_side ? preview_w : 0.0f, 0.0f),
                      side_by_side ? ImGuiChildFlags_None : ImGuiChildFlags_AutoResizeY);
    RenderPreview();
    ImGui::EndChild();
    if (side_by_side)
        ImGui::PopItemWidth();
    ImGui::EndGroup();
}

}  // namespace cyxwiz
