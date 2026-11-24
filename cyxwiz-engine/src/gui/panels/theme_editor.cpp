#include "theme_editor.h"

#include "../appearance_settings.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/appearance_options.h"
#include "../../core/file_dialogs.h"

#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace gui {

namespace fs = std::filesystem;

// The surfaces the theme draws with. Check marks, slider grabs, the selection
// washes, the keyboard cursor, drag targets, the docking preview and the
// selected tab's line follow the accent (Theme::ApplyEngineColours): no rows.
const std::vector<ThemeEditorPanel::ColourGroup> ThemeEditorPanel::kSurfaceGroups = {
    {"Text", {{ImGuiCol_Text, "Text"}, {ImGuiCol_TextDisabled, "Dim text"}}},
    {"Window", {{ImGuiCol_WindowBg, "Window"},
                {ImGuiCol_PopupBg, "Menus and popups"},
                {ImGuiCol_Border, "Window border"},
                {ImGuiCol_Separator, "Separator"}}},
    {"Fields", {{ImGuiCol_FrameBg, "Field"}, {ImGuiCol_FrameBgHovered, "Field hovered"},
                {ImGuiCol_FrameBgActive, "Field in use"}}},
    {"Buttons", {{ImGuiCol_Button, "Button"}, {ImGuiCol_ButtonHovered, "Button hovered"},
                 {ImGuiCol_ButtonActive, "Button pressed"}}},
    {"Bars", {{ImGuiCol_TitleBg, "Title bar"}, {ImGuiCol_TitleBgActive, "Title bar (focused)"},
              {ImGuiCol_MenuBarBg, "Menu bar"}}},
    {"Tabs", {{ImGuiCol_Tab, "Tab"}, {ImGuiCol_TabHovered, "Tab hovered"}, {ImGuiCol_TabSelected, "Tab selected"},
              {ImGuiCol_TabDimmed, "Tab (unfocused window)"},
              {ImGuiCol_TabDimmedSelected, "Tab selected (unfocused window)"}}},
    {"Lists", {{ImGuiCol_Header, "Selected row (plain lists)"}}},
    {"Scrollbars", {{ImGuiCol_ScrollbarGrab, "Scrollbar"}, {ImGuiCol_ScrollbarGrabHovered, "Scrollbar hovered"},
                    {ImGuiCol_ScrollbarGrabActive, "Scrollbar dragged"}}},
    {"Tables", {{ImGuiCol_TableHeaderBg, "Header"}, {ImGuiCol_TableBorderStrong, "Outer line"},
                {ImGuiCol_TableBorderLight, "Inner line"}, {ImGuiCol_TableRowBgAlt, "Alternate row"}}},
    {"Simple plots", {{ImGuiCol_PlotLines, "Line"}, {ImGuiCol_PlotLinesHovered, "Line hovered"},
                      {ImGuiCol_PlotHistogram, "Bar"}, {ImGuiCol_PlotHistogramHovered, "Bar hovered"}}},
    {"Dialogs", {{ImGuiCol_ModalWindowDimBg, "Dim behind a dialog"}}},
};

namespace {

// Node family names, in cyxwiz::ui::NodeHue order.
constexpr const char* kFamilyNames[] = {
    "Data sources", "Data transform", "Analytics", "Layers", "Activation", "Pooling", "Normalization", "Attention",
    "Recurrent", "Shape ops", "Merge ops", "Upsampling", "Training", "Regularization", "Preprocessing",
    "Data pipeline", "Text processing", "Time series", "Audio", "DNN", "RL", "Signal", "Plugin", "Neutral"};
static_assert(std::size(kFamilyNames) == cyxwiz::ui::Tokens::kNodeHueCount, "a name for every node family");

// Data types (pins and links) in Tokens::series order (PinTypeColour).
constexpr const char* kDataTypeNames[] = {"Optimizer", "Tensor", "Parameters", "Labels", "Loss", "Dataset"};
static_assert(std::size(kDataTypeNames) == cyxwiz::ui::Tokens::kSeriesCount, "a name for every data type");

float Contrast(const ImVec4& a, const ImVec4& b) {
    const float la = cyxwiz::appearance::Luminance(a.x, a.y, a.z);
    const float lb = cyxwiz::appearance::Luminance(b.x, b.y, b.z);
    return (std::max(la, lb) + 0.05f) / (std::min(la, lb) + 0.05f);
}

bool MatchesFilter(const char* text, const std::string& filter) {
    if (filter.empty()) return true;
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower.find(filter) != std::string::npos;
}

// The label column fits this table's longest label (one width for every
// table cut the field's last numbers off in a narrow window).
template <typename Labels>
float LabelWidth(const Labels& labels) {
    float width = 0.0f;
    for (const char* label : labels) width = std::max(width, ImGui::CalcTextSize(label).x);
    return width + 16.0f;
}

bool BeginColourTable(const char* id, float label_width) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, label_width);
    ImGui::TableSetupColumn("field", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

}  // namespace

ThemeEditorPanel::ThemeEditorPanel() : Panel("Theme Editor", false) {}

bool ThemeEditorPanel::ColourRow(const char* label, ImVec4* colour, bool with_alpha, const char* note,
                                 const ImVec4* note_colour) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    const bool changed = cyxwiz::ui::ColourField(label, colour, with_alpha);
    if (note && note[0]) {
        const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
        ImGui::PushStyleColor(ImGuiCol_Text, note_colour ? *note_colour : t.text_dim);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(note);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
    if (changed) MarkChanged();
    return changed;
}

void ThemeEditorPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(720, 680), ImGuiCond_FirstUseEver);
    // Wide enough for a whole colour row (swatch, hex, R G B with - / +) beside
    // its label: nothing cut at the right edge.
    ImGui::SetNextWindowSizeConstraints(ImVec2(700.0f, 320.0f), ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::Begin("Theme Editor", &visible_)) {
        RenderPresetSelector();
        ImGui::Spacing();
        if (ImGui::BeginTabBar("ThemeEditorTabs")) {
            if (ImGui::BeginTabItem("Engine colours")) {
                RenderEngineColoursTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Surfaces")) {
                RenderSurfacesTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Canvas")) {
                RenderCanvasTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Shape")) {
                RenderShapeTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Saved themes")) {
                RenderSavedThemesTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
    RenderSaveDialog();
}

void ThemeEditorPanel::RenderPresetSelector() {
    auto& theme = GetTheme();
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();

    // Base theme: the preset the edits start from. Picking one applies it
    // Engine-wide (the Appearance page shows every theme as cards).
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Base theme");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::BeginCombo("##PresetCombo", Theme::GetPresetName(theme.GetCurrentPreset()))) {
        for (const auto& preset : Theme::GetAvailablePresets()) {
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
    const std::string custom = ActiveCustomThemeName();
    ImGui::TextDisabled("%s", custom.empty() ? "Browse all themes in File > Preferences > Appearance."
                                             : ("Custom theme in use: " + custom).c_str());

    // Edits apply live but are only kept once saved.
    if (has_unsaved_changes_) {
        ImGui::TextColored(t.warning, ICON_FA_CIRCLE_EXCLAMATION " Unsaved changes: save them to keep them after a restart.");
        if (cyxwiz::ui::PrimaryButton(ICON_FA_FLOPPY_DISK " Save as custom theme...", true, nullptr,
                                      cyxwiz::ui::ButtonSize::Small)) {
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
        ImGui::TextColored(status_error_ ? t.error : t.success, "%s", status_message_.c_str());
    }
}

void ThemeEditorPanel::RenderEngineColoursTab() {
    auto& theme = GetTheme();
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    cyxwiz::ui::EngineColours e = theme.GetEngineColours();
    bool changed = false;
    ImGui::BeginChild("EngineColours", ImVec2(0, 0));

    cyxwiz::ui::SettingsSection("Accent");
    if (BeginColourTable("##accent", LabelWidth(std::array<const char*, 1>{"Accent"}))) {
        changed |= ColourRow("Accent", &e.accent, false,
                             "Train, OK and every primary button, selection, check marks, links, the selected "
                             "tab's line, group frames on the canvas.");
        ImGui::EndTable();
    }

    cyxwiz::ui::SettingsSection("Node families");
    ImGui::TextDisabled("The fill of each kind of node on the canvas.");
    if (BeginColourTable("##families", LabelWidth(kFamilyNames))) {
        for (int i = 0; i < cyxwiz::ui::Tokens::kNodeHueCount; ++i) changed |= ColourRow(kFamilyNames[i], &e.node[i]);
        ImGui::EndTable();
    }

    cyxwiz::ui::SettingsSection("Data types");
    ImGui::TextDisabled("Pins and links by what flows through them; also the colours of plot series, in this order.");
    if (BeginColourTable("##data", LabelWidth(kDataTypeNames))) {
        for (int i = 0; i < cyxwiz::ui::Tokens::kSeriesCount; ++i) changed |= ColourRow(kDataTypeNames[i], &e.data[i]);
        ImGui::EndTable();
    }

    cyxwiz::ui::SettingsSection("Status words");
    ImGui::TextDisabled("Ready, warnings, errors, running ... wherever a state is shown.");
    if (BeginColourTable("##status", LabelWidth(std::array<const char*, 8>{"Success", "Warning", "Failed", "Error",
                                                                          "Critical", "Info", "Not yet", "Running"}))) {
        const std::pair<const char*, ImVec4*> status[] = {
            {"Success", &e.success}, {"Warning", &e.warning}, {"Failed", &e.caution}, {"Error", &e.error},
            {"Critical", &e.critical}, {"Info", &e.info}, {"Not yet", &e.pending}, {"Running", &e.running}};
        for (const auto& [name, colour] : status) {
            const float ratio = Contrast(*colour, t.bg_window);
            char note[96] = "";
            if (ratio < 4.5f)
                std::snprintf(note, sizeof(note), "Hard to read on this window: %.1f:1 (4.5:1 or more reads well).", ratio);
            changed |= ColourRow(name, colour, false, note, &t.warning);
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (cyxwiz::ui::LinkButton("Reset the Engine colours to the base theme's")) {
        const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        e = cyxwiz::ui::DefaultEngineColours(cyxwiz::appearance::IsLightBackground(bg.x, bg.y, bg.z));
        e.accent = Theme::GetPresetSwatch(theme.GetCurrentPreset()).accent;
        changed = true;
        MarkChanged();
    }
    ImGui::EndChild();
    if (changed) theme.SetEngineColours(e);
}

void ThemeEditorPanel::RenderSurfacesTab() {
    cyxwiz::ui::SearchField("##surface_filter", surface_filter_, sizeof(surface_filter_), "Filter colours", -1.0f);
    std::string filter(surface_filter_);
    std::transform(filter.begin(), filter.end(), filter.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    ImGui::BeginChild("Surfaces", ImVec2(0, 0));
    ImGuiStyle& style = ImGui::GetStyle();
    for (const auto& group : kSurfaceGroups) {
        bool any = false;
        for (const auto& [id, name] : group.colours) any |= MatchesFilter(name, filter) || MatchesFilter(group.name, filter);
        if (!any) continue;
        cyxwiz::ui::SettingsSection(group.name);
        ImGui::PushID(group.name);
        std::vector<const char*> labels;
        for (const auto& [id, name] : group.colours) labels.push_back(name);
        if (BeginColourTable("##group", LabelWidth(labels))) {
            for (const auto& [id, name] : group.colours) {
                if (!MatchesFilter(name, filter) && !MatchesFilter(group.name, filter)) continue;
                ImVec4 colour = style.Colors[id];
                if (ColourRow(name, &colour, colour.w < 1.0f || id == ImGuiCol_ModalWindowDimBg ||
                                                 id == ImGuiCol_TableRowBgAlt))
                    style.Colors[id] = colour;
            }
            ImGui::EndTable();
        }
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Check marks, selection, the keyboard cursor and the selected tab's line follow the accent "
                        "(Engine colours).");
    ImGui::EndChild();
}

void ThemeEditorPanel::RenderCanvasTab() {
    ImGui::BeginChild("Canvas", ImVec2(0, 0));
    cyxwiz::ui::SettingsSection("Grid");
    auto& nodes = ImNodes::GetStyle();
    if (BeginColourTable("##grid", LabelWidth(std::array<const char*, 3>{"Background", "Line", "Major line"}))) {
        const std::pair<int, const char*> rows[] = {{ImNodesCol_GridBackground, "Background"},
                                                    {ImNodesCol_GridLine, "Line"},
                                                    {ImNodesCol_GridLinePrimary, "Major line"}};
        for (const auto& [id, name] : rows) {
            ImVec4 colour = ImGui::ColorConvertU32ToFloat4(nodes.Colors[id]);
            if (ColourRow(name, &colour)) nodes.Colors[id] = ImGui::ColorConvertFloat4ToU32(colour);
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Nodes are filled by their family, pins and links by their data type (Engine colours); "
                        "the selection box and group frames use the accent.");
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
}

void ThemeEditorPanel::RenderShapeTab() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::BeginChild("Shape", ImVec2(0, 0));
    const auto row = [this](const char* label, float* value, double max, double step) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableNextColumn();
        cyxwiz::ui::NumberFieldOptions o;
        o.min = 0.0;
        o.max = max;
        o.step = step;
        o.format = "%.1f";
        o.width = 56.0f;
        o.hint = "Pixels";
        if (cyxwiz::ui::NumberField(label, value, o)) MarkChanged();
    };
    const auto pair_row = [this](const char* label, ImVec2* value, double max) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableNextColumn();
        cyxwiz::ui::NumberFieldOptions o;
        o.min = 0.0;
        o.max = max;
        o.step = 1.0;
        o.format = "%.0f";
        o.width = 44.0f;
        ImGui::PushID(label);
        o.hint = "Across (pixels)";
        if (cyxwiz::ui::NumberField("##x", &value->x, o)) MarkChanged();
        ImGui::SameLine();
        o.hint = "Down (pixels)";
        if (cyxwiz::ui::NumberField("##y", &value->y, o)) MarkChanged();
        ImGui::PopID();
    };
    const auto table = [](const char* id) {
        if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Item inner spacing").x + 16.0f);
        ImGui::TableSetupColumn("field", ImGuiTableColumnFlags_WidthStretch);
        return true;
    };

    cyxwiz::ui::SettingsSection("Rounding");
    if (table("##rounding")) {
        row("Windows", &style.WindowRounding, 16.0, 1.0);
        row("Fields and buttons", &style.FrameRounding, 16.0, 1.0);
        row("Menus and popups", &style.PopupRounding, 16.0, 1.0);
        row("Scrollbars", &style.ScrollbarRounding, 16.0, 1.0);
        row("Grabs", &style.GrabRounding, 16.0, 1.0);
        row("Tabs", &style.TabRounding, 16.0, 1.0);
        ImGui::EndTable();
    }
    cyxwiz::ui::SettingsSection("Lines");
    ImGui::TextDisabled("Fields and buttons have no outline (a fill marks them); windows keep a line.");
    if (table("##borders")) {
        row("Window border", &style.WindowBorderSize, 3.0, 0.5);
        row("Popup border", &style.PopupBorderSize, 3.0, 0.5);
        ImGui::EndTable();
    }
    cyxwiz::ui::SettingsSection("Padding and spacing");
    if (table("##padding")) {
        pair_row("Window padding", &style.WindowPadding, 24.0);
        pair_row("Field padding", &style.FramePadding, 24.0);
        pair_row("Item spacing", &style.ItemSpacing, 24.0);
        pair_row("Item inner spacing", &style.ItemInnerSpacing, 24.0);
        pair_row("Table cell padding", &style.CellPadding, 24.0);
        ImGui::EndTable();
    }
    cyxwiz::ui::SettingsSection("Sizes");
    if (table("##sizes")) {
        row("Scrollbar width", &style.ScrollbarSize, 24.0, 1.0);
        row("Smallest grab", &style.GrabMinSize, 24.0, 1.0);
        row("Indent", &style.IndentSpacing, 32.0, 1.0);
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

void ThemeEditorPanel::RenderSavedThemesTab() {
    ImGui::BeginChild("SavedThemes", ImVec2(0, 0));
    ImGui::TextUnformatted("Your custom themes");
    ImGui::TextDisabled("%s", CustomThemesDirectory().string().c_str());
    ImGui::Spacing();

    const auto themes = ListCustomThemes();
    const std::string active = ActiveCustomThemeName();
    if (themes.empty()) {
        ImGui::TextDisabled("No custom themes yet. Change colours or shape, then save.");
    } else if (ImGui::BeginTable("##custom_themes", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthFixed,
                                cyxwiz::ui::ButtonWidth("Apply", cyxwiz::ui::ButtonSize::Small));
        for (const auto& path : themes) {
            const std::string name = path.stem().string();
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            if (name == active)
                ImGui::Text(ICON_FA_CIRCLE_CHECK " %s", name.c_str());
            else
                ImGui::TextUnformatted(name.c_str());
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
    if (cyxwiz::ui::SecondaryButton(ICON_FA_FOLDER_OPEN " Load theme file...")) LoadThemeFromFileDialog();

    ImGui::Spacing();
    ImGui::TextDisabled("Edits apply live. A saved or loaded theme is re-applied at startup;");
    ImGui::TextDisabled("choosing a base theme replaces it.");
    ImGui::EndChild();
}

void ThemeEditorPanel::RenderSaveDialog() {
    if (show_save_dialog_) {
        ImGui::OpenPopup("Save Theme");
        show_save_dialog_ = false;
    }
    if (!ImGui::BeginPopupModal("Save Theme", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    ImGui::TextUnformatted("Save the current colours and shape as a custom theme.");
    ImGui::TextDisabled("Stored in %s", CustomThemesDirectory().string().c_str());
    ImGui::SetNextItemWidth(320.0f);
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    const bool enter = ImGui::InputTextWithHint("##theme_name", "Theme name", theme_name_buffer_,
                                                sizeof(theme_name_buffer_), ImGuiInputTextFlags_EnterReturnsTrue);
    if (!status_message_.empty() && status_error_) ImGui::TextColored(t.error, "%s", status_message_.c_str());
    ImGui::Spacing();
    const bool has_name = theme_name_buffer_[0] != '\0';
    if (cyxwiz::ui::PrimaryButton("Save", has_name, "Type a name first.", cyxwiz::ui::ButtonSize::Regular) ||
        (enter && has_name)) {
        if (SaveTheme(theme_name_buffer_)) ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (cyxwiz::ui::SecondaryButton("Cancel", true, nullptr, cyxwiz::ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

bool ThemeEditorPanel::SaveTheme(const std::string& name) {
    fs::path saved;
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
    status_message_ = "Loaded " + fs::path(path).stem().string();
    status_error_ = false;
    return true;
}

void ThemeEditorPanel::LoadThemeFromFileDialog() {
    const auto dir = CustomThemesDirectory().string();
    if (const auto result = cyxwiz::FileDialogs::OpenTheme(dir.c_str())) LoadTheme(*result);
}

}  // namespace gui
