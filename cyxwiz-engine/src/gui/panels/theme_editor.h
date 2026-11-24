#pragma once

#include "../panel.h"
#include "../theme.h"

#include <string>
#include <utility>
#include <vector>

namespace gui {

// Theme Editor (TOFIX129 A4, one colour system). Every colour the Engine draws
// is in it and every row changes the screen:
//  - Engine colours: the accent, the node families, the data types (pins,
//    links, plot series) and the status words (with a contrast warning);
//  - Surfaces: window, text, fields, buttons, tabs, tables (what follows the
//    accent - check marks, selection, the tab line - has no row of its own);
//  - Canvas: the grid (nodes, pins and links use the Engine colours);
//  - Shape: rounding, borders, padding and sizes as typed numbers;
//  - Saved themes: custom themes kept over the base preset.
// Edits apply at once; Save as custom theme keeps them after a restart.
class ThemeEditorPanel : public cyxwiz::Panel {
public:
    ThemeEditorPanel();
    ~ThemeEditorPanel() override = default;

    void Render() override;

private:
    void RenderPresetSelector();
    void RenderEngineColoursTab();
    void RenderSurfacesTab();
    void RenderCanvasTab();
    void RenderShapeTab();
    void RenderSavedThemesTab();
    void RenderSaveDialog();

    bool SaveTheme(const std::string& name);
    bool LoadTheme(const std::string& path);
    void LoadThemeFromFileDialog();

    // One row of a two-column colour table: the label, then the typed field;
    // `note` (may be empty) under the field, in `note_colour`.
    bool ColourRow(const char* label, ImVec4* colour, bool with_alpha = false, const char* note = nullptr,
                   const ImVec4* note_colour = nullptr);
    void MarkChanged() { has_unsaved_changes_ = true; }

    bool has_unsaved_changes_ = false;
    bool show_save_dialog_ = false;
    std::string status_message_;
    bool status_error_ = false;
    char theme_name_buffer_[256] = "";
    char surface_filter_[128] = "";

    struct ColourGroup {
        const char* name;
        std::vector<std::pair<ImGuiCol_, const char*>> colours;
    };
    static const std::vector<ColourGroup> kSurfaceGroups;
};

}  // namespace gui
