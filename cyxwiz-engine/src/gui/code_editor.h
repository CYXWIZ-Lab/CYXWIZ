#pragma once

// The Script Editor's code view (TOFIX133 P1/P2, decision D1): draws an
// editor::Document with ImGui and turns keys and mouse into its commands.
// Look approved on the mockup boards: one gutter (change bar, breakpoint,
// line number, fold arrow), current-line tone, multiple cursors, colours from
// the Engine theme, scrollbars in the surface tone, no outlines, a minimap,
// optional soft wrap.
//
// Keys handled here are the ones Preferences > Shortcuts marks as handled
// by the window (typing, moving, selecting, clipboard, undo, Tab); commands
// with Engine-wide chords (duplicate line, comment, find...) reach the
// Script Editor through the shortcut table and call the document directly.

#include "../core/editor/folding.h"
#include "../core/editor/python_highlight.h"
#include "../core/editor/text_document.h"

#include <imgui.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

class CodeEditor {
public:
    struct Mark {  // a search hit or other range drawn behind the text
        editor::Pos a;
        editor::Pos b;
        bool current = false;
    };

    editor::Document& Doc() { return doc_; }
    const editor::Document& Doc() const { return doc_; }
    editor::FoldState& Folds() { return folds_; }

    void SetText(std::string_view text);
    std::string GetText() const { return doc_.Text(); }

    void SetReadOnly(bool read_only) { read_only_ = read_only; }
    bool IsReadOnly() const { return read_only_; }
    void SetShowWhitespace(bool show) { show_whitespace_ = show; }
    void SetColorize(bool colorize) { colorize_ = colorize; }
    void SetTabSize(int size) { doc_.Settings().tab_size = size; }
    void SetKeyboardEnabled(bool enabled) { keyboard_enabled_ = enabled; }
    void SetLanguageIsPython(bool python) { python_ = python; }
    // Soft wrap at the view's width (Up/Down still move by document lines).
    void SetWordWrap(bool wrap) { wrap_ = wrap; }

    // Breakpoints are 1-based line numbers owned by the caller; a gutter
    // click reports the 0-based line.
    void SetBreakpoints(const std::vector<int>* lines) { breakpoints_ = lines; }
    std::function<void(int line)> on_gutter_click;

    void SetMarks(std::vector<Mark> marks) { marks_ = std::move(marks); }
    void SetDebugLine(int line) { debug_line_ = line; }  // 0-based, -1 none

    // Minimap at the right: the text's shape in its colours, the visible part
    // in a lighter tone; click or drag to scroll.
    void SetShowMinimap(bool show) { show_minimap_ = show; }

    // Notebook cells (TOFIX133 P4 step 4.3b, board 4): no line numbers or
    // fold arrows (a narrow margin keeps breakpoints and change bars), the
    // caller's surface colour (0 = the editor's own), and no blank rows after
    // the last line so a view sized to RowCount() rows never scrolls.
    void SetShowLineNumbers(bool show) { show_line_numbers_ = show; }
    void SetBackground(ImU32 colour) { background_ = colour; }
    void SetScrollPastEnd(bool on) { scroll_past_end_ = on; }
    // Screen rows at the last Render (wrapped pieces count); 0 before it.
    int RowCount() const { return static_cast<int>(rows_.size()); }
    // Line height the view uses with the current font.
    static float LineHeightFor(float font_size) { return std::floor(font_size * 1.4f); }

    void RequestFocus() { request_focus_ = true; }
    bool IsFocused() const { return focused_; }
    void ScrollToCursor() { scroll_to_cursor_ = true; }
    // Moves the cursor to a line (0-based), opening folds and scrolling there.
    void GoToLine(int line);

    // A right click in the text since the last call (the caller opens its menu).
    bool TakeContextMenuRequest() {
        const bool r = context_menu_requested_;
        context_menu_requested_ = false;
        return r;
    }

    // Draws the editor and handles input. Returns true when the text changed.
    bool Render(const char* id, const ImVec2& size);

    // Bottom-left of the primary cursor on screen, from the last Render.
    ImVec2 CursorScreenPos() const { return cursor_screen_; }

private:
    struct Palette {
        ImU32 bg, current_line, text, line_number, line_number_current, selection, selection_inactive, caret,
            breakpoint, fold, fold_hover, mark, mark_current, debug_line, whitespace, pill_bg, pill_text,
            scroll, scroll_hover, change_bar;
        ImU32 tokens[11];
    };
    // One screen row: a whole line, or a wrapped piece of one.
    struct Row {
        int line = 0;
        int start = 0;  // byte range of the line shown on this row
        int end = 0;
        bool first = true;
        bool last = true;
    };

    Palette BuildPalette() const;
    void BuildRows(const std::vector<int>& visible, int wrap_columns);
    int RowOfPos(editor::Pos p) const;
    void HandleKeyboard(float page_height, float line_height);
    void HandleMouse(const ImVec2& origin, float gutter, float advance, float line_height);
    editor::Pos MouseToPos(const ImVec2& mouse, const ImVec2& origin, float gutter, float advance, float line_height) const;
    void TrackChanges(const std::vector<editor::LineEdit>& edits);
    void CopySelection(bool cut);
    void Touch() { last_activity_ = ImGui::GetTime(); }
    float MaxLineWidthColumns();
    void RenderMinimap(const Palette& pal, float height);

    editor::Document doc_;
    editor::FoldState folds_;
    editor::Highlighter highlighter_;
    std::vector<Mark> marks_;
    const std::vector<int>* breakpoints_ = nullptr;
    int debug_line_ = -1;
    bool read_only_ = false;
    bool show_whitespace_ = false;
    bool colorize_ = true;
    bool python_ = true;
    bool wrap_ = false;
    bool keyboard_enabled_ = true;
    bool focused_ = false;
    bool request_focus_ = false;
    bool scroll_to_cursor_ = false;
    bool scroll_to_top_ = false;
    bool dragging_ = false;
    bool context_menu_requested_ = false;
    double last_activity_ = 0.0;
    ImVec2 cursor_screen_{0.0f, 0.0f};
    uint64_t width_version_ = ~0ull;
    float max_columns_ = 0.0f;

    // Lines changed since the text was loaded or saved (change bars).
    std::vector<uint8_t> changed_;

    // Rows from the last Render, rebuilt when the text, folds or wrap width change.
    std::vector<Row> rows_;
    uint64_t rows_version_ = ~0ull;
    int rows_wrap_ = -1;
    std::vector<int> rows_visible_;

    bool show_minimap_ = false;
    bool show_line_numbers_ = true;
    ImU32 background_ = 0;
    bool scroll_past_end_ = true;
    float pending_scroll_y_ = -1.0f;  // set by the minimap, applied in the code view
    std::vector<int> visible_;        // visible lines from the last Render
    float last_scroll_y_ = 0.0f;
    float last_view_h_ = 0.0f;
    float last_line_h_ = 1.0f;
    float last_content_h_ = 0.0f;
};

}  // namespace cyxwiz
