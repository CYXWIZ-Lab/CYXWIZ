#pragma once

// The Script Editor's code view (TOFIX133 P1, decision D1): draws an
// editor::Document with ImGui and turns keys and mouse into its commands.
// Look approved on the mockup board "Text mode": one gutter (breakpoint,
// line number, fold arrow), current-line tone, multiple cursors, colours from
// the Engine theme, scrollbars in the surface tone, no outlines.
//
// Keys handled here are the ones Preferences > Shortcuts marks as handled
// by the window (typing, moving, selecting, clipboard, undo, Tab); commands
// with Engine-wide chords (duplicate line, comment, find...) reach the
// Script Editor through the shortcut table and call the document directly.

#include "../core/editor/folding.h"
#include "../core/editor/python_highlight.h"
#include "../core/editor/text_document.h"

#include <imgui.h>

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

    // Breakpoints are 1-based line numbers owned by the caller; a gutter
    // click reports the 0-based line.
    void SetBreakpoints(const std::vector<int>* lines) { breakpoints_ = lines; }
    std::function<void(int line)> on_gutter_click;

    void SetMarks(std::vector<Mark> marks) { marks_ = std::move(marks); }
    void SetDebugLine(int line) { debug_line_ = line; }  // 0-based, -1 none

    // Minimap at the right: the text's shape in its colours, the visible part
    // in a lighter tone; click or drag to scroll.
    void SetShowMinimap(bool show) { show_minimap_ = show; }

    void RequestFocus() { request_focus_ = true; }
    bool IsFocused() const { return focused_; }
    void ScrollToCursor() { scroll_to_cursor_ = true; }
    // Moves the cursor to a line (0-based), opening folds and scrolling there.
    void GoToLine(int line);

    // Draws the editor and handles input. Returns true when the text changed.
    bool Render(const char* id, const ImVec2& size);

    // Bottom-left of the primary cursor on screen, from the last Render.
    ImVec2 CursorScreenPos() const { return cursor_screen_; }

private:
    struct Palette {
        ImU32 bg, current_line, text, line_number, line_number_current, selection, selection_inactive, caret,
            breakpoint, fold, fold_hover, mark, mark_current, debug_line, whitespace, pill_bg, pill_text,
            scroll, scroll_hover;
        ImU32 tokens[11];
    };
    Palette BuildPalette() const;
    void HandleKeyboard(float page_height, float line_height);
    void HandleMouse(const ImVec2& origin, float gutter, float advance, float line_height,
                     const std::vector<int>& visible);
    editor::Pos MouseToPos(const ImVec2& mouse, const ImVec2& origin, float gutter, float advance, float line_height,
                           const std::vector<int>& visible) const;
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
    bool keyboard_enabled_ = true;
    bool focused_ = false;
    bool request_focus_ = false;
    bool scroll_to_cursor_ = false;
    bool scroll_to_top_ = false;
    bool dragging_ = false;
    double last_activity_ = 0.0;
    ImVec2 cursor_screen_{0.0f, 0.0f};
    uint64_t width_version_ = ~0ull;
    float max_columns_ = 0.0f;

    bool show_minimap_ = false;
    float pending_scroll_y_ = -1.0f;  // set by the minimap, applied in the code view
    std::vector<int> visible_;        // visible lines from the last Render
    float last_scroll_y_ = 0.0f;
    float last_view_h_ = 0.0f;
    float last_line_h_ = 1.0f;
    float last_content_h_ = 0.0f;
};

}  // namespace cyxwiz
