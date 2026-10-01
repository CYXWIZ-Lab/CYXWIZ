#include "code_editor.h"

#include "ui_tokens.h"

#include <imgui_internal.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

using editor::Motion;
using editor::Pos;
using editor::TokenKind;

namespace {
int Digits(int n) {
    int d = 1;
    while (n >= 10) {
        n /= 10;
        ++d;
    }
    return d;
}

void AppendUtf8(std::string& out, unsigned int c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}

}  // namespace

void CodeEditor::SetText(std::string_view text) {
    doc_.SetText(text);
    folds_.UnfoldAll();
    scroll_to_top_ = true;
}

void CodeEditor::GoToLine(int line) {
    line = std::clamp(line, 0, doc_.LineCount() - 1);
    folds_.Reveal(line);
    doc_.SetCursor({line, 0});
    scroll_to_cursor_ = true;
    request_focus_ = true;
}

CodeEditor::Palette CodeEditor::BuildPalette() const {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec4 black(0, 0, 0, 1);
    const ImVec4 white(1, 1, 1, 1);
    // The code surface is one step darker (lighter on light themes) than the window.
    const ImVec4 bg = t.light ? ui::Mix(t.bg_window, white, 0.6f) : ui::Mix(t.bg_window, black, 0.18f);
    Palette p{};
    p.bg = ui::ToU32(bg);
    p.current_line = ui::ToU32(ui::Mix(bg, t.light ? black : white, 0.035f));
    p.text = ui::ToU32(t.text);
    p.line_number = ui::ToU32(ui::Mix(t.text_dim, bg, 0.35f));
    p.line_number_current = ui::ToU32(t.text);
    p.selection = ui::ToU32(ui::WithAlpha(t.accent, t.light ? 0.22f : 0.30f));
    p.selection_inactive = ui::ToU32(ui::WithAlpha(t.text_dim, 0.18f));
    p.caret = ui::ToU32(t.text_bright);
    p.breakpoint = ui::ToU32(t.error);
    p.fold = ui::ToU32(ui::Mix(t.text_dim, bg, 0.3f));
    p.fold_hover = ui::ToU32(t.text);
    p.mark = ui::ToU32(ui::WithAlpha(t.warning, 0.20f));
    p.mark_current = ui::ToU32(ui::WithAlpha(t.warning, 0.45f));
    p.debug_line = ui::ToU32(ui::WithAlpha(t.warning, 0.16f));
    p.whitespace = ui::ToU32(ui::Mix(t.text_dim, bg, 0.6f));
    p.pill_bg = ui::ToU32(ui::Mix(bg, t.light ? black : white, 0.08f));
    p.pill_text = ui::ToU32(t.text_dim);
    p.scroll = ui::ToU32(ui::Mix(bg, t.light ? black : white, 0.07f));
    p.scroll_hover = ui::ToU32(ui::Mix(bg, t.light ? black : white, 0.13f));
    p.change_bar = ui::ToU32(ui::WithAlpha(t.accent_text, 0.55f));
    const ImVec4 fn = t.light ? ImVec4(0.0f, 0.42f, 0.58f, 1.0f) : ImVec4(0.50f, 0.78f, 0.91f, 1.0f);
    const ImVec4 konst = t.light ? ImVec4(0.52f, 0.20f, 0.72f, 1.0f) : ImVec4(0.85f, 0.63f, 1.0f, 1.0f);
    p.tokens[static_cast<int>(TokenKind::Default)] = ui::ToU32(t.text);
    p.tokens[static_cast<int>(TokenKind::Keyword)] = ui::ToU32(t.accent_text);
    p.tokens[static_cast<int>(TokenKind::Builtin)] = ui::ToU32(t.success);
    p.tokens[static_cast<int>(TokenKind::String)] = ui::ToU32(t.warning);
    p.tokens[static_cast<int>(TokenKind::Number)] = ui::ToU32(t.caution);
    p.tokens[static_cast<int>(TokenKind::Comment)] = ui::ToU32(t.text_dim);
    p.tokens[static_cast<int>(TokenKind::Function)] = ui::ToU32(fn);
    p.tokens[static_cast<int>(TokenKind::Decorator)] = ui::ToU32(t.caution);
    p.tokens[static_cast<int>(TokenKind::Constant)] = ui::ToU32(konst);
    p.tokens[static_cast<int>(TokenKind::Punctuation)] = ui::ToU32(ui::Mix(t.text, bg, 0.2f));
    p.tokens[static_cast<int>(TokenKind::CellMarker)] = ui::ToU32(t.accent_text);
    return p;
}

float CodeEditor::MaxLineWidthColumns() {
    if (width_version_ != doc_.Version()) {
        width_version_ = doc_.Version();
        int best = 0;
        for (int i = 0; i < doc_.LineCount(); ++i) {
            const std::string& line = doc_.Line(i);
            best = std::max(best, editor::VisualColumn(line, static_cast<int>(line.size()), doc_.Settings().tab_size));
        }
        max_columns_ = static_cast<float>(best);
    }
    return max_columns_;
}

void CodeEditor::CopySelection(bool cut) {
    std::string text = doc_.SelectedText();
    if (text.empty()) {
        // No selection: the whole line, as other editors do.
        std::vector<int> lines;
        for (const auto& s : doc_.Selections()) lines.push_back(s.head.line);
        for (const int l : lines) text += doc_.Line(l) + "\n";
        ImGui::SetClipboardText(text.c_str());
        if (cut && !read_only_) doc_.DeleteLines();
        return;
    }
    ImGui::SetClipboardText(text.c_str());
    if (cut && !read_only_) doc_.Backspace();
}

void CodeEditor::HandleKeyboard(float page_height, float line_height) {
    ImGuiIO& io = ImGui::GetIO();
    const bool ctrl = io.KeyCtrl;
    const bool shift = io.KeyShift;
    const bool alt = io.KeyAlt;
    io.WantCaptureKeyboard = true;
    io.WantTextInput = true;
    const uint64_t before = doc_.Version();
    const Pos before_head = doc_.Primary().head;
    const int before_count = doc_.CursorCount();
    const int page = std::max(1, static_cast<int>(page_height / line_height) - 1);
    auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, true); };

    // Moving and selecting.
    if (!alt) {
        if (pressed(ImGuiKey_LeftArrow)) doc_.Move(ctrl ? Motion::WordLeft : Motion::Left, shift);
        if (pressed(ImGuiKey_RightArrow)) doc_.Move(ctrl ? Motion::WordRight : Motion::Right, shift);
        if (!ctrl && pressed(ImGuiKey_UpArrow)) doc_.Move(Motion::Up, shift);
        if (!ctrl && pressed(ImGuiKey_DownArrow)) doc_.Move(Motion::Down, shift);
        if (pressed(ImGuiKey_Home)) doc_.Move(ctrl ? Motion::DocStart : Motion::Home, shift);
        if (pressed(ImGuiKey_End)) doc_.Move(ctrl ? Motion::DocEnd : Motion::End, shift);
        if (pressed(ImGuiKey_PageUp)) doc_.Move(Motion::PageUp, shift, page);
        if (pressed(ImGuiKey_PageDown)) doc_.Move(Motion::PageDown, shift, page);
    }
    if (ctrl && alt && pressed(ImGuiKey_UpArrow)) doc_.AddCursorVertical(-1);
    if (ctrl && alt && pressed(ImGuiKey_DownArrow)) doc_.AddCursorVertical(1);
    if (ctrl && !alt && !shift && pressed(ImGuiKey_A)) doc_.SelectAll();
    if (ctrl && shift && !alt && pressed(ImGuiKey_L)) doc_.SelectAllOccurrences();
    if (!ctrl && !alt && !shift && pressed(ImGuiKey_Escape) && doc_.CursorCount() > 1) {
        const Pos p = doc_.Primary().head;
        doc_.SetCursor(p);
    }
    if (ctrl && shift && !alt && pressed(ImGuiKey_LeftBracket)) folds_.Fold(doc_.Primary().head.line);
    if (ctrl && shift && !alt && pressed(ImGuiKey_RightBracket)) folds_.Unfold(doc_.Primary().head.line);

    // Clipboard and undo work on read-only text too (copy only).
    if (ctrl && !alt && !shift && (pressed(ImGuiKey_C) || pressed(ImGuiKey_Insert))) CopySelection(false);

    if (!read_only_) {
        if (ctrl && !alt && !shift && pressed(ImGuiKey_X)) CopySelection(true);
        if (((ctrl && !alt && !shift) && pressed(ImGuiKey_V)) || (shift && !ctrl && pressed(ImGuiKey_Insert))) {
            if (const char* clip = ImGui::GetClipboardText()) doc_.Paste(clip);
        }
        if (ctrl && !alt && !shift && pressed(ImGuiKey_Z)) doc_.Undo();
        if (ctrl && !alt && ((!shift && pressed(ImGuiKey_Y)) || (shift && pressed(ImGuiKey_Z)))) doc_.Redo();
        if (!ctrl && !alt && (pressed(ImGuiKey_Enter) || pressed(ImGuiKey_KeypadEnter))) doc_.Newline();
        if (!alt && pressed(ImGuiKey_Backspace)) {
            if (ctrl) doc_.DeleteWordLeft();
            else doc_.Backspace();
        }
        if (!alt && !shift && pressed(ImGuiKey_Delete)) {
            if (ctrl) doc_.DeleteWordRight();
            else doc_.DeleteForward();
        }
        if (!ctrl && !alt && pressed(ImGuiKey_Tab)) {
            if (shift) doc_.Outdent();
            else doc_.Tab();
        }
        // Typed characters (AltGr arrives as Ctrl+Alt and still types).
        if (!ctrl || alt) {
            std::string typed;
            for (int i = 0; i < io.InputQueueCharacters.Size; ++i) {
                const unsigned int c = io.InputQueueCharacters[i];
                if (c >= 32 && c != 127) AppendUtf8(typed, c);
            }
            if (!typed.empty()) doc_.Type(typed);
        }
    }
    io.InputQueueCharacters.resize(0);

    if (doc_.Version() != before || doc_.Primary().head != before_head || doc_.CursorCount() != before_count) {
        folds_.Reveal(doc_.Primary().head.line);
        scroll_to_cursor_ = true;
        Touch();
    }
}

void CodeEditor::RenderMinimap(const Palette& pal, float height) {
    constexpr float kWidth = 96.0f;
    constexpr float kRow = 2.5f;    // pixels per line
    constexpr float kChar = 0.85f;  // pixels per column
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##minimap", ImVec2(kWidth, std::max(1.0f, height)));
    const bool active = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + kWidth, p.y + height), pal.bg);

    const int rows = static_cast<int>(visible_.size());
    const float total = static_cast<float>(rows) * kRow;
    const float max_scroll = std::max(1.0f, last_content_h_ - last_view_h_);
    const float offset = total > height ? (last_scroll_y_ / max_scroll) * (total - height) : 0.0f;
    const int tab = doc_.Settings().tab_size;
    const int first = std::max(0, static_cast<int>(offset / kRow));
    const int last = std::min(rows - 1, static_cast<int>((offset + height) / kRow) + 1);

    // The part the code view shows, in a lighter tone (no outline).
    const float view_top = p.y + (last_scroll_y_ / last_line_h_) * kRow - offset;
    const float view_h = (last_view_h_ / last_line_h_) * kRow;
    dl->AddRectFilled(ImVec2(p.x, view_top), ImVec2(p.x + kWidth, view_top + view_h),
                      ImGui::IsItemHovered() || active ? pal.scroll_hover : pal.scroll);

    dl->PushClipRect(p, ImVec2(p.x + kWidth, p.y + height), true);
    for (int r = first; r <= last; ++r) {
        const int line_no = visible_[static_cast<size_t>(r)];
        const std::string& line = doc_.Line(line_no);
        const float y = p.y + static_cast<float>(r) * kRow - offset;
        auto bar = [&](int a, int b, ImU32 colour) {
            const float x1 = p.x + 6.0f + static_cast<float>(editor::VisualColumn(line, a, tab)) * kChar;
            const float x2 = p.x + 6.0f + static_cast<float>(editor::VisualColumn(line, b, tab)) * kChar;
            if (x2 > x1) dl->AddRectFilled(ImVec2(x1, y), ImVec2(std::min(x2, p.x + kWidth - 4.0f), y + kRow - 0.8f),
                                           (colour & 0x00FFFFFFu) | 0x99000000u);
        };
        // Words in the line's colours; spaces stay empty.
        int at = 0;
        const int n = static_cast<int>(line.size());
        auto plain = [&](int a, int b) {
            int i = a;
            while (i < b) {
                while (i < b && (line[static_cast<size_t>(i)] == ' ' || line[static_cast<size_t>(i)] == '\t')) ++i;
                int j = i;
                while (j < b && line[static_cast<size_t>(j)] != ' ' && line[static_cast<size_t>(j)] != '\t') ++j;
                if (j > i) bar(i, j, pal.text);
                i = j;
            }
        };
        if (colorize_ && python_) {
            for (const auto& span : highlighter_.Spans(line_no)) {
                if (span.start > at) plain(at, span.start);
                bar(span.start, span.start + span.length, pal.tokens[static_cast<int>(span.kind)]);
                at = span.start + span.length;
            }
        }
        if (at < n) plain(at, n);
    }
    dl->PopClipRect();

    if (active) {
        // Centre the view on the line under the mouse.
        const float row = (ImGui::GetIO().MousePos.y - p.y + offset) / kRow;
        const float target = row * last_line_h_ - last_view_h_ * 0.5f;
        pending_scroll_y_ = std::clamp(target, 0.0f, max_scroll);
    }
}

void CodeEditor::BuildRows(const std::vector<int>& visible, int wrap_columns) {
    if (rows_version_ == doc_.Version() && rows_wrap_ == wrap_columns && rows_visible_ == visible) return;
    rows_version_ = doc_.Version();
    rows_wrap_ = wrap_columns;
    rows_visible_ = visible;
    rows_.clear();
    rows_.reserve(visible.size());
    const int tab = doc_.Settings().tab_size;
    for (const int line_no : visible) {
        const std::string& line = doc_.Line(line_no);
        const int n = static_cast<int>(line.size());
        if (wrap_columns <= 0 || editor::VisualColumn(line, n, tab) <= wrap_columns) {
            rows_.push_back({line_no, 0, n, true, true});
            continue;
        }
        // Break at the last space that fits, else at the width.
        int start = 0;
        while (start < n) {
            const int base = editor::VisualColumn(line, start, tab);
            int end = start;
            int last_space = -1;
            while (end < n) {
                const int next = end + 1;
                if (editor::VisualColumn(line, next, tab) - base > wrap_columns) break;
                if (line[static_cast<size_t>(end)] == ' ') last_space = next;
                end = next;
            }
            if (end < n && last_space > start) end = last_space;
            if (end == start) end = start + 1;  // one wide character
            rows_.push_back({line_no, start, end, start == 0, end >= n});
            start = end;
        }
    }
    if (rows_.empty()) rows_.push_back({0, 0, 0, true, true});
}

int CodeEditor::RowOfPos(Pos p) const {
    // Rows are sorted by (line, start).
    int lo = 0;
    int hi = static_cast<int>(rows_.size()) - 1;
    int found = 0;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const Row& r = rows_[static_cast<size_t>(mid)];
        if (r.line < p.line || (r.line == p.line && r.start <= p.col)) {
            found = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return found;
}

void CodeEditor::TrackChanges(const std::vector<editor::LineEdit>& edits) {
    for (const auto& e : edits) {
        if (e.reset) {
            changed_.assign(static_cast<size_t>(doc_.LineCount()), 0);
            continue;
        }
        const size_t at = static_cast<size_t>(std::max(0, e.line));
        if (changed_.size() < at + static_cast<size_t>(e.removed) + 1) changed_.resize(at + static_cast<size_t>(e.removed) + 1, 0);
        changed_.erase(changed_.begin() + static_cast<long long>(at), changed_.begin() + static_cast<long long>(at + e.removed + 1));
        changed_.insert(changed_.begin() + static_cast<long long>(at), static_cast<size_t>(e.added) + 1, 1);
    }
    changed_.resize(static_cast<size_t>(doc_.LineCount()), 0);
    // Saved again (or undone back to the saved text): nothing is changed.
    if (!doc_.Modified()) std::fill(changed_.begin(), changed_.end(), 0);
}

Pos CodeEditor::MouseToPos(const ImVec2& mouse, const ImVec2& origin, float gutter, float advance, float line_height) const {
    const int row_index = std::clamp(static_cast<int>(std::floor((mouse.y - origin.y) / line_height)), 0,
                                     static_cast<int>(rows_.size()) - 1);
    const Row& row = rows_[static_cast<size_t>(row_index)];
    const std::string& line = doc_.Line(row.line);
    const int tab = doc_.Settings().tab_size;
    const float x = (mouse.x - origin.x - gutter) / advance;
    const int visual = std::max(0, static_cast<int>(std::lround(x))) + editor::VisualColumn(line, row.start, tab);
    int col = editor::ByteForVisual(line, visual, tab);
    col = std::clamp(col, row.start, row.last ? row.end : std::max(row.start, row.end - 1));
    return {row.line, col};
}

void CodeEditor::HandleMouse(const ImVec2& origin, float gutter, float advance, float line_height) {
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const ImVec2 win = ImGui::GetWindowPos();
    const bool in_gutter = mouse.x < win.x + gutter;

    if (ImGui::IsWindowHovered()) {
        if (!in_gutter) ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const Pos p = MouseToPos(mouse, origin, gutter, advance, line_height);
            if (in_gutter) {
                const float fold_x = win.x + gutter - advance * 2.0f;
                if (show_line_numbers_ && mouse.x >= fold_x && folds_.CanFold(p.line)) folds_.Toggle(p.line);
                else if (on_gutter_click) on_gutter_click(p.line);
            } else {
                const int clicks = io.MouseClickedCount[ImGuiMouseButton_Left];
                if (clicks == 2) doc_.SelectWord(p);
                else if (clicks >= 3) doc_.SelectLine(p.line);
                else if (io.KeyAlt) doc_.AddCursor(p);
                else doc_.SetCursor(p, io.KeyShift);
                dragging_ = clicks == 1 && !io.KeyAlt;
                Touch();
            }
        }
        if (!in_gutter && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            // Keep the selection when the click is inside it, else move the cursor there.
            const Pos p = MouseToPos(mouse, origin, gutter, advance, line_height);
            bool inside = false;
            for (const auto& s : doc_.Selections()) inside = inside || (!s.Empty() && s.Start() <= p && p <= s.End());
            if (!inside) doc_.SetCursor(p);
            ImGui::SetWindowFocus();
            context_menu_requested_ = true;
            Touch();
        }
    }
    if (dragging_) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
                const Pos p = MouseToPos(mouse, origin, gutter, advance, line_height);
                if (p != doc_.Primary().head) {
                    doc_.SetCursor(p, true);
                    scroll_to_cursor_ = true;
                    Touch();
                }
            }
        } else {
            dragging_ = false;
        }
    }
}

bool CodeEditor::Render(const char* id, const ImVec2& size) {
    const uint64_t version_before = doc_.Version();
    {
        const auto edits = doc_.TakeLineEdits();
        folds_.Update(doc_, edits);
        TrackChanges(edits);
    }
    if (colorize_ && python_) highlighter_.Update(doc_);
    Palette pal = BuildPalette();
    if (background_ != 0) pal.bg = background_;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, pal.bg);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, pal.bg);
    // Scrollbars in the surface tone (owner rule): grab a few percent off it.
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, pal.scroll);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, pal.scroll_hover);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, pal.scroll_hover);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 3.0f);
    if (request_focus_) {
        ImGui::SetNextWindowFocus();
        request_focus_ = false;
    }
    constexpr float kMinimapWidth = 96.0f;
    ImVec2 code_size = size;
    if (show_minimap_) {
        const float full = size.x > 0.0f ? size.x : ImGui::GetContentRegionAvail().x + size.x;
        code_size.x = std::max(80.0f, full - kMinimapWidth);
    }
    const ImGuiWindowFlags flags =
        (wrap_ ? ImGuiWindowFlags_None : ImGuiWindowFlags_HorizontalScrollbar) | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNavInputs;
    const bool open = ImGui::BeginChild(id, code_size, ImGuiChildFlags_None, flags);
    if (!open) {
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(5);
        return false;
    }
    focused_ = ImGui::IsWindowFocused();

    const float font_size = ImGui::GetFontSize();
    ImFont* font = ImGui::GetFont();
    const float advance = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, " ").x;
    const float line_height = std::floor(font_size * 1.4f);
    const int tab = doc_.Settings().tab_size;
    const float gutter = show_line_numbers_
                             ? std::floor(advance * (2.0f + static_cast<float>(std::max(3, Digits(doc_.LineCount()))) + 3.5f))
                             : 12.0f;
    const ImVec2 avail = ImGui::GetContentRegionAvail();

    if (focused_ && keyboard_enabled_) HandleKeyboard(avail.y, line_height);

    // Rows: lines, or wrapped pieces of them.
    const std::vector<int> visible = folds_.VisibleLines(doc_.LineCount());
    const float text_width = ImGui::GetWindowWidth() - gutter - ImGui::GetStyle().ScrollbarSize - advance;
    const int wrap_columns = wrap_ ? std::max(20, static_cast<int>(text_width / advance)) : 0;
    BuildRows(visible, wrap_columns);

    // Content size defines the scroll range (a little room after the last row).
    const float content_w = wrap_ ? 0.0f : gutter + (MaxLineWidthColumns() + 4.0f) * advance;
    const float content_h = (static_cast<float>(rows_.size()) + (scroll_past_end_ ? 2.0f : 0.0f)) * line_height;
    ImGui::SetCursorPos(ImVec2(0, 0));
    ImGui::Dummy(ImVec2(content_w, content_h));

    if (pending_scroll_y_ >= 0.0f) {
        ImGui::SetScrollY(pending_scroll_y_);
        pending_scroll_y_ = -1.0f;
    }
    if (scroll_to_top_) {
        ImGui::SetScrollY(0.0f);
        ImGui::SetScrollX(0.0f);
        scroll_to_top_ = false;
    }
    const ImVec2 win = ImGui::GetWindowPos();
    const float scroll_x = wrap_ ? 0.0f : ImGui::GetScrollX();
    const float scroll_y = ImGui::GetScrollY();
    const ImVec2 origin(win.x - scroll_x, win.y - scroll_y);  // document (0,0) on screen, before the gutter
    const float view_h = ImGui::GetWindowHeight();
    const float view_w = ImGui::GetWindowWidth();

    HandleMouse(origin, gutter, advance, line_height);
    visible_ = visible;
    last_scroll_y_ = scroll_y;
    last_view_h_ = view_h;
    last_line_h_ = line_height;
    last_content_h_ = content_h;

    if (scroll_to_cursor_) {
        const Pos head = doc_.Primary().head;
        const float y = static_cast<float>(RowOfPos(head)) * line_height;
        if (y < scroll_y) ImGui::SetScrollY(y);
        else if (y + line_height > scroll_y + view_h - ImGui::GetStyle().ScrollbarSize)
            ImGui::SetScrollY(y + line_height - view_h + ImGui::GetStyle().ScrollbarSize);
        if (!wrap_) {
            const float x = static_cast<float>(editor::VisualColumn(doc_.Line(head.line), head.col, tab)) * advance;
            const float text_w = view_w - gutter - ImGui::GetStyle().ScrollbarSize;
            if (x < scroll_x) ImGui::SetScrollX(std::max(0.0f, x - advance * 4.0f));
            else if (x > scroll_x + text_w - advance * 2.0f) ImGui::SetScrollX(x - text_w + advance * 6.0f);
        }
        scroll_to_cursor_ = false;
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float text_x0 = win.x + gutter - scroll_x;
    const int first_row = std::max(0, static_cast<int>(scroll_y / line_height));
    const int last_row = std::min(static_cast<int>(rows_.size()) - 1, static_cast<int>((scroll_y + view_h) / line_height) + 1);
    const auto& selections = doc_.Selections();
    const editor::Selection primary = doc_.Primary();
    const bool blink_on = !focused_ ? false : std::fmod(ImGui::GetTime() - last_activity_, 1.0) < 0.6;

    // Text area, clipped right of the gutter.
    dl->PushClipRect(ImVec2(win.x + gutter, win.y), ImVec2(win.x + view_w, win.y + view_h), true);
    for (int r = first_row; r <= last_row; ++r) {
        const Row& row = rows_[static_cast<size_t>(r)];
        const int line_no = row.line;
        const std::string& line = doc_.Line(line_no);
        const float y = origin.y + static_cast<float>(r) * line_height;
        const float text_y = y + (line_height - font_size) * 0.5f;
        const int base = editor::VisualColumn(line, row.start, tab);
        auto x_of = [&](int byte) {
            return text_x0 + static_cast<float>(editor::VisualColumn(line, std::clamp(byte, row.start, row.end), tab) - base) * advance;
        };
        auto in_row = [&](Pos a, Pos b, float& x1, float& x2, float tail) {
            // The part of [a, b) on this row, as screen x; `tail` when it runs on past the row.
            if (b.line < line_no || a.line > line_no) return false;
            const int a_col = a.line < line_no ? row.start : std::max(a.col, row.start);
            const bool runs_on = b.line > line_no || b.col > row.end;
            const int b_col = runs_on ? row.end : std::min(b.col, row.end);
            if (a_col > b_col || (a_col == b_col && !runs_on)) return false;
            x1 = x_of(a_col);
            x2 = x_of(b_col) + (runs_on ? tail : 0.0f);
            return true;
        };

        if (line_no == debug_line_) {
            dl->AddRectFilled(ImVec2(win.x + gutter, y), ImVec2(win.x + view_w, y + line_height), pal.debug_line);
        } else if (focused_ && primary.Empty() && line_no == primary.head.line) {
            dl->AddRectFilled(ImVec2(win.x + gutter, y), ImVec2(win.x + view_w, y + line_height), pal.current_line);
        }
        float x1 = 0.0f;
        float x2 = 0.0f;
        for (const auto& m : marks_)
            if (in_row(m.a, m.b, x1, x2, advance))
                dl->AddRectFilled(ImVec2(x1, y + 1), ImVec2(x2, y + line_height - 1), m.current ? pal.mark_current : pal.mark, 2.0f);
        for (const auto& s : selections)
            if (!s.Empty() && in_row(s.Start(), s.End(), x1, x2, advance * 0.5f))
                dl->AddRectFilled(ImVec2(x1, y), ImVec2(x2, y + line_height), focused_ ? pal.selection : pal.selection_inactive);

        // Text: coloured spans clipped to the row, tabs expanded.
        auto draw_run = [&](int from, int to, ImU32 colour) {
            from = std::max(from, row.start);
            to = std::min(to, row.end);
            int a = from;
            while (a < to) {
                int b = a;
                while (b < to && line[static_cast<size_t>(b)] != '\t') ++b;
                if (b > a) dl->AddText(font, font_size, ImVec2(x_of(a), text_y), colour, line.data() + a, line.data() + b);
                a = b + (b < to ? 1 : 0);
            }
        };
        int at = row.start;
        if (colorize_ && python_) {
            for (const auto& span : highlighter_.Spans(line_no)) {
                if (span.start + span.length <= row.start) continue;
                if (span.start >= row.end) break;
                if (span.start > at) draw_run(at, span.start, pal.tokens[0]);
                draw_run(span.start, span.start + span.length, pal.tokens[static_cast<int>(span.kind)]);
                at = span.start + span.length;
            }
        }
        if (at < row.end) draw_run(at, row.end, pal.text);

        if (show_whitespace_) {
            for (int i = row.start; i < row.end; ++i) {
                if (line[static_cast<size_t>(i)] != ' ' && line[static_cast<size_t>(i)] != '\t') continue;
                dl->AddCircleFilled(ImVec2(x_of(i) + advance * 0.5f, y + line_height * 0.5f), 1.2f, pal.whitespace);
            }
        }

        if (row.last) {
            if (const int hidden = folds_.HiddenCount(line_no); hidden > 0) {
                char label[32];
                std::snprintf(label, sizeof(label), "\xC2\xB7\xC2\xB7\xC2\xB7 %d lines", hidden);
                const ImVec2 ts = font->CalcTextSizeA(font_size * 0.85f, FLT_MAX, 0.0f, label);
                const float px = x_of(row.end) + advance;
                dl->AddRectFilled(ImVec2(px, y + 2), ImVec2(px + ts.x + advance * 1.5f, y + line_height - 2), pal.pill_bg, line_height * 0.5f);
                dl->AddText(font, font_size * 0.85f, ImVec2(px + advance * 0.75f, y + (line_height - ts.y) * 0.5f), pal.pill_text, label);
            }
        }

        for (const auto& s : selections) {
            if (s.head.line != line_no || RowOfPos(s.head) != r) continue;
            const float cx = x_of(s.head.col);
            if (blink_on && !read_only_) dl->AddRectFilled(ImVec2(cx, y + 2), ImVec2(cx + 2.0f, y + line_height - 2), pal.caret);
            if (s.head == primary.head) cursor_screen_ = ImVec2(cx, y + line_height);
        }
    }
    dl->PopClipRect();

    // Gutter: change bar, breakpoints, line numbers, fold arrows. Same tone as the text.
    dl->PushClipRect(ImVec2(win.x, win.y), ImVec2(win.x + gutter, win.y + view_h), true);
    dl->AddRectFilled(ImVec2(win.x, win.y), ImVec2(win.x + gutter, win.y + view_h), pal.bg);
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool gutter_hover = ImGui::IsWindowHovered() && mouse.x < win.x + gutter;
    for (int r = first_row; r <= last_row; ++r) {
        const Row& row = rows_[static_cast<size_t>(r)];
        const int line_no = row.line;
        const float y = origin.y + static_cast<float>(r) * line_height;
        if (line_no < static_cast<int>(changed_.size()) && changed_[static_cast<size_t>(line_no)])
            dl->AddRectFilled(ImVec2(win.x, y), ImVec2(win.x + 3.0f, y + line_height), pal.change_bar);
        if (!row.first) continue;  // numbers, breakpoints and arrows on a line's first row
        if (!show_line_numbers_) {
            if (breakpoints_ &&
                std::find(breakpoints_->begin(), breakpoints_->end(), line_no + 1) != breakpoints_->end())
                dl->AddCircleFilled(ImVec2(win.x + gutter * 0.5f, y + line_height * 0.5f), 3.5f, pal.breakpoint);
            continue;
        }
        if (breakpoints_ &&
            std::find(breakpoints_->begin(), breakpoints_->end(), line_no + 1) != breakpoints_->end()) {
            dl->AddCircleFilled(ImVec2(win.x + advance * 1.2f, y + line_height * 0.5f), advance * 0.45f, pal.breakpoint);
        }
        char num[16];
        std::snprintf(num, sizeof(num), "%d", line_no + 1);
        const float nw = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, num).x;
        const float num_right = win.x + gutter - advance * 2.5f;
        const bool current = line_no == primary.head.line;
        dl->AddText(font, font_size, ImVec2(num_right - nw, y + (line_height - font_size) * 0.5f),
                    current ? pal.line_number_current : pal.line_number, num);
        if (folds_.CanFold(line_no)) {
            const bool folded = folds_.IsFolded(line_no);
            const bool hot = gutter_hover && mouse.y >= y && mouse.y < y + line_height;
            const ImU32 c = folded || hot || current ? pal.fold_hover : pal.fold;
            const float cx = win.x + gutter - advance * 1.3f;
            const float cy = y + line_height * 0.5f;
            const float rr = advance * 0.32f;
            if (folded) {
                dl->AddTriangleFilled(ImVec2(cx - rr * 0.6f, cy - rr), ImVec2(cx - rr * 0.6f, cy + rr), ImVec2(cx + rr * 0.8f, cy), c);
            } else {
                dl->AddTriangleFilled(ImVec2(cx - rr, cy - rr * 0.6f), ImVec2(cx + rr, cy - rr * 0.6f), ImVec2(cx, cy + rr * 0.8f), c);
            }
        }
    }
    dl->PopClipRect();

    ImGui::EndChild();
    if (show_minimap_) {
        ImGui::SameLine(0.0f, 0.0f);
        RenderMinimap(pal, last_view_h_);
    }
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(5);
    return doc_.Version() != version_before;
}

}  // namespace cyxwiz
