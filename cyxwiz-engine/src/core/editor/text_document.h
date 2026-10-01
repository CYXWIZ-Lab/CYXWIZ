// The Script Editor's own text model (TOFIX133 P1, decision D1): lines,
// several cursors, editing commands and grouped undo. No ImGui; the editor
// widget draws it and feeds it keys. Columns are byte offsets into a line
// (UTF-8); VisualColumn/ByteForVisual convert for drawing and Up/Down.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cyxwiz::editor {

struct Pos {
    int line = 0;
    int col = 0;  // byte offset in the line
    friend bool operator==(const Pos& a, const Pos& b) { return a.line == b.line && a.col == b.col; }
    friend bool operator!=(const Pos& a, const Pos& b) { return !(a == b); }
    friend bool operator<(const Pos& a, const Pos& b) { return a.line != b.line ? a.line < b.line : a.col < b.col; }
    friend bool operator<=(const Pos& a, const Pos& b) { return !(b < a); }
};

// Lines replaced by an edit: `removed` lines after `line` went away and
// `added` new ones took their place (both 0 for an edit inside one line).
// `reset` marks SetText (everything replaced; not a change by the user).
struct LineEdit {
    int line = 0;
    int removed = 0;
    int added = 0;
    bool reset = false;
};

struct Selection {
    Pos anchor;
    Pos head;      // where the caret is
    int goal = -1; // visual column kept by Up/Down; -1 = from head
    Pos Start() const { return anchor < head ? anchor : head; }
    Pos End() const { return anchor < head ? head : anchor; }
    bool Empty() const { return anchor == head; }
};

enum class Motion { Left, Right, Up, Down, WordLeft, WordRight, Home, End, DocStart, DocEnd, PageUp, PageDown };

struct Options {
    int tab_size = 4;
    bool auto_indent = true;  // keep indentation, indent after ':' and open brackets
    bool auto_close = true;   // brackets and quotes come in pairs
};

// Visual column of `byte` in `line` (tabs to the next stop, one per UTF-8
// character) and the byte offset for a visual column (clamped to the line).
int VisualColumn(std::string_view line, int byte, int tab_size);
int ByteForVisual(std::string_view line, int visual, int tab_size);

class Document {
public:
    Document();
    explicit Document(std::string_view text);

    // Replaces everything; clears undo; one cursor at the start; not modified.
    void SetText(std::string_view text);
    std::string Text() const;
    int LineCount() const { return static_cast<int>(lines_.size()); }
    const std::string& Line(int index) const { return lines_[static_cast<size_t>(index)]; }
    std::string GetRange(Pos a, Pos b) const;
    uint64_t Version() const { return version_; }  // changes on every edit, undo and redo
    Options& Settings() { return options_; }
    const Options& Settings() const { return options_; }
    Pos Clamp(Pos p) const;

    // ---- Cursors. Selections are kept sorted and never overlap; the last one
    // added is the primary (scrolled to, shown in the status bar).
    const std::vector<Selection>& Selections() const { return selections_; }
    const Selection& Primary() const { return selections_[static_cast<size_t>(primary_)]; }
    void SetSelections(std::vector<Selection> selections, int primary = -1);
    void SetCursor(Pos p, bool extend = false);  // one cursor (extend: keep its anchor)
    void AddCursor(Pos p);
    void AddCursorVertical(int direction);       // a cursor on the line above (-1) / below (+1)
    void SelectAll();
    void SelectWord(Pos p);
    void SelectLine(int line);
    void SelectAllOccurrences();                 // of the primary selection, or the word at it
    void Move(Motion motion, bool extend, int page_lines = 20);
    std::string SelectedText() const;            // one line per selection, joined with '\n'
    int CursorCount() const { return static_cast<int>(selections_.size()); }

    // ---- Editing; each applies at every cursor and is one undo step
    // (typing letters merges into one step until the cursor moves).
    void Type(std::string_view text);
    void Paste(std::string_view text);  // N lines onto N cursors: one line each
    void Newline();
    void Backspace();
    void DeleteForward();
    void DeleteWordLeft();
    void DeleteWordRight();
    void Tab();      // indent the lines of a multi-line selection, else spaces to the next stop
    void Outdent();
    void IndentLines();
    void ToggleLineComment(std::string_view prefix = "#");
    void DuplicateLines();
    void MoveLines(int direction);
    void DeleteLines();
    // One undo step that replaces [a, b) with `text`; cursors follow.
    void Replace(Pos a, Pos b, std::string_view text);
    // One undo step: each non-empty selection becomes transform(its text) and
    // stays selected.
    void TransformSelections(const std::function<std::string(const std::string&)>& transform);

    // ---- Undo.
    bool CanUndo() const { return !undo_.empty(); }
    bool CanRedo() const { return !redo_.empty(); }
    bool Undo();
    bool Redo();
    void BreakUndoGroup() { merge_open_ = false; }
    bool Modified() const;
    void MarkSaved();

    // Line edits since the last call (folds and per-line caches follow them).
    std::vector<LineEdit> TakeLineEdits() { return std::exchange(line_edits_, {}); }

private:
    struct Change {
        Pos start;
        std::string removed;
        std::string inserted;
    };
    enum class GroupKind { Typing, Other };
    struct Group {
        std::vector<Change> changes;
        std::vector<Selection> before;
        std::vector<Selection> after;
        int primary_before = 0;
        int primary_after = 0;
        GroupKind kind = GroupKind::Other;
        uint64_t id = 0;
    };
    struct Edit {
        Pos a;
        Pos b;
        std::string text;
    };
    enum class CaretRule { EndOfInsert, KeepMapped };

    Pos RawReplace(Pos a, Pos b, std::string_view text);
    // Applies non-overlapping edits as one undo step and maps the cursors.
    // `carets` (one per edit, optional) are offsets into each inserted text
    // where that edit's cursor lands.
    void ApplyEdits(std::vector<Edit> edits, GroupKind kind, CaretRule rule,
                    const std::vector<int>* caret_offsets = nullptr);
    void Normalize();
    Pos MovePos(Pos p, Motion motion, int& goal, int page_lines) const;
    Pos PrevChar(Pos p) const;
    Pos NextChar(Pos p) const;
    Pos WordLeft(Pos p) const;
    Pos WordRight(Pos p) const;
    std::vector<std::pair<int, int>> LineBlocks() const;  // merged line ranges of all selections
    std::string IndentUnit() const { return std::string(static_cast<size_t>(options_.tab_size), ' '); }

    std::vector<std::string> lines_;
    std::vector<Selection> selections_;
    int primary_ = 0;
    Options options_;
    std::vector<Group> undo_;
    std::vector<Group> redo_;
    bool merge_open_ = false;
    uint64_t version_ = 0;
    uint64_t next_group_id_ = 1;
    uint64_t saved_group_id_ = 0;  // id of the top undo group when saved (0 = none)
    std::vector<LineEdit> line_edits_;
};

}  // namespace cyxwiz::editor
