// The Script Editor's text model (TOFIX133 P1, D1): editing commands, several
// cursors, auto-indent and pairs, line commands, grouped undo, saved state.
#include "../src/core/editor/text_document.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::editor;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Expect(const Document& d, const std::string& text, const std::string& what) {
    if (d.Text() != text) {
        std::cerr << "FAIL: " << what << "\n--- got ---\n" << d.Text() << "\n--- want ---\n" << text << '\n';
        std::exit(1);
    }
}

Pos Caret(const Document& d, int i = 0) { return d.Selections()[static_cast<size_t>(i)].head; }
}  // namespace

int main() {
    // Text in and out; CRLF and lone CR become lines.
    Document d("a\r\nb\rc");
    Expect(d, "a\nb\nc", "line endings split");
    Check(d.LineCount() == 3 && !d.Modified(), "three lines, not modified");

    // Typing merges into one undo step; a space starts the next.
    d.SetText("");
    for (const char* c : {"p", "r", "i", "n", "t"}) d.Type(c);
    d.Type(" ");
    d.Type("x");
    Expect(d, "print x", "typed text");
    Check(d.Modified(), "modified after typing");
    d.Undo();
    Expect(d, "print", "undo removes the last word step");
    d.Undo();
    Expect(d, "", "undo removes the first word in one step");
    Check(!d.Modified(), "back to the saved state");
    d.Redo();
    Expect(d, "print", "redo");

    // Auto-close, type-over, wrap, backspace of an empty pair.
    d.SetText("");
    d.Type("f");
    d.Type("(");
    Expect(d, "f()", "( closes");
    Check(Caret(d).col == 2, "caret between the pair");
    d.Type(")");
    Expect(d, "f()", ") steps over the closer");
    Check(Caret(d).col == 3, "caret after )");
    d.SetText("x = ");
    d.SetCursor({0, 4});
    d.Type("\"");
    Expect(d, "x = \"\"", "quote pairs after a space");
    d.Backspace();
    Expect(d, "x = ", "backspace removes the empty pair");
    d.SetText("it");
    d.SetCursor({0, 2});
    d.Type("'");
    Expect(d, "it'", "no pair after a word (it's)");
    d.SetText("value");
    d.SelectAll();
    d.Type("[");
    Expect(d, "[value]", "a bracket wraps the selection");

    // Newline: keep indentation, indent after ':' and between brackets.
    d.SetText("def f():");
    d.SetCursor({0, 8});
    d.Newline();
    Expect(d, "def f():\n    ", "indent after ':'");
    d.Type("return 1");
    d.Newline();
    Expect(d, "def f():\n    return 1\n    ", "keeps indentation");
    d.SetText("call()");
    d.SetCursor({0, 5});
    d.Newline();
    Expect(d, "call(\n    \n)", "closer moves to its own line");
    Check(Caret(d).line == 1 && Caret(d).col == 4, "caret on the inner line");

    // Backspace in indentation goes back one tab stop.
    d.SetText("        x");
    d.SetCursor({0, 8});
    d.Backspace();
    Expect(d, "    x", "backspace removes four spaces");
    d.SetText("ab");
    d.SetCursor({0, 0});
    d.Backspace();
    Expect(d, "ab", "backspace at the start does nothing");

    // Several cursors type at once.
    d.SetText("one\ntwo\nthree");
    d.SetCursor({0, 0});
    d.AddCursor({1, 0});
    d.AddCursor({2, 0});
    d.Type("# ");
    Expect(d, "# one\n# two\n# three", "three cursors type");
    Check(d.CursorCount() == 3 && Caret(d, 2).col == 2, "cursors after the insert");
    d.Undo();
    Expect(d, "one\ntwo\nthree", "one undo for all cursors");
    Check(d.CursorCount() == 3, "undo restores the cursors");

    // Paste: one line per cursor when the counts match.
    d.SetText("a\nb");
    d.SetCursor({0, 1});
    d.AddCursor({1, 1});
    d.Paste("1\r\n2\r\n");
    Expect(d, "a1\nb2", "lines distributed over cursors");
    d.SetCursor({1, 2});
    d.Paste("x\ny");
    Expect(d, "a1\nb2x\ny", "multi-line paste at one cursor");

    // Up/Down keep the visual column through tabs and short lines.
    d.SetText("\tabc\nx\n    abcd");
    d.SetCursor({0, 2});  // after the tab and 'a': visual column 5
    d.Move(Motion::Down, false);
    Check(Caret(d).line == 1 && Caret(d).col == 1, "short line clamps");
    d.Move(Motion::Down, false);
    Check(Caret(d).line == 2 && Caret(d).col == 5, "goal column comes back");
    d.SetCursor({2, 6});
    d.Move(Motion::Home, false);
    Check(Caret(d).col == 4, "Home: first non-blank");
    d.Move(Motion::Home, false);
    Check(Caret(d).col == 0, "Home again: column 0");

    // Words and UTF-8.
    d.SetText("x = foo_bar(caf\xC3\xA9)");
    d.SetCursor({0, 0});
    d.Move(Motion::WordRight, false);
    d.Move(Motion::WordRight, false);
    d.Move(Motion::WordRight, false);
    Check(Caret(d).col == 11, "word right stops after foo_bar");
    d.Move(Motion::End, false);
    d.Move(Motion::Left, false);
    d.Move(Motion::Left, false);
    Check(Caret(d).col == 15, "left steps over a two-byte character");
    Check(VisualColumn(d.Line(0), 17, 4) == 16, "visual column counts characters");
    Check(ByteForVisual("\tx", 4, 4) == 1 && ByteForVisual("\tx", 2, 4) == 0, "byte for visual column with a tab");

    // Selecting words and occurrences.
    d.SetText("text = text.strip()\nprint(text)");
    d.SelectWord({0, 2});
    Check(d.SelectedText() == "text", "double-click selects the word");
    d.SelectAllOccurrences();
    Check(d.CursorCount() == 3, "three occurrences");
    d.Type("s");
    Expect(d, "s = s.strip()\nprint(s)", "rename at all occurrences");

    // Tab and Shift+Tab on lines; Tab to the next stop.
    d.SetText("a\nb\n\nc");
    d.SetSelections({Selection{{0, 0}, {3, 1}, -1}});
    d.Tab();
    Expect(d, "    a\n    b\n\n    c", "indent lines (blank lines stay empty)");
    d.Outdent();
    Expect(d, "a\nb\n\nc", "outdent lines");
    d.SetText("ab");
    d.SetCursor({0, 1});
    d.Tab();
    Expect(d, "a   b", "tab to the next stop");

    // Comments.
    d.SetText("    x = 1\n    y = 2");
    d.SetSelections({Selection{{0, 0}, {1, 3}, -1}});
    d.ToggleLineComment();
    Expect(d, "    # x = 1\n    # y = 2", "comment at the common indentation");
    d.ToggleLineComment();
    Expect(d, "    x = 1\n    y = 2", "uncomment");

    // Line commands.
    d.SetText("a\nb\nc");
    d.SetCursor({1, 1});
    d.DuplicateLines();
    Expect(d, "a\nb\nb\nc", "duplicate line");
    Check(Caret(d).line == 2, "cursor on the copy below");
    d.MoveLines(-1);
    d.MoveLines(-1);
    Expect(d, "b\na\nb\nc", "move line up twice");
    Check(Caret(d).line == 0 && Caret(d).col == 1, "cursor moved with the line");
    d.MoveLines(-1);
    Expect(d, "b\na\nb\nc", "top line does not move further up");
    d.DeleteLines();
    Expect(d, "a\nb\nc", "delete line");
    d.SetCursor({2, 0});
    d.DeleteLines();
    Expect(d, "a\nb", "delete the last line");

    // Replace keeps an undo step and moves later cursors.
    d.SetText("alpha beta");
    d.SetCursor({0, 10});
    d.Replace({0, 0}, {0, 5}, "a");
    Expect(d, "a beta", "replace");
    Check(Caret(d).col == 6, "cursor after the replacement moved");
    d.Undo();
    Expect(d, "alpha beta", "replace undone");

    // Transform selections (upper case) as one step, text stays selected.
    d.SetText("ab cd ab");
    d.SetSelections({Selection{{0, 0}, {0, 2}, -1}, Selection{{0, 6}, {0, 8}, -1}});
    d.TransformSelections([](const std::string& s) { std::string u = s; for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return u; });
    Expect(d, "AB cd AB", "upper case at two selections");
    Check(d.SelectedText() == "AB\nAB", "new text selected");
    d.Undo();
    Expect(d, "ab cd ab", "one undo");

    // Saved state follows undo.
    d.SetText("v");
    d.SetCursor({0, 1});
    d.Type("1");
    d.MarkSaved();
    Check(!d.Modified(), "saved");
    d.Type("2");
    Check(d.Modified(), "edited after save");
    d.Undo();
    Check(!d.Modified(), "undo back to the saved text is not modified");

    // Cursor on line 0 adds a cursor below at the same column.
    d.SetText("abc\nabc");
    d.SetCursor({0, 2});
    d.AddCursorVertical(1);
    Check(d.CursorCount() == 2 && Caret(d, 1).line == 1 && Caret(d, 1).col == 2, "cursor added below");

    std::cout << "text document: typing, pairs, indent, cursors, paste, motion, lines, comments, undo. OK\n";
    return 0;
}
