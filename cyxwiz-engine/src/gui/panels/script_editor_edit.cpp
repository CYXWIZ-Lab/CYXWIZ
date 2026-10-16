// Script Editor edit, line and text operations (menus and Engine-wide
// shortcuts). Each is one undo step on the tab's editor::Document
// (TOFIX133 P1); the code view handles typing, moving and the clipboard keys.

#include "script_editor.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {
// Whole lines covered by the primary selection: [first, last].
void PrimaryLines(const editor::Document& doc, int& first, int& last) {
    const editor::Selection& s = doc.Primary();
    first = s.Start().line;
    last = s.End().line;
    if (last > first && s.End().col == 0) --last;
}

std::vector<std::string> LinesOf(const editor::Document& doc, int first, int last) {
    std::vector<std::string> out;
    for (int l = first; l <= last; ++l) out.push_back(doc.Line(l));
    return out;
}

void ReplaceLines(editor::Document& doc, int first, int last, const std::string& text) {
    doc.Replace({first, 0}, {last, static_cast<int>(doc.Line(last).size())}, text);
}

std::string Join(const std::vector<std::string>& lines, const char* sep) {
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) out += sep;
        out += lines[i];
    }
    return out;
}
}  // namespace

// Every operation below acts only on the text the user sees (TOFIX133 P0
// item 14): not in notebook mode, the large-file view or a failed load.
#define CYXWIZ_TEXT_DOC                                   \
    if (!IsActiveTabTextMode()) return;                   \
    editor::Document& doc = tabs_[active_tab_index_]->editor.Doc()

void ScriptEditorPanel::Undo() {
    CYXWIZ_TEXT_DOC;
    doc.Undo();
}

void ScriptEditorPanel::Redo() {
    CYXWIZ_TEXT_DOC;
    doc.Redo();
}

void ScriptEditorPanel::Cut() {
    CYXWIZ_TEXT_DOC;
    // No selection: the whole line, as other editors do.
    std::string text = doc.SelectedText();
    if (text.empty()) {
        for (const auto& s : doc.Selections()) text += doc.Line(s.head.line) + "\n";
        ImGui::SetClipboardText(text.c_str());
        doc.DeleteLines();
        return;
    }
    ImGui::SetClipboardText(text.c_str());
    doc.Backspace();
}

void ScriptEditorPanel::Copy() {
    CYXWIZ_TEXT_DOC;
    std::string text = doc.SelectedText();
    if (text.empty())
        for (const auto& s : doc.Selections()) text += doc.Line(s.head.line) + "\n";
    ImGui::SetClipboardText(text.c_str());
}

void ScriptEditorPanel::Paste() {
    CYXWIZ_TEXT_DOC;
    if (const char* clip = ImGui::GetClipboardText()) doc.Paste(clip);
}

void ScriptEditorPanel::Delete() {
    CYXWIZ_TEXT_DOC;
    if (doc.SelectedText().empty()) doc.DeleteLines();
    else doc.Backspace();
}

void ScriptEditorPanel::SelectAll() {
    CYXWIZ_TEXT_DOC;
    doc.SelectAll();
}

void ScriptEditorPanel::GoToLine(int line_number) {
    if (!IsActiveTabTextMode()) return;
    tabs_[active_tab_index_]->editor.GoToLine(line_number - 1);
}

void ScriptEditorPanel::DuplicateLine() {
    CYXWIZ_TEXT_DOC;
    doc.DuplicateLines();
}

void ScriptEditorPanel::MoveLineUp() {
    CYXWIZ_TEXT_DOC;
    doc.MoveLines(-1);
}

void ScriptEditorPanel::MoveLineDown() {
    CYXWIZ_TEXT_DOC;
    doc.MoveLines(1);
}

void ScriptEditorPanel::Indent() {
    CYXWIZ_TEXT_DOC;
    doc.IndentLines();
}

void ScriptEditorPanel::Outdent() {
    CYXWIZ_TEXT_DOC;
    doc.Outdent();
}

void ScriptEditorPanel::TransformToUppercase() {
    CYXWIZ_TEXT_DOC;
    doc.TransformSelections([](const std::string& s) {
        std::string out = s;
        for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return out;
    });
}

void ScriptEditorPanel::TransformToLowercase() {
    CYXWIZ_TEXT_DOC;
    doc.TransformSelections([](const std::string& s) {
        std::string out = s;
        for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return out;
    });
}

void ScriptEditorPanel::TransformToTitleCase() {
    CYXWIZ_TEXT_DOC;
    doc.TransformSelections([](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool next_upper = true;
        for (const char c : s) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (std::isspace(u)) {
                next_upper = true;
                out += c;
            } else {
                out += static_cast<char>(next_upper ? std::toupper(u) : std::tolower(u));
                next_upper = false;
            }
        }
        return out;
    });
}

void ScriptEditorPanel::SortLinesAscending() {
    CYXWIZ_TEXT_DOC;
    int first = 0;
    int last = 0;
    PrimaryLines(doc, first, last);
    if (last <= first) return;
    auto lines = LinesOf(doc, first, last);
    std::sort(lines.begin(), lines.end());
    ReplaceLines(doc, first, last, Join(lines, "\n"));
}

void ScriptEditorPanel::SortLinesDescending() {
    CYXWIZ_TEXT_DOC;
    int first = 0;
    int last = 0;
    PrimaryLines(doc, first, last);
    if (last <= first) return;
    auto lines = LinesOf(doc, first, last);
    std::sort(lines.begin(), lines.end(), std::greater<>());
    ReplaceLines(doc, first, last, Join(lines, "\n"));
}

void ScriptEditorPanel::JoinLines() {
    CYXWIZ_TEXT_DOC;
    int first = 0;
    int last = 0;
    PrimaryLines(doc, first, last);
    if (last == first) {
        if (last + 1 >= doc.LineCount()) return;
        ++last;  // no range: join with the next line
    }
    std::string joined;
    for (int l = first; l <= last; ++l) {
        std::string line = doc.Line(l);
        if (l > first) line.erase(0, line.find_first_not_of(" \t") == std::string::npos ? line.size() : line.find_first_not_of(" \t"));
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (!joined.empty() && !line.empty()) joined += ' ';
        joined += line;
    }
    ReplaceLines(doc, first, last, joined);
}

void ScriptEditorPanel::ToggleLineComment() {
    CYXWIZ_TEXT_DOC;
    doc.ToggleLineComment("#");
}

void ScriptEditorPanel::ToggleBlockComment() {
    CYXWIZ_TEXT_DOC;
    doc.TransformSelections([](const std::string& s) {
        if (s.size() >= 6 && s.compare(0, 3, "\"\"\"") == 0 && s.compare(s.size() - 3, 3, "\"\"\"") == 0)
            return s.substr(3, s.size() - 6);
        return "\"\"\"" + s + "\"\"\"";
    });
}

#undef CYXWIZ_TEXT_DOC

}  // namespace cyxwiz
