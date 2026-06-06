#include "script_editor.h"
#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

// ============================================================================
// Edit Operations
// ============================================================================

void ScriptEditorPanel::Undo() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (editor.CanUndo()) {
        editor.Undo();
        tabs_[active_tab_index_]->is_modified = true;
    }
}

void ScriptEditorPanel::Redo() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (editor.CanRedo()) {
        editor.Redo();
        tabs_[active_tab_index_]->is_modified = true;
    }
}

void ScriptEditorPanel::Cut() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    if (!editor.HasSelection()) {
        auto cursor = editor.GetCursorPosition();
        int line = cursor.mLine;
        int totalLines = editor.GetTotalLines();

        TextEditor::Coordinates start(line, 0);
        TextEditor::Coordinates end;

        if (line + 1 < totalLines) {
            end = TextEditor::Coordinates(line + 1, 0);
        } else {
            auto lines = editor.GetTextLines();
            int lineLen = (line < static_cast<int>(lines.size())) ? static_cast<int>(lines[line].size()) : 0;
            end = TextEditor::Coordinates(line, lineLen);
        }

        editor.SetSelection(start, end);
    }

    editor.Cut();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Copy() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.Copy();
}

void ScriptEditorPanel::Paste() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.Paste();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Delete() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    if (!editor.HasSelection()) {
        auto cursor = editor.GetCursorPosition();
        int line = cursor.mLine;
        int totalLines = editor.GetTotalLines();

        TextEditor::Coordinates start(line, 0);
        TextEditor::Coordinates end;

        if (line + 1 < totalLines) {
            end = TextEditor::Coordinates(line + 1, 0);
        } else {
            auto lines = editor.GetTextLines();
            int lineLen = (line < static_cast<int>(lines.size())) ? static_cast<int>(lines[line].size()) : 0;
            end = TextEditor::Coordinates(line, lineLen);
        }

        editor.SetSelection(start, end);
    }

    editor.Delete();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::SelectAll() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.SelectAll();
}

// ============================================================================
// Navigation
// ============================================================================

void ScriptEditorPanel::GoToLine(int line_number) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    int total_lines = editor.GetTotalLines();

    if (line_number < 1) line_number = 1;
    if (line_number > total_lines) line_number = total_lines;

    TextEditor::Coordinates pos(line_number - 1, 0);
    editor.SetCursorPosition(pos);

    auto lines = editor.GetTextLines();
    int line_idx = line_number - 1;
    int line_len = (line_idx < static_cast<int>(lines.size())) ? static_cast<int>(lines[line_idx].size()) : 0;
    editor.SetSelection(pos, TextEditor::Coordinates(line_idx, line_len));
}

// ============================================================================
// Line Operations
// ============================================================================

void ScriptEditorPanel::DuplicateLine() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    auto lines = editor.GetTextLines();
    if (line < 0 || line >= static_cast<int>(lines.size())) {
        return;
    }

    std::string line_text = lines[line];
    int line_len = static_cast<int>(line_text.size());
    editor.SetCursorPosition(TextEditor::Coordinates(line, line_len));
    editor.InsertText("\n" + line_text);
    editor.SetCursorPosition(TextEditor::Coordinates(line + 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::MoveLineUp() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    if (line <= 0) {
        return;
    }

    auto lines = editor.GetTextLines();
    if (line >= static_cast<int>(lines.size())) {
        return;
    }

    std::string current_line = lines[line];
    std::string above_line = lines[line - 1];

    editor.SetSelection(
        TextEditor::Coordinates(line - 1, 0),
        TextEditor::Coordinates(line, static_cast<int>(current_line.size()))
    );

    editor.Delete();
    editor.InsertText(current_line + "\n" + above_line);
    editor.SetCursorPosition(TextEditor::Coordinates(line - 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::MoveLineDown() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    auto lines = editor.GetTextLines();
    if (line < 0 || line >= static_cast<int>(lines.size()) - 1) {
        return;
    }

    std::string current_line = lines[line];
    std::string below_line = lines[line + 1];

    editor.SetSelection(
        TextEditor::Coordinates(line, 0),
        TextEditor::Coordinates(line + 1, static_cast<int>(below_line.size()))
    );

    editor.Delete();
    editor.InsertText(below_line + "\n" + current_line);
    editor.SetCursorPosition(TextEditor::Coordinates(line + 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Indent() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    if (editor.HasSelection()) {
        auto sel_start = editor.GetSelectionStart();
        auto sel_end = editor.GetSelectionEnd();

        auto lines = editor.GetTextLines();
        std::string indent_str(tab_size_, ' ');

        std::string new_text;
        for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
            new_text += indent_str + lines[i];
            if (i < sel_end.mLine) {
                new_text += "\n";
            }
        }

        int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
            static_cast<int>(lines[sel_end.mLine].size()) : 0;
        editor.SetSelection(
            TextEditor::Coordinates(sel_start.mLine, 0),
            TextEditor::Coordinates(sel_end.mLine, last_line_len)
        );
        editor.Delete();
        editor.InsertText(new_text);

        editor.SetSelection(
            TextEditor::Coordinates(sel_start.mLine, 0),
            TextEditor::Coordinates(sel_end.mLine, last_line_len + tab_size_)
        );
    } else {
        std::string indent_str(tab_size_, ' ');
        editor.InsertText(indent_str);
    }

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Outdent() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();

    auto lines = editor.GetTextLines();
    int start_line = cursor.mLine;
    int end_line = cursor.mLine;

    if (editor.HasSelection()) {
        start_line = editor.GetSelectionStart().mLine;
        end_line = editor.GetSelectionEnd().mLine;
    }

    std::string new_text;
    for (int i = start_line; i <= end_line && i < static_cast<int>(lines.size()); ++i) {
        std::string line = lines[i];
        int spaces_to_remove = 0;
        for (int j = 0; j < tab_size_ && j < static_cast<int>(line.size()); ++j) {
            if (line[j] == ' ') {
                spaces_to_remove++;
            } else {
                break;
            }
        }
        if (spaces_to_remove > 0) {
            line = line.substr(spaces_to_remove);
        }
        new_text += line;
        if (i < end_line) {
            new_text += "\n";
        }
    }

    int last_line_len = (end_line < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[end_line].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(start_line, 0),
        TextEditor::Coordinates(end_line, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

// ============================================================================
// Text Transformation
// ============================================================================

void ScriptEditorPanel::TransformToUppercase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string upper;
    upper.reserve(selected.size());
    for (char c : selected) {
        upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    editor.Delete();
    editor.InsertText(upper);
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::TransformToLowercase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string lower;
    lower.reserve(selected.size());
    for (char c : selected) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    editor.Delete();
    editor.InsertText(lower);
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::TransformToTitleCase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string title;
    title.reserve(selected.size());
    bool capitalize_next = true;

    for (char c : selected) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            capitalize_next = true;
            title += c;
        } else if (capitalize_next) {
            title += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalize_next = false;
        } else {
            title += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }

    editor.Delete();
    editor.InsertText(title);
    tabs_[active_tab_index_]->is_modified = true;
}

// ============================================================================
// Multi-line Operations
// ============================================================================

void ScriptEditorPanel::SortLinesAscending() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    std::vector<std::string> selected_lines;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        selected_lines.push_back(lines[i]);
    }

    std::sort(selected_lines.begin(), selected_lines.end());

    std::string new_text;
    for (size_t i = 0; i < selected_lines.size(); ++i) {
        new_text += selected_lines[i];
        if (i < selected_lines.size() - 1) {
            new_text += "\n";
        }
    }

    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::SortLinesDescending() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    std::vector<std::string> selected_lines;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        selected_lines.push_back(lines[i]);
    }

    std::sort(selected_lines.begin(), selected_lines.end(), std::greater<std::string>());

    std::string new_text;
    for (size_t i = 0; i < selected_lines.size(); ++i) {
        new_text += selected_lines[i];
        if (i < selected_lines.size() - 1) {
            new_text += "\n";
        }
    }

    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::JoinLines() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    std::string joined;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        std::string line = lines[i];
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if (!joined.empty() && !line.empty()) {
            joined += " ";
        }
        joined += line;
    }

    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(joined);

    tabs_[active_tab_index_]->is_modified = true;
}

// ==================== Comment Operations ====================

void ScriptEditorPanel::ToggleLineComment() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // Get current line
    auto cursor = editor.GetCursorPosition();
    auto lines = editor.GetTextLines();

    if (cursor.mLine >= static_cast<int>(lines.size())) {
        return;
    }

    std::string line = lines[cursor.mLine];

    // Find first non-whitespace character
    size_t first_char = line.find_first_not_of(" \t");

    if (first_char != std::string::npos && line.substr(first_char, 1) == "#") {
        // Remove comment
        std::string new_line = line.substr(0, first_char) + line.substr(first_char + 1);
        // Remove space after # if present
        if (first_char < new_line.length() && new_line[first_char] == ' ') {
            new_line = new_line.substr(0, first_char) + new_line.substr(first_char + 1);
        }

        // Replace line
        TextEditor::Coordinates start(cursor.mLine, 0);
        TextEditor::Coordinates end(cursor.mLine, static_cast<int>(line.length()));
        editor.SetSelection(start, end);
        editor.Delete();
        editor.InsertText(new_line);
    } else {
        // Add comment
        std::string new_line;
        if (first_char != std::string::npos) {
            new_line = line.substr(0, first_char) + "# " + line.substr(first_char);
        } else {
            new_line = "# " + line;
        }

        // Replace line
        TextEditor::Coordinates start(cursor.mLine, 0);
        TextEditor::Coordinates end(cursor.mLine, static_cast<int>(line.length()));
        editor.SetSelection(start, end);
        editor.Delete();
        editor.InsertText(new_line);
    }

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::ToggleBlockComment() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    if (!editor.HasSelection()) {
        spdlog::info("Select text to toggle block comment");
        return;
    }

    std::string selected = editor.GetSelectedText();

    // Check if already a block comment (Python uses triple quotes)
    if (selected.length() >= 6 &&
        selected.substr(0, 3) == "\"\"\"" &&
        selected.substr(selected.length() - 3) == "\"\"\"") {
        // Remove block comment
        selected = selected.substr(3, selected.length() - 6);
    } else {
        // Add block comment
        selected = "\"\"\"" + selected + "\"\"\"";
    }

    editor.Delete();
    editor.InsertText(selected);

    tabs_[active_tab_index_]->is_modified = true;
}

} // namespace cyxwiz
