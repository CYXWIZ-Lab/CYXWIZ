// Script Editor find and replace operations. Matching lives in
// core/text_search (TOFIX133 P0 item 13); this file maps it to the editor.

#include "script_editor.h"

#include "../../core/text_search.h"

#include <string>

#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {
textsearch::Options MakeOptions(bool case_sensitive, bool whole_word, bool use_regex) {
    textsearch::Options o;
    o.case_sensitive = case_sensitive;
    o.whole_word = whole_word;
    o.regex = use_regex;
    return o;
}

TextEditor::Coordinates ToCoordinates(const std::string& text, size_t offset, int tab_size) {
    const auto p = textsearch::ToPosition(text, offset, tab_size);
    return TextEditor::Coordinates(p.line, p.column);
}

size_t ToOffset(const std::string& text, const TextEditor::Coordinates& c, int tab_size) {
    return textsearch::ToOffset(text, {c.mLine, c.mColumn}, tab_size);
}

void SelectMatch(TextEditor& editor, const std::string& text, const textsearch::Match& m) {
    const int tab = editor.GetTabSize();
    const auto start = ToCoordinates(text, m.pos, tab);
    const auto end = ToCoordinates(text, m.pos + m.len, tab);
    editor.SetSelection(start, end);
    editor.SetCursorPosition(end);
}
}  // namespace

// ==================== Find/Replace Operations ====================

bool ScriptEditorPanel::FindInEditor(const std::string& search_text, bool case_sensitive, bool whole_word, bool use_regex) {
    if (!IsActiveTabTextMode() || search_text.empty()) {
        return false;
    }
    last_search_text_ = search_text;
    last_case_sensitive_ = case_sensitive;
    last_whole_word_ = whole_word;
    last_use_regex_ = use_regex;

    auto& editor = tabs_[active_tab_index_]->editor;
    const std::string text = editor.GetText();
    const int tab = editor.GetTabSize();
    // From the start of the selection, so a fresh search finds the match
    // under the cursor; FindNext starts after it.
    const size_t from = ToOffset(text, editor.HasSelection() ? editor.GetSelectionStart() : editor.GetCursorPosition(), tab);
    std::string error;
    const auto m = textsearch::FindNext(text, search_text, from, MakeOptions(case_sensitive, whole_word, use_regex), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) {
        spdlog::info("'{}' not found", search_text);
        return false;
    }
    SelectMatch(editor, text, *m);
    return true;
}

bool ScriptEditorPanel::FindNext() {
    if (last_search_text_.empty() || !IsActiveTabTextMode()) {
        return false;
    }
    auto& editor = tabs_[active_tab_index_]->editor;
    const std::string text = editor.GetText();
    const size_t from = ToOffset(text, editor.GetCursorPosition(), editor.GetTabSize());
    std::string error;
    const auto m = textsearch::FindNext(text, last_search_text_, from,
                                        MakeOptions(last_case_sensitive_, last_whole_word_, last_use_regex_), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) return false;
    SelectMatch(editor, text, *m);
    return true;
}

bool ScriptEditorPanel::FindPreviousOf(const std::string& search_text, bool case_sensitive, bool whole_word,
                                       bool use_regex) {
    if (search_text.empty()) return false;
    last_search_text_ = search_text;
    last_case_sensitive_ = case_sensitive;
    last_whole_word_ = whole_word;
    last_use_regex_ = use_regex;
    return FindPrevious();
}

bool ScriptEditorPanel::FindPrevious() {
    if (last_search_text_.empty() || !IsActiveTabTextMode()) {
        return false;
    }
    auto& editor = tabs_[active_tab_index_]->editor;
    const std::string text = editor.GetText();
    const int tab = editor.GetTabSize();
    // Before the current match (the selection), else before the cursor.
    const size_t before = ToOffset(text, editor.HasSelection() ? editor.GetSelectionStart() : editor.GetCursorPosition(), tab);
    std::string error;
    const auto m = textsearch::FindPrevious(text, last_search_text_, before,
                                            MakeOptions(last_case_sensitive_, last_whole_word_, last_use_regex_), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) return false;
    SelectMatch(editor, text, *m);
    editor.SetCursorPosition(ToCoordinates(text, m->pos, tab));
    return true;
}

bool ScriptEditorPanel::Replace(const std::string& search_text, const std::string& replace_text,
                                 bool case_sensitive, bool whole_word, bool use_regex) {
    if (!IsActiveTabTextMode()) {
        return false;
    }
    auto& editor = tabs_[active_tab_index_]->editor;
    const auto options = MakeOptions(case_sensitive, whole_word, use_regex);

    // Replace the selection when it is a whole match (regex groups expanded),
    // then move to the next match; otherwise only move to the next match.
    if (editor.HasSelection()) {
        if (const auto replacement =
                textsearch::ReplacementFor(editor.GetSelectedText(), search_text, replace_text, options)) {
            editor.Delete();
            editor.InsertText(*replacement);
            tabs_[active_tab_index_]->is_modified = true;
        }
    }
    return FindInEditor(search_text, case_sensitive, whole_word, use_regex);
}

int ScriptEditorPanel::ReplaceAll(const std::string& search_text, const std::string& replace_text,
                                   bool case_sensitive, bool whole_word, bool use_regex) {
    if (!IsActiveTabTextMode() || search_text.empty()) {
        return 0;
    }
    auto& editor = tabs_[active_tab_index_]->editor;
    int count = 0;
    std::string error;
    const std::string result = textsearch::ReplaceAll(editor.GetText(), search_text, replace_text,
                                                      MakeOptions(case_sensitive, whole_word, use_regex), &count, &error);
    if (!error.empty()) {
        spdlog::warn("{}", error);
        return 0;
    }
    if (count > 0) {
        const auto cursor = editor.GetCursorPosition();
        editor.SetText(result);
        editor.SetCursorPosition(cursor);
        tabs_[active_tab_index_]->is_modified = true;
        spdlog::info("Replaced {} occurrences of '{}' with '{}'", count, search_text, replace_text);
    }
    return count;
}

} // namespace cyxwiz
