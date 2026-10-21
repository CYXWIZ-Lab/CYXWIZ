// Script Editor find and replace operations. Matching lives in
// core/text_search (TOFIX133 P0 item 13); this file maps it to the document.

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

// The document's columns are byte offsets, so a text offset maps directly.
editor::Pos ToPos(const std::string& text, size_t offset) {
    editor::Pos p;
    size_t line_start = 0;
    for (size_t i = 0; i < offset && i < text.size(); ++i) {
        if (text[i] == '\n') {
            ++p.line;
            line_start = i + 1;
        }
    }
    p.col = static_cast<int>(std::min(offset, text.size()) - line_start);
    return p;
}

size_t ToOffset(const editor::Document& doc, editor::Pos p) {
    size_t offset = 0;
    for (int l = 0; l < p.line && l < doc.LineCount(); ++l) offset += doc.Line(l).size() + 1;
    return offset + static_cast<size_t>(p.col);
}

void SelectMatch(CodeEditor& code, const std::string& text, const textsearch::Match& m, bool caret_at_start) {
    const editor::Pos a = ToPos(text, m.pos);
    const editor::Pos b = ToPos(text, m.pos + m.len);
    code.Doc().SetSelections({editor::Selection{caret_at_start ? b : a, caret_at_start ? a : b, -1}});
    code.Folds().Reveal(a.line);
    code.ScrollToCursor();
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

    CodeEditor& code = tabs_[active_tab_index_]->editor;
    const std::string text = code.Doc().Text();
    // From the start of the selection, so a fresh search finds the match
    // under the cursor; FindNext starts after it.
    const size_t from = ToOffset(code.Doc(), code.Doc().Primary().Start());
    std::string error;
    const auto m = textsearch::FindNext(text, search_text, from, MakeOptions(case_sensitive, whole_word, use_regex), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) {
        spdlog::info("'{}' not found", search_text);
        return false;
    }
    SelectMatch(code, text, *m, false);
    return true;
}

bool ScriptEditorPanel::FindNext() {
    if (last_search_text_.empty() || !IsActiveTabTextMode()) {
        return false;
    }
    CodeEditor& code = tabs_[active_tab_index_]->editor;
    const std::string text = code.Doc().Text();
    const size_t from = ToOffset(code.Doc(), code.Doc().Primary().End());
    std::string error;
    const auto m = textsearch::FindNext(text, last_search_text_, from,
                                        MakeOptions(last_case_sensitive_, last_whole_word_, last_use_regex_), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) return false;
    SelectMatch(code, text, *m, false);
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
    CodeEditor& code = tabs_[active_tab_index_]->editor;
    const std::string text = code.Doc().Text();
    // Before the current match (the selection), else before the cursor.
    const size_t before = ToOffset(code.Doc(), code.Doc().Primary().Start());
    std::string error;
    const auto m = textsearch::FindPrevious(text, last_search_text_, before,
                                            MakeOptions(last_case_sensitive_, last_whole_word_, last_use_regex_), &error);
    if (!error.empty()) spdlog::warn("{}", error);
    if (!m) return false;
    SelectMatch(code, text, *m, true);
    return true;
}

bool ScriptEditorPanel::Replace(const std::string& search_text, const std::string& replace_text,
                                 bool case_sensitive, bool whole_word, bool use_regex) {
    if (!IsActiveTabTextMode()) {
        return false;
    }
    editor::Document& doc = tabs_[active_tab_index_]->editor.Doc();
    const auto options = MakeOptions(case_sensitive, whole_word, use_regex);

    // Replace the selection when it is a whole match (regex groups expanded),
    // then move to the next match; otherwise only move to the next match.
    const std::string selected = doc.SelectedText();
    if (!selected.empty()) {
        if (const auto replacement = textsearch::ReplacementFor(selected, search_text, replace_text, options)) {
            doc.Paste(*replacement);
        }
    }
    return FindInEditor(search_text, case_sensitive, whole_word, use_regex);
}

int ScriptEditorPanel::ReplaceAll(const std::string& search_text, const std::string& replace_text,
                                   bool case_sensitive, bool whole_word, bool use_regex) {
    if (!IsActiveTabTextMode() || search_text.empty()) {
        return 0;
    }
    editor::Document& doc = tabs_[active_tab_index_]->editor.Doc();
    int count = 0;
    std::string error;
    const std::string text = doc.Text();
    const std::string result = textsearch::ReplaceAll(text, search_text, replace_text,
                                                      MakeOptions(case_sensitive, whole_word, use_regex), &count, &error);
    if (!error.empty()) {
        spdlog::warn("{}", error);
        return 0;
    }
    if (count > 0) {
        // One undoable step (it used to replace the whole text and lose undo).
        const editor::Pos cursor = doc.Primary().head;
        doc.Replace({0, 0}, {doc.LineCount() - 1, static_cast<int>(doc.Line(doc.LineCount() - 1).size())}, result);
        doc.SetCursor(doc.Clamp(cursor));
        spdlog::info("Replaced {} occurrences of '{}' with '{}'", count, search_text, replace_text);
    }
    return count;
}

} // namespace cyxwiz
