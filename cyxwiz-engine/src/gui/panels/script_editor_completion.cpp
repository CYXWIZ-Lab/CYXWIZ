// Script Editor auto-completion: when to ask, and what comes back (TOFIX133
// P0 item 6, P3). The list itself is drawn in script_editor_language.cpp.

#include "script_editor.h"
#include "../../core/language_results.h"
#include "../../scripting/scripting_engine.h"
#include "../../scripting/script_manager.h"

#include <algorithm>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {
std::string KindName(scripting::CompletionItem::Kind kind) {
    using K = scripting::CompletionItem::Kind;
    switch (kind) {
        case K::Keyword: return "keyword";
        case K::Builtin:
        case K::Function:
        case K::Method: return "function";
        case K::Module: return "module";
        case K::Class: return "class";
        case K::Property: return "property";
        case K::Variable: return "variable";
        case K::Snippet: break;
    }
    return "text";
}
}  // namespace

void ScriptEditorPanel::UpdateAutoCompletion(bool force) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        CloseCompletionPopup();
        return;
    }
    auto& tab = tabs_[active_tab_index_];
    CodeEditor* code = ActiveCodeEditor();  // the text, or the notebook cell being edited
    if (tab->is_loading || !code) {
        CloseCompletionPopup();
        return;
    }

    // The document's columns are byte offsets into the line.
    const editor::Document& doc = code->Doc();
    const editor::Pos cursor_pos = doc.Primary().head;
    const std::string current_line = doc.Line(cursor_pos.line);
    const int col = cursor_pos.col;

    // Check if we should show completions (allow empty line/col=0 for force mode)
    if (!force && (col <= 0 || current_line.empty())) {
        CloseCompletionPopup();
        return;
    }
    const char last_char = (col > 0 && col <= static_cast<int>(current_line.length())) ? current_line[col - 1] : '\0';
    const std::string word = scripting::ScriptManager::GetWordAtCursor(current_line, col);
    if (!force && !script_manager_.ShouldTriggerCompletion(last_char)) {
        // Only close if we're not in an identifier
        if (word.empty()) {
            if (show_completion_popup_) CloseCompletionPopup();
            return;
        }
    }

    completion_prefix_ = word;
    completion_file_stem_ = tab->filepath.empty() ? std::string() : std::filesystem::path(tab->filepath).stem().string();
    completion_start_pos_ = {cursor_pos.line, col - static_cast<int>(word.length())};
    completion_request_pos_ = cursor_pos;
    completion_request_version_ = doc.Version();

    // Jedi (bundled, TOFIX133 P3) answers on its worker; the list opens when
    // its result arrives (PollLanguageResults). Without the tools, the old
    // keyword completer.
    if (LanguageReady()) {
        completion_request_ =
            scripting_engine_->Language().Submit(LanguageRequest(scripting::LanguageService::Kind::Complete, *code, cursor_pos));
        if (completion_request_ != 0) return;
    }
    completion_entries_.clear();
    for (const auto& item : script_manager_.GetCompletions(current_line, col)) {
        lang::Completion c;
        c.name = item.label;
        c.complete = item.label.size() >= word.size() ? item.label.substr(word.size()) : item.label;
        c.kind = KindName(item.kind);
        c.detail = item.detail == "keyword" || item.detail == "builtin" ? std::string() : item.detail;
        completion_entries_.push_back(std::move(c));
    }
    OpenCompletionList(true);
}

void ScriptEditorPanel::OpenCompletionList(bool fallback) {
    if (completion_entries_.empty()) {
        CloseCompletionPopup();
        return;
    }
    completion_entries_from_fallback_ = fallback;
    show_completion_popup_ = true;
    completion_just_opened_ = true;  // Prevent immediate close from Ctrl+Space inserting space
    selected_completion_ = 0;
    completion_scroll_to_selected_ = true;
    completion_details_for_.clear();
    completion_details_ = {};
}

void ScriptEditorPanel::PollLanguageResults() {
    if (!scripting_engine_) return;
    for (auto& result : scripting_engine_->Language().Poll()) {
        using Kind = scripting::LanguageService::Kind;
        if (result.kind == Kind::Describe) {
            if (result.id == completion_details_request_) completion_details_ = lang::ParseDescription(result.json);
            continue;
        }
        if (HandleLanguageResult(result)) continue;  // problems, signatures, hover, definitions
        if (result.kind != Kind::Complete || result.id != completion_request_) continue;
        completion_request_ = 0;
        // Stale when the text or the cursor moved since it was asked.
        CodeEditor* code = ActiveCodeEditor();
        if (!code) continue;
        const editor::Document& doc = code->Doc();
        if (doc.Version() != completion_request_version_ || !(doc.Primary().head == completion_request_pos_)) continue;
        completion_entries_ = lang::ParseCompletions(result.json);
        OpenCompletionList(false);
    }
}

void ScriptEditorPanel::CloseCompletionPopup() {
    show_completion_popup_ = false;
    completion_entries_.clear();
    selected_completion_ = 0;
    completion_request_ = 0;
}
} // namespace cyxwiz
