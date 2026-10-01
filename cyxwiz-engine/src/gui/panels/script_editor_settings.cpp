// Script Editor settings and tab accessors.

#include "script_editor.h"

#include <string>
#include <vector>

namespace cyxwiz {

// ========== Settings ==========

void ScriptEditorPanel::ConfigureEditor(CodeEditor& editor) const {
    editor.SetTabSize(tab_size_);
    editor.SetShowWhitespace(show_whitespace_);
    editor.SetColorize(syntax_highlighting_);
    editor.Doc().Settings().auto_indent = auto_indent_;
}

void ScriptEditorPanel::SetTabSize(int size) {
    if (size >= 1 && size <= 8) {
        tab_size_ = size;
        ApplyTabSizeToAllTabs();
    }
}

void ScriptEditorPanel::SetShowWhitespace(bool show) {
    show_whitespace_ = show;
    for (auto& tab : tabs_) tab->editor.SetShowWhitespace(show);
}

void ScriptEditorPanel::SetWordWrap(bool wrap) {
    // Kept for Preferences; the code view wraps from TOFIX133 P2.
    word_wrap_ = wrap;
}

void ScriptEditorPanel::SetAutoIndent(bool indent) {
    auto_indent_ = indent;
    for (auto& tab : tabs_) tab->editor.Doc().Settings().auto_indent = indent;
}

void ScriptEditorPanel::SetSyntaxHighlighting(bool enabled) {
    syntax_highlighting_ = enabled;
    ApplySyntaxHighlightingToAllTabs();
}

std::vector<std::string> ScriptEditorPanel::GetOpenFilePaths() const {
    std::vector<std::string> result;
    result.reserve(tabs_.size());
    for (const auto& tab : tabs_) {
        if (tab && !tab->filepath.empty()) {
            result.push_back(tab->filepath);
        }
    }
    return result;
}

void ScriptEditorPanel::SetActiveTabIndex(int index) {
    if (index < 0 || index >= static_cast<int>(tabs_.size())) {
        return;
    }
    active_tab_index_ = index;
    request_window_focus_ = true;
}

} // namespace cyxwiz
