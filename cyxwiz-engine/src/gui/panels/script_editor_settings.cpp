// Script Editor settings and tab accessors.

#include "script_editor.h"

#include <string>
#include <vector>

namespace cyxwiz {

// ========== Settings ==========

void ScriptEditorPanel::SetTabSize(int size) {
    if (size >= 1 && size <= 8) {
        tab_size_ = size;
        ApplyTabSizeToAllTabs();
    }
}

void ScriptEditorPanel::SetShowWhitespace(bool show) {
    show_whitespace_ = show;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetShowWhitespaces(show);
    }
}

void ScriptEditorPanel::SetWordWrap(bool wrap) {
    word_wrap_ = wrap;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetWordWrap(wrap);
    }
}

void ScriptEditorPanel::SetAutoIndent(bool indent) {
    auto_indent_ = indent;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetAutoIndent(indent);
    }
}

void ScriptEditorPanel::SetSyntaxHighlighting(bool enabled) {
    syntax_highlighting_ = enabled;
    ApplySyntaxHighlightingToAllTabs();
}

void ScriptEditorPanel::SetTheme(int theme_index) {
    if (theme_index >= 0 && theme_index <= 6) {
        current_theme_ = static_cast<EditorTheme>(theme_index);
        ApplyThemeToAllTabs();
    }
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
