// Script Editor settings and tab accessors.

#include "script_editor.h"
#include "../separate_windows.h"

#include <string>
#include <vector>

namespace cyxwiz {

// ========== Settings ==========

void ScriptEditorPanel::ConfigureEditor(CodeEditor& editor) const {
    editor.SetTabSize(tab_size_);
    editor.SetShowWhitespace(show_whitespace_);
    editor.SetColorize(syntax_highlighting_);
    editor.SetWordWrap(word_wrap_);
    editor.Doc().Settings().auto_indent = auto_indent_;
}

void ScriptEditorPanel::ApplyEditorSettings(const core::EditorConfig& settings) {
    SetTabSize(settings.tab_size);
    SetShowWhitespace(settings.show_whitespace);
    SetSyntaxHighlighting(settings.syntax_colours);
    SetShowMinimap(settings.minimap);
    SetWordWrap(settings.word_wrap);
    SetAutoIndent(settings.auto_indent);
    new_file_line_endings_ = settings.new_file_line_endings;
    trim_trailing_whitespace_ = settings.trim_trailing_whitespace;
    final_newline_ = settings.final_newline;
}

core::EditorConfig ScriptEditorPanel::EditorSettings() const {
    core::EditorConfig settings;
    settings.tab_size = tab_size_;
    settings.show_whitespace = show_whitespace_;
    settings.syntax_colours = syntax_highlighting_;
    settings.minimap = show_minimap_;
    settings.word_wrap = word_wrap_;
    settings.auto_indent = auto_indent_;
    settings.new_file_line_endings = new_file_line_endings_;
    settings.trim_trailing_whitespace = trim_trailing_whitespace_;
    settings.final_newline = final_newline_;
    return settings;
}

void ScriptEditorPanel::TidyBeforeSave(EditorTab& tab, const std::string& path) {
    if (tab.cell_mode || scriptfile::IsNotebookJson(path)) return;
    tab.editor.Doc().TidyForSave(trim_trailing_whitespace_, final_newline_);
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
    word_wrap_ = wrap;
    for (auto& tab : tabs_) tab->editor.SetWordWrap(wrap);
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
    ::gui::BringToFront(GetName());
}

} // namespace cyxwiz
