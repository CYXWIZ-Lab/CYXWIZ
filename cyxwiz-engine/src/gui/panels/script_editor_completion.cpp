// Script Editor auto-completion popup and insertion handling.

#include "script_editor.h"
#include "../../core/text_search.h"
#include "../../scripting/script_manager.h"

#include <algorithm>
#include <string>

#include <imgui.h>

namespace cyxwiz {
// ============================================================================
// Auto-Completion Implementation
// ============================================================================

void ScriptEditorPanel::UpdateAutoCompletion(bool force) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        CloseCompletionPopup();
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    if (tab->cell_mode || tab->is_loading) {
        CloseCompletionPopup();
        return;
    }

    // Get current cursor position and line. The editor's column is visual
    // (tabs expanded, one per UTF-8 character); the completer needs a byte
    // index into the line (TOFIX133 P0 item 6).
    auto cursor_pos = tab->editor.GetCursorPosition();
    std::string current_line = tab->editor.GetCurrentLineText();
    const int tab_size = tab->editor.GetTabSize();
    int col = static_cast<int>(textsearch::ToOffset(current_line, {0, cursor_pos.mColumn}, tab_size));

    // Check if we should show completions (allow empty line/col=0 for force mode)
    if (!force && (col <= 0 || current_line.empty())) {
        CloseCompletionPopup();
        return;
    }

    // Get the character just typed
    char last_char = (col > 0 && col <= static_cast<int>(current_line.length()))
                     ? current_line[col - 1] : '\0';

    // Check if we should trigger completion (skip check if forced via Ctrl+Space)
    if (!force && !script_manager_.ShouldTriggerCompletion(last_char)) {
        // Only close if we're not in an identifier
        std::string word = scripting::ScriptManager::GetWordAtCursor(current_line, col);
        if (word.empty() && !show_completion_popup_) {
            return;
        }
        if (word.empty() && show_completion_popup_) {
            CloseCompletionPopup();
            return;
        }
    }

    // Get completions
    completion_items_ = script_manager_.GetCompletions(current_line, col);

    if (completion_items_.empty()) {
        CloseCompletionPopup();
        return;
    }

    // Get prefix and start position
    completion_prefix_ = scripting::ScriptManager::GetWordAtCursor(current_line, col);
    completion_start_pos_ = cursor_pos;
    completion_start_pos_.mColumn =
        textsearch::ToPosition(current_line, static_cast<size_t>(col) - completion_prefix_.length(), tab_size).column;

    show_completion_popup_ = true;
    completion_just_opened_ = true;  // Prevent immediate close from Ctrl+Space inserting space
    selected_completion_ = 0;
}

void ScriptEditorPanel::RenderCompletionPopup() {
    if (!show_completion_popup_ || completion_items_.empty()) {
        return;
    }

    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // NO keyboard handling here - it interferes with the text editor!
    // Keyboard shortcuts are handled in HandleKeyboardShortcuts() instead

    // Under the cursor, from where the editor drew it this frame (the old
    // position assumed a 45 px gutter and an 80 px header).
    const ImVec2 cursor_screen = tab->editor.GetCursorScreenPos();
    const ImVec2 display_size = ImGui::GetIO().DisplaySize;
    const float popup_x = std::min(cursor_screen.x, display_size.x - 320.0f);
    const float popup_y = std::min(cursor_screen.y + 2.0f, display_size.y - 250.0f);
    ImGui::SetNextWindowPos(ImVec2(popup_x, popup_y), ImGuiCond_Always);

    // Popup flags - NO focus stealing!
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.4f, 0.4f, 0.5f, 0.8f));

    if (ImGui::Begin("##completion_popup", nullptr, flags)) {
        // Header with hint
        ImGui::TextDisabled("Tab: insert | Up/Down: choose | Esc: close");
        ImGui::Separator();

        // Render completion list
        for (int i = 0; i < static_cast<int>(completion_items_.size()) && i < 10; ++i) {
            const auto& item = completion_items_[i];
            bool is_selected = (i == selected_completion_);

            // Kind icon
            const char* icon = scripting::GetCompletionKindIcon(item.kind);

            ImGui::PushID(i);

            // Highlight selected item
            if (is_selected) {
                ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.8f, 0.7f));
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.5f, 0.8f, 0.8f));
            }

            if (ImGui::Selectable("##item", is_selected, ImGuiSelectableFlags_None, ImVec2(280, 0))) {
                ApplyCompletion(item);
                CloseCompletionPopup();
            }

            if (is_selected) {
                ImGui::PopStyleColor(2);
            }

            ImGui::SameLine(0, 0);
            ImGui::SetCursorPosX(8);

            // Icon with color based on kind
            ImVec4 kind_color;
            switch (item.kind) {
                case scripting::CompletionItem::Kind::Keyword:  kind_color = ImVec4(0.8f, 0.4f, 0.8f, 1.0f); break;
                case scripting::CompletionItem::Kind::Builtin:  kind_color = ImVec4(0.4f, 0.8f, 0.8f, 1.0f); break;
                case scripting::CompletionItem::Kind::Module:   kind_color = ImVec4(0.8f, 0.6f, 0.2f, 1.0f); break;
                case scripting::CompletionItem::Kind::Function: kind_color = ImVec4(0.4f, 0.7f, 1.0f, 1.0f); break;
                default: kind_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f); break;
            }
            ImGui::TextColored(kind_color, "[%s]", icon);
            ImGui::SameLine();

            // Label
            ImGui::Text("%s", item.label.c_str());

            // Detail (if any)
            if (!item.detail.empty() && item.detail != "keyword" && item.detail != "builtin") {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", item.detail.c_str());
            }

            ImGui::PopID();
        }

        // Show more items indicator
        if (completion_items_.size() > 10) {
            ImGui::Separator();
            ImGui::TextDisabled("... and %zu more", completion_items_.size() - 10);
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

void ScriptEditorPanel::ApplyCompletion(const scripting::CompletionItem& item) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Get the text to insert
    std::string text_to_insert = item.insert_text.empty() ? item.label : item.insert_text;

    // Select the prefix text (from completion_start_pos_ to current cursor)
    auto cursor_pos = tab->editor.GetCursorPosition();
    tab->editor.SetSelection(completion_start_pos_, cursor_pos);

    // Delete the selected prefix, then insert completion
    if (tab->editor.HasSelection()) {
        tab->editor.Delete();  // Deletes selected text
    }
    tab->editor.InsertText(text_to_insert);
}

void ScriptEditorPanel::CloseCompletionPopup() {
    show_completion_popup_ = false;
    completion_items_.clear();
    selected_completion_ = 0;
}
} // namespace cyxwiz
