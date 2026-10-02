// Script Editor auto-completion popup and insertion handling.

#include "script_editor.h"
#include "../../core/language_results.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"
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
    const editor::Document& doc = tab->editor.Doc();
    const editor::Pos cursor_pos = doc.Primary().head;
    std::string current_line = doc.Line(cursor_pos.line);
    int col = cursor_pos.col;  // the document's columns are byte offsets already

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

    // Jedi (bundled, TOFIX133 P3) answers on its worker; the list opens when
    // its result arrives (PollLanguageResults). Without the tools, the old
    // keyword completer.
    if (RequestLanguageCompletion(*tab, cursor_pos, scripting::ScriptManager::GetWordAtCursor(current_line, col))) {
        return;
    }
    completion_items_ = script_manager_.GetCompletions(current_line, col);

    if (completion_items_.empty()) {
        CloseCompletionPopup();
        return;
    }

    // Get prefix and start position
    completion_prefix_ = scripting::ScriptManager::GetWordAtCursor(current_line, col);
    completion_start_pos_ = {cursor_pos.line, col - static_cast<int>(completion_prefix_.length())};

    show_completion_popup_ = true;
    completion_just_opened_ = true;  // Prevent immediate close from Ctrl+Space inserting space
    selected_completion_ = 0;
}

namespace {
// Jedi columns count characters; the editor's are UTF-8 byte offsets.
int CharacterColumn(const std::string& line, int byte_col) {
    int chars = 0;
    for (int i = 0; i < byte_col && i < static_cast<int>(line.size()); ++i)
        if ((static_cast<unsigned char>(line[static_cast<size_t>(i)]) & 0xC0) != 0x80) ++chars;
    return chars;
}

scripting::CompletionItem::Kind ItemKind(const std::string& kind) {
    using K = scripting::CompletionItem::Kind;
    if (kind == "function") return K::Function;
    if (kind == "class") return K::Class;
    if (kind == "module") return K::Module;
    if (kind == "keyword") return K::Keyword;
    if (kind == "property") return K::Property;
    return K::Variable;
}
}  // namespace

bool ScriptEditorPanel::RequestLanguageCompletion(EditorTab& tab, const editor::Pos& cursor, const std::string& prefix) {
    if (!scripting_engine_) return false;
    if (!scripting_engine_->IsInitialized()) {
        // Python starts on the UI thread, as a first run starts it.
        std::string why;
        if (!scripting_engine_->StartPython(&why)) return false;
    }
    const std::string tools_error = scripting_engine_->LanguageToolsError();
    if (!tools_error.empty()) return false;
    const editor::Document& doc = tab.editor.Doc();
    scripting::LanguageService::Request request;
    request.kind = scripting::LanguageService::Kind::Complete;
    request.source = doc.Text();
    request.line = cursor.line + 1;
    request.column = CharacterColumn(doc.Line(cursor.line), cursor.col);
    request.path = tab.filepath;
    if (ProjectManager::Instance().HasActiveProject()) request.project_root = ProjectManager::Instance().GetProjectRoot();
    completion_request_ = scripting_engine_->Language().Submit(std::move(request));
    completion_request_pos_ = cursor;
    completion_request_version_ = doc.Version();
    completion_prefix_ = prefix;
    completion_start_pos_ = {cursor.line, cursor.col - static_cast<int>(prefix.length())};
    return completion_request_ != 0;
}

void ScriptEditorPanel::PollLanguageResults() {
    if (!scripting_engine_) return;
    for (auto& result : scripting_engine_->Language().Poll()) {
        if (result.kind != scripting::LanguageService::Kind::Complete || result.id != completion_request_) continue;
        completion_request_ = 0;
        if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) continue;
        auto& tab = tabs_[active_tab_index_];
        // Stale when the text or the cursor moved since it was asked.
        const editor::Document& doc = tab->editor.Doc();
        if (doc.Version() != completion_request_version_ || !(doc.Primary().head == completion_request_pos_)) continue;
        completion_items_.clear();
        for (const auto& c : lang::ParseCompletions(result.json)) {
            scripting::CompletionItem item(c.name, ItemKind(c.kind), c.detail);
            completion_items_.push_back(std::move(item));
        }
        if (completion_items_.empty()) {
            CloseCompletionPopup();
            continue;
        }
        show_completion_popup_ = true;
        completion_just_opened_ = true;
        selected_completion_ = 0;
    }
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
    const ImVec2 cursor_screen = tab->editor.CursorScreenPos();
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
    editor::Document& doc = tab->editor.Doc();
    const editor::Pos cursor_pos = doc.Primary().head;
    doc.SetSelections({editor::Selection{completion_start_pos_, cursor_pos, -1}});

    // Replace the typed prefix with the completion (one undo step).
    doc.Paste(text_to_insert);
}

void ScriptEditorPanel::CloseCompletionPopup() {
    show_completion_popup_ = false;
    completion_items_.clear();
    selected_completion_ = 0;
}
} // namespace cyxwiz
