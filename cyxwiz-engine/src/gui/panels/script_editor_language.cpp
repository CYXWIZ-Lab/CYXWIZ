// Script Editor language intelligence UI (TOFIX133 P3, approved boards 6-8):
// the completion list with the selected item's details. Results come from
// the bundled Jedi on the language worker (scripting/language_service).

#include "script_editor.h"

#include "../../core/language_results.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <string>

namespace cyxwiz {

namespace {
// Jedi columns count characters; the editor's are UTF-8 byte offsets.
int CharacterColumn(const std::string& line, int byte_col) {
    int chars = 0;
    for (int i = 0; i < byte_col && i < static_cast<int>(line.size()); ++i)
        if ((static_cast<unsigned char>(line[static_cast<size_t>(i)]) & 0xC0) != 0x80) ++chars;
    return chars;
}

ImVec4 KindColour(int role) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (role) {
        case 0: return t.info;          // function
        case 1: return t.warning;       // class
        case 2: return t.accent_text;   // module
        case 3: return t.text;          // variable
        case 4: return t.accent_text;   // keyword
        default: return t.text_dim;
    }
}
}  // namespace

CodeEditor* ScriptEditorPanel::ActiveCodeEditor() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) return nullptr;
    auto& tab = *tabs_[active_tab_index_];
    if (!tab.cell_mode) return &tab.editor;
    if (tab.editing_cell < 0 || tab.editing_cell >= tab.cell_manager.GetCellCount()) return nullptr;
    Cell& cell = tab.cell_manager.GetCell(tab.editing_cell);
    return cell.type == CellType::Code ? &cell.editor : nullptr;
}

scripting::LanguageService::Request ScriptEditorPanel::LanguageRequest(scripting::LanguageService::Kind kind,
                                                                       const CodeEditor& code, const editor::Pos& pos) {
    auto& tab = *tabs_[active_tab_index_];
    scripting::LanguageService::Request request;
    request.kind = kind;
    request.source = code.Doc().Text();
    request.line = pos.line + 1;
    request.column = CharacterColumn(code.Doc().Line(pos.line), pos.col);
    request.path = tab.filepath;
    if (ProjectManager::Instance().HasActiveProject()) request.project_root = ProjectManager::Instance().GetProjectRoot();
    // A notebook cell completes against the notebook's live variables.
    if (tab.cell_mode) request.namespace_key = tab.cell_manager.NamespaceKey();
    return request;
}

bool ScriptEditorPanel::LanguageReady() {
    if (!scripting_engine_) return false;
    if (!scripting_engine_->IsInitialized()) {
        // Python starts on the UI thread, as a first run starts it.
        std::string why;
        if (!scripting_engine_->StartPython(&why)) return false;
    }
    return scripting_engine_->LanguageToolsError().empty();
}

void ScriptEditorPanel::RequestCompletionDetails() {
    if (!completion_details_shown_ || selected_completion_ < 0 ||
        selected_completion_ >= static_cast<int>(completion_entries_.size()) || !scripting_engine_)
        return;
    const std::string& name = completion_entries_[static_cast<size_t>(selected_completion_)].name;
    if (name == completion_details_for_ || completion_entries_from_fallback_) return;
    CodeEditor* code = ActiveCodeEditor();
    if (!code) return;
    auto request = LanguageRequest(scripting::LanguageService::Kind::Describe, *code, completion_request_pos_);
    request.name = name;
    completion_details_for_ = name;
    completion_details_ = {};
    completion_details_request_ = scripting_engine_->Language().Submit(std::move(request));
}

void ScriptEditorPanel::RenderCompletionPopup() {
    if (!show_completion_popup_ || completion_entries_.empty()) return;
    CodeEditor* code = ActiveCodeEditor();
    if (!code) return;
    const ui::Tokens& t = ui::CurrentTokens();
    RequestCompletionDetails();

    const float row_h = std::floor(ImGui::GetFontSize() * 2.0f);
    const int rows = std::min(10, static_cast<int>(completion_entries_.size()));
    const float list_w = 330.0f;
    const float details_w = 420.0f;
    const bool details = completion_details_shown_ && !completion_details_.name.empty();
    const float hint_h = ImGui::GetFontSize() + 12.0f;
    // The details keep their own height (measured last frame, capped).
    const float list_h = rows * row_h + 8.0f + hint_h;
    const float details_h = details ? std::min(360.0f, completion_details_height_ + 24.0f) : 0.0f;
    const ImVec2 size(list_w + (details ? details_w : 0.0f), std::max(list_h, details_h));

    // Under the word (the list's first column lines up with the text).
    const ImVec2 cursor = code->CursorScreenPos();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    float x = cursor.x - 34.0f - ImGui::CalcTextSize(completion_prefix_.c_str()).x;
    float y = cursor.y + 2.0f;
    x = std::clamp(x, 4.0f, std::max(4.0f, display.x - size.x - 4.0f));
    if (y + size.y > display.y - 4.0f) y = std::max(4.0f, cursor.y - ImGui::GetTextLineHeightWithSpacing() - size.y - 2.0f);

    // A soft shadow under the raised list (drawn on the editor's window, below the popup).
    ImDrawList* shadow = ImGui::GetWindowDrawList();
    for (int i = 1; i <= 6; ++i) {
        const float s = static_cast<float>(i) * 1.5f;
        shadow->AddRectFilled(ImVec2(x - s + 2.0f, y - s + 6.0f), ImVec2(x + size.x + s - 2.0f, y + size.y + s + 4.0f),
                              IM_COL32(0, 0, 0, 14), 8.0f + s);
    }

    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, t.bg_panel);
    if (ImGui::Begin("##completion_popup", nullptr, flags)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetWindowPos();
        ImFont* mono = gui::GetCodeFont() ? gui::GetCodeFont() : ImGui::GetFont();
        const float font = ImGui::GetFontSize();

        // The list: scrolls to keep the selected row visible.
        ImGui::SetCursorPos(ImVec2(4.0f, 4.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
        ImGui::BeginChild("##completion_list", ImVec2(list_w - 8.0f, rows * row_h), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar);
        const int count = static_cast<int>(completion_entries_.size());
        if (completion_scroll_to_selected_) {
            const float top = selected_completion_ * row_h;
            if (top < ImGui::GetScrollY()) ImGui::SetScrollY(top);
            if (top + row_h > ImGui::GetScrollY() + rows * row_h) ImGui::SetScrollY(top + row_h - rows * row_h);
            completion_scroll_to_selected_ = false;
        }
        ImDrawList* ldl = ImGui::GetWindowDrawList();
        for (int i = 0; i < count; ++i) {
            const lang::Completion& c = completion_entries_[static_cast<size_t>(i)];
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            ImGui::PushID(i);
            if (ImGui::InvisibleButton("##row", ImVec2(w, row_h))) {
                selected_completion_ = i;
                AcceptCompletion();
                ImGui::PopID();
                break;
            }
            const bool hovered = ImGui::IsItemHovered();
            ImGui::PopID();
            if (i == selected_completion_)
                ldl->AddRectFilled(p, ImVec2(p.x + w, p.y + row_h), ui::ToU32(ui::WithAlpha(t.accent, 0.35f)), 4.0f);
            else if (hovered)
                ldl->AddRectFilled(p, ImVec2(p.x + w, p.y + row_h), ui::ToU32(t.hover), 4.0f);
            // Kind chip, name (typed part highlighted), where it comes from.
            const lang::KindChip chip = lang::ChipFor(c.kind);
            const ImVec4 kc = KindColour(chip.role);
            const float chip_s = std::floor(font * 1.35f);
            const ImVec2 cp(p.x + 8.0f, p.y + (row_h - chip_s) * 0.5f);
            ldl->AddRectFilled(cp, ImVec2(cp.x + chip_s, cp.y + chip_s), ui::ToU32(ui::WithAlpha(kc, 0.14f)), 4.0f);
            const ImVec2 ls = mono->CalcTextSizeA(font - 1.0f, FLT_MAX, 0.0f, chip.letter);
            ldl->AddText(mono, font - 1.0f, ImVec2(cp.x + (chip_s - ls.x) * 0.5f, cp.y + (chip_s - ls.y) * 0.5f), ui::ToU32(kc),
                         chip.letter);
            const float tx = cp.x + chip_s + 8.0f;
            const float ty = p.y + (row_h - font) * 0.5f;
            const size_t typed = std::min(c.name.size(), c.name.size() >= c.complete.size() ? c.name.size() - c.complete.size() : 0);
            const std::string head = c.name.substr(0, typed);
            const std::string tail = c.name.substr(typed);
            const float hw = mono->CalcTextSizeA(font, FLT_MAX, 0.0f, head.c_str()).x;
            ldl->AddText(mono, font, ImVec2(tx, ty), ui::ToU32(t.accent_text), head.c_str());
            ldl->AddText(mono, font, ImVec2(tx + hw, ty), ui::ToU32(i == selected_completion_ ? t.text_bright : t.text), tail.c_str());
            // Right side: where the name comes from (board 6); not for names of this file.
            std::string side = c.module;
            if (side == "__main__" || side == completion_file_stem_ || c.kind == "module") side.clear();
            if (side.size() > 28) side = side.substr(0, 25) + "...";
            if (!side.empty()) {
                const float sw = ImGui::CalcTextSize(side.c_str()).x;
                const float nx = tx + mono->CalcTextSizeA(font, FLT_MAX, 0.0f, c.name.c_str()).x + 12.0f;
                const float sx = std::max(nx, p.x + w - 8.0f - sw);
                if (sx + sw <= p.x + w - 4.0f)
                    ldl->AddText(ImVec2(sx, p.y + (row_h - font) * 0.5f), ui::ToU32(t.text_dim), side.c_str());
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        // Keys hint, faint, under the list.
        const char* hint = completion_entries_from_fallback_
                               ? "Enter or Tab insert \xC2\xB7 Esc close"
                               : "Enter or Tab insert \xC2\xB7 Esc close \xC2\xB7 Ctrl+Space details";
        dl->AddText(ImVec2(origin.x + 12.0f, origin.y + 4.0f + rows * row_h + 6.0f), ui::ToU32(t.text_faint), hint);

        // Details of the selected item: signature, docstring, module.
        if (details) {
            const ImVec2 d0(origin.x + list_w, origin.y);
            dl->AddRectFilled(d0, ImVec2(origin.x + size.x, origin.y + size.y), ui::ToU32(ui::Mix(t.bg_panel, t.bg_window, 0.35f)),
                              6.0f, ImDrawFlags_RoundCornersRight);
            ImGui::SetCursorScreenPos(ImVec2(d0.x + 14.0f, d0.y + 12.0f));
            ImGui::BeginChild("##completion_details", ImVec2(details_w - 28.0f, size.y - 24.0f), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
            if (!completion_details_.signature.empty()) {
                ImGui::PushFont(mono);
                ImGui::TextColored(t.text, "%s", completion_details_.signature.c_str());
                ImGui::PopFont();
                ImGui::Dummy(ImVec2(0, 6));
            }
            if (!completion_details_.doc.empty()) {
                const std::string doc = lang::ReflowDoc(completion_details_.doc, 2);
                ImGui::TextColored(ui::Mix(t.text, t.text_dim, 0.25f), "%s", doc.c_str());
                ImGui::Dummy(ImVec2(0, 6));
            }
            if (!completion_details_.module.empty()) ImGui::TextColored(t.text_dim, "%s", completion_details_.module.c_str());
            completion_details_height_ = ImGui::GetCursorPosY();
            ImGui::PopTextWrapPos();
            ImGui::EndChild();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(4);
}

bool ScriptEditorPanel::HandleLanguageResult(const scripting::LanguageService::Result& result) {
    (void)result;
    return false;
}

void ScriptEditorPanel::AcceptCompletion() {
    if (selected_completion_ >= 0 && selected_completion_ < static_cast<int>(completion_entries_.size())) {
        CodeEditor* code = ActiveCodeEditor();
        if (code) {
            editor::Document& doc = code->Doc();
            const editor::Pos cursor = doc.Primary().head;
            doc.SetSelections({editor::Selection{completion_start_pos_, cursor, -1}});
            doc.Paste(completion_entries_[static_cast<size_t>(selected_completion_)].name);  // one undo step
        }
    }
    CloseCompletionPopup();
}

}  // namespace cyxwiz
