// Script Editor language cards (TOFIX133 P3 steps 3.5-3.6, approved boards
// 7-8): signature help above the call being typed, the hover card (a name's
// signature and docstring, or the problem under the mouse), and go to
// definition (F12, Ctrl+click, the right-click menu).

#include "script_editor.h"

#include "../../core/language_results.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <filesystem>
#include <string>

namespace cyxwiz {

namespace {
using Kind = scripting::LanguageService::Kind;

// Jedi and pyflakes columns count characters; the editor's are UTF-8 bytes.
int ByteColumnOf(const std::string& line, int char_col) {
    int chars = 0;
    for (int i = 0; i < static_cast<int>(line.size()); ++i) {
        if ((static_cast<unsigned char>(line[static_cast<size_t>(i)]) & 0xC0) == 0x80) continue;
        if (chars == char_col) return i;
        ++chars;
    }
    return static_cast<int>(line.size());
}

bool IsWordChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || (static_cast<unsigned char>(c) & 0x80);
}

// The identifier around a column: [first, second), empty when not on one.
std::pair<int, int> WordAt(const std::string& line, int col) {
    if (col < 0 || col >= static_cast<int>(line.size()) || !IsWordChar(line[static_cast<size_t>(col)])) return {col, col};
    int a = col;
    int b = col;
    while (a > 0 && IsWordChar(line[static_cast<size_t>(a - 1)])) --a;
    while (b < static_cast<int>(line.size()) && IsWordChar(line[static_cast<size_t>(b)])) ++b;
    return {a, b};
}

bool SamePath(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;
    auto norm = [](const std::string& p) {
        std::string s = std::filesystem::path(p).lexically_normal().generic_string();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    return norm(a) == norm(b);
}

// A raised card: soft shadow on the editor's window, then a borderless window.
bool BeginCard(const char* id, ImVec2 pos, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(width, FLT_MAX));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 4.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ui::Mix(t.bg_panel, t.bg_window, 0.2f));
    return ImGui::Begin(id, nullptr, flags);
}

void EndCard() {
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(4);
}

// The shadow under a card of the given rect (drawn behind it, on the window below).
void CardShadow(ImDrawList* dl, ImVec2 a, ImVec2 b) {
    for (int i = 1; i <= 6; ++i) {
        const float s = static_cast<float>(i) * 1.5f;
        dl->AddRectFilled(ImVec2(a.x - s + 2.0f, a.y - s + 6.0f), ImVec2(b.x + s - 2.0f, b.y + s + 4.0f), IM_COL32(0, 0, 0, 14),
                          8.0f + s);
    }
}
}  // namespace

// ---------------------------------------------------------------------------
// Where the language cards point: the script, or one notebook cell.

ScriptEditorPanel::CodeTarget ScriptEditorPanel::ActiveCodeTarget() {
    CodeTarget target;
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) return target;
    auto& tab = *tabs_[active_tab_index_];
    target.document_id = tab.document_id;
    if (tab.cell_mode && tab.editing_cell >= 0 && tab.editing_cell < tab.cell_manager.GetCellCount())
        target.cell_id = tab.cell_manager.GetCell(tab.editing_cell).id;
    return target;
}

CodeEditor* ScriptEditorPanel::CodeEditorFor(const CodeTarget& target) {
    const int index = FindTabIndex(target.document_id);
    if (index < 0 || index != active_tab_index_) return nullptr;
    auto& tab = *tabs_[index];
    if (target.cell_id.empty()) return tab.cell_mode ? nullptr : &tab.editor;
    for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
        Cell& cell = tab.cell_manager.GetCell(i);
        if (cell.id == target.cell_id) return cell.type == CellType::Code ? &cell.editor : nullptr;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Signature help (step 3.5, board 7): opens when "(" or "," is typed or on
// Ctrl+Shift+Space, follows the cursor while it stays in the call, Esc closes.

void ScriptEditorPanel::RequestSignatures(bool manual) {
    CodeEditor* code = ActiveCodeEditor();
    if (!code || !LanguageReady()) return;
    const editor::Pos pos = code->Doc().Primary().head;
    signature_target_ = ActiveCodeTarget();
    signature_request_ = scripting_engine_->Language().Submit(LanguageRequest(Kind::Signatures, *code, pos));
    signature_request_version_ = code->Doc().Version();
    signature_request_pos_ = pos;
    if (manual) signature_manual_ = true;
}

void ScriptEditorPanel::CloseSignatureHelp() {
    signature_open_ = false;
    signature_manual_ = false;
    signature_request_ = 0;
    signatures_.clear();
}

void ScriptEditorPanel::UpdateSignatureHelp() {
    CodeEditor* code = ActiveCodeEditor();
    const CodeTarget target = ActiveCodeTarget();
    if (!code || !code->IsFocused()) {
        if (signature_open_ || signature_request_) CloseSignatureHelp();
        signature_seen_version_ = 0;
        return;
    }
    const editor::Document& doc = code->Doc();
    const editor::Pos pos = doc.Primary().head;
    const bool moved_editor = !(target == signature_seen_target_);
    const bool changed = doc.Version() != signature_seen_version_;
    const bool moved = !(pos == signature_seen_pos_);
    signature_seen_target_ = target;
    signature_seen_version_ = doc.Version();
    signature_seen_pos_ = pos;
    if (moved_editor) {
        CloseSignatureHelp();
        return;
    }
    if (!changed && !moved) return;
    // Typing "(" or "," asks; while the card is open every edit or move
    // asks again (it closes when the cursor leaves the call).
    bool ask = signature_open_ || signature_request_ != 0;
    if (changed && pos.col > 0) {
        const std::string& line = doc.Line(pos.line);
        const char before = pos.col <= static_cast<int>(line.size()) ? line[static_cast<size_t>(pos.col - 1)] : '\0';
        if (before == '(' || before == ',') ask = true;
        if (before == ')' && !signature_manual_) ask = signature_open_;
    }
    if (ask) RequestSignatures(false);
}

bool ScriptEditorPanel::HandleSignaturesResult(const scripting::LanguageService::Result& result) {
    if (result.id != signature_request_) return true;  // replaced by a newer request
    signature_request_ = 0;
    CodeEditor* code = CodeEditorFor(signature_target_);
    if (!code || code->Doc().Version() != signature_request_version_ || !(code->Doc().Primary().head == signature_request_pos_))
        return true;  // the text moved on; the next request answers
    signatures_ = lang::ParseSignatures(result.json);
    signature_open_ = !signatures_.empty();
    if (!signature_open_) signature_manual_ = false;
    return true;
}

void ScriptEditorPanel::RenderSignatureCard() {
    if (!signature_open_ || signatures_.empty()) return;
    CodeEditor* code = CodeEditorFor(signature_target_);
    if (!code) {
        CloseSignatureHelp();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const lang::Signature& sig = signatures_.front();
    ImFont* mono = gui::GetCodeFont() ? gui::GetCodeFont() : ImGui::GetFont();
    const float font = ImGui::GetFontSize();
    const float max_w = std::min(680.0f, ImGui::GetIO().DisplaySize.x - 16.0f);

    // Lay the call out in pieces, wrapping between them (board 7).
    const auto pieces = lang::SignaturePieces(sig);
    struct Placed {
        const lang::SignaturePiece* piece;
        float x, y, w;
    };
    std::vector<Placed> placed;
    const float line_h = font + 4.0f;
    const float inner = max_w - 28.0f;
    float x = 0.0f, y = 0.0f, width = 0.0f;
    for (const auto& piece : pieces) {
        const float w = mono->CalcTextSizeA(font, FLT_MAX, 0.0f, piece.text.c_str()).x;
        if (x > 0.0f && x + w > inner) {
            x = 24.0f;  // continuation lines indent under the name
            y += line_h;
        }
        placed.push_back({&piece, x, y, w});
        x += w;
        width = std::max(width, x);
    }
    std::string footer = lang::SignatureFooter(sig);
    if (signatures_.size() > 1) footer += " \xC2\xB7 " + std::to_string(signatures_.size()) + " signatures";
    width = std::max(width, ImGui::CalcTextSize(footer.c_str()).x);
    const ImVec2 size(width + 28.0f, y + line_h + 6.0f + font + 20.0f);

    // Above the cursor's line (completions open below it).
    const ImVec2 cursor = code->CursorScreenPos();
    const float row = CodeEditor::LineHeightFor(font);
    float cx = std::clamp(cursor.x - 24.0f, 8.0f, std::max(8.0f, ImGui::GetIO().DisplaySize.x - size.x - 8.0f));
    float cy = cursor.y - row - size.y - 4.0f;
    if (cy < 4.0f) cy = cursor.y + 4.0f;
    CardShadow(ImGui::GetWindowDrawList(), ImVec2(cx, cy), ImVec2(cx + size.x, cy + size.y));
    if (BeginCard("##signature_card", ImVec2(cx, cy), size.x)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetCursorScreenPos();
        for (const auto& p : placed) {
            const ImVec2 at(o.x + p.x, o.y + p.y);
            dl->AddText(mono, font, at, ui::ToU32(p.piece->active ? t.accent_text : t.text), p.piece->text.c_str());
            if (p.piece->active)
                dl->AddLine(ImVec2(at.x, at.y + font + 1.0f), ImVec2(at.x + p.w, at.y + font + 1.0f),
                            ui::ToU32(ui::WithAlpha(t.accent_text, 0.7f)), 1.0f);
        }
        ImGui::Dummy(ImVec2(width, y + line_h + 2.0f));
        ImGui::TextColored(t.text_dim, "%s", footer.c_str());
    }
    EndCard();
}

// ---------------------------------------------------------------------------
// Hover (step 3.6, board 8): after the mouse rests half a second on a name,
// its card; on an underlined problem, the problem.

void ScriptEditorPanel::UpdateHover(CodeEditor& code, const std::vector<lang::Problem>& problems, const std::string& cell_id) {
    editor::Pos pos;
    ImVec2 below;
    if (!code.HoverPos(pos, below)) return;
    const auto& tab = *tabs_[active_tab_index_];
    CodeTarget target{tab.document_id, cell_id};
    const std::string& line = code.Doc().Line(pos.line);
    const auto word = WordAt(line, pos.col);
    // The problem under the mouse, if any (its underline's columns).
    int problem = -1;
    for (size_t i = 0; i < problems.size(); ++i) {
        const lang::Problem& p = problems[i];
        if (p.line - 1 != pos.line || (!p.error && !tab.problems_show_warnings)) continue;
        lang::Problem bytes = p;
        bytes.column = ByteColumnOf(line, p.column);
        const auto range = lang::ProblemRange(line, bytes);
        if (pos.col >= range.first && pos.col < std::max(range.second, range.first + 1)) {
            problem = static_cast<int>(i);
            break;
        }
    }
    const int key_col = problem >= 0 ? -2 - problem : word.first;
    if (word.first == word.second && problem < 0) return;  // not on a name: the card closes
    hover_seen_frame_ = ImGui::GetFrameCount();
    const bool same = hover_.target == target && hover_.line == pos.line && hover_.key_col == key_col &&
                      hover_.version == code.Doc().Version();
    if (same) return;
    hover_ = {};
    hover_.target = target;
    hover_.line = pos.line;
    hover_.key_col = key_col;
    hover_.pos = {pos.line, word.first == word.second ? pos.col : word.first};
    hover_.version = code.Doc().Version();
    hover_.since = ImGui::GetTime();
    hover_.below = below;
    if (problem >= 0) {
        hover_.problem = problems[static_cast<size_t>(problem)];
        hover_.is_problem = true;
    }
}

bool ScriptEditorPanel::HandleHoverResult(const scripting::LanguageService::Result& result) {
    if (result.id != hover_.request) return true;
    hover_.request = 0;
    hover_.info = lang::ParseHover(result.json);
    // Keywords (True, for, ...) have nothing worth a card.
    hover_.state = hover_.info.name.empty() || hover_.info.kind == "keyword" ? HoverState::Nothing : HoverState::Shown;
    return true;
}

void ScriptEditorPanel::RenderHoverCard() {
    if (hover_.target.document_id == 0) return;
    const bool over_card = hover_.state == HoverState::Shown && ImGui::IsMouseHoveringRect(hover_.card_min, hover_.card_max, false);
    CodeEditor* code = CodeEditorFor(hover_.target);
    // Typing, clicking, the completion list, or the mouse leaving close it.
    if (!code || code->Doc().Version() != hover_.version || show_completion_popup_ ||
        (hover_seen_frame_ != ImGui::GetFrameCount() && !over_card) ||
        (!over_card && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)))) {
        hover_ = {};
        return;
    }
    if (hover_.state == HoverState::Resting) {
        if (ImGui::GetTime() - hover_.since < 0.5) return;
        if (hover_.is_problem) {
            hover_.state = HoverState::Shown;
        } else if (LanguageReady()) {
            hover_.request = scripting_engine_->Language().Submit(LanguageRequest(Kind::Hover, *code, hover_.pos));
            hover_.state = hover_.request ? HoverState::Waiting : HoverState::Nothing;
        } else {
            hover_.state = HoverState::Nothing;
        }
    }
    if (hover_.state != HoverState::Shown) return;

    const ui::Tokens& t = ui::CurrentTokens();
    ImFont* mono = gui::GetCodeFont() ? gui::GetCodeFont() : ImGui::GetFont();
    const float max_w = std::min(560.0f, ImGui::GetIO().DisplaySize.x - 16.0f);
    const ImVec2 last = hover_.card_max.x > hover_.card_min.x
                            ? ImVec2(hover_.card_max.x - hover_.card_min.x, hover_.card_max.y - hover_.card_min.y)
                            : ImVec2(max_w, 120.0f);
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    float x = std::clamp(hover_.below.x - 20.0f, 8.0f, std::max(8.0f, display.x - last.x - 8.0f));
    float y = hover_.below.y + 4.0f;
    if (y + last.y > display.y - 8.0f) y = std::max(4.0f, hover_.below.y - CodeEditor::LineHeightFor(ImGui::GetFontSize()) - last.y - 4.0f);
    CardShadow(ImGui::GetWindowDrawList(), ImVec2(x, y), ImVec2(x + last.x, y + last.y));
    if (BeginCard("##hover_card", ImVec2(x, y), max_w)) {
        ImGui::PushTextWrapPos(max_w - 28.0f);
        if (hover_.is_problem) {
            const lang::Problem& p = hover_.problem;
            ImGui::TextColored(p.error ? t.error : t.warning, "%s", p.error ? ICON_FA_CIRCLE_XMARK : ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::SameLine();
            ImGui::TextUnformatted(p.message.c_str());
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::TextColored(t.text_faint, "pyflakes");
        } else {
            const lang::Hover& h = hover_.info;
            ImGui::PushFont(mono);
            ImGui::TextColored(t.text, "%s", lang::HoverHeadline(h).c_str());
            ImGui::PopFont();
            ImGui::Dummy(ImVec2(0.0f, 2.0f));
            if (!h.doc.empty()) ImGui::TextColored(ui::Mix(t.text, t.text_dim, 0.25f), "%s", lang::ReflowDoc(h.doc, 2).c_str());
            else ImGui::TextColored(t.text_dim, "%s", "No docstring.");
            const std::string where = lang::LocationLabel(h.path, h.line, h.module);
            if (!where.empty() && (h.line > 0 || !h.path.empty())) {
                ImGui::Dummy(ImVec2(0.0f, 2.0f));
                if (ui::LinkButton(where.c_str())) {
                    CodeEditor* c = CodeEditorFor(hover_.target);
                    if (c) GoToDefinition(hover_.target, *c, hover_.pos);
                    hover_ = {};
                }
                ImGui::TextColored(t.text_faint, "%s", "Go to definition: F12 or Ctrl+click");
            } else if (!where.empty()) {
                ImGui::TextColored(t.text_dim, "%s", where.c_str());
            }
        }
        ImGui::PopTextWrapPos();
        if (hover_.target.document_id != 0) {
            hover_.card_min = ImGui::GetWindowPos();
            hover_.card_max = ImVec2(hover_.card_min.x + ImGui::GetWindowWidth(), hover_.card_min.y + ImGui::GetWindowHeight());
        }
    }
    EndCard();
}

// ---------------------------------------------------------------------------
// Go to definition: in this text (or cell) the cursor moves there; another
// file opens at the line; a built-in says so for a moment.

void ScriptEditorPanel::GoToDefinition(const CodeTarget& target, CodeEditor& code, const editor::Pos& pos) {
    if (!LanguageReady()) {
        ShowLanguageNote(code, "Go to definition needs the Python tools (see the Python status).");
        return;
    }
    definition_target_ = target;
    definition_request_ = scripting_engine_->Language().Submit(LanguageRequest(Kind::Definition, code, pos));
    const auto word = WordAt(code.Doc().Line(pos.line), pos.col);
    definition_name_ = word.first < word.second ? code.Doc().Line(pos.line).substr(static_cast<size_t>(word.first),
                                                                                    static_cast<size_t>(word.second - word.first))
                                                : std::string();
}

bool ScriptEditorPanel::HandleDefinitionResult(const scripting::LanguageService::Result& result) {
    if (result.id != definition_request_) return true;
    definition_request_ = 0;
    CodeEditor* code = CodeEditorFor(definition_target_);
    const int index = FindTabIndex(definition_target_.document_id);
    if (!code || index < 0) return true;
    auto& tab = *tabs_[index];
    const auto locations = lang::ParseLocations(result.json);
    const std::string name = definition_name_.empty() ? std::string("this name") : "'" + definition_name_ + "'";
    if (locations.empty()) {
        ShowLanguageNote(*code, "No definition found for " + name + ".");
        return true;
    }
    const lang::Location& at = locations.front();
    const bool here = at.path.empty() ? at.line > 0 : (definition_target_.cell_id.empty() && SamePath(at.path, tab.filepath));
    if (here) {
        const int line = std::clamp(at.line - 1, 0, code->Doc().LineCount() - 1);
        code->GoToLine(line);
        code->Doc().SetCursor({line, ByteColumnOf(code->Doc().Line(line), at.column)});
        code->ScrollToCursor();
        code->RequestFocus();
        return true;
    }
    if (at.path.empty() || at.line <= 0) {
        ShowLanguageNote(*code, name + " is built in; there is no Python source to open.");
        return true;
    }
    // Another file: opened next frame (this runs before the tabs draw).
    deferred_open_path_ = at.path;
    deferred_open_line_ = at.line;
    return true;
}

void ScriptEditorPanel::ShowLanguageNote(const CodeEditor& code, std::string text) {
    language_note_ = std::move(text);
    language_note_until_ = ImGui::GetTime() + 2.5;
    language_note_at_ = code.CursorScreenPos();
}

void ScriptEditorPanel::RenderLanguageNote() {
    if (language_note_.empty()) return;
    if (ImGui::GetTime() > language_note_until_) {
        language_note_.clear();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at(language_note_at_.x - 12.0f, language_note_at_.y + 4.0f);
    const ImVec2 size(ImGui::CalcTextSize(language_note_.c_str()).x + 28.0f, ImGui::GetFontSize() + 20.0f);
    CardShadow(ImGui::GetWindowDrawList(), at, ImVec2(at.x + size.x, at.y + size.y));
    if (BeginCard("##language_note", at, 640.0f)) ImGui::TextColored(t.text_dim, "%s", language_note_.c_str());
    EndCard();
}

bool ScriptEditorPanel::HandleCardResult(const scripting::LanguageService::Result& result) {
    switch (result.kind) {
        case Kind::Signatures: return HandleSignaturesResult(result);
        case Kind::Hover: return HandleHoverResult(result);
        case Kind::Definition: return HandleDefinitionResult(result);
        default: return false;
    }
}

void ScriptEditorPanel::AfterCodeRender(CodeEditor& code, const std::vector<lang::Problem>& problems, const std::string& cell_id) {
    UpdateHover(code, problems, cell_id);
    editor::Pos pos;
    if (code.TakeCtrlClick(pos)) {
        hover_ = {};
        GoToDefinition(CodeTarget{tabs_[active_tab_index_]->document_id, cell_id}, code, pos);
    }
}

void ScriptEditorPanel::RenderLanguageCards() {
    RenderSignatureCard();
    RenderHoverCard();
    RenderLanguageNote();
}

}  // namespace cyxwiz
