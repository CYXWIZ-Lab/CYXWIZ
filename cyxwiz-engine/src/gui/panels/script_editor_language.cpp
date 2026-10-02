// Script Editor language intelligence UI (TOFIX133 P3, approved boards 6-8):
// the completion list with the selected item's details. Results come from
// the bundled Jedi on the language worker (scripting/language_service).

#include "script_editor.h"

#include "../../core/language_results.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdio>
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
    if (result.kind == scripting::LanguageService::Kind::Diagnostics) return HandleDiagnosticsResult(result);
    return HandleCardResult(result);  // signatures, hover, definitions
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


// ---------------------------------------------------------------------------
// Problems (step 3.4, boards 6-7): pyflakes half a second after typing
// stops, underlines in the code, counts in the status bar, a Problems panel.

namespace {
// pyflakes columns count characters; the editor's are UTF-8 byte offsets.
int ByteColumn(const std::string& line, int char_col) {
    int chars = 0;
    for (int i = 0; i < static_cast<int>(line.size()); ++i) {
        if ((static_cast<unsigned char>(line[static_cast<size_t>(i)]) & 0xC0) == 0x80) continue;
        if (chars == char_col) return i;
        ++chars;
    }
    return static_cast<int>(line.size());
}

void SetSquigglesFrom(CodeEditor& code, const std::vector<lang::Problem>& problems, bool show_warnings) {
    std::vector<CodeEditor::Squiggle> marks;
    const editor::Document& doc = code.Doc();
    for (const auto& p : problems) {
        if (!p.error && !show_warnings) continue;
        const int line = p.line - 1;
        if (line < 0 || line >= doc.LineCount()) continue;
        const std::string& text = doc.Line(line);
        lang::Problem bytes = p;
        bytes.column = ByteColumn(text, p.column);
        const auto range = lang::ProblemRange(text, bytes);
        marks.push_back({{line, range.first}, {line, std::min(range.second, static_cast<int>(text.size()) + 1)}, p.error});
    }
    code.SetSquiggles(std::move(marks));
}
}  // namespace

void ScriptEditorPanel::UpdateDiagnostics(EditorTab& tab) {
    if (!scripting_engine_) return;
    const double now = ImGui::GetTime();
    // The code to check: the script, or the notebook cell being edited.
    CodeEditor* code = nullptr;
    std::string cell_id;
    std::vector<std::string> known;
    if (!tab.cell_mode) {
        if (tab.is_loading || tab.is_large_file || tab.load_failed) return;
        code = &tab.editor;
    } else {
        if (tab.editing_cell < 0 || tab.editing_cell >= tab.cell_manager.GetCellCount()) return;
        Cell& cell = tab.cell_manager.GetCell(tab.editing_cell);
        if (cell.type != CellType::Code) return;
        code = &cell.editor;
        cell_id = cell.id;
        // Names the notebook already has are not "undefined" in a cell.
        if (tab.variables_view)
            for (auto& n : tab.variables_view->Names()) known.push_back(std::move(n));
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
            const Cell& other = tab.cell_manager.GetCell(i);
            if (other.type != CellType::Code || other.id == cell.id) continue;
            // Names assigned or imported in other cells (a light scan).
            size_t start = 0;
            while (start < other.source.size()) {
                size_t end = other.source.find('\n', start);
                if (end == std::string::npos) end = other.source.size();
                const std::string line = other.source.substr(start, end - start);
                size_t i0 = 0;
                while (i0 < line.size() && (std::isalnum(static_cast<unsigned char>(line[i0])) || line[i0] == '_')) ++i0;
                if (i0 > 0 && line.find('=') != std::string::npos && line.find_first_not_of(" \t", i0) == line.find('=') &&
                    line.compare(line.find('='), 2, "==") != 0)
                    known.push_back(line.substr(0, i0));
                for (const char* kw : {"import ", "def ", "class "}) {
                    const size_t k = line.rfind(kw, 0) == 0 ? 0 : std::string::npos;
                    if (k == 0) {
                        size_t a = std::string(kw).size();
                        size_t b = a;
                        while (b < line.size() && (std::isalnum(static_cast<unsigned char>(line[b])) || line[b] == '_')) ++b;
                        if (b > a) known.push_back(line.substr(a, b - a));
                    }
                }
                const size_t imp = line.find(" import ");
                if (line.rfind("from ", 0) == 0 && imp != std::string::npos) {
                    std::string names = line.substr(imp + 8);
                    size_t s0 = 0;
                    while (s0 < names.size()) {
                        size_t comma = names.find(',', s0);
                        std::string n = names.substr(s0, comma == std::string::npos ? std::string::npos : comma - s0);
                        const size_t as = n.find(" as ");
                        if (as != std::string::npos) n = n.substr(as + 4);
                        n.erase(0, n.find_first_not_of(" ("));
                        n.erase(n.find_last_not_of(" )") + 1);
                        if (!n.empty()) known.push_back(n);
                        if (comma == std::string::npos) break;
                        s0 = comma + 1;
                    }
                }
                start = end + 1;
            }
        }
    }
    const std::uint64_t version = code->Doc().Version();
    if (diag_seen_doc_ != tab.document_id || diag_seen_cell_ != cell_id || diag_seen_version_ != version) {
        diag_seen_doc_ = tab.document_id;
        diag_seen_cell_ = cell_id;
        diag_seen_version_ = version;
        diag_changed_at_ = now;
        return;
    }
    const std::uint64_t have = cell_id.empty() ? tab.problems_version : tab.cell_manager.GetCell(tab.editing_cell).problems_version;
    if (have == version + 1 || now - diag_changed_at_ < 0.5) return;  // up to date, or still typing
    if (diag_in_flight_version_ == version + 1 && diag_in_flight_doc_ == tab.document_id && diag_in_flight_cell_ == cell_id) return;
    if (!LanguageReady()) return;
    auto request = LanguageRequest(scripting::LanguageService::Kind::Diagnostics, *code, code->Doc().Primary().head);
    request.known_names = std::move(known);
    request.namespace_key.clear();  // pyflakes reads the text only; names come in known_names
    const std::uint64_t id = scripting_engine_->Language().Submit(std::move(request));
    if (id == 0) return;
    // Versions are stored +1 so 0 means "never checked".
    pending_diagnostics_[id] = {tab.document_id, cell_id, version + 1};
    diag_in_flight_doc_ = tab.document_id;
    diag_in_flight_cell_ = cell_id;
    diag_in_flight_version_ = version + 1;
}

bool ScriptEditorPanel::HandleDiagnosticsResult(const scripting::LanguageService::Result& result) {
    auto it = pending_diagnostics_.find(result.id);
    if (it == pending_diagnostics_.end()) return true;  // replaced by a newer request
    const PendingDiagnostics target = it->second;
    pending_diagnostics_.erase(it);
    if (diag_in_flight_version_ == target.version && diag_in_flight_doc_ == target.document_id) diag_in_flight_version_ = 0;
    const int index = FindTabIndex(target.document_id);
    if (index < 0) return true;
    auto& tab = *tabs_[index];
    std::vector<lang::Problem> problems = lang::ParseProblems(result.json);
    if (target.cell_id.empty()) {
        tab.problems = std::move(problems);
        tab.problems_version = target.version;
    } else {
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
            Cell& cell = tab.cell_manager.GetCell(i);
            if (cell.id != target.cell_id) continue;
            cell.problems = std::move(problems);
            cell.problems_version = target.version;
        }
    }
    return true;
}

void ScriptEditorPanel::ApplyProblemSquiggles(const EditorTab& tab, CodeEditor& code, const std::vector<lang::Problem>& problems) {
    // The last check's problems stay until the next one (0.5 s after typing stops).
    SetSquigglesFrom(code, problems, tab.problems_show_warnings);
}

ScriptEditorPanel::ProblemCounts ScriptEditorPanel::CountProblems(const EditorTab& tab) const {
    ProblemCounts c;
    auto add = [&](const std::vector<lang::Problem>& list) {
        for (const auto& p : list) (p.error ? c.errors : c.warnings)++;
    };
    if (!tab.cell_mode) add(tab.problems);
    else
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) add(tab.cell_manager.GetCell(i).problems);
    return c;
}

float ScriptEditorPanel::ProblemCountsWidth(const EditorTab& tab) const {
    const ProblemCounts c = CountProblems(tab);
    char text[64];
    std::snprintf(text, sizeof(text), "%s %d   %s %d", ICON_FA_CIRCLE_XMARK, c.errors, ICON_FA_TRIANGLE_EXCLAMATION, c.warnings);
    return ImGui::CalcTextSize(text).x + 16.0f;
}

// The counts in the status bar (board 6): coloured icons, click for the panel.
bool ScriptEditorPanel::ProblemCountsItem(EditorTab& tab) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ProblemCounts c = CountProblems(tab);
    char errors[16], warnings[16];
    std::snprintf(errors, sizeof(errors), "%d", c.errors);
    std::snprintf(warnings, sizeof(warnings), "%d", c.warnings);
    const float w = ProblemCountsWidth(tab);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight();
    const bool clicked = ImGui::InvisibleButton("##problem_counts", ImVec2(w, h));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (hovered || tab.show_problems) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ui::ToU32(hovered ? t.hover : ui::WithAlpha(t.text, 0.06f)), 4.0f);
    const float y = p.y + (h - ImGui::GetFontSize()) * 0.5f;
    float x = p.x + 8.0f;
    auto part = [&](const char* icon, const ImVec4& colour, const char* count) {
        dl->AddText(ImVec2(x, y), ui::ToU32(colour), icon);
        x += ImGui::CalcTextSize(icon).x + ImGui::CalcTextSize(" ").x;
        dl->AddText(ImVec2(x, y), ui::ToU32(t.text_dim), count);
        x += ImGui::CalcTextSize(count).x + ImGui::CalcTextSize("   ").x;
    };
    part(ICON_FA_CIRCLE_XMARK, c.errors ? t.error : t.text_dim, errors);
    part(ICON_FA_TRIANGLE_EXCLAMATION, c.warnings ? t.warning : t.text_dim, warnings);
    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        std::vector<lang::Problem> all;
        if (!tab.cell_mode) all = tab.problems;
        else
            for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i)
                for (const auto& pr : tab.cell_manager.GetCell(i).problems) all.push_back(pr);
        ImGui::SetTooltip("%s (pyflakes). Click to show the Problems panel.", lang::ProblemSummary(all).c_str());
    }
    if (clicked) tab.show_problems = !tab.show_problems;
    return clicked;
}

// Problems panel (board 7): under the code, one row per problem; click a
// row to go there.
void ScriptEditorPanel::RenderProblemsPanel(EditorTab& tab, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    struct Row {
        lang::Problem p;
        int cell = -1;  // notebook cell index
        int count = 0;  // its [n]
    };
    std::vector<Row> rows;
    if (!tab.cell_mode) {
        for (const auto& p : tab.problems) rows.push_back({p});
    } else {
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
            const Cell& cell = tab.cell_manager.GetCell(i);
            for (const auto& p : cell.problems) rows.push_back({p, i, cell.execution_count});
        }
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.p.error && !b.p.error; });
    int errors = 0, warnings = 0;
    for (const auto& r : rows) (r.p.error ? errors : warnings)++;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.bg_panel, 0.6f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 6.0f));
    ImGui::BeginChild("##problems_panel", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar);
    ImGui::AlignTextToFramePadding();
    {
        ui::FontScope bold(ui::Font::Bold);
        ImGui::TextUnformatted("Problems");
    }
    ImGui::SameLine(0.0f, 12.0f);
    std::vector<lang::Problem> plain;
    for (const auto& r : rows) plain.push_back(r.p);
    ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s", tab.cell_mode ? "this notebook" : "this file", lang::ProblemSummary(plain).c_str());
    const float close_w = ui::ButtonWidth(ICON_FA_XMARK, ui::ButtonSize::Small);
    const float check_w = ImGui::GetFrameHeight() + ImGui::CalcTextSize("Show warnings").x + 16.0f;
    ui::SameLineRight(close_w + check_w + 8.0f);
    ImGui::Checkbox("Show warnings", &tab.problems_show_warnings);
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_XMARK "##close_problems")) tab.show_problems = false;
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Close Problems");

    ImGui::BeginChild("##problem_rows", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None);
    if (rows.empty()) {
        ImGui::TextColored(t.text_dim, "%s", "No problems found by pyflakes.");
    }
    ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        if (!r.p.error && !tab.problems_show_warnings) continue;
        ImGui::PushID(static_cast<int>(i));
        const ImVec2 start = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        if (ImGui::Selectable("##row", false, ImGuiSelectableFlags_None, ImVec2(w, ImGui::GetFrameHeight()))) {
            // Go there: the line in the script, or the cell in a notebook.
            if (r.cell >= 0) {
                tab.selected_cell = r.cell;
                tab.editing_cell = r.cell;
                tab.last_editing_cell = r.cell;
                tab.scroll_to_cell = r.cell;
                Cell& cell = tab.cell_manager.GetCell(r.cell);
                cell.SyncEditorFromSource();
                cell.editor.GoToLine(std::max(0, r.p.line - 1));
                const int line = std::clamp(r.p.line - 1, 0, cell.editor.Doc().LineCount() - 1);
                cell.editor.Doc().SetCursor({line, ByteColumn(cell.editor.Doc().Line(line), r.p.column)});
                cell.editor.RequestFocus();
            } else {
                tab.editor.GoToLine(std::max(0, r.p.line - 1));
                const int line = std::clamp(r.p.line - 1, 0, tab.editor.Doc().LineCount() - 1);
                tab.editor.Doc().SetCursor({line, ByteColumn(tab.editor.Doc().Line(line), r.p.column)});
                request_focus_ = true;
            }
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float y = start.y + (ImGui::GetFrameHeight() - ImGui::GetFontSize()) * 0.5f;
        dl->AddText(ImVec2(start.x + 4.0f, y), ui::ToU32(r.p.error ? t.error : t.warning),
                    r.p.error ? ICON_FA_CIRCLE_XMARK : ICON_FA_TRIANGLE_EXCLAMATION);
        dl->AddText(ImVec2(start.x + 28.0f, y), ui::ToU32(t.text), r.p.message.c_str());
        char where[64];
        if (r.cell >= 0) std::snprintf(where, sizeof(where), "Cell %d, Ln %d", r.cell + 1, r.p.line);
        else std::snprintf(where, sizeof(where), "Ln %d, Col %d", r.p.line, r.p.column + 1);
        const float ww = ImGui::CalcTextSize(where).x;
        const float sw = ImGui::CalcTextSize("pyflakes").x;
        dl->AddText(ImVec2(start.x + w - ww - 8.0f, y), ui::ToU32(t.text_dim), where);
        dl->AddText(ImVec2(start.x + w - ww - sw - 28.0f, y), ui::ToU32(t.text_faint), "pyflakes");
        ImGui::PopID();
    }
    ImGui::PopStyleColor(3);
    ImGui::EndChild();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    (void)errors;
    (void)warnings;
}

}  // namespace cyxwiz
