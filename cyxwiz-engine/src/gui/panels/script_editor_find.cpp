// Script Editor inline find and replace (TOFIX133 P2, approved board 1):
// a raised panel at the top right of the code (a full-width row when the
// editor is narrow, board 3) with match case / whole word / regex, "3 of
// 13", previous / next / close, and a replace row. Every match is marked in
// the code; the current one is stronger. Matching is core/text_search.

#include "script_editor.h"

#include "../../core/text_search.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace cyxwiz {

namespace {
// A small toggle: tinted when on, no outline.
bool FindToggle(const char* label, const char* tooltip, bool* value) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, *value ? ui::WithAlpha(t.accent, 0.40f) : ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, *value ? ui::WithAlpha(t.accent, 0.55f) : t.hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ui::WithAlpha(t.accent, 0.60f));
    ImGui::PushStyleColor(ImGuiCol_Text, *value ? t.text_bright : t.text_dim);
    const bool clicked = ImGui::SmallButton(label);
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", tooltip);
    if (clicked) *value = !*value;
    return clicked;
}

bool IconButton(const char* icon, const char* tooltip, bool enabled = true) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ui::WithAlpha(t.text, 0.15f));
    if (!enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::SmallButton(icon);
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", tooltip);
    return clicked && enabled;
}
}  // namespace

void ScriptEditorPanel::OpenFind(bool replace) {
    if (!IsActiveTabTextMode()) return;
    find_.open = true;
    find_.show_replace = find_.show_replace || replace;
    find_.focus_input = replace ? 2 : 1;
    // Start from the selected text when it is one line, as other editors do.
    const std::string selected = tabs_[active_tab_index_]->editor.Doc().SelectedText();
    if (!selected.empty() && selected.find('\n') == std::string::npos && selected.size() < sizeof(find_.query)) {
        std::strncpy(find_.query, selected.c_str(), sizeof(find_.query) - 1);
        find_.query[sizeof(find_.query) - 1] = 0;
    }
    find_.version = ~0ull;  // recount
    request_window_focus_ = true;
}

void ScriptEditorPanel::CloseFind() {
    find_.open = false;
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        tabs_[active_tab_index_]->editor.SetMarks({});
        tabs_[active_tab_index_]->editor.RequestFocus();
    }
}

void ScriptEditorPanel::UpdateFindMarks(CodeEditor& code) {
    const editor::Document& doc = code.Doc();
    const std::string key = std::string(find_.query) + (find_.case_sensitive ? "1" : "0") + (find_.whole_word ? "1" : "0") +
                            (find_.regex ? "1" : "0");
    if (find_.version != doc.Version() || key != find_.key || find_.doc_id != tabs_[active_tab_index_]->document_id) {
        find_.version = doc.Version();
        find_.key = key;
        find_.doc_id = tabs_[active_tab_index_]->document_id;
        find_.error.clear();
        find_.matches.clear();
        if (find_.query[0]) {
            textsearch::Options o;
            o.case_sensitive = find_.case_sensitive;
            o.whole_word = find_.whole_word;
            o.regex = find_.regex;
            const std::string text = doc.Text();
            for (const auto& m : textsearch::FindAll(text, find_.query, o, &find_.error)) {
                // Offsets to document positions (byte columns).
                editor::Pos a{0, 0};
                size_t line_start = 0;
                for (size_t i = 0; i < m.pos; ++i)
                    if (text[i] == '\n') {
                        ++a.line;
                        line_start = i + 1;
                    }
                a.col = static_cast<int>(m.pos - line_start);
                editor::Pos b = a;
                for (size_t i = m.pos; i < m.pos + m.len; ++i) {
                    if (text[i] == '\n') {
                        ++b.line;
                        b.col = 0;
                    } else {
                        ++b.col;
                    }
                }
                find_.matches.push_back({a, b});
            }
        }
    }
    // The current match is the one the selection covers.
    const editor::Selection& sel = doc.Primary();
    find_.current = -1;
    std::vector<CodeEditor::Mark> marks;
    marks.reserve(find_.matches.size());
    for (size_t i = 0; i < find_.matches.size(); ++i) {
        const auto& m = find_.matches[i];
        const bool current = sel.Start() == m.first && sel.End() == m.second;
        if (current) find_.current = static_cast<int>(i);
        marks.push_back({m.first, m.second, current});
    }
    code.SetMarks(std::move(marks));
}

void ScriptEditorPanel::FindStep(bool forward) {
    if (!find_.query[0]) return;
    if (forward) {
        if (!FindInEditor(find_.query, find_.case_sensitive, find_.whole_word, find_.regex)) return;
        // FindInEditor starts at the selection; step past a match already selected.
        const auto& code = tabs_[active_tab_index_]->editor;
        if (find_.current >= 0 && code.Doc().Primary().Start() == find_.matches[static_cast<size_t>(find_.current)].first)
            FindNext();
    } else {
        FindPreviousOf(find_.query, find_.case_sensitive, find_.whole_word, find_.regex);
    }
}

void ScriptEditorPanel::RenderFindWidget(CodeEditor& code, const ImVec2& code_min, float code_width, bool narrow) {
    const ui::Tokens& t = ui::CurrentTokens();
    UpdateFindMarks(code);

    const float width = narrow ? code_width : std::min(code_width - 24.0f, 460.0f);
    const ImVec2 pos = narrow ? code_min : ImVec2(code_min.x + code_width - width - 120.0f, code_min.y + 6.0f);
    const ImVec2 at(std::max(code_min.x, pos.x), pos.y);
    if (!narrow && find_.last_height > 0.0f) {
        // A soft shadow lifts it off the code (no outline).
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (int i = 1; i <= 3; ++i) {
            const float o = static_cast<float>(i) * 2.0f;
            dl->AddRectFilled(ImVec2(at.x - o + 2.0f, at.y + o), ImVec2(at.x + width + o - 2.0f, at.y + find_.last_height + o),
                              IM_COL32(0, 0, 0, 28), 6.0f + o);
        }
    }
    ImGui::SetCursorScreenPos(at);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1), 0.05f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, narrow ? 0.0f : 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 4.0f));
    ImGui::BeginChild("##find_widget", ImVec2(width, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    // Row 1: expand, query with its toggles, count, previous / next / close.
    if (IconButton(find_.show_replace ? ICON_FA_CHEVRON_DOWN : ICON_FA_CHEVRON_RIGHT,
                   find_.show_replace ? "Hide replace" : "Show replace (Ctrl+H)")) {
        find_.show_replace = !find_.show_replace;
    }
    ImGui::SameLine();
    const float toggles_w = ImGui::CalcTextSize("Aa ab .*").x + 34.0f;
    const float tail_w = ImGui::CalcTextSize("999 of 999").x + 3.0f * ImGui::GetFrameHeight() + 20.0f;
    const float input_w = std::max(80.0f, ImGui::GetContentRegionAvail().x - toggles_w - tail_w);
    if (find_.focus_input == 1) {
        ImGui::SetKeyboardFocusHere();
        find_.focus_input = 0;
    }
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ui::Mix(t.bg_window, ImVec4(0, 0, 0, 1), t.light ? 0.0f : 0.25f));
    if (!find_.error.empty()) ImGui::PushStyleColor(ImGuiCol_Text, t.error);
    ImGui::SetNextItemWidth(input_w);
    const bool enter = ImGui::InputTextWithHint("##find_query", "Find", find_.query, sizeof(find_.query),
                                                ImGuiInputTextFlags_EnterReturnsTrue);
    if (!find_.error.empty()) {
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", find_.error.c_str());
    }
    ImGui::PopStyleColor();
    if (enter) {
        FindStep(!ImGui::GetIO().KeyShift);
        ImGui::SetKeyboardFocusHere(-1);  // keep typing in the box
    }
    ImGui::SameLine();
    FindToggle("Aa", "Match case", &find_.case_sensitive);
    ImGui::SameLine(0.0f, 2.0f);
    FindToggle("ab", "Whole word", &find_.whole_word);
    ImGui::SameLine(0.0f, 2.0f);
    FindToggle(".*", "Regular expression", &find_.regex);
    ImGui::SameLine();
    char count[48];
    if (!find_.query[0]) std::snprintf(count, sizeof(count), " ");
    else if (find_.matches.empty()) std::snprintf(count, sizeof(count), "No results");
    else if (find_.current >= 0) std::snprintf(count, sizeof(count), "%d of %d", find_.current + 1, static_cast<int>(find_.matches.size()));
    else std::snprintf(count, sizeof(count), "%d results", static_cast<int>(find_.matches.size()));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(find_.matches.empty() && find_.query[0] ? t.warning : t.text_dim, "%s", count);
    ImGui::SameLine();
    const bool any = !find_.matches.empty();
    if (IconButton(ICON_FA_ARROW_UP, "Previous match (Shift+Enter, Shift+F3)", any)) FindStep(false);
    ImGui::SameLine(0.0f, 0.0f);
    if (IconButton(ICON_FA_ARROW_DOWN, "Next match (Enter, F3)", any)) FindStep(true);
    ImGui::SameLine(0.0f, 0.0f);
    bool close = IconButton(ICON_FA_XMARK, "Close (Escape)");

    // Row 2: replace.
    if (find_.show_replace) {
        ImGui::Dummy(ImVec2(ImGui::GetFrameHeight() - 4.0f, 0.0f));
        ImGui::SameLine();
        if (find_.focus_input == 2) {
            ImGui::SetKeyboardFocusHere();
            find_.focus_input = 0;
        }
        const float buttons_w = ui::ButtonWidth("Replace", ui::ButtonSize::Small) + ui::ButtonWidth("Replace all", ui::ButtonSize::Small) + 8.0f;
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ui::Mix(t.bg_window, ImVec4(0, 0, 0, 1), t.light ? 0.0f : 0.25f));
        ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - buttons_w));
        const bool replace_enter = ImGui::InputTextWithHint("##find_replace", "Replace", find_.replacement,
                                                            sizeof(find_.replacement), ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        if (ui::SecondaryButton("Replace", any, "Nothing to replace") || (replace_enter && any)) {
            Replace(find_.query, find_.replacement, find_.case_sensitive, find_.whole_word, find_.regex);
            if (replace_enter) ImGui::SetKeyboardFocusHere(-1);
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Replace all", any, "Nothing to replace")) {
            ReplaceAll(find_.query, find_.replacement, find_.case_sensitive, find_.whole_word, find_.regex);
        }
    }

    // Keys while the widget has focus.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) close = true;
        if (ImGui::IsKeyPressed(ImGuiKey_F3)) FindStep(!ImGui::GetIO().KeyShift);
    }
    find_.last_height = ImGui::GetWindowHeight();
    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    if (close) CloseFind();
}

}  // namespace cyxwiz
