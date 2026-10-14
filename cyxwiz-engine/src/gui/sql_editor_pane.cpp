// SQL editor pane (TOFIX148 step 2; was the SQL Workbench editing of TOFIX146
// step 4b-2): the editor with its find bar (Ctrl+F), completion (typing or
// Ctrl+Space) and Format.
#include "sql_editor_pane.h"

#include "../core/text_search.h"
#include "editor_fonts.h"
#include "icons.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>
#include <imgui_internal.h>  // BringWindowToDisplayFront: the list stays above the editor

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqledit = cyxwiz::sqledit;
using namespace sqlwb;

constexpr int kCompletionRows = 10;

const char* KindLabel(sqledit::CompletionKind kind) {
    switch (kind) {
        case sqledit::CompletionKind::Column: return "column";
        case sqledit::CompletionKind::Part: return "part";
        case sqledit::CompletionKind::Name: return "name";
        case sqledit::CompletionKind::Function: return "function";
        default: return "keyword";
    }
}

cyxwiz::editor::Pos DocumentEnd(const cyxwiz::editor::Document& doc) {
    const int last = std::max(0, doc.LineCount() - 1);
    return {last, static_cast<int>(doc.Line(last).size())};
}
}  // namespace

SqlEditorPane::SqlEditorPane(std::string id) : id_(std::move(id)) {
    editor_.SetLanguage(cyxwiz::editor::Language::Sql);
    editor_.SetTabSize(4);
}

void SqlEditorPane::Refresh() {
    if (editor_.Doc().Version() == outline_version_) return;
    outline_version_ = editor_.Doc().Version();
    text_ = editor_.GetText();
    outline_ = cyxwiz::sqlstep::OutlineSqlQuery(text_);
}

// ---- Find ---------------------------------------------------------------

void SqlEditorPane::OpenFind() {
    find_.open = true;
    find_.focus = true;
    // A one-line selection becomes the search text.
    const std::string selected = editor_.Doc().SelectedText();
    if (!selected.empty() && selected.find('\n') == std::string::npos && selected.size() < sizeof(find_.query)) {
        std::snprintf(find_.query, sizeof(find_.query), "%s", selected.c_str());
    }
    find_.version = ~0ull;
}

void SqlEditorPane::CloseFind() {
    find_.open = false;
    find_.matches.clear();
    editor_.SetMarks({});
}

void SqlEditorPane::UpdateFind() {
    const auto& doc = editor_.Doc();
    const std::string key = std::string(find_.query) + (find_.case_sensitive ? "1" : "0") +
                            (find_.whole_word ? "1" : "0") + (find_.regex ? "1" : "0");
    if (find_.version != doc.Version() || key != find_.key) {
        find_.version = doc.Version();
        find_.key = key;
        find_.error.clear();
        find_.matches.clear();
        if (find_.query[0]) {
            cyxwiz::textsearch::Options options;
            options.case_sensitive = find_.case_sensitive;
            options.whole_word = find_.whole_word;
            options.regex = find_.regex;
            const std::string text = doc.Text();
            // Offsets to positions in one pass over the text.
            cyxwiz::editor::Pos at{0, 0};
            size_t offset = 0;
            const auto advance = [&](size_t to) {
                for (; offset < to; ++offset) {
                    if (text[offset] == '\n') {
                        ++at.line;
                        at.col = 0;
                    } else {
                        ++at.col;
                    }
                }
            };
            for (const auto& match : cyxwiz::textsearch::FindAll(text, find_.query, options, &find_.error)) {
                advance(match.pos);
                const cyxwiz::editor::Pos start = at;
                advance(match.pos + match.len);
                find_.matches.push_back({start, at});
            }
        }
    }
    // The current match is the one the selection covers.
    const auto& selection = doc.Primary();
    find_.current = -1;
    std::vector<cyxwiz::CodeEditor::Mark> marks;
    marks.reserve(find_.matches.size());
    for (size_t i = 0; i < find_.matches.size(); ++i) {
        const auto& [a, b] = find_.matches[i];
        const bool current = selection.Start() == a && selection.End() == b;
        if (current) find_.current = static_cast<int>(i);
        marks.push_back({a, b, current});
    }
    editor_.SetMarks(std::move(marks));
}

void SqlEditorPane::FindStep(bool forward) {
    const int count = static_cast<int>(find_.matches.size());
    if (count == 0) return;
    const auto head = editor_.Doc().Primary().head;
    int pick = -1;
    if (find_.current >= 0) {
        pick = (find_.current + (forward ? 1 : count - 1)) % count;
    } else if (forward) {
        // The first match at or after the cursor, else wrap to the first.
        pick = 0;
        for (int i = 0; i < count; ++i) {
            if (head <= find_.matches[static_cast<size_t>(i)].first) {
                pick = i;
                break;
            }
        }
    } else {
        // The last match before the cursor, else wrap to the last.
        pick = count - 1;
        for (int i = count; i-- > 0;) {
            if (find_.matches[static_cast<size_t>(i)].second <= head) {
                pick = i;
                break;
            }
        }
    }
    cyxwiz::editor::Selection selection;
    selection.anchor = find_.matches[static_cast<size_t>(pick)].first;
    selection.head = find_.matches[static_cast<size_t>(pick)].second;
    editor_.Doc().SetSelections({selection});
    editor_.ScrollToCursor();
    find_.version = ~0ull;  // recompute the current mark
}

void SqlEditorPane::RenderFindBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushID(("##find_" + id_).c_str());
    Coloured(ICON_FA_MAGNIFYING_GLASS, t.text_dim);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(280.0f);
    if (find_.focus) {
        ImGui::SetKeyboardFocusHere();
        find_.focus = false;
    }
    const bool entered = ImGui::InputTextWithHint("##query", "Find in the query", find_.query, sizeof(find_.query),
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    const bool input_focused = ImGui::IsItemFocused() || ImGui::IsItemActive();
    if (entered) {
        FindStep(!ImGui::GetIO().KeyShift);
        find_.focus = true;  // keep typing after Enter
    }
    if (input_focused && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        CloseFind();
        editor_.RequestFocus();
        ImGui::PopID();
        return;
    }
    ImGui::SameLine();
    if (ui::GhostButton("Aa", true, nullptr, find_.case_sensitive)) find_.case_sensitive = !find_.case_sensitive;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Match case");
    ImGui::SameLine();
    if (ui::GhostButton("W", true, nullptr, find_.whole_word)) find_.whole_word = !find_.whole_word;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Whole word");
    ImGui::SameLine();
    if (ui::GhostButton(".*", true, nullptr, find_.regex)) find_.regex = !find_.regex;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Regular expression");
    ImGui::SameLine();
    if (!find_.error.empty()) {
        Coloured(find_.error, t.caution);
    } else if (!find_.query[0]) {
        Dim("Enter next, Shift+Enter previous, Esc close");
    } else if (find_.matches.empty()) {
        Coloured("no matches", t.warning);
    } else {
        const std::string count = (find_.current >= 0 ? std::to_string(find_.current + 1) + " of " : "") +
                                  std::to_string(find_.matches.size()) +
                                  (find_.current >= 0 ? "" : find_.matches.size() == 1 ? " match" : " matches");
        Dim(count);
    }
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_CHEVRON_UP, !find_.matches.empty())) FindStep(false);
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_CHEVRON_DOWN, !find_.matches.empty())) FindStep(true);
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_XMARK)) CloseFind();
    ImGui::PopID();
}

// ---- Completion -----------------------------------------------------------

void SqlEditorPane::UpdateCompletion(bool forced) {
    const auto& doc = editor_.Doc();
    const auto& selection = doc.Primary();
    if (doc.CursorCount() != 1 || !selection.Empty()) {
        completion_open_ = false;
        return;
    }
    const auto head = selection.head;
    const std::string prefix = sqledit::WordBefore(doc.Line(head.line), static_cast<size_t>(head.col));
    if (!forced && !completion_open_ && prefix.size() < 2) {
        completion_open_ = false;
        return;
    }
    completion_items_ = sqledit::SqlCompletions(prefix, columns_, text_, outline_);
    completion_open_ = !completion_items_.empty();
    completion_start_ = {head.line, head.col - static_cast<int>(prefix.size())};
    completion_selected_ = std::clamp(completion_selected_, 0, std::max(0, static_cast<int>(completion_items_.size()) - 1));
    completion_version_ = doc.Version();
}

void SqlEditorPane::AcceptCompletion() {
    if (!completion_open_ || completion_items_.empty()) return;
    const auto head = editor_.Doc().Primary().head;
    editor_.Doc().Replace(completion_start_, head, completion_items_[static_cast<size_t>(completion_selected_)].text);
    completion_open_ = false;
    completion_version_ = editor_.Doc().Version();
}

void SqlEditorPane::RenderCompletion() {
    if (!completion_open_) return;
    const ui::Tokens& t = ui::CurrentTokens();
    ImFont* mono = cyxwiz::gui::GetCodeFont();
    const ImVec2 caret = editor_.CursorScreenPos();
    ImGui::SetNextWindowPos(ImVec2(caret.x, caret.y + line_height_));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, t.bg_panel);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                                   ImGuiWindowFlags_NoDocking;
    const std::string id = "##sql_completion_" + id_;
    if (ImGui::Begin(id.c_str(), nullptr, flags)) {
        ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
        const int count = static_cast<int>(completion_items_.size());
        const int first = std::clamp(completion_selected_ - kCompletionRows + 1, 0, std::max(0, count - kCompletionRows));
        ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        for (int i = first; i < std::min(count, first + kCompletionRows); ++i) {
            const auto& item = completion_items_[static_cast<size_t>(i)];
            ImGui::PushID(i);
            if (mono) ImGui::PushFont(mono);
            if (ImGui::Selectable(item.text.c_str(), i == completion_selected_, ImGuiSelectableFlags_None,
                                  ImVec2(260.0f, 0.0f))) {
                completion_selected_ = i;
                AcceptCompletion();
            }
            if (mono) ImGui::PopFont();
            ImGui::SameLine(270.0f);
            Dim(KindLabel(item.kind));
            ImGui::PopID();
        }
        ImGui::PopStyleColor(2);
        if (count > kCompletionRows) Dim(std::to_string(count) + " matches; keep typing to narrow");
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

// ---- The editor ---------------------------------------------------------------

void SqlEditorPane::Render(float height) {
    Refresh();
    // Ctrl+F (the window has focus) opens the find bar.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        OpenFind();
    }
    if (find_.open) {
        RenderFindBar();
        height -= ImGui::GetFrameHeightWithSpacing();
    }

    // Keys for the completion list stay out of the editor for this frame.
    bool keys_taken = false;
    if (completion_open_) {
        const int count = static_cast<int>(completion_items_.size());
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            completion_selected_ = (completion_selected_ + 1) % std::max(1, count);
            keys_taken = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            completion_selected_ = (completion_selected_ - 1 + std::max(1, count)) % std::max(1, count);
            keys_taken = true;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
            if (!ImGui::GetIO().KeyCtrl) {
                AcceptCompletion();
                keys_taken = true;
            }
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            completion_open_ = false;
            keys_taken = true;
        }
    }
    const bool ctrl_space = ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Space, false) && editor_.IsFocused();
    if (ctrl_space) keys_taken = true;

    const uint64_t version_before = editor_.Doc().Version();
    const auto head_before = editor_.Doc().Primary().head;
    editor_.SetKeyboardEnabled(!keys_taken);
    ImFont* mono = cyxwiz::gui::GetCodeFont();
    if (mono) ImGui::PushFont(mono);
    line_height_ = cyxwiz::CodeEditor::LineHeightFor(ImGui::GetFontSize());
    editor_.Render(("##editor_" + id_).c_str(), ImVec2(ImGui::GetContentRegionAvail().x, height));
    if (mono) ImGui::PopFont();
    editor_.SetKeyboardEnabled(true);

    if (find_.open) UpdateFind();

    // Completion: opened by typing a word or Ctrl+Space; it follows the
    // word and closes when the cursor leaves it.
    const auto& doc = editor_.Doc();
    const auto head = doc.Primary().head;
    if (ctrl_space) {
        UpdateCompletion(true);
    } else if (doc.Version() != version_before && editor_.IsFocused()) {
        const bool typed_word = head.col > 0 &&
                                head.line < doc.LineCount() &&
                                (std::isalnum(static_cast<unsigned char>(doc.Line(head.line)[static_cast<size_t>(head.col) - 1])) ||
                                 doc.Line(head.line)[static_cast<size_t>(head.col) - 1] == '_');
        if (typed_word) UpdateCompletion(false);
        else completion_open_ = false;
    } else if (completion_open_ && (head.line != completion_start_.line || head.col < completion_start_.col ||
                                    (head != head_before && doc.Version() == version_before))) {
        completion_open_ = false;
    }
    if (!editor_.IsFocused() && !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) completion_open_ = false;
    RenderCompletion();
}

// ---- Format -------------------------------------------------------------------

bool SqlEditorPane::Format() {
    auto& doc = editor_.Doc();
    const std::string text = doc.Text();
    const std::string formatted = sqledit::FormatSqlKeywords(text);
    if (formatted == text) return false;
    doc.Replace({0, 0}, DocumentEnd(doc), formatted);
    return true;
}

}  // namespace gui
