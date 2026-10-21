// SQL Workbench editing (TOFIX146 step 4b-2): Format and Compare with saved.
// The editor itself, its find bar and completion are the SQL editor pane
// (gui/sql_editor_pane.cpp, TOFIX148).
#include "sql_workbench.h"

#include "editor_fonts.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqledit = cyxwiz::sqledit;
using namespace sqlwb;
}  // namespace

// ---- Format and Compare ------------------------------------------------------

void SqlWorkbenchDialog::FormatDraft() {
    if (pane_.Format()) AddMessage("Formatted: keywords upper case, trailing spaces removed (Ctrl+Z undoes it).", false);
    else AddMessage("The query is already formatted (keywords upper case, no trailing spaces).", false);
}

void SqlWorkbenchDialog::RenderCompare() {
    if (!compare_open_) return;
    const ui::Tokens& t = ui::CurrentTokens();
    if (compare_version_ != pane_.Editor().Doc().Version() || compare_saved_ != saved_query_ || compare_shown_view_ != compare_view_) {
        compare_version_ = pane_.Editor().Doc().Version();
        compare_saved_ = saved_query_;
        compare_shown_view_ = compare_view_;
        compare_lines_ = sqledit::DiffLines(saved_query_, pane_.Text());
        // Changes with 3 lines around them; the rest folds to a count.
        compare_rows_.clear();
        const int count = static_cast<int>(compare_lines_.size());
        std::vector<bool> keep(static_cast<size_t>(count), compare_view_ == 1);
        for (int i = 0; i < count; ++i) {
            if (compare_lines_[static_cast<size_t>(i)].kind == sqledit::DiffKind::Same) continue;
            for (int k = std::max(0, i - 3); k <= std::min(count - 1, i + 3); ++k) keep[static_cast<size_t>(k)] = true;
        }
        for (int i = 0; i < count;) {
            if (keep[static_cast<size_t>(i)]) {
                compare_rows_.push_back(i++);
                continue;
            }
            int skipped = 0;
            while (i < count && !keep[static_cast<size_t>(i)]) {
                ++skipped;
                ++i;
            }
            compare_rows_.push_back(-skipped);
        }
    }
    ImGui::SetNextWindowSize(ImVec2(980.0f, 640.0f), ImGuiCond_FirstUseEver);
    const std::string title = "Compare with saved - " + step_name_ + "###sql_compare_" + std::to_string(node_id_);
    if (ImGui::Begin(title.c_str(), &compare_open_, ImGuiWindowFlags_NoCollapse)) {
        int added = 0, removed = 0;
        for (const auto& line : compare_lines_) {
            added += line.kind == sqledit::DiffKind::Added ? 1 : 0;
            removed += line.kind == sqledit::DiffKind::Removed ? 1 : 0;
        }
        Coloured("+" + std::to_string(added), t.success);
        ImGui::SameLine();
        Coloured("-" + std::to_string(removed), t.caution);
        ImGui::SameLine();
        Dim(added + removed == 0 ? "the draft is the step's saved query" : "lines, the step's saved query -> the draft");
        static const char* const kViews[] = {"Changes only", "Whole query"};
        const float views_w = ImGui::CalcTextSize("Changes onlyWhole query").x + 48.0f;
        const float reset_w = ui::ButtonWidth("Reset to saved", ui::ButtonSize::Small);
        AlignRight(views_w + reset_w + ImGui::GetStyle().ItemSpacing.x);
        ui::SegmentedControl("##compare_view", kViews, 2, &compare_view_);
        ImGui::SameLine();
        if (ui::SecondaryButton("Reset to saved", added + removed > 0, "Nothing to reset")) Reset();
        ImFont* mono = cyxwiz::gui::GetCodeFont();
        if (ImGui::BeginChild("##compare_lines", ImGui::GetContentRegionAvail(), ImGuiChildFlags_None,
                              ImGuiWindowFlags_HorizontalScrollbar)) {
            if (mono) ImGui::PushFont(mono);
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(compare_rows_.size()));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    const int i = compare_rows_[static_cast<size_t>(row)];
                    if (i < 0) {
                        Dim("          ... " + std::to_string(-i) + " unchanged line" + (i == -1 ? "" : "s"));
                        continue;
                    }
                    const auto& line = compare_lines_[static_cast<size_t>(i)];
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    if (line.kind != sqledit::DiffKind::Same) {
                        const ImVec4 tint = ui::WithAlpha(line.kind == sqledit::DiffKind::Added ? t.success : t.caution, 0.14f);
                        dl->AddRectFilled(at, ImVec2(at.x + ImGui::GetContentRegionAvail().x + ImGui::GetScrollMaxX() + 2000.0f,
                                                     at.y + ImGui::GetTextLineHeightWithSpacing()),
                                          ui::ToU32(tint));
                    }
                    char numbers[32];
                    std::snprintf(numbers, sizeof(numbers), "%4s %4s ",
                                  line.old_line ? std::to_string(line.old_line).c_str() : "",
                                  line.new_line ? std::to_string(line.new_line).c_str() : "");
                    Dim(numbers);
                    ImGui::SameLine(0.0f, 0.0f);
                    const char* sign = line.kind == sqledit::DiffKind::Added ? "+ " : line.kind == sqledit::DiffKind::Removed ? "- " : "  ";
                    Coloured(sign + line.text, line.kind == sqledit::DiffKind::Added
                                                   ? t.success
                                                   : line.kind == sqledit::DiffKind::Removed ? t.caution : t.text);
                }
            }
            if (mono) ImGui::PopFont();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

}  // namespace gui
