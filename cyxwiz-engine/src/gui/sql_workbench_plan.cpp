// SQL Workbench Plan tab (TOFIX146, mockup v5 board 4): Explain (estimate)
// and Run with timing, summary cards, where the time goes, operators in the
// order rows flow (Tree, Diagram, DuckDB's Text) and the selected operator's
// details.
#include "sql_workbench.h"

#include "../core/file_dialogs.h"
#include "editor_fonts.h"
#include "icons.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqlplan = cyxwiz::sqlplan;
using cyxwiz::sqlstep::FormatCount;
using namespace sqlwb;

constexpr double kHotShare = 0.10;  // operators at or over 10% of the time are marked

std::string Percent(double share) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.1f%%", share * 100.0);
    return text;
}

bool EstimateOff(const sqlplan::PlanOperator& op) {
    if (op.estimated < 0 || op.rows < 0) return false;
    const double actual = static_cast<double>(std::max<int64_t>(op.rows, 1));
    const double guess = static_cast<double>(std::max<int64_t>(op.estimated, 1));
    return actual / guess >= 100.0 || guess / actual >= 100.0;
}

std::string PartsText(const sqlplan::PlanOperator& op) {
    std::string text;
    for (const auto& part : op.parts) text += (text.empty() ? "" : ", ") + part;
    return text.empty() ? std::string("-") : text;
}

// A card: a value over its label.
void Card(const char* id, const std::string& value, const std::string& label, const ImVec4& colour, float width) {
    const auto& t = ui::CurrentTokens();
    if (BeginPanel(id, ImVec2(width, 0.0f), t.bg_raised, ImGuiChildFlags_AutoResizeY)) {
        ColouredWrapped(value, colour);
        DimWrapped(label);
    }
    EndPanel();
}
}  // namespace

int SqlWorkbenchDialog::PartIndex(const std::string& name) const {
    for (size_t i = 0; i < pane_.Outline().parts.size(); ++i) {
        if (pane_.Outline().parts[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

void SqlWorkbenchDialog::RenderPlan() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool pipeline = node_editor_ && node_editor_->IsDataPipelineRunning();
    const bool can = !run_ && !pipeline;
    const char* busy = run_ ? "A query is running" : "A pipeline run is in progress";

    // Actions, mode note, views.
    const bool estimate_shown = !plan_.ops.empty() && !plan_.timed;
    if (estimate_shown ? ui::PrimaryButton("Explain (estimate)", can, busy, ui::ButtonSize::Small)
                       : ui::SecondaryButton("Explain (estimate)", can, busy)) {
        StartRun(RunKind::Explain);
    }
    if (ImGui::IsItemHovered() && can) ImGui::SetTooltip("DuckDB plans the query without running it (instant)");
    ImGui::SameLine();
    if (!estimate_shown ? ui::PrimaryButton(ICON_FA_PLAY " Run with timing", can, busy, ui::ButtonSize::Small)
                        : ui::SecondaryButton(ICON_FA_PLAY " Run with timing", can, busy)) {
        StartRun(RunKind::Profile);
    }
    if (ImGui::IsItemHovered() && can) {
        ImGui::SetTooltip("Runs the query once, read-only, and times every operator (EXPLAIN ANALYZE)");
    }
    ImGui::SameLine();
    if (plan_.ops.empty()) {
        Dim("Explain shows how DuckDB will run the query; Run with timing shows where the time goes.");
    } else {
        std::string note = plan_.timed ? "Timed run at " + plan_when_ + "  EXPLAIN ANALYZE, same inputs as Run step"
                                       : "Estimate at " + plan_when_ + ": DuckDB planned the query without running it";
        if (plan_draft_ != pane_.Text()) note += "  (an older draft)";
        Dim(note);
    }
    static const char* const kViews[] = {"Tree", "Diagram", "Text"};
    const float views_w = ImGui::CalcTextSize("TreeDiagramText").x + 3 * 24.0f;
    const float copy_w = ui::ButtonWidth("Copy plan", ui::ButtonSize::Small);
    const float export_w = ui::ButtonWidth("Export JSON", ui::ButtonSize::Small);
    AlignRight(views_w + copy_w + export_w + ImGui::GetStyle().ItemSpacing.x * 2.0f);
    ui::SegmentedControl("##plan_view", kViews, 3, &plan_view_);
    ImGui::SameLine();
    if (ui::SecondaryButton("Copy plan", !plan_text_.empty(), "Explain the query first")) {
        ImGui::SetClipboardText(plan_text_.c_str());
        AddMessage("Copied DuckDB's text plan.", false);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export JSON", !plan_json_.empty(), "Explain the query first")) {
        const cyxwiz::FileDialogs::FilterList filters = {{"JSON", "json"}};
        const std::string name = step_name_ + (plan_.timed ? "_timed_plan.json" : "_plan.json");
        if (const auto path = cyxwiz::FileDialogs::SaveFile("Export plan", filters, nullptr, name.c_str())) {
            std::ofstream file(*path, std::ios::binary);
            file << plan_json_;
            AddMessage(file ? "Exported the plan to " + *path : "Could not write " + *path, !file);
        }
    }

    // A plan run in progress.
    if (run_ && (running_kind_ == RunKind::Explain || running_kind_ == RunKind::Profile)) {
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        if (BeginPanel("##plan_running", ImVec2(0.0f, 0.0f), t.bg_raised, ImGuiChildFlags_AutoResizeY)) {
            char elapsed[32];
            std::snprintf(elapsed, sizeof(elapsed), "%.1f s so far", ImGui::GetTime() - running_since_);
            Coloured(ICON_FA_SPINNER, t.running);
            ImGui::SameLine();
            ImGui::TextUnformatted(running_kind_ == RunKind::Profile ? "Running the query with timing..."
                                                                       : "Planning the query...");
            ImGui::SameLine();
            Dim(std::string(elapsed) + (plan_.timed && plan_.operator_seconds > 0
                                            ? "  the last timed run took " + sqlplan::FormatSeconds(plan_.operator_seconds)
                                            : "") +
                "  read-only, nothing is saved");
            ImGui::SameLine();
            if (ui::LinkButton("Stop")) StopRun();
        }
        EndPanel();
    }

    if (!plan_.error.empty()) {
        ColouredWrapped("The plan could not be read: " + plan_.error, t.caution);
        return;
    }
    if (plan_.ops.empty()) return;

    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    RenderPlanSummary();
    if (plan_.timed) RenderPlanTimeBar();
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));

    // Operators (left) and the selected one's details (right).
    const float details_w = std::clamp(ImGui::GetContentRegionAvail().x * 0.36f, 380.0f, 600.0f);
    const float body_h = ImGui::GetContentRegionAvail().y;
    if (BeginPanel("##plan_ops", ImVec2(ImGui::GetContentRegionAvail().x - details_w - ImGui::GetStyle().ItemSpacing.x,
                                        body_h),
                   t.bg_window)) {
        if (plan_view_ == 0) RenderPlanTree();
        else if (plan_view_ == 1) RenderPlanDiagram();
        else RenderPlanText();
    }
    EndPanel();
    ImGui::SameLine();
    if (BeginPanel("##plan_details", ImVec2(details_w, body_h), t.bg_window)) RenderPlanDetails();
    EndPanel();
}

void SqlWorkbenchDialog::RenderPlanSummary() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const int count = plan_.timed ? 5 : 3;
    const float width = (ImGui::GetContentRegionAvail().x - spacing * static_cast<float>(count - 1)) /
                        static_cast<float>(count);
    const std::string operators = std::to_string(plan_.ops.size());
    const std::string internal = "operators (" + std::to_string(plan_.internal_count) + " internal)";
    int64_t input_rows = 0;
    for (const auto& input : inputs_) input_rows += input.table ? input.table->num_rows() : 0;
    if (plan_.timed) {
        const auto& slow = plan_.ops[static_cast<size_t>(std::max(0, plan_.slowest))];
        const double share = plan_.operator_seconds > 0 ? slow.seconds / plan_.operator_seconds : 0.0;
        Card("##c1", sqlplan::FormatSeconds(plan_.operator_seconds), "in operators", t.text_bright, width);
        ImGui::SameLine();
        Card("##c2", FormatCount(plan_.rows_out), "rows out, from " + FormatCount(input_rows) + " input rows",
             t.text_bright, width);
        ImGui::SameLine();
        Card("##c3", operators, internal, t.text_bright, width);
        ImGui::SameLine();
        Card("##c4", slow.name + " " + sqlplan::FormatSeconds(slow.seconds), "slowest operator, " + Percent(share),
             share >= kHotShare ? t.caution : t.text_bright, width);
        ImGui::SameLine();
        Card("##c5", std::to_string(plan_.estimates_off), "estimates 100x off",
             plan_.estimates_off > 0 ? t.warning : t.text_bright, width);
    } else {
        Card("##c1", "not run", "estimate only: nothing executed", t.text_bright, width);
        ImGui::SameLine();
        Card("##c2", operators, internal, t.text_bright, width);
        ImGui::SameLine();
        Card("##c3", FormatCount(input_rows), "input rows; Run with timing for rows and time", t.text_bright, width);
    }
}

void SqlWorkbenchDialog::RenderPlanTimeBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (plan_.operator_seconds <= 0.0) return;
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    ImGui::TextUnformatted("Where the time goes");
    ImGui::SameLine();
    Dim("by operator, " + sqlplan::FormatSeconds(plan_.operator_seconds) + " in operators; click a part to select it");
    // The four slowest operators over 3% get their own segment.
    std::vector<int> order(plan_.ops.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return plan_.ops[static_cast<size_t>(a)].seconds > plan_.ops[static_cast<size_t>(b)].seconds; });
    std::vector<int> shown;
    for (const int i : order) {
        if (shown.size() == 4 || plan_.ops[static_cast<size_t>(i)].seconds / plan_.operator_seconds < 0.03) break;
        shown.push_back(i);
    }
    std::sort(shown.begin(), shown.end());
    const ImVec4 colours[4] = {t.caution, t.warning, t.accent_text, t.accent};
    std::vector<int> by_time = shown;
    std::stable_sort(by_time.begin(), by_time.end(),
                     [&](int a, int b) { return plan_.ops[static_cast<size_t>(a)].seconds > plan_.ops[static_cast<size_t>(b)].seconds; });
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetFrameHeight();
    double rest = plan_.operator_seconds;
    float x = origin.x;
    const auto segment = [&](const std::string& label, double seconds, const ImVec4& fill, const ImVec4& ink, int pick) {
        const float w = std::max(2.0f, static_cast<float>(seconds / plan_.operator_seconds) * width - 2.0f);
        ImGui::SetCursorScreenPos(ImVec2(x, origin.y));
        ImGui::PushID(pick);
        if (ImGui::InvisibleButton("##seg", ImVec2(w, height)) && pick >= 0) plan_picked_ = pick;
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        dl->AddRectFilled(ImVec2(x, origin.y), ImVec2(x + w, origin.y + height),
                          ui::ToU32(hovered ? ui::Mix(fill, t.text_bright, 0.15f) : fill), 4.0f);
        const ImVec2 size = ImGui::CalcTextSize(label.c_str());
        if (size.x + 12.0f < w) {
            dl->AddText(ImVec2(x + 6.0f, origin.y + (height - size.y) * 0.5f), ui::ToU32(ink), label.c_str());
        }
        if (hovered) ImGui::SetTooltip("%s", label.c_str());
        x += w + 2.0f;
    };
    for (const int i : shown) {
        const auto& op = plan_.ops[static_cast<size_t>(i)];
        const size_t rank = static_cast<size_t>(std::find(by_time.begin(), by_time.end(), i) - by_time.begin());
        const std::string label = op.name + " " + PartsText(op) + "  " + sqlplan::FormatSeconds(op.seconds) + "  " +
                                  Percent(op.seconds / plan_.operator_seconds);
        segment(label, op.seconds, colours[std::min<size_t>(rank, 3)], t.bg_window, i);
        rest -= op.seconds;
    }
    if (rest > 0.0) {
        const size_t others = plan_.ops.size() - shown.size();
        segment(std::to_string(others) + " other operators  " + Percent(rest / plan_.operator_seconds), rest,
                t.bg_raised, t.text, -1);
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + height + ImGui::GetStyle().ItemSpacing.y));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

void SqlWorkbenchDialog::RenderPlanTree() {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    // Selection and hover in the theme's own tones (not the header colour).
    ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
    if (!ImGui::BeginTable("##plan_tree", 5, flags, ImGui::GetContentRegionAvail())) {
        ImGui::PopStyleColor(3);
        return;
    }
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000").x);
    ImGui::TableSetupColumn("Operator, in the order rows flow", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableSetupColumn("Part", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Rows", ImGuiTableColumnFlags_WidthStretch, 0.9f);
    ImGui::TableSetupColumn(plan_.timed ? "Time and share" : "Time", ImGuiTableColumnFlags_WidthStretch, 1.9f);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(plan_.ops.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const auto& op = plan_.ops[static_cast<size_t>(i)];
            const double share = plan_.timed && plan_.operator_seconds > 0 ? op.seconds / plan_.operator_seconds : 0.0;
            const bool hot = plan_.timed && share >= kHotShare;
            ImGui::PushID(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable("##op", plan_picked_ == i,
                                  ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
                plan_picked_ = i;
            }
            ImGui::SameLine(0.0f, 0.0f);
            Dim(std::to_string(i + 1));
            ImGui::TableNextColumn();
            ImGui::Indent(static_cast<float>(op.depth) * 18.0f);
            ImFont* mono = cyxwiz::gui::GetEditorMonoFont(1.0f);
            if (mono) ImGui::PushFont(mono);
            Coloured(op.name, op.internal ? t.text_faint : (hot ? t.caution : t.text_bright));
            if (mono) ImGui::PopFont();
            ImGui::SameLine();
            Dim(op.summary);
            if (plan_.timed && EstimateOff(op)) {
                ImGui::SameLine();
                Coloured(ICON_FA_TRIANGLE_EXCLAMATION, t.warning);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("DuckDB estimated %s rows; %s came", FormatCount(op.estimated).c_str(),
                                      FormatCount(op.rows).c_str());
                }
            }
            ImGui::Unindent(static_cast<float>(op.depth) * 18.0f);
            ImGui::TableNextColumn();
            Coloured(op.internal ? std::string() : PartsText(op), t.accent_text);
            ImGui::TableNextColumn();
            if (plan_.timed) {
                ImGui::TextUnformatted(FormatCount(op.rows).c_str());
                if (op.estimated >= 0) {
                    ImGui::SameLine();
                    Dim("est " + FormatCount(op.estimated));
                }
            } else {
                Dim(op.estimated >= 0 ? "est " + FormatCount(op.estimated) : "est -");
            }
            ImGui::TableNextColumn();
            if (plan_.timed) {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float bar_w = std::max(40.0f, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("00.00 s  100.0%").x - 8.0f);
                const float bar_h = 6.0f;
                const float y = at.y + (ImGui::GetTextLineHeight() - bar_h) * 0.5f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + bar_w, y + bar_h), ui::ToU32(t.bg_raised), 3.0f);
                dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + std::max(2.0f, bar_w * static_cast<float>(share)), y + bar_h),
                                  ui::ToU32(hot ? t.caution : t.accent), 3.0f);
                ImGui::Dummy(ImVec2(bar_w, ImGui::GetTextLineHeight()));
                ImGui::SameLine();
                Coloured(sqlplan::FormatSeconds(op.seconds) + "  " + Percent(share), hot ? t.caution : t.text);
            } else {
                Dim("-");
            }
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
    ImGui::PopStyleColor(3);
}

void SqlWorkbenchDialog::RenderPlanDiagram() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ImGui::BeginChild("##plan_diagram", ImGui::GetContentRegionAvail())) {
        ImGui::EndChild();
        return;
    }
    DimWrapped("Each box is one operator, in the order rows flow; the hotter its colour, the more time it takes.");
    const float box_w = 168.0f;
    const float arrow_w = ImGui::CalcTextSize(" " ICON_FA_ARROW_RIGHT " ").x;
    const float avail = ImGui::GetContentRegionAvail().x;
    float x = 0.0f;
    for (size_t i = 0; i < plan_.ops.size(); ++i) {
        const auto& op = plan_.ops[i];
        const double share = plan_.timed && plan_.operator_seconds > 0 ? op.seconds / plan_.operator_seconds : 0.0;
        if (i > 0) {
            if (x + arrow_w + box_w > avail) {
                x = 0.0f;
            } else {
                ImGui::SameLine();
                Coloured(ICON_FA_ARROW_RIGHT, t.text_faint);
                ImGui::SameLine();
            }
        }
        ImGui::PushID(static_cast<int>(i));
        const ImVec4 heat = ui::Mix(t.bg_raised, t.caution, static_cast<float>(std::min(1.0, share * 2.5)));
        const ImVec4 fill = plan_picked_ == static_cast<int>(i) ? t.selection : heat;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, fill);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
        const float box_h = ImGui::GetTextLineHeightWithSpacing() * 4.0f + 16.0f;
        if (ImGui::BeginChild("##box", ImVec2(box_w, box_h), ImGuiChildFlags_AlwaysUseWindowPadding,
                              ImGuiWindowFlags_NoScrollbar)) {
            Coloured(op.name, op.internal ? t.text_faint : t.text_bright);
            Coloured(op.internal ? std::string("(internal)") : PartsText(op), t.accent_text);
            Dim(plan_.timed ? FormatCount(op.rows) + " rows" : (op.estimated >= 0 ? "est " + FormatCount(op.estimated) : "est -"));
            if (plan_.timed) Coloured(sqlplan::FormatSeconds(op.seconds) + "  " + Percent(share), share >= kHotShare ? t.caution : t.text);
            if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) plan_picked_ = static_cast<int>(i);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        ImGui::PopID();
        x += (x == 0.0f ? 0.0f : arrow_w) + box_w + ImGui::GetStyle().ItemSpacing.x;
    }
    ImGui::EndChild();
}

void SqlWorkbenchDialog::RenderPlanText() {
    const ui::Tokens& t = ui::CurrentTokens();
    Dim("DuckDB's own text plan, read from the result back to the input");
    ImFont* mono = cyxwiz::gui::GetCodeFont();
    if (ImGui::BeginChild("##plan_text", ImGui::GetContentRegionAvail(), ImGuiChildFlags_None,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        if (mono) ImGui::PushFont(mono);
        Coloured(plan_text_, t.text);
        if (mono) ImGui::PopFont();
    }
    ImGui::EndChild();
}

void SqlWorkbenchDialog::RenderPlanDetails() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (plan_picked_ < 0 || plan_picked_ >= static_cast<int>(plan_.ops.size())) {
        Dim("Select an operator to see its details.");
        return;
    }
    const auto& op = plan_.ops[static_cast<size_t>(plan_picked_)];
    ImFont* mono = cyxwiz::gui::GetEditorMonoFont(1.0f);
    Coloured(op.name, t.text_bright);
    ImGui::SameLine();
    Dim("operator " + std::to_string(plan_picked_ + 1) + " of " + std::to_string(plan_.ops.size()));
    if (!op.parts.empty()) {
        const int part = PartIndex(op.parts.front());
        if (part >= 0) {
            const auto& p = pane_.Outline().parts[static_cast<size_t>(part)];
            const std::string link = "Show in query (lines " + std::to_string(p.first_line) + "-" +
                                     std::to_string(p.last_line) + ")";
            AlignRight(ImGui::CalcTextSize(link.c_str()).x);
            if (ui::LinkButton(link.c_str())) MarkPart(static_cast<size_t>(part));
        }
    }
    ColouredWrapped(op.summary, t.text);
    if (!op.parts.empty()) {
        Dim("from the part" + std::string(op.parts.size() > 1 ? "s " : " "));
        for (const auto& part : op.parts) {
            ImGui::SameLine();
            Coloured(part, t.accent_text);
        }
    }
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));

    // Figures.
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float width = (ImGui::GetContentRegionAvail().x - spacing * 2.0f) / 3.0f;
    const double share = plan_.timed && plan_.operator_seconds > 0 ? op.seconds / plan_.operator_seconds : 0.0;
    if (plan_.timed) {
        Card("##f1", op.rows_in >= 0 ? FormatCount(op.rows_in) + " -> " + FormatCount(op.rows) : FormatCount(op.rows),
             op.rows_in >= 0 ? "rows in -> out" : "rows out", t.text_bright, width);
        ImGui::SameLine();
        Card("##f2", sqlplan::FormatSeconds(op.seconds), "time", share >= kHotShare ? t.caution : t.text_bright, width);
        ImGui::SameLine();
        Card("##f3", Percent(share), "of operator time", t.text_bright, width);
    } else {
        Card("##f1", op.estimated >= 0 ? FormatCount(op.estimated) : std::string("none"), "estimated rows",
             t.text_bright, width);
        ImGui::SameLine();
        Card("##f2", "-", "time (Run with timing)", t.text_dim, width);
        ImGui::SameLine();
        Card("##f3", op.internal ? std::string("internal") : std::string("query"), "kind", t.text_bright, width);
    }

    // Notes.
    for (size_t i = 0; i < op.notes.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (BeginPanel("##note", ImVec2(0.0f, 0.0f), ui::Mix(t.bg_window, t.warning, 0.12f), ImGuiChildFlags_AutoResizeY)) {
            Coloured(ICON_FA_TRIANGLE_EXCLAMATION, t.warning);
            ImGui::SameLine();
            ColouredWrapped(op.notes[i], t.text);
        }
        EndPanel();
        ImGui::PopID();
    }

    // DuckDB's details, then the columns out.
    for (size_t i = 0; i < op.details.size(); ++i) {
        const auto& [key, value] = op.details[i];
        if (key == "Projections" && !op.columns.empty()) continue;
        ImGui::PushID(static_cast<int>(100 + i));
        Dim(key + (key == "Expression" ? " (as DuckDB runs it)" : ""));
        if (BeginPanel("##detail", ImVec2(0.0f, 0.0f), t.bg_input, ImGuiChildFlags_AutoResizeY)) {
            if (mono) ImGui::PushFont(mono);
            ColouredWrapped(value, t.text);
            if (mono) ImGui::PopFont();
        }
        EndPanel();
        ImGui::PopID();
    }
    if (!op.columns.empty()) {
        Dim("Columns out (" + std::to_string(op.columns.size()) + ")");
        std::string list;
        for (const auto& column : op.columns) list += (list.empty() ? "" : "   ") + column;
        if (mono) ImGui::PushFont(mono);
        ColouredWrapped(list, t.text);
        if (mono) ImGui::PopFont();
    }
}

}  // namespace gui
