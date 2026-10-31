#include "missing_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <arrow/api.h>

namespace cyxwiz {

namespace {

namespace ev = explore_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

// A cell as text; false for a missing value.
bool CellText(const std::shared_ptr<arrow::Table>& table, int column, int64_t row, std::string& out) {
    auto chunked = table->column(column);
    auto s = chunked->GetScalar(row);
    if (!s.ok() || !(*s)->is_valid) return false;
    if ((*s)->type->id() == arrow::Type::STRING) out = std::static_pointer_cast<arrow::StringScalar>(*s)->ToString();
    else out = (*s)->ToString();
    return true;
}

const std::vector<std::string>& TextsFor(const ExploreContext& c, const std::string& column) {
    static const std::vector<std::string> kNone;
    auto it = c.missing_texts.find(column);
    return it == c.missing_texts.end() ? kNone : it->second;
}

// Coloured text parts that wrap to the next line when one does not fit.
void FlowText(ui::FlowRow& row, const ImVec4& colour, const std::string& text) {
    row.Next(ImGui::CalcTextSize(text.c_str()).x);
    ImGui::TextColored(colour, "%s", text.c_str());
}

// Bullets that wrap (BulletText does not).
void Bullets(const std::vector<std::string>& lines) {
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& line : lines) {
        ImGui::Bullet();
        ImGui::TextWrapped("%s", line.c_str());
    }
    ImGui::PopTextWrapPos();
}

// A table of `rows` rows with its header (and room for a horizontal scrollbar).
float ColumnsHeight(size_t rows) {
    return ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(rows) + 1.0f) + ImGui::GetStyle().ScrollbarSize + 4.0f;
}

// A bar of `share` (0..1) in the remaining width of a table cell.
void ShareBar(float share, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight() * 0.6f;
    const float y = at.y + (ImGui::GetTextLineHeight() - h) * 0.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + width, y + h), ui::ToU32(t.bg_input), 3.0f);
    if (share > 0) dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + std::max(2.0f, width * share), y + h), ui::ToU32(t.warning), 3.0f);
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
}

}  // namespace

void MissingView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to see its missing values.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        // A new profile (another dataset, or the data re-loaded): ask again.
        seen_ = context.profile;
        summary_.Reset();
        rows_.Reset();
        message_.clear();
        offset_ = 0;
        if (!p.Find(column_)) column_.clear();
    }
    if (column_.empty()) {
        // The column missing the most.
        const ProfiledColumn* most = nullptr;
        for (const auto& c : p.columns)
            if (c.missing > 0 && (!most || c.missing > most->missing)) most = &c;
        if (most) column_ = most->facts.name;
        else if (!p.columns.empty()) column_ = p.columns.front().facts.name;
    }

    const std::string table = context.dataset;
    const auto texts = context.missing_texts;
    const std::shared_ptr<const DatasetProfile> profile = context.profile;
    summary_.Ask("Missing values of " + context.shown, table, [profile, texts](LambdaTask&) {
        return explore::SummariseMissing(*profile, texts, RunNow);
    });

    // Summary line.
    const size_t cells = p.rows * p.columns.size();
    const size_t missing = p.MissingCells();
    {
        ui::FlowRow row;
        FlowText(row, t.text, context.shown);
        FlowText(row, missing ? t.warning : t.success,
                 ev::Count(missing, "cell", "cells") + " of " + ev::Number(cells) + " missing (" + ev::Share(missing, cells) + ")");
        if (const auto* s = summary_.Answer(table)) {
            FlowText(row, t.text_dim, ev::Count(s->complete_rows, "complete row", "complete rows"));
            FlowText(row, s->incomplete_rows ? t.warning : t.text_dim,
                     ev::Count(s->incomplete_rows, "row", "rows") + " incomplete (" + ev::Share(s->incomplete_rows, p.rows) + ")");
        } else if (summary_.Running()) {
            FlowText(row, t.text_faint, "counting complete rows...");
        } else if (!summary_.Error().empty()) {
            FlowText(row, t.error, summary_.Error());
        }
    }
    size_t text_columns = 0;
    for (const auto& [column, list] : context.missing_texts)
        if (!list.empty() && p.Find(column)) ++text_columns;
    {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "%s", ev::MissingRule(text_columns).c_str());
        ImGui::PopTextWrapPos();
        ui::FlowRow row;
        row.first = false;
        const char* link = text_columns ? "change them in Profile##rule" : "set them in Profile##rule";
        row.Next(ImGui::CalcTextSize(link, nullptr, true).x);
        if (ui::LinkButton(link, static_cast<bool>(on_open_profile))) on_open_profile(column_);
    }
    ImGui::Spacing();

    if (missing == 0) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " No missing values in %s: every row is complete.", context.shown.c_str());
        return;
    }

    // Columns and patterns beside the picked column's pane; stacked when narrow.
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool stacked = avail < 640.0f;
    const float pane_w = stacked ? avail : std::min(420.0f, avail * 0.42f);
    if (!stacked) {
        ImGui::BeginChild("##missing_left", ImVec2(avail - pane_w - ImGui::GetStyle().ItemSpacing.x, 0), ImGuiChildFlags_None);
        RenderColumns(context, std::min(ImGui::GetContentRegionAvail().y * 0.6f, ColumnsHeight(p.columns.size())));
        RenderPatterns(context);
        ImGui::EndChild();
        ImGui::SameLine();
    } else {
        RenderColumns(context, std::min(220.0f, ColumnsHeight(p.columns.size())));
        RenderPatterns(context);
        ImGui::Spacing();
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    // Stacked under the columns, the pane is as tall as what it shows.
    ImGui::BeginChild("##missing_column", ImVec2(stacked ? 0.0f : pane_w, 0.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding | (stacked ? ImGuiChildFlags_AutoResizeY : ImGuiChildFlags_None));
    ImGui::PopStyleColor();
    RenderColumn(context);
    ImGui::EndChild();
}

void MissingView::RenderColumns(const ExploreContext& context, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const DatasetProfile& p = *context.profile;
    std::vector<const ProfiledColumn*> order;
    size_t most = 0;
    for (const auto& c : p.columns) {
        order.push_back(&c);
        most = std::max(most, c.missing);
    }
    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Sortable |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_NoSavedSettings;
    if (!ImGui::BeginTable("##missing_columns", 4, flags, ImVec2(0, height))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("column", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("missing", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending);
    ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_NoSort);
    ImGui::TableSetupColumn("##bar", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0) {
        sort_ = specs->Specs[0].ColumnIndex == 0 ? 0 : 1;
        descending_ = specs->Specs[0].SortDirection == ImGuiSortDirection_Descending;
    }
    // A strict order: ties keep the table's column order.
    std::stable_sort(order.begin(), order.end(), [&](const ProfiledColumn* a, const ProfiledColumn* b) {
        if (sort_ == 0) return descending_ ? a->facts.name > b->facts.name : a->facts.name < b->facts.name;
        return descending_ ? a->missing > b->missing : a->missing < b->missing;
    });
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(order.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const ProfiledColumn& c = *order[static_cast<size_t>(i)];
            ImGui::PushID(c.facts.name.c_str());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool none = c.missing == 0;
            if (none) ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
            if (ImGui::Selectable(c.facts.name.c_str(), c.facts.name == column_, ImGuiSelectableFlags_SpanAllColumns)) {
                column_ = c.facts.name;
                offset_ = 0;
                message_.clear();
            }
            if (none) ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::TextColored(none ? t.text_faint : t.warning, "%s", ev::Number(c.missing).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(none ? t.text_faint : t.text, "%s", ev::Share(c.missing, p.rows).c_str());
            ImGui::TableNextColumn();
            ShareBar(most ? static_cast<float>(c.missing) / static_cast<float>(most) : 0.0f, 110.0f);
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

void MissingView::RenderPatterns(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto* s = summary_.Answer(context.dataset);
    if (!s || s->patterns.empty()) return;
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "MISSING TOGETHER");
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& pattern : s->patterns) {
        std::string names;
        for (const auto& c : pattern.columns) names += (names.empty() ? "" : ", ") + c;
        ImGui::TextColored(t.warning, "%s", ev::Count(pattern.rows, "row", "rows").c_str());
        ImGui::SameLine();
        ImGui::TextUnformatted(pattern.columns.size() == 1 ? (names + " only").c_str() : names.c_str());
    }
    if (s->patterns.size() == explore::kPatterns)
        ImGui::TextColored(t.text_faint, "The %zu most common; other rows are missing other mixes.", explore::kPatterns);
    ImGui::PopTextWrapPos();
}

void MissingView::RenderColumn(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    const DatasetProfile& p = *context.profile;
    const ProfiledColumn* c = p.Find(column_);
    if (!c) return;
    ImGui::TextColored(t.text_dim, "%s", column_.c_str());
    if (c->missing == 0) {
        ImGui::TextColored(t.success, "No missing values in %s.", column_.c_str());
        return;
    }
    ImGui::TextColored(t.warning, "%s", ev::Count(c->missing, "row", "rows").c_str());
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::Text("missing (%s)", ev::Share(c->missing, p.rows).c_str());
    if (c->missing_text) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "%s as text", ev::Number(c->missing_text).c_str());
    }
    Bullets(ev::MissingAdvice(100.0 * static_cast<double>(c->missing) / static_cast<double>(std::max<size_t>(1, p.rows))));

    const auto& texts = TextsFor(context, column_);
    const std::string blocker = !on_drop_rows ? "Prepare is not available" : drop_blocker ? drop_blocker() : std::string();
    const std::string drop_reason = blocker.empty() ? std::string() : blocker;
    if (ui::SecondaryButton("Drop these rows in Prepare", blocker.empty(), drop_reason.c_str())) {
        message_ = on_drop_rows(column_, texts);
        message_error_ = message_.rfind("Added", 0) != 0;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && blocker.empty())
        ImGui::SetTooltip("%s", ev::StepTip("Keep rows", "the rows where " + column_ + " is not empty" +
                                                              (texts.empty() ? "" : " and is not one of the texts marked as missing")).c_str());
    ImGui::SameLine();
    if (ui::LinkButton("Open in Query", static_cast<bool>(on_open_query)))
        on_open_query(ev::MissingRowsSql(context.shown, column_, texts));
    // Fill in Prepare (TOFIX129 D7a G2): a Fill Missing Values step on this
    // column. Numbers take a statistic or a constant; other columns the mode
    // or a constant. The step learns the statistic from the Train rows.
    {
        const bool numeric = c->numeric;
        static const char* kNumeric[] = {"mean", "median", "mode", "constant"};
        static const char* kOther[] = {"mode", "constant"};
        const char* const* strategies = numeric ? kNumeric : kOther;
        const int count = numeric ? 4 : 2;
        if (fill_strategy_ >= count) fill_strategy_ = 0;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Fill with");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("constant").x + ImGui::GetFrameHeight() + 20.0f);
        if (ImGui::BeginCombo("##fill_strategy", strategies[fill_strategy_])) {
            for (int i = 0; i < count; ++i)
                if (ImGui::Selectable(strategies[i], i == fill_strategy_)) fill_strategy_ = i;
            ImGui::EndCombo();
        }
        const bool constant = std::string(strategies[fill_strategy_]) == "constant";
        if (constant) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputText("##fill_value", fill_value_, sizeof(fill_value_));
        }
        const std::string fill_blocker = !on_fill ? "Prepare is not available" : blocker;
        const std::string fill_reason = fill_blocker.empty() && constant && fill_value_[0] == '\0' ? "Type the value to fill with" : fill_blocker;
        ExploreNextButton("Fill in Prepare");
        if (ui::SecondaryButton("Fill in Prepare", fill_reason.empty(), fill_reason.c_str())) {
            message_ = on_fill(column_, strategies[fill_strategy_], constant ? fill_value_ : "");
            message_error_ = message_.rfind("Added", 0) != 0;
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && fill_reason.empty()) {
            const std::string with = constant ? std::string(" with ") + fill_value_
                                              : " with the " + std::string(strategies[fill_strategy_]) + " of its Train rows";
            ImGui::SetTooltip("%s\nIt fills the empty cells of %s%s; texts marked as missing are values to it (drop those rows, or make them empty first).",
                              ev::StepTip("Fill Missing Values", column_ + " with its empty cells filled").c_str(), column_.c_str(), with.c_str());
        }
    }
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }

    // The rows, a page at a time.
    const std::string key = context.dataset + "\x1f" + column_ + "\x1f" + std::to_string(offset_);
    const std::string table = context.dataset, column = column_;
    const size_t offset = offset_;
    rows_.Ask("Missing rows of " + column_, key, [table, column, texts, offset](LambdaTask&) {
        return explore::MissingRows(table, column, texts, offset, RunNow);
    });
    ImGui::Spacing();
    const explore::RowPage* page = rows_.Answer(key);
    if (!page) {
        if (!rows_.Error().empty()) ImGui::TextColored(t.error, "%s", rows_.Error().c_str());
        else ImGui::TextColored(t.text_faint, "Reading the rows...");
    } else if (page->rows && page->rows->num_columns() > 0) {
        const auto& rows = page->rows;
        const int ncols = rows->num_columns();
        const ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_NoSavedSettings;
        if (ImGui::BeginTable("##missing_rows", std::min(ncols, 64), flags,
                              ImVec2(0, ColumnsHeight(static_cast<size_t>(std::min<int64_t>(rows->num_rows(), 15)))))) {
            ImGui::TableSetupScrollFreeze(1, 1);
            for (int col = 0; col < std::min(ncols, 64); ++col) ImGui::TableSetupColumn(rows->field(col)->name().c_str());
            ImGui::TableHeadersRow();
            for (int64_t r = 0; r < rows->num_rows(); ++r) {
                ImGui::TableNextRow();
                for (int col = 0; col < std::min(ncols, 64); ++col) {
                    ImGui::TableNextColumn();
                    std::string text;
                    if (!CellText(rows, col, r, text)) ImGui::TextColored(t.text_faint, "empty");
                    else if (col == 0) ImGui::TextColored(t.text_dim, "%s", text.c_str());
                    else ImGui::TextUnformatted(text.c_str());
                }
            }
            ImGui::EndTable();
        }
        const size_t shown = static_cast<size_t>(rows->num_rows());
        ImGui::TextColored(t.text_faint, "Rows %zu to %zu of %s%s", offset_ + 1, offset_ + shown, ev::Number(c->missing).c_str(),
                           ncols > 64 ? " (the first 63 columns)" : "");
        if (offset_ > 0 || offset_ + shown < c->missing) {
            if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT " Previous", offset_ > 0, "This is the first page")) offset_ -= std::min(offset_, explore::kRowPage);
            ImGui::SameLine();
            if (ui::SecondaryButton("Next " ICON_FA_CHEVRON_RIGHT, offset_ + shown < c->missing, "This is the last page")) offset_ += explore::kRowPage;
        }
    }

    ImGui::Spacing();
    if (ImGui::TreeNodeEx("General recommendations", ImGuiTreeNodeFlags_SpanAvailWidth)) {
        Bullets(ev::GeneralMissingAdvice());
        ImGui::TreePop();
    }
}

}  // namespace cyxwiz
