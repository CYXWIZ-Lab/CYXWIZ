#include "outliers_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <arrow/api.h>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

namespace ev = explore_view;
using explore::OutlierRule;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

// A number column the rules can measure (it has a range).
bool Measurable(const ProfiledColumn& c) {
    return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min;
}

bool CellText(const std::shared_ptr<arrow::Table>& table, int column, int64_t row, std::string& out) {
    auto s = table->column(column)->GetScalar(row);
    if (!s.ok() || !(*s)->is_valid) return false;
    if ((*s)->type->id() == arrow::Type::STRING) out = std::static_pointer_cast<arrow::StringScalar>(*s)->ToString();
    else out = (*s)->ToString();
    return true;
}

double CellNumber(const std::shared_ptr<arrow::Table>& table, int column, int64_t row) {
    auto s = table->column(column)->GetScalar(row);
    if (!s.ok() || !(*s)->is_valid) return NAN;
    auto d = (*s)->CastTo(arrow::float64());
    return d.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*d)->value : NAN;
}

float TableHeight(size_t rows) {
    return ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(rows) + 1.0f) + ImGui::GetStyle().ScrollbarSize + 4.0f;
}

// A dashed vertical line.
void Dashed(ImDrawList* dl, float x, float y0, float y1, ImU32 colour) {
    for (float y = y0; y < y1; y += 7.0f) dl->AddLine(ImVec2(x, y), ImVec2(x, std::min(y + 4.0f, y1)), colour, 2.0f);
}

}  // namespace

std::string OutliersView::Key(const std::string& dataset) const {
    char f[32];
    std::snprintf(f, sizeof(f), "%.6g", factor_);
    return dataset + "\x1f" + column_ + "\x1f" + explore::OutlierRuleName(rule_) + "\x1f" + f;
}

void OutliersView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to find its outliers.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        mad_.Reset();
        counts_.Reset();
        rows_.Reset();
        bins_.Reset();
        offset_ = 0;
    }
    const ProfiledColumn* c = p.Find(column_);
    if (!c || !Measurable(*c)) {
        c = nullptr;
        for (const auto& col : p.columns)
            if (Measurable(col)) {
                c = &col;
                break;
            }
        column_ = c ? c->facts.name : std::string();
        offset_ = 0;
    }
    if (!c) {
        ui::EmptyState(ICON_FA_CHART_SIMPLE, "No number column with a range",
                       "Outliers are found in number columns that hold more than one value.");
        return;
    }
    if (RenderControls(p)) {
        offset_ = 0;
        c = p.Find(column_);
        if (!c) return;
    }

    // Modified Z needs the median absolute deviation (one query per column).
    const std::string dataset = context.dataset, column = column_;
    const ProfiledColumn column_copy = *c;
    double mad = 0;
    bool mad_ready = true;
    if (rule_ == OutlierRule::ModifiedZ) {
        const std::string mad_key = dataset + "\x1f" + column;
        mad_.Ask("Median absolute deviation of " + column, mad_key, [dataset, column_copy](LambdaTask&) {
            explore::Mad m;
            m.value = explore::MedianAbsoluteDeviation(dataset, column_copy, RunNow, &m.error);
            return m;
        });
        if (const auto* m = mad_.Answer(mad_key)) mad = m->value;
        else mad_ready = false;
    }
    const explore::OutlierBounds bounds = explore::BoundsFor(*c, rule_, factor_, mad);

    const std::string key = Key(dataset);
    const explore::OutlierCounts* counts = nullptr;
    if (mad_ready && bounds.ok()) {
        counts_.Ask("Outliers in " + column, key, [dataset, column_copy, bounds](LambdaTask&) {
            return explore::CountOutliers(dataset, column_copy, bounds, RunNow);
        });
        counts = counts_.Answer(key);
    }

    // Summary line.
    {
        ui::FlowRow row;
        const auto part = [&](const ImVec4& colour, const std::string& text) {
            row.Next(ImGui::CalcTextSize(text.c_str()).x);
            ImGui::TextColored(colour, "%s", text.c_str());
        };
        const size_t values = c->facts.non_null;
        char rule[64];
        std::snprintf(rule, sizeof(rule), "%s %s", explore::OutlierRuleName(rule_), ev::Figure(factor_).c_str());
        if (!mad_ready) part(t.text_faint, "Measuring the spread...");
        else if (!bounds.ok()) part(t.caution, std::string(rule) + ": no spread to measure against");
        else if (counts) {
            part(counts->total() ? t.warning : t.success,
                 ev::Count(counts->total(), "outlier", "outliers") + " in " + column + " (" + ev::Share(counts->total(), values) + " of " +
                     ev::Number(values) + " values)");
            part(t.text_dim, ev::Number(counts->low) + " low \xC2\xB7 " + ev::Number(counts->high) + " high");
            part(t.text_dim, std::string(rule) + ": below " + ev::Figure(bounds.lo) + " or above " + ev::Figure(bounds.hi));
        } else if (!counts_.Error().empty()) part(t.error, counts_.Error());
        else part(t.text_faint, "Counting...");
        if (!p.exact) part(t.text_faint, "(quartiles approximate on this table)");
    }
    if (mad_ready && bounds.ok() && counts) RenderStepButtons(context, bounds, counts->total());
    ImGui::Spacing();

    const float avail = ImGui::GetContentRegionAvail().x;
    const bool stacked = avail < 640.0f;
    const float pane_w = stacked ? avail : std::min(380.0f, avail * 0.36f);
    if (!stacked) {
        ImGui::BeginChild("##outliers_left", ImVec2(avail - pane_w - ImGui::GetStyle().ItemSpacing.x, 0), ImGuiChildFlags_None);
    }
    RenderDistribution(context, *c, bounds.ok() && mad_ready ? &bounds : nullptr);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "ROWS");
    if (!mad_ready || (bounds.ok() && !counts)) {
        ImGui::TextColored(t.text_faint, "Counting...");
    } else if (!bounds.ok()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.caution, "%s", bounds.unavailable.c_str());
        ImGui::TextColored(t.text_faint, "Pick another rule: %s", rule_ == OutlierRule::ZScore ? "IQR or Modified Z may still measure it." :
                                                                  "Z-score uses the mean and std instead.");
        ImGui::PopTextWrapPos();
    } else if (counts->total() == 0) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " No outliers in %s with %s %s.", column_.c_str(), explore::OutlierRuleName(rule_),
                           ev::Figure(factor_).c_str());
        ImGui::TextColored(t.text_faint, "Every value lies between %s and %s. A smaller factor finds more.", ev::Figure(bounds.lo).c_str(),
                           ev::Figure(bounds.hi).c_str());
    } else {
        RenderRows(context, *c, bounds, counts->total());
    }
    if (!stacked) {
        ImGui::EndChild();
        ImGui::SameLine();
    } else {
        ImGui::Spacing();
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##outliers_figures", ImVec2(stacked ? 0.0f : pane_w, 0.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding | (stacked ? ImGuiChildFlags_AutoResizeY : ImGuiChildFlags_None));
    ImGui::PopStyleColor();
    RenderFigures(p, *c, mad_ready ? &bounds : nullptr);
    ImGui::EndChild();
}

bool OutliersView::RenderControls(const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    bool changed = false;
    std::vector<plot::ColumnSummary> numbers;
    for (const auto& c : p.columns) {
        if (!Measurable(c)) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = true;
        numbers.push_back(std::move(s));
    }
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Column").x + 250.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Column");
    ImGui::SameLine();
    if (picker_.Pick("##outlier_column", column_, numbers, true, nullptr, 240.0f)) changed = true;
    static const char* kRules[] = {"IQR", "Z-score", "Modified Z"};
    row.Next(ImGui::CalcTextSize("Rule").x + ImGui::CalcTextSize("IQRZ-scoreModified Z").x + 60.0f);
    ImGui::TextColored(t.text_dim, "Rule");
    ImGui::SameLine();
    int rule = static_cast<int>(rule_);
    if (ui::SegmentedControl("##outlier_rule", kRules, 3, &rule)) {
        rule_ = static_cast<OutlierRule>(rule);
        factor_ = explore::DefaultOutlierFactor(rule_);
        changed = true;
    }
    const float step_w = ui::ButtonWidth("+", ui::ButtonSize::Small);
    row.Next(ImGui::CalcTextSize("Factor").x + 90.0f + 2.0f * step_w + 24.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Factor");
    ImGui::SameLine();
    // Typed; the count follows when the field is left or - / + is pressed.
    const std::string hint = "k in the bounds: smaller finds more outliers; default " + ev::Figure(explore::DefaultOutlierFactor(rule_)) +
                             " for " + explore::OutlierRuleName(rule_);
    ui::NumberFieldOptions factor;
    factor.min = 0.1;
    factor.max = 20.0;
    factor.step = 0.1;
    factor.width = 70.0f;
    factor.hint = hint.c_str();
    if (ui::NumberField("##outlier_factor", &factor_, factor)) changed = true;
    return changed;
}

void OutliersView::RenderDistribution(const ExploreContext& context, const ProfiledColumn& c, const explore::OutlierBounds* bounds) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "DISTRIBUTION");
    std::vector<double> edges = c.hist_edges;
    std::vector<size_t> counts = c.hist_counts;
    if (counts.empty()) {
        // Past the profile's first 200 columns: ask for the bins.
        const std::string key = context.dataset + "\x1f" + c.facts.name;
        const std::string dataset = context.dataset, column = c.facts.name;
        const double lo = c.min, hi = c.max;
        bins_.Ask("Histogram of " + column, key, [dataset, column, lo, hi](LambdaTask&) {
            return explore::CountBins(dataset, column, lo, hi, 20, RunNow);
        });
        const auto* b = bins_.Answer(key);
        if (!b) {
            ImGui::TextColored(bins_.Error().empty() ? t.text_faint : t.error, "%s",
                               bins_.Error().empty() ? "Counting the values..." : bins_.Error().c_str());
            return;
        }
        edges = b->edges;
        counts = b->counts;
    }
    if (counts.empty() || edges.size() != counts.size() + 1) return;
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 120.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const size_t most = std::max<size_t>(1, *std::max_element(counts.begin(), counts.end()));
    const double span = edges.back() - edges.front();
    const auto X = [&](double v) { return at.x + static_cast<float>((v - edges.front()) / span) * width; };
    const float top = at.y + ImGui::GetTextLineHeight() + 2.0f;  // room for the bound labels
    const float base = top + height;
    int hovered = -1;
    for (size_t i = 0; i < counts.size(); ++i) {
        const float x0 = X(edges[i]) + 1.0f, x1 = X(edges[i + 1]) - 1.0f;
        const float y = base - height * static_cast<float>(counts[i]) / static_cast<float>(most);
        const bool out = bounds && (edges[i + 1] <= bounds->lo || edges[i] >= bounds->hi);
        dl->AddRectFilled(ImVec2(x0, std::min(y, base - 1.0f)), ImVec2(std::max(x1, x0 + 1.0f), base),
                          ui::ToU32(out ? t.warning : ui::WithAlpha(t.accent_text, 0.55f)), 2.0f);
        if (ImGui::IsMouseHoveringRect(ImVec2(x0, top), ImVec2(x1, base)) && ImGui::IsWindowHovered()) hovered = static_cast<int>(i);
    }
    if (bounds) {
        const ImU32 fence = ui::ToU32(t.warning);
        for (const auto& [value, label] : {std::pair<double, const char*>{bounds->lo, "lower "}, {bounds->hi, "upper "}}) {
            if (value <= edges.front() || value >= edges.back()) continue;
            const float x = X(value);
            Dashed(dl, x, top, base, fence);
            const std::string text = label + ev::Figure(value);
            const float tw = ImGui::CalcTextSize(text.c_str()).x;
            const float tx = label[0] == 'l' ? std::max(at.x, x - tw - 4.0f) : std::min(at.x + width - tw, x + 4.0f);
            dl->AddText(ImVec2(tx, at.y), fence, text.c_str());
        }
    }
    ImGui::Dummy(ImVec2(width, base - at.y + 2.0f));
    const std::string lo = ev::Figure(edges.front()), mid = ev::Figure((edges.front() + edges.back()) / 2), hi = ev::Figure(edges.back());
    ImGui::TextColored(t.text_faint, "%s", lo.c_str());
    ImGui::SameLine(std::max(0.0f, width / 2 - ImGui::CalcTextSize(mid.c_str()).x / 2));
    ImGui::TextColored(t.text_faint, "%s", mid.c_str());
    ImGui::SameLine(std::max(0.0f, width - ImGui::CalcTextSize(hi.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hi.c_str());
    if (bounds && (bounds->lo <= edges.front() || bounds->hi >= edges.back())) {
        const bool both = bounds->lo <= edges.front() && bounds->hi >= edges.back();
        ImGui::TextColored(t.text_faint, "%s", both ? "Both bounds lie beyond the values."
                                                    : bounds->lo <= edges.front() ? "The lower bound lies below the smallest value."
                                                                                   : "The upper bound lies above the largest value.");
    }
    if (hovered >= 0) {
        const size_t i = static_cast<size_t>(hovered);
        ImGui::SetTooltip("%s to %s: %s", ev::Figure(edges[i]).c_str(), ev::Figure(edges[i + 1]).c_str(), ev::Number(counts[i]).c_str());
    }
}

void OutliersView::RenderRows(const ExploreContext& context, const ProfiledColumn& c, const explore::OutlierBounds& bounds, size_t total) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string key = Key(context.dataset) + "\x1f" + std::to_string(offset_);
    const std::string dataset = context.dataset;
    const ProfiledColumn column_copy = c;
    const OutlierRule rule = rule_;
    const size_t offset = offset_;
    rows_.Ask("Outlier rows of " + c.facts.name, key, [dataset, column_copy, rule, bounds, offset](LambdaTask&) {
        return explore::OutlierRows(dataset, column_copy, rule, bounds, offset, RunNow);
    });
    const explore::RowPage* page = rows_.Answer(key);
    if (!page) {
        ImGui::TextColored(rows_.Error().empty() ? t.text_faint : t.error, "%s", rows_.Error().empty() ? "Reading the rows..." : rows_.Error().c_str());
        return;
    }
    if (!page->rows || page->rows->num_columns() < 2) return;
    const auto& rows = page->rows;
    const int ncols = std::min(rows->num_columns(), 64);
    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit |
                                  ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_NoSavedSettings;
    // row, score, low / high, then the table's columns.
    if (ImGui::BeginTable("##outlier_rows", ncols + 1, flags, ImVec2(0, TableHeight(static_cast<size_t>(std::min<int64_t>(rows->num_rows(), 15)))))) {
        // row, score, low / high, the column checked, then the others.
        const int value_col = rows->schema()->GetFieldIndex(c.facts.name);
        std::vector<int> order;
        if (value_col >= 2 && value_col < ncols) order.push_back(value_col);
        for (int col = 2; col < ncols; ++col)
            if (col != value_col) order.push_back(col);
        ImGui::TableSetupScrollFreeze(4, 1);
        ImGui::TableSetupColumn("row");
        ImGui::TableSetupColumn(explore::OutlierScoreName(rule_));
        ImGui::TableSetupColumn("##side");
        for (int col : order) ImGui::TableSetupColumn(rows->field(col)->name().c_str());
        ImGui::TableHeadersRow();
        for (int64_t r = 0; r < rows->num_rows(); ++r) {
            ImGui::TableNextRow();
            std::string text;
            ImGui::TableNextColumn();
            if (CellText(rows, 0, r, text)) ImGui::TextColored(t.text_dim, "%s", text.c_str());
            const double score = CellNumber(rows, 1, r);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", score);
            ImGui::TableNextColumn();
            ImGui::TextColored(score > 0 ? t.error : t.info, "%s", score > 0 ? "high" : "low");
            for (int col : order) {
                ImGui::TableNextColumn();
                if (!CellText(rows, col, r, text)) continue;
                if (col == value_col) ImGui::TextColored(t.warning, "%s", text.c_str());
                else ImGui::TextUnformatted(text.c_str());
            }
        }
        ImGui::EndTable();
    }
    const size_t shown = static_cast<size_t>(rows->num_rows());
    ImGui::TextColored(t.text_faint, "Rows %zu to %zu of %s, largest score first%s", offset_ + 1, offset_ + shown, ev::Number(total).c_str(),
                       rows->num_columns() > 64 ? " (the first 62 columns)" : "");
    ImGui::SameLine();
    if (ui::LinkButton("Open in Query", static_cast<bool>(on_open_query)))
        on_open_query(ev::OutlierRowsSql(context.shown, c.facts.name, bounds.lo, bounds.hi));
    if (offset_ > 0 || offset_ + shown < total) {
        if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT " Previous", offset_ > 0, "This is the first page")) offset_ -= std::min(offset_, explore::kRowPage);
        ImGui::SameLine();
        if (ui::SecondaryButton("Next " ICON_FA_CHEVRON_RIGHT, offset_ + shown < total, "This is the last page")) offset_ += explore::kRowPage;
    }
}

void OutliersView::RenderFigures(const DatasetProfile& p, const ProfiledColumn& c, const explore::OutlierBounds* bounds) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "FIGURES");
    const bool ok = bounds && bounds->ok();
    const std::string none = "-";
    const std::vector<std::pair<std::string, std::string>> rows = {
        {"Lower bound", ok ? ev::Figure(bounds->lo) : none},
        {"Upper bound", ok ? ev::Figure(bounds->hi) : none},
        {"Mean", ev::Figure(c.mean)},
        {"Median", ev::Figure(c.median)},
        {"Std", ev::Figure(c.std)},
        {"MAD", rule_ == OutlierRule::ModifiedZ ? (bounds ? ev::Figure(bounds->mad) : std::string("...")) : std::string("Modified Z only")},
        {"Q1", ev::Figure(c.q1)},
        {"Q3", ev::Figure(c.q3)},
        {"Values", ev::Number(c.facts.non_null)},
        {"Missing", ev::Number(c.missing)},
    };
    ExploreFigures("##outlier_figures", rows);
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s", explore::OutlierRuleExplain(rule_));
    if (!p.exact) ImGui::TextColored(t.text_faint, "This table is profiled approximately: Q1, median and Q3 are close estimates.");
    ImGui::PopTextWrapPos();
}

void OutliersView::RenderStepButtons(const ExploreContext& context, const explore::OutlierBounds& bounds, size_t outliers) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string blocker = !on_add_step ? "Prepare is not available" : step_blocker ? step_blocker() : std::string();
    const bool enabled = blocker.empty();
    const char* reason = enabled ? nullptr : blocker.c_str();
    const std::string rule = std::string(explore::OutlierRuleName(rule_)) + " " + ev::Figure(factor_);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "In Prepare:");
    ImGui::SameLine();
    const auto add = [&](const char* label, const char* action, const std::string& gives) {
        if (ui::SecondaryButton(label, enabled, reason)) {
            message_ = on_add_step(column_, rule_, factor_, action);
            message_error_ = message_.rfind("Added", 0) != 0;
        }
        if (enabled && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
            ImGui::SetTooltip("%s\nThe step learns %s's bounds from the Train rows (every row without a Data Split), so its count can differ from the %s found here.",
                              ev::StepTip(std::string("Outlier Detector (") + rule + ")", gives).c_str(), column_.c_str(),
                              ev::Count(outliers, "outlier", "outliers").c_str());
        }
    };
    add("Flag rows", "flag", "an is_outlier column (1 for the rows outside " + ev::Figure(bounds.lo) + " .. " + ev::Figure(bounds.hi) + ")");
    ExploreNextButton("Clip values");
    add("Clip values", "clip", column_ + " with every value pulled into " + ev::Figure(bounds.lo) + " .. " + ev::Figure(bounds.hi));
    ExploreNextButton("Drop rows");
    add("Drop rows", "remove", "the rows without the outliers (" + ev::Count(outliers, "row", "rows") + " here)");
    (void)context;
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
}

}  // namespace cyxwiz
