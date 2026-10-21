#include "correlation_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <tuple>

namespace cyxwiz {

namespace {

namespace ev = explore_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) {
    return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min;
}

// `text` cut to `width` with an ellipsis of dots.
std::string Fit(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    std::string cut = text;
    while (!cut.empty() && ImGui::CalcTextSize((cut + "..").c_str()).x > width) cut.pop_back();
    return cut + "..";
}

}  // namespace

void CorrelationView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to see how its number columns move together.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        matrix_.Reset();
        chosen_.clear();
        pick_i_ = 0;
        pick_j_ = 1;
        message_.clear();
    }
    // The profile's matrix, or the columns picked (asked for here).
    const ColumnCorrelations* matrix = nullptr;
    if (chosen_.empty() || chosen_ == p.correlation.columns) {
        matrix = p.correlation.columns.size() >= 2 ? &p.correlation : nullptr;
    } else if (chosen_.size() >= 2) {
        std::string key = context.dataset;
        for (const auto& c : chosen_) key += "\x1f" + c;
        const std::string dataset = context.dataset;
        const std::vector<std::string> columns = chosen_;
        matrix_.Ask("Correlations of " + context.shown, key, [dataset, columns](LambdaTask&) {
            return Correlate(dataset, columns, RunNow);
        });
        matrix = matrix_.Answer(key);
    }

    RenderControls(context, p, matrix);
    size_t numbers = 0;
    for (const auto& c : p.columns)
        if (Measurable(c)) ++numbers;
    if (numbers < 2) {
        ui::EmptyState(ICON_FA_TABLE, "Fewer than two number columns", "Correlation compares number columns that hold more than one value.");
        return;
    }
    if (!matrix) {
        if (chosen_.size() == 1) ImGui::TextColored(t.text_faint, "Pick at least two columns.");
        else if (!matrix_.Error().empty()) ImGui::TextColored(t.error, "%s", matrix_.Error().c_str());
        else ImGui::TextColored(t.text_faint, "Computing the correlations...");
        return;
    }
    {
        ui::FlowRow row;
        const auto part = [&](const ImVec4& colour, const std::string& text) {
            row.Next(ImGui::CalcTextSize(text.c_str()).x);
            ImGui::TextColored(colour, "%s", text.c_str());
        };
        part(t.text, context.shown);
        part(t.text_dim, "Pearson r of " + ev::Count(matrix->size(), "number column", "number columns") + ", over the rows where both values are present");
        if (chosen_.empty() && numbers > matrix->size())
            part(t.text_faint, "(the first " + ev::Number(matrix->size()) + " of " + ev::Number(numbers) + "; pick others in Columns)");
    }
    if (!message_.empty()) ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
    ImGui::Spacing();

    if (pick_i_ >= static_cast<int>(matrix->size()) || pick_j_ >= static_cast<int>(matrix->size())) {
        pick_i_ = 0;
        pick_j_ = 1;
    }
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool stacked = avail < 640.0f;
    const float pane_w = stacked ? avail : std::min(360.0f, avail * 0.34f);
    ImGui::BeginChild("##corr_grid", ImVec2(stacked ? 0.0f : avail - pane_w - ImGui::GetStyle().ItemSpacing.x, stacked ? 360.0f : 0.0f),
                      ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    RenderHeatmap(*matrix);
    ImGui::EndChild();
    if (!stacked) ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##corr_pair", ImVec2(stacked ? 0.0f : pane_w, 0.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding | (stacked ? ImGuiChildFlags_AutoResizeY : ImGuiChildFlags_None));
    ImGui::PopStyleColor();
    RenderPair(*matrix);
    ImGui::EndChild();
}

void CorrelationView::RenderControls(const ExploreContext& context, const DatasetProfile& p, const ColumnCorrelations* matrix) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plot::ColumnSummary> numbers;
    for (const auto& c : p.columns) {
        if (!Measurable(c)) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = true;
        numbers.push_back(std::move(s));
    }
    ui::FlowRow row;
    row.Next(ui::ButtonWidth("Display", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Display")) ImGui::OpenPopup("##corr_display");
    if (ImGui::BeginPopup("##corr_display")) {
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderFloat("Cell size", &cell_, 28.0f, 90.0f, "%.0f px");
        ImGui::Checkbox("Show values", &show_values_);
        ImGui::Checkbox("Show legend", &show_legend_);
        ImGui::EndPopup();
    }
    const char* reason = matrix ? nullptr : "No matrix yet";
    row.Next(ui::ButtonWidth("Export CSV...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Export CSV...", matrix != nullptr, reason) && matrix) {
        const std::string name = context.shown + "_correlations.csv";
        if (const auto path = FileDialogs::SaveFile("Export correlations", {{"CSV", "csv"}}, nullptr, name.c_str())) {
            std::ofstream out(*path, std::ios::binary);
            out << ev::CorrelationCsv(matrix->columns, matrix->r);
            message_error_ = !out.good();
            message_ = message_error_ ? "Could not write " + *path : "Saved " + *path;
        }
    }
    row.Next(ImGui::CalcTextSize("Columns").x + 330.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Columns");
    ImGui::SameLine();
    std::vector<std::string> picked = chosen_.empty() ? p.correlation.columns : chosen_;
    if (picker_.PickMany("##corr_columns", picked, numbers, true, 320.0f)) {
        chosen_ = picked;
        pick_i_ = 0;
        pick_j_ = 1;
    }
}

void CorrelationView::RenderHeatmap(const ColumnCorrelations& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t n = m.size();
    float label_w = 0.0f;
    for (const auto& c : m.columns) label_w = std::max(label_w, ImGui::CalcTextSize(c.c_str()).x);
    label_w = std::min(label_w + 8.0f, 170.0f);
    const float line = ImGui::GetTextLineHeight();
    const float header = line + 6.0f;
    const float gap = 2.0f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 grid(origin.x + label_w, origin.y + header);
    const float total_w = label_w + static_cast<float>(n) * (cell_ + gap);
    const float total_h = header + static_cast<float>(n) * (cell_ + gap);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 clip_min = ImGui::GetWindowPos();
    const ImVec2 clip_max(clip_min.x + ImGui::GetWindowSize().x, clip_min.y + ImGui::GetWindowSize().y);
    const bool hover_ok = ImGui::IsWindowHovered();
    const ImVec2 mouse = ImGui::GetMousePos();
    int hover_i = -1, hover_j = -1;

    for (size_t j = 0; j < n; ++j) {
        const float x = grid.x + static_cast<float>(j) * (cell_ + gap);
        if (x + cell_ < clip_min.x || x > clip_max.x) continue;
        const std::string label = Fit(m.columns[j], cell_);
        dl->AddText(ImVec2(x + (cell_ - ImGui::CalcTextSize(label.c_str()).x) / 2, origin.y), ui::ToU32(t.text_dim), label.c_str());
        if (hover_ok && mouse.x >= x && mouse.x < x + cell_ && mouse.y >= origin.y && mouse.y < grid.y) ImGui::SetTooltip("%s", m.columns[j].c_str());
    }
    for (size_t i = 0; i < n; ++i) {
        const float y = grid.y + static_cast<float>(i) * (cell_ + gap);
        if (y + cell_ < clip_min.y || y > clip_max.y) continue;  // only the rows in view are drawn
        const std::string label = Fit(m.columns[i], label_w - 8.0f);
        dl->AddText(ImVec2(origin.x, y + (cell_ - line) / 2), ui::ToU32(t.text_dim), label.c_str());
        for (size_t j = 0; j < n; ++j) {
            const float x = grid.x + static_cast<float>(j) * (cell_ + gap);
            if (x + cell_ < clip_min.x || x > clip_max.x) continue;
            const double r = m.At(i, j);
            const ImVec4 fill = std::isfinite(r) ? ui::DivergingAt(t, static_cast<float>(r)) : t.bg_input;
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cell_, y + cell_), ui::ToU32(fill), 3.0f);
            const bool picked = static_cast<int>(i) == pick_i_ && static_cast<int>(j) == pick_j_;
            if (picked) dl->AddRect(ImVec2(x, y), ImVec2(x + cell_, y + cell_), ui::ToU32(ui::ReadableOn(t, fill)), 3.0f, 0, 2.0f);
            if (show_values_ && cell_ >= 34.0f) {
                char text[16];
                if (std::isfinite(r)) std::snprintf(text, sizeof(text), "%.2f", r);
                else std::snprintf(text, sizeof(text), "-");
                const ImVec2 ts = ImGui::CalcTextSize(text);
                dl->AddText(ImVec2(x + (cell_ - ts.x) / 2, y + (cell_ - ts.y) / 2), ui::ToU32(ui::ReadableOn(t, fill)), text);
            }
            if (hover_ok && mouse.x >= x && mouse.x < x + cell_ && mouse.y >= y && mouse.y < y + cell_) {
                hover_i = static_cast<int>(i);
                hover_j = static_cast<int>(j);
            }
        }
    }
    ImGui::Dummy(ImVec2(total_w, total_h));
    if (hover_i >= 0) {
        const double r = m.At(static_cast<size_t>(hover_i), static_cast<size_t>(hover_j));
        if (hover_i == hover_j) ImGui::SetTooltip("%s with itself: r = 1", m.columns[static_cast<size_t>(hover_i)].c_str());
        else
            ImGui::SetTooltip("%s \xC2\xB7 %s\nr = %+.4f (%s)\n%s rows have both", m.columns[static_cast<size_t>(hover_i)].c_str(),
                              m.columns[static_cast<size_t>(hover_j)].c_str(), r, ev::CorrelationStrength(r),
                              ev::Number(m.PairRows(static_cast<size_t>(hover_i), static_cast<size_t>(hover_j))).c_str());
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            pick_i_ = hover_i;
            pick_j_ = hover_j;
        }
    }
    if (show_legend_) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_faint, "-1");
        ImGui::SameLine();
        const ImVec2 bar = ImGui::GetCursorScreenPos();
        const float w = 160.0f, h = line * 0.6f;
        for (int k = 0; k < 64; ++k) {
            const float v = -1.0f + 2.0f * static_cast<float>(k) / 63.0f;
            const float x0 = bar.x + w * static_cast<float>(k) / 64.0f;
            dl->AddRectFilled(ImVec2(x0, bar.y + (line - h) / 2), ImVec2(x0 + w / 64.0f + 0.5f, bar.y + (line + h) / 2),
                              ui::ToU32(ui::DivergingAt(t, v)));
        }
        ImGui::Dummy(ImVec2(w, line));
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "+1");
    }
}

void CorrelationView::RenderPair(const ColumnCorrelations& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t i = static_cast<size_t>(pick_i_), j = static_cast<size_t>(pick_j_);
    ImGui::PushTextWrapPos(0.0f);
    if (i == j) {
        ImGui::TextColored(t.text_dim, "%s", m.columns[i].c_str());
        ImGui::TextColored(t.text_dim, "A column with itself: r = 1.");
    } else {
        ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s", m.columns[i].c_str(), m.columns[j].c_str());
        const double r = m.At(i, j);
        if (std::isfinite(r)) ImGui::Text("r = %+.4f", r);
        ImGui::TextUnformatted(ev::CorrelationSentence(r).c_str());
        ImGui::TextColored(t.text_dim, "%s rows have both values.", ev::Number(m.PairRows(i, j)).c_str());
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "STRONGEST PAIRS");
    std::vector<std::tuple<double, size_t, size_t>> pairs;
    for (size_t a = 0; a < m.size(); ++a)
        for (size_t b = a + 1; b < m.size(); ++b)
            if (std::isfinite(m.At(a, b))) pairs.emplace_back(m.At(a, b), a, b);
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto& x, const auto& y) { return std::fabs(std::get<0>(x)) > std::fabs(std::get<0>(y)); });
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(pairs.size()));
    while (clipper.Step()) {
        for (int k = clipper.DisplayStart; k < clipper.DisplayEnd; ++k) {
            const auto& [r, a, b] = pairs[static_cast<size_t>(k)];
            char label[32];
            std::snprintf(label, sizeof(label), "%+.2f", r);
            ImGui::PushID(k);
            const bool strong = std::fabs(r) >= 0.6;
            ImGui::PushStyleColor(ImGuiCol_Text, strong ? t.text : t.text_dim);
            const std::string text = std::string(label) + "  " + m.columns[a] + " \xC2\xB7 " + m.columns[b];
            const bool picked = (static_cast<size_t>(pick_i_) == a && static_cast<size_t>(pick_j_) == b) ||
                                (static_cast<size_t>(pick_i_) == b && static_cast<size_t>(pick_j_) == a);
            if (ImGui::Selectable(text.c_str(), picked)) {
                pick_i_ = static_cast<int>(a);
                pick_j_ = static_cast<int>(b);
            }
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
    }
}

}  // namespace cyxwiz
