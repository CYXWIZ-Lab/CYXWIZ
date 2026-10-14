#include "table_viewer.h"
#include "../../core/async_task_manager.h"
#include "../../core/dataset_profiler.h"
#include "../../data/arrow_data_table.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>

namespace cyxwiz {

void TableViewerPanel::ComputeColumnKinds(TableTab* tab) {
    if (!tab || !tab->HasData()) return;
    const size_t cols = tab->GetColumnCount();
    const size_t rows = tab->GetRowCount();
    tab->column_kinds.assign(cols, TableColumnKind{});
    for (size_t c = 0; c < cols; c++) {
        bool numeric = false;
        double min_v = std::numeric_limits<double>::max();
        double max_v = std::numeric_limits<double>::lowest();
        for (size_t r = 0; r < rows; r++) {
            const auto cell = tab->GetCell(r, c);
            double v = 0.0;
            if (const auto* d = std::get_if<double>(&cell)) v = *d;
            else if (const auto* n = std::get_if<int64_t>(&cell)) v = static_cast<double>(*n);
            else continue;
            if (std::isnan(v) || std::isinf(v)) continue;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            numeric = true;
        }
        auto& kind = tab->column_kinds[c];
        kind.type = numeric ? "Numeric" : "Text";
        if (numeric) {
            kind.min_val = min_v;
            kind.max_val = max_v;
        }
    }
}

void TableViewerPanel::StartProfile(TableTab* tab) {
    if (!tab || !tab->table || tab->profile_task) return;
    tab->profile_stale = false;
    // Converted here: the worker must not read the table while a cell is edited.
    auto table = ArrowFromDataTable(*tab->table);
    auto result = std::make_shared<DatasetProfile>();
    const int uid = tab->uid;
    std::weak_ptr<int> alive = alive_;
    tab->profile_task = AsyncTaskManager::Instance().RunAsync(
        "Profile " + tab->filename,
        [table, result](LambdaTask& task) {
            ProfileOptions options;
            options.should_stop = [&task] { return task.ShouldStop(); };
            if (table) *result = ProfileArrow(table, options);
            else result->error = "The table could not be read for profiling.";
        },
        nullptr,
        [this, alive, uid, result](bool, const std::string&) {
            if (alive.expired()) return;
            for (auto& t : tabs_) {
                if (t->uid != uid) continue;
                t->profile_task = 0;
                t->profile = result;
            }
        },
        alive_);
}

void TableViewerPanel::SortByColumn(TableTab* tab, int column) {
    if (!tab || !tab->HasData() || column < 0) return;

    size_t rows = tab->GetRowCount();

    tab->sorted_indices.resize(rows);
    std::iota(tab->sorted_indices.begin(), tab->sorted_indices.end(), size_t(0));

    bool is_numeric = (column < static_cast<int>(tab->column_kinds.size()) &&
                       tab->column_kinds[column].type == "Numeric");
    bool ascending = tab->sort_ascending;

    std::sort(tab->sorted_indices.begin(), tab->sorted_indices.end(),
        [&](size_t a, size_t b) {
            if (is_numeric) {
                auto get_val = [&](size_t r) -> double {
                    auto cell = tab->GetCell(r, column);
                    if (std::holds_alternative<double>(cell))
                        return std::get<double>(cell);
                    if (std::holds_alternative<int64_t>(cell))
                        return static_cast<double>(std::get<int64_t>(cell));
                    return ascending ? std::numeric_limits<double>::max()
                                    : std::numeric_limits<double>::lowest();
                };
                double va = get_val(a), vb = get_val(b);
                return ascending ? va < vb : va > vb;
            } else {
                std::string sa = tab->GetCellAsString(a, column);
                std::string sb = tab->GetCellAsString(b, column);
                return ascending ? sa < sb : sa > sb;
            }
        });

    spdlog::info("Sorted by column {} ({})", column, ascending ? "ascending" : "descending");
}

void TableViewerPanel::RenderStatsSidebar(TableTab* tab) {
    // A tinted area, no outline (owner rule: no lines around areas).
    const ui::Tokens& tk = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(tk.bg_window, tk.text, 0.03f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
    ImGui::BeginChild("StatsSidebar", ImVec2(stats_sidebar_width_, -30.0f), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();

    // Cell selection info at top
    ImGui::TextDisabled("Selection");
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text(ICON_FA_CROSSHAIRS " Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }
    ImGui::Spacing();
    ImGui::Spacing();

    const int column = tab->selected_column;
    if (column < 0 || column >= static_cast<int>(tab->column_kinds.size())) {
        ImGui::TextDisabled("Click a cell to view");
        ImGui::TextDisabled("column statistics");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        return;
    }
    if (tab->profile_stale) StartProfile(tab);

    const auto& kind = tab->column_kinds[column];
    const bool numeric = kind.type == "Numeric";
    ImGui::Text(ICON_FA_CHART_COLUMN " %s", tab->GetHeaders()[column].c_str());
    ImGui::Spacing();
    ImGui::TextColored(numeric ? tk.info : tk.caution, "%s %s", numeric ? ICON_FA_HASHTAG : ICON_FA_FONT, kind.type.c_str());
    ImGui::Spacing();

    const ProfiledColumn* pc = tab->profile && tab->profile->ok() && column < static_cast<int>(tab->profile->columns.size())
                                   ? &tab->profile->columns[static_cast<size_t>(column)]
                                   : nullptr;
    const auto row = [](const char* label, const std::string& value) {
        ImGui::TextDisabled("%s", label);
        ImGui::SameLine(90);
        ImGui::TextUnformatted(value.c_str());
    };
    const auto number = [](double v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        return std::string(buf);
    };
    if (!pc) {
        if (tab->profile_task) ImGui::TextColored(tk.info, ICON_FA_SPINNER " Profiling");
        else if (tab->profile && !tab->profile->ok()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(tk.error, "%s", tab->profile->error.c_str());
            ImGui::PopTextWrapPos();
        }
    } else {
        row("Rows", std::to_string(pc->facts.rows));
        row("Missing", std::to_string(pc->missing));
        row("Distinct", std::to_string(pc->facts.distinct));
        if (pc->empty_text > 0) row("Empty text", std::to_string(pc->empty_text));
        ImGui::Spacing();
        if (pc->numeric) {
            row("Min", number(pc->min));
            row("Max", number(pc->max));
            row("Mean", number(pc->mean));
            row("Std Dev", number(pc->std));
            ImGui::Spacing();
            ImGui::TextDisabled("Percentiles:");
            row("  25%", number(pc->q1));
            row("  50%", number(pc->median));
            row("  75%", number(pc->q3));
            ImGui::Spacing();
            ImGui::TextDisabled("Distribution:");
            RenderMiniHistogram(tab, column);
        } else {
            ImGui::TextDisabled("Top Values:");
            for (size_t i = 0; i < std::min<size_t>(pc->top.size(), 5); ++i) {
                const auto& [val, cnt] = pc->top[i];
                std::string display = val.length() > 15 ? val.substr(0, 15) + "..." : val;
                ImGui::BulletText("%s (%zu)", display.c_str(), cnt);
            }
        }
        if (tab->profile_task) ImGui::TextColored(tk.info, ICON_FA_SPINNER " Profiling again");
    }
    if (numeric) {
        ImGui::Spacing();
        // Plot: the Plot window on this column's histogram.
        if (ui::SecondaryButton(ICON_FA_CHART_BAR " Plot")) {
            OpenPlot(tab, plot::Kind::Histogram, column);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}


void TableViewerPanel::RenderMiniHistogram(TableTab* tab, int column) {
    if (!tab || !tab->profile || column < 0 || column >= static_cast<int>(tab->profile->columns.size())) return;
    const ProfiledColumn& pc = tab->profile->columns[static_cast<size_t>(column)];
    if (!pc.numeric || pc.hist_counts.empty()) return;

    // Small ImPlot histogram
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(2, 2));

    if (ImPlot::BeginPlot("##MiniHist", ImVec2(stats_sidebar_width_ - 20, 60),
        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
        ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {

        ImPlot::SetupAxes(nullptr, nullptr,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_AutoFit);

        std::vector<double> bins_d;
        bins_d.reserve(pc.hist_counts.size());
        for (size_t n : pc.hist_counts) bins_d.push_back(static_cast<double>(n));
        ImPlot::SetNextFillStyle(ui::WithAlpha(ui::CurrentTokens().series[0], 0.7f));
        ImPlot::PlotBars("##bins", bins_d.data(), static_cast<int>(bins_d.size()), 0.8);

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleVar();

    // Min/max labels
    ImGui::TextDisabled("%.2g", pc.min);
    ImGui::SameLine(stats_sidebar_width_ - 60);
    ImGui::TextDisabled("%.2g", pc.max);
}

std::vector<double> TableViewerPanel::GetColumnAsDoubles(TableTab* tab, int column) const {
    std::vector<double> result;
    if (!tab || !tab->table || column < 0) return result;

    size_t row_count = tab->table->GetRowCount();
    result.reserve(row_count);

    for (size_t r = 0; r < row_count; r++) {
        auto cell = tab->table->GetCell(r, column);
        if (std::holds_alternative<double>(cell)) {
            double val = std::get<double>(cell);
            if (!std::isnan(val) && !std::isinf(val)) {
                result.push_back(val);
            }
        } else if (std::holds_alternative<int64_t>(cell)) {
            result.push_back(static_cast<double>(std::get<int64_t>(cell)));
        }
    }

    return result;
}


}  // namespace cyxwiz

