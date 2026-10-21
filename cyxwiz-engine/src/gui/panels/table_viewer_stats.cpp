#include "table_viewer.h"
#include "../../core/async_task_manager.h"
#include "../../core/column_role_store.h"
#include "../../core/dataset_catalog.h"
#include "../../core/dataset_profiler.h"
#include "../../core/session_query_service.h"
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

void TableViewerPanel::StartProfile(TableTab* tab) {
    if (!tab || !tab->grid.table || tab->profile_task) return;
    tab->profile_stale = false;
    // A whole loaded dataset, unedited: profiled as Data Studio profiles it
    // (its queries through the session query service, with the project's
    // missing texts), so the answers either one saved serve the other.
    std::string dataset;
    ProfileOptions options;
    if (!tab->dataset.empty() && !tab->is_dirty) {
        if (const auto entry = DatasetCatalog::Instance().Resolve(tab->dataset)) {
            dataset = tab->dataset;
            if (const auto* roles = ProjectColumnRoles().Find(RoleSourceKey(entry->source_path, entry->Shown())))
                options.missing_text = roles->missing_text;
        }
    }
    // Otherwise the rows as read while the table is unedited; after an edit,
    // converted here: the worker must not read the table while a cell is edited.
    std::shared_ptr<arrow::Table> table;
    if (dataset.empty()) table = tab->source && !tab->is_dirty ? tab->source : ArrowFromDataTable(*tab->grid.table);
    auto result = std::make_shared<DatasetProfile>();
    const int uid = tab->uid;
    std::weak_ptr<int> alive = alive_;
    tab->profile_task = AsyncTaskManager::Instance().RunAsync(
        "Profile " + tab->filename,
        [dataset, table, options, result](LambdaTask& task) {
            ProfileOptions opts = options;
            opts.should_stop = [&task] { return task.ShouldStop(); };
            opts.progress = [&task](float f, const std::string& what) { task.ReportProgress(f, what); };
            if (!dataset.empty())
                *result = ProfileTable(dataset, [](const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); },
                                       opts);
            else if (table) *result = ProfileArrow(table, opts);
            else result->error = "The table could not be read for profiling.";
            if (!result->ok()) task.MarkFailed(result->error);
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

void TableViewerPanel::RenderStatsSidebar(TableTab* tab) {
    // A tinted area, no outline (owner rule: no lines around areas).
    const ui::Tokens& tk = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(tk.bg_window, tk.text, 0.03f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
    ImGui::BeginChild("StatsSidebar", ImVec2(stats_sidebar_width_, -30.0f), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();

    // Cell selection info at top
    ImGui::TextDisabled("Selection");
    if (tab->grid.selected_row >= 0 && tab->grid.selected_col >= 0) {
        ImGui::Text(ICON_FA_CROSSHAIRS " Row %d, Col %d", tab->grid.selected_row + 1, tab->grid.selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }
    ImGui::Spacing();
    ImGui::Spacing();

    const int column = tab->grid.selected_column;
    if (column < 0 || column >= static_cast<int>(tab->grid.column_kinds.size())) {
        ImGui::TextDisabled("Click a cell to view");
        ImGui::TextDisabled("column statistics");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        return;
    }
    if (tab->profile_stale) StartProfile(tab);

    const auto& kind = tab->grid.column_kinds[column];
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

}  // namespace cyxwiz

