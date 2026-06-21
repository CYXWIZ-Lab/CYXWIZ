#include "table_viewer.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace cyxwiz {

void TableViewerPanel::ComputeColumnStats(TableTab* tab) {
    if (!tab || !tab->HasData()) return;

    size_t cols = tab->GetColumnCount();
    size_t rows = tab->GetRowCount();

    tab->column_stats.resize(cols);

    // For lazy loading with many rows, sample instead of scanning all
    bool sample_mode = tab->use_lazy_loading && rows > 10000;
    size_t sample_size = sample_mode ? 5000 : rows;
    size_t step = sample_mode ? rows / sample_size : 1;

    for (size_t c = 0; c < cols; c++) {
        auto& stats = tab->column_stats[c];
        stats.count = rows;
        stats.null_count = 0;

        bool has_numeric = false;
        double min_v = std::numeric_limits<double>::max();
        double max_v = std::numeric_limits<double>::lowest();
        double sum_v = 0;
        std::vector<double> numeric_values;  // For std dev, median, histogram
        std::map<std::string, int> value_counts;

        numeric_values.reserve(sample_mode ? sample_size : rows);

        for (size_t i = 0; i < sample_size; i++) {
            size_t r = sample_mode ? i * step : i;
            if (r >= rows) break;

            auto cell = tab->GetCell(r, c);
            if (std::holds_alternative<std::monostate>(cell)) {
                stats.null_count++;
            } else if (std::holds_alternative<double>(cell)) {
                double v = std::get<double>(cell);
                if (!std::isnan(v) && !std::isinf(v)) {
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    sum_v += v;
                    numeric_values.push_back(v);
                    has_numeric = true;
                } else {
                    stats.null_count++;
                }
            } else if (std::holds_alternative<int64_t>(cell)) {
                double v = static_cast<double>(std::get<int64_t>(cell));
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                sum_v += v;
                numeric_values.push_back(v);
                has_numeric = true;
            } else {
                std::string s = tab->GetCellAsString(r, c);
                value_counts[s]++;
            }
        }

        size_t numeric_count = numeric_values.size();
        if (has_numeric && numeric_count > 0) {
            stats.type = "Numeric";
            stats.min_val = min_v;
            stats.max_val = max_v;
            stats.sum = sum_v;
            stats.avg = sum_v / numeric_count;

            // Compute standard deviation
            double sq_sum = 0;
            for (double v : numeric_values) {
                sq_sum += (v - stats.avg) * (v - stats.avg);
            }
            stats.std_dev = std::sqrt(sq_sum / numeric_count);

            // Compute median and quartiles (sort a copy)
            std::vector<double> sorted_vals = numeric_values;
            std::sort(sorted_vals.begin(), sorted_vals.end());
            size_t n = sorted_vals.size();
            stats.median = sorted_vals[n / 2];
            stats.q1 = sorted_vals[n / 4];
            stats.q3 = sorted_vals[3 * n / 4];

            // Compute histogram bins (16 bins)
            constexpr int NUM_BINS = 16;
            stats.histogram_bins.resize(NUM_BINS, 0);
            double range = max_v - min_v;
            if (range > 0) {
                for (double v : numeric_values) {
                    int bin = static_cast<int>((v - min_v) / range * (NUM_BINS - 1));
                    bin = std::clamp(bin, 0, NUM_BINS - 1);
                    stats.histogram_bins[bin]++;
                }
            } else {
                // All values are the same
                stats.histogram_bins[NUM_BINS / 2] = static_cast<int>(numeric_count);
            }
        } else {
            stats.type = "Text";
            // Get top 5 values
            std::vector<std::pair<std::string, int>> sorted_values(
                value_counts.begin(), value_counts.end());
            std::sort(sorted_values.begin(), sorted_values.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
            size_t top_n = std::min(size_t(5), sorted_values.size());
            stats.top_values.assign(sorted_values.begin(), sorted_values.begin() + top_n);
        }
        stats.computed = true;
    }

    spdlog::info("Computed column stats for {} columns", cols);
}

void TableViewerPanel::SortByColumn(TableTab* tab, int column) {
    if (!tab || !tab->HasData() || column < 0) return;

    size_t rows = tab->GetRowCount();

    // For very large lazy-loaded files, sorting is disabled (too slow)
    if (tab->use_lazy_loading && rows > 100000) {
        spdlog::warn("Sorting disabled for lazy-loaded files with >100K rows");
        return;
    }

    tab->sorted_indices.resize(rows);
    std::iota(tab->sorted_indices.begin(), tab->sorted_indices.end(), size_t(0));

    bool is_numeric = (column < static_cast<int>(tab->column_stats.size()) &&
                       tab->column_stats[column].type == "Numeric");
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
    ImGui::BeginChild("StatsSidebar", ImVec2(stats_sidebar_width_, 0), true);

    // Cell selection info at top
    ImGui::TextDisabled("Selection");
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text(ICON_FA_CROSSHAIRS " Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }
    ImGui::Separator();
    ImGui::Spacing();

    if (tab->selected_column < 0 || tab->selected_column >= static_cast<int>(tab->column_stats.size())) {
        ImGui::TextDisabled("Click a cell to view");
        ImGui::TextDisabled("column statistics");
        ImGui::EndChild();
        return;
    }

    auto& stats = tab->column_stats[tab->selected_column];
    auto& headers = tab->table->GetHeaders();

    // Column name with icon
    ImGui::Text(ICON_FA_CHART_COLUMN " %s", headers[tab->selected_column].c_str());
    ImGui::Separator();

    // Type badge
    ImVec4 type_color = (stats.type == "Numeric")
        ? ImVec4(0.2f, 0.6f, 0.9f, 1.0f)  // Blue
        : ImVec4(0.9f, 0.6f, 0.2f, 1.0f);  // Orange
    ImGui::TextColored(type_color, "%s %s",
        stats.type == "Numeric" ? ICON_FA_HASHTAG : ICON_FA_FONT,
        stats.type.c_str());

    ImGui::Spacing();

    // Count info
    ImGui::TextDisabled("Count");
    ImGui::SameLine(90);
    ImGui::Text("%zu", stats.count);

    ImGui::TextDisabled("Nulls");
    ImGui::SameLine(90);
    ImGui::Text("%zu", stats.null_count);

    ImGui::Separator();

    if (stats.type == "Numeric") {
        ImGui::TextDisabled("Min");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.min_val);

        ImGui::TextDisabled("Max");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.max_val);

        ImGui::TextDisabled("Mean");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.avg);

        ImGui::TextDisabled("Std Dev");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.std_dev);

        ImGui::TextDisabled("Median");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.median);

        ImGui::Separator();

        // Percentiles
        ImGui::TextDisabled("Percentiles:");
        ImGui::TextDisabled("  25%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.q1);
        ImGui::TextDisabled("  50%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.median);
        ImGui::TextDisabled("  75%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.q3);

        ImGui::Separator();

        // Mini histogram
        ImGui::TextDisabled("Distribution:");
        RenderMiniHistogram(tab, tab->selected_column);

        ImGui::Separator();

        // Quick plot button
        if (ImGui::Button(ICON_FA_CHART_BAR " Plot", ImVec2(-1, 0))) {
            plot_popup_.type = QuickPlotType::Histogram;
            plot_popup_.x_column = tab->selected_column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Histogram of " + headers[tab->selected_column];
            plot_popup_.x_data = GetColumnAsDoubles(tab, tab->selected_column);
            show_plot_popup_ = true;
        }
    } else {
        ImGui::TextDisabled("Top Values:");
        for (auto& [val, cnt] : stats.top_values) {
            std::string display = val.length() > 15 ? val.substr(0, 15) + "..." : val;
            ImGui::BulletText("%s (%d)", display.c_str(), cnt);
        }
    }

    ImGui::EndChild();
}


void TableViewerPanel::RenderMiniHistogram(TableTab* tab, int column) {
    if (!tab || column < 0 || column >= static_cast<int>(tab->column_stats.size())) return;

    auto& stats = tab->column_stats[column];
    if (stats.type != "Numeric" || stats.histogram_bins.empty()) return;

    // Small ImPlot histogram
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(2, 2));

    if (ImPlot::BeginPlot("##MiniHist", ImVec2(stats_sidebar_width_ - 20, 60),
        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
        ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {

        ImPlot::SetupAxes(nullptr, nullptr,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_AutoFit);

        std::vector<double> bins_d(stats.histogram_bins.begin(), stats.histogram_bins.end());
        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.5f, 0.9f, 0.7f));
        ImPlot::PlotBars("##bins", bins_d.data(), static_cast<int>(bins_d.size()), 0.8);

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleVar();

    // Min/max labels
    ImGui::TextDisabled("%.2g", stats.min_val);
    ImGui::SameLine(stats_sidebar_width_ - 60);
    ImGui::TextDisabled("%.2g", stats.max_val);
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

