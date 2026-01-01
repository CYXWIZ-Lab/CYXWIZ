#include "visualization_panel.h"
#include <cyxwiz/stats_utils.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz {

VisualizationPanel::VisualizationPanel()
    : Panel("Visualizer", false) {
    spdlog::info("VisualizationPanel initialized");
}

VisualizationPanel::~VisualizationPanel() = default;

void VisualizationPanel::SetData(const std::vector<std::string>& column_names,
                                  const std::vector<std::vector<std::string>>& rows) {
    column_names_ = column_names;
    rows_ = rows;
    viz_x_column_ = 0;
    viz_y_column_ = column_names_.size() > 1 ? 1 : 0;
    spdlog::info("VisualizationPanel: Received {} columns, {} rows",
                 column_names_.size(), rows_.size());
}

void VisualizationPanel::ClearData() {
    column_names_.clear();
    rows_.clear();
    viz_x_column_ = 0;
    viz_y_column_ = 1;
}

std::vector<double> VisualizationPanel::GetColumnAsDoubles(int col_index) const {
    std::vector<double> result;
    if (col_index < 0 || col_index >= (int)column_names_.size()) {
        return result;
    }

    result.reserve(rows_.size());
    for (const auto& row : rows_) {
        if (col_index < (int)row.size()) {
            try {
                double val = std::stod(row[col_index]);
                if (!std::isnan(val) && !std::isinf(val)) {
                    result.push_back(val);
                }
            } catch (...) {
                // Skip non-numeric values
            }
        }
    }
    return result;
}

void VisualizationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_SIMPLE " Visualizer###Visualizer", &visible_)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

        if (!HasData()) {
            ImGui::TextDisabled("No data loaded.");
            ImGui::TextDisabled("Use 'Open in Visualizer' from Data Explorer to send data here.");
        } else {
            // Compact toolbar
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 4));
            RenderToolbar();
            ImGui::PopStyleVar();

            ImGui::Separator();
            ImGui::Spacing();

            // Chart area - use all available space
            float chart_height = ImGui::GetContentRegionAvail().y - 10.0f;
            chart_height = std::max(chart_height, 300.0f);

            ImGui::BeginChild("ChartArea", ImVec2(-1, chart_height), false);
            {
                switch (chart_type_) {
                    case VizChartType::Histogram: RenderHistogramChart(); break;
                    case VizChartType::Scatter: RenderScatterChart(); break;
                    case VizChartType::Bar: RenderBarChart(); break;
                    case VizChartType::Box: RenderBoxChart(); break;
                }
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void VisualizationPanel::RenderToolbar() {
    // Chart type selector with icons
    ImGui::AlignTextToFramePadding();
    ImGui::Text(ICON_FA_CHART_SIMPLE " Chart:");
    ImGui::SameLine();

    ImGui::PushID("VizChartTypeSelector");
    if (ImGui::RadioButton(ICON_FA_CHART_BAR " Histogram", chart_type_ == VizChartType::Histogram))
        chart_type_ = VizChartType::Histogram;
    ImGui::SameLine();
    if (ImGui::RadioButton(ICON_FA_CHART_SCATTER " Scatter", chart_type_ == VizChartType::Scatter))
        chart_type_ = VizChartType::Scatter;
    ImGui::SameLine();
    if (ImGui::RadioButton(ICON_FA_CHART_COLUMN " Bar", chart_type_ == VizChartType::Bar))
        chart_type_ = VizChartType::Bar;
    ImGui::SameLine();
    if (ImGui::RadioButton(ICON_FA_CUBE " Box", chart_type_ == VizChartType::Box))
        chart_type_ = VizChartType::Box;
    ImGui::PopID();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Column selectors
    auto ColumnCombo = [&](const char* label, int& selected, const char* combo_id) {
        ImGui::SetNextItemWidth(180);
        ImGui::PushID(combo_id);
        if (ImGui::BeginCombo("##col",
            selected >= 0 && selected < (int)column_names_.size()
                ? column_names_[selected].c_str()
                : "Select column...")) {
            for (int i = 0; i < (int)column_names_.size(); i++) {
                ImGui::PushID(i);
                if (ImGui::Selectable(column_names_[i].c_str(), selected == i)) {
                    selected = i;
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::PopID();
    };

    ImGui::Text("X:");
    ImGui::SameLine();
    ColumnCombo("##XCol", viz_x_column_, "VizXColCombo");

    if (chart_type_ == VizChartType::Scatter) {
        ImGui::SameLine();
        ImGui::Text("Y:");
        ImGui::SameLine();
        ColumnCombo("##YCol", viz_y_column_, "VizYColCombo");
    }

    // Data info
    ImGui::SameLine();
    ImGui::TextDisabled("| %zu rows", rows_.size());
}

void VisualizationPanel::RenderHistogramChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    // Compute statistics
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    std::string col_name = column_names_[viz_x_column_];
    std::string title = "Distribution of " + col_name;

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.7f);

    ImGui::PushID("VizHistogramPlot");
    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxes(col_name.c_str(), "Frequency", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        // Plot histogram
        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.5f, 0.9f, 0.7f));
        ImPlot::PlotHistogram(col_name.c_str(), data.data(), (int)data.size(), 30);

        // Mean line
        double mean_x[] = {mean, mean};
        double mean_y[] = {0, (double)data.size() / 10.0};
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), 2.0f);
        ImPlot::PlotLine("Mean", mean_x, mean_y, 2);

        ImPlot::Annotation(mean, mean_y[1] * 0.8, ImVec4(1,1,1,1), ImVec2(5, -5), true,
                          "Mean: %.2f", mean);

        ImPlot::EndPlot();
    }
    ImGui::PopID();

    ImPlot::PopStyleVar();

    // Stats footer
    ImGui::TextDisabled("Min: %.4g | Max: %.4g | Mean: %.4g | Count: %zu",
                       min_val, max_val, mean, data.size());
}

void VisualizationPanel::RenderScatterChart() {
    std::vector<double> x_data = GetColumnAsDoubles(viz_x_column_);
    std::vector<double> y_data = GetColumnAsDoubles(viz_y_column_);

    if (x_data.empty() || y_data.empty()) {
        ImGui::TextDisabled("Selected columns have no numeric data");
        return;
    }

    // Match sizes
    size_t n = std::min(x_data.size(), y_data.size());
    x_data.resize(n);
    y_data.resize(n);

    std::string x_name = column_names_[viz_x_column_];
    std::string y_name = column_names_[viz_y_column_];
    std::string title = y_name + " vs " + x_name;

    // Compute correlation
    double corr = stats::PearsonCorrelation(x_data, y_data);

    ImGui::PushID("VizScatterPlot");
    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1), ImPlotFlags_None)) {
        ImPlot::SetupAxes(x_name.c_str(), y_name.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.2f, 0.6f, 1.0f, 0.8f), IMPLOT_AUTO, ImVec4(0.2f, 0.6f, 1.0f, 1.0f));
        ImPlot::PlotScatter("Data Points", x_data.data(), y_data.data(), (int)n);

        ImPlot::EndPlot();
    }
    ImGui::PopID();

    ImGui::TextDisabled("Points: %zu | Correlation (r): %.4f", n, corr);
}

void VisualizationPanel::RenderBarChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    int n = std::min((int)data.size(), 100);
    std::string col_name = column_names_[viz_x_column_];
    std::string title = col_name + " Values";

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.8f);

    ImGui::PushID("VizBarPlot");
    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1), ImPlotFlags_None)) {
        ImPlot::SetupAxes("Row Index", col_name.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        ImPlot::SetNextFillStyle(ImVec4(0.4f, 0.7f, 0.4f, 0.8f));
        ImPlot::PlotBars(col_name.c_str(), data.data(), n, 0.67);

        ImPlot::EndPlot();
    }
    ImGui::PopID();

    ImPlot::PopStyleVar();

    double sum = std::accumulate(data.begin(), data.begin() + n, 0.0);
    ImGui::TextDisabled("Showing: %d bars | Sum: %.4g | Avg: %.4g", n, sum, sum / n);
}

void VisualizationPanel::RenderBoxChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    // Compute statistics
    std::sort(data.begin(), data.end());
    size_t n = data.size();

    double min_val = data.front();
    double max_val = data.back();
    double q1 = data[n / 4];
    double median = data[n / 2];
    double q3 = data[3 * n / 4];
    double iqr = q3 - q1;
    double lower_fence = q1 - 1.5 * iqr;
    double upper_fence = q3 + 1.5 * iqr;
    double lower_whisker = std::max(min_val, lower_fence);
    double upper_whisker = std::min(max_val, upper_fence);

    std::string col_name = column_names_[viz_x_column_];
    std::string title = "Box Plot of " + col_name;

    // Count outliers
    int outlier_count = 0;
    std::vector<double> outlier_vals;
    for (double v : data) {
        if (v < lower_fence || v > upper_fence) {
            outlier_count++;
            if (outlier_vals.size() < 100) outlier_vals.push_back(v);
        }
    }

    ImGui::PushID("VizBoxPlot");
    if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1), ImPlotFlags_None)) {
        ImPlot::SetupAxes("", col_name.c_str(), ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, -1, 1);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        // IQR box
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.3f, 0.5f, 0.9f, 0.6f));
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
        double box_x[] = {-0.3, 0.3, 0.3, -0.3, -0.3};
        double box_y[] = {q1, q1, q3, q3, q1};
        ImPlot::PlotLine("IQR Box", box_x, box_y, 5);
        ImPlot::PopStyleColor(2);

        // Median line
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 3.0f);
        double med_x[] = {-0.3, 0.3};
        double med_y[] = {median, median};
        ImPlot::PlotLine("Median", med_x, med_y, 2);

        // Whiskers
        ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 2.0f);
        double whisker_x[] = {0, 0};
        double whisker_lo[] = {lower_whisker, q1};
        ImPlot::PlotLine("##LowerWhisker", whisker_x, whisker_lo, 2);

        ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 2.0f);
        double whisker_hi[] = {q3, upper_whisker};
        ImPlot::PlotLine("##UpperWhisker", whisker_x, whisker_hi, 2);

        // Whisker caps
        ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 2.0f);
        double cap_x[] = {-0.15, 0.15};
        double cap_lo[] = {lower_whisker, lower_whisker};
        double cap_hi[] = {upper_whisker, upper_whisker};
        ImPlot::PlotLine("##LowerCap", cap_x, cap_lo, 2);
        ImPlot::PlotLine("##UpperCap", cap_x, cap_hi, 2);

        // Outliers
        if (!outlier_vals.empty()) {
            std::vector<double> outlier_x(outlier_vals.size(), 0.0);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(1.0f, 0.3f, 0.3f, 0.8f));
            ImPlot::PlotScatter("Outliers", outlier_x.data(), outlier_vals.data(), (int)outlier_vals.size());
        }

        ImPlot::EndPlot();
    }
    ImGui::PopID();

    ImGui::TextDisabled("Q1: %.4g | Median: %.4g | Q3: %.4g | IQR: %.4g | Outliers: %d",
                       q1, median, q3, iqr, outlier_count);
}

} // namespace cyxwiz
