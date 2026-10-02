#include "table_viewer.h"
#include "../../core/plot_script.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace cyxwiz {

void TableViewerPanel::ShowQuickPlotPopup(TableTab* tab) {
    if (!tab) return;
    show_plot_popup_ = true;
}

void TableViewerPanel::RenderQuickPlot() {
    if (!show_plot_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_SIMPLE " Quick Plot", &show_plot_popup_)) {
        TableTab* tab = GetActiveTab();

        // Chart type selector - Row 1: Basic charts
        ImGui::Text("Chart Type:");
        ImGui::SameLine();

        if (ImGui::RadioButton("Histogram", plot_popup_.type == QuickPlotType::Histogram)) {
            plot_popup_.type = QuickPlotType::Histogram;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Bar", plot_popup_.type == QuickPlotType::Bar)) {
            plot_popup_.type = QuickPlotType::Bar;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Line", plot_popup_.type == QuickPlotType::Line)) {
            plot_popup_.type = QuickPlotType::Line;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scatter", plot_popup_.type == QuickPlotType::Scatter)) {
            plot_popup_.type = QuickPlotType::Scatter;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Box", plot_popup_.type == QuickPlotType::Box)) {
            plot_popup_.type = QuickPlotType::Box;
        }

        // Row 2: Extended charts
        ImGui::Text("          ");
        ImGui::SameLine();
        if (ImGui::RadioButton("Pie", plot_popup_.type == QuickPlotType::Pie)) {
            plot_popup_.type = QuickPlotType::Pie;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Stairs", plot_popup_.type == QuickPlotType::Stairs)) {
            plot_popup_.type = QuickPlotType::Stairs;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Stem", plot_popup_.type == QuickPlotType::Stem)) {
            plot_popup_.type = QuickPlotType::Stem;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Area", plot_popup_.type == QuickPlotType::Area)) {
            plot_popup_.type = QuickPlotType::Area;
        }

        // Column selector(s)
        if (tab && tab->table) {
            const auto& headers = tab->table->GetHeaders();

            ImGui::Text("X Column:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200);
            if (ImGui::BeginCombo("##XColumn",
                (plot_popup_.x_column >= 0 && plot_popup_.x_column < static_cast<int>(headers.size()))
                    ? headers[plot_popup_.x_column].c_str() : "Select...")) {
                for (int i = 0; i < static_cast<int>(headers.size()); i++) {
                    if (ImGui::Selectable(headers[i].c_str(), plot_popup_.x_column == i)) {
                        plot_popup_.x_column = i;
                        plot_popup_.x_data = GetColumnAsDoubles(tab, i);
                    }
                }
                ImGui::EndCombo();
            }

            // Y column for scatter
            if (plot_popup_.type == QuickPlotType::Scatter) {
                ImGui::SameLine();
                ImGui::Text("Y Column:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(200);
                if (ImGui::BeginCombo("##YColumn",
                    (plot_popup_.y_column >= 0 && plot_popup_.y_column < static_cast<int>(headers.size()))
                        ? headers[plot_popup_.y_column].c_str() : "Select...")) {
                    for (int i = 0; i < static_cast<int>(headers.size()); i++) {
                        if (ImGui::Selectable(headers[i].c_str(), plot_popup_.y_column == i)) {
                            plot_popup_.y_column = i;
                            plot_popup_.y_data = GetColumnAsDoubles(tab, i);
                        }
                    }
                    ImGui::EndCombo();
                }
            }
        }

        ImGui::Separator();

        // Plot area
        float plot_height = ImGui::GetContentRegionAvail().y - 40;
        if (plot_height < 200) plot_height = 200;

        if (!plot_popup_.x_data.empty()) {
            switch (plot_popup_.type) {
                case QuickPlotType::Histogram: {
                    if (ImPlot::BeginPlot("##Histogram", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Value", "Frequency", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.5f, 0.9f, 0.7f));
                        ImPlot::PlotHistogram("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()), 30);
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Bar: {
                    if (ImPlot::BeginPlot("##Bar", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextFillStyle(ImVec4(0.4f, 0.7f, 0.4f, 0.8f));
                        int n = std::min(static_cast<int>(plot_popup_.x_data.size()), 100);
                        ImPlot::PlotBars("Data", plot_popup_.x_data.data(), n, 0.67);
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Line: {
                    if (ImPlot::BeginPlot("##Line", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::PlotLine("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()));
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Scatter: {
                    if (!plot_popup_.y_data.empty()) {
                        if (ImPlot::BeginPlot("##Scatter", ImVec2(-1, plot_height))) {
                            ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                            int n = static_cast<int>(std::min(plot_popup_.x_data.size(), plot_popup_.y_data.size()));
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.2f, 0.6f, 1.0f, 0.8f));
                            ImPlot::PlotScatter("Data", plot_popup_.x_data.data(), plot_popup_.y_data.data(), n);
                            ImPlot::EndPlot();
                        }
                    } else {
                        ImGui::TextDisabled("Select Y column for scatter plot");
                    }
                    break;
                }

                case QuickPlotType::Box: {
                    if (ImPlot::BeginPlot("##Box", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("", "Value", ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_AutoFit);

                        // Compute box plot statistics
                        std::vector<double> sorted_data = plot_popup_.x_data;
                        std::sort(sorted_data.begin(), sorted_data.end());
                        size_t n = sorted_data.size();

                        if (n > 0) {
                            double q1 = sorted_data[n / 4];
                            double median = sorted_data[n / 2];
                            double q3 = sorted_data[3 * n / 4];
                            double iqr = q3 - q1;
                            double lower = std::max(sorted_data.front(), q1 - 1.5 * iqr);
                            double upper = std::min(sorted_data.back(), q3 + 1.5 * iqr);

                            // Draw box
                            double box_x[] = {-0.3, 0.3, 0.3, -0.3, -0.3};
                            double box_y[] = {q1, q1, q3, q3, q1};
                            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
                            ImPlot::PlotLine("##Box", box_x, box_y, 5);
                            ImPlot::PopStyleColor();

                            // Draw median
                            double med_x[] = {-0.3, 0.3};
                            double med_y[] = {median, median};
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 3.0f);
                            ImPlot::PlotLine("Median", med_x, med_y, 2);

                            // Draw whiskers
                            double whisker_x[] = {0, 0};
                            double whisker_lo[] = {lower, q1};
                            double whisker_hi[] = {q3, upper};
                            ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 2.0f);
                            ImPlot::PlotLine("##LowerWhisker", whisker_x, whisker_lo, 2);
                            ImPlot::PlotLine("##UpperWhisker", whisker_x, whisker_hi, 2);
                        }

                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Pie: {
                    if (ImPlot::BeginPlot("##Pie", ImVec2(-1, plot_height), ImPlotFlags_Equal)) {
                        // For pie chart, bin the data into categories
                        int num_bins = std::min(8, static_cast<int>(plot_popup_.x_data.size()));
                        if (num_bins > 0) {
                            double min_val = *std::min_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
                            double max_val = *std::max_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
                            double bin_width = (max_val - min_val) / num_bins;

                            std::vector<double> bin_counts(num_bins, 0);
                            std::vector<const char*> labels;
                            std::vector<std::string> label_strings;

                            for (double val : plot_popup_.x_data) {
                                int bin = static_cast<int>((val - min_val) / bin_width);
                                bin = std::clamp(bin, 0, num_bins - 1);
                                bin_counts[bin]++;
                            }

                            // Create labels
                            for (int i = 0; i < num_bins; i++) {
                                double lo = min_val + i * bin_width;
                                double hi = lo + bin_width;
                                label_strings.push_back(std::to_string(static_cast<int>(lo)) + "-" + std::to_string(static_cast<int>(hi)));
                            }
                            for (const auto& s : label_strings) {
                                labels.push_back(s.c_str());
                            }

                            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
                            ImPlot::PlotPieChart(labels.data(), bin_counts.data(), num_bins, 0, 0, 0.9, "%.0f", 90);
                        }
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Stairs: {
                    if (ImPlot::BeginPlot("##Stairs", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.7f, 0.3f, 1.0f), 2.0f);
                        ImPlot::PlotStairs("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()));
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Stem: {
                    if (ImPlot::BeginPlot("##Stem", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5, ImVec4(0.8f, 0.3f, 0.2f, 1.0f));
                        ImPlot::PlotStems("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()));
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Area: {
                    if (ImPlot::BeginPlot("##Area", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.6f, 0.9f, 0.5f));
                        ImPlot::PlotShaded("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()), 0);
                        // Draw line on top
                        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.5f, 0.8f, 1.0f), 2.0f);
                        ImPlot::PlotLine("##Line", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()));
                        ImPlot::EndPlot();
                    }
                    break;
                }

                default:
                    ImGui::TextDisabled("Select a chart type");
                    break;
            }
        } else {
            ImGui::TextDisabled("No data to plot. Select a column.");
        }

        // Stats footer
        if (!plot_popup_.x_data.empty()) {
            ImGui::Separator();
            double sum = std::accumulate(plot_popup_.x_data.begin(), plot_popup_.x_data.end(), 0.0);
            double mean = sum / plot_popup_.x_data.size();
            double min_val = *std::min_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
            double max_val = *std::max_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
            ImGui::Text("Points: %zu | Min: %.4g | Max: %.4g | Mean: %.4g",
                plot_popup_.x_data.size(), min_val, max_val, mean);

            // Action buttons
            ImGui::Separator();

            if (ImGui::Button(ICON_FA_CHART_LINE " Open in Visualizer")) {
                // Send selected column data to VisualizationPanel
                SendToVisualizer(plot_popup_.x_column);
                show_plot_popup_ = false;
            }
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_CODE " Plot with Python")) {
                // TOFIX134 P0 item 4: core/plot_script writes valid Python (all
                // values, literal title, full digits); it used to put "..." in
                // the data after 101 values and paste the title unescaped.
                using plotscript::Kind;
                Kind kind = Kind::Line;
                switch (plot_popup_.type) {
                    case QuickPlotType::Histogram: kind = Kind::Histogram; break;
                    case QuickPlotType::Bar: kind = Kind::Bar; break;
                    case QuickPlotType::Line: kind = Kind::Line; break;
                    case QuickPlotType::Scatter: kind = Kind::Scatter; break;
                    case QuickPlotType::Box: kind = Kind::Box; break;
                    case QuickPlotType::Pie: kind = Kind::Pie; break;
                    case QuickPlotType::Stairs: kind = Kind::Stairs; break;
                    case QuickPlotType::Stem: kind = Kind::Stem; break;
                    case QuickPlotType::Area: kind = Kind::Area; break;
                    default: kind = Kind::Line; break;
                }
                const std::string script =
                    plotscript::MatplotlibScript(kind, plot_popup_.title, plot_popup_.x_data, plot_popup_.y_data);
                ImGui::SetClipboardText(script.c_str());
                spdlog::info("Python matplotlib script copied to clipboard ({} bytes)", script.size());
            }
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_XMARK " Close")) {
                show_plot_popup_ = false;
            }
        }
    }
    ImGui::End();
}


}  // namespace cyxwiz

