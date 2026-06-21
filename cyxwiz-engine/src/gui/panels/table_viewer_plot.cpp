#include "table_viewer.h"
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
                // Generate matplotlib script
                std::string plot_type;
                switch (plot_popup_.type) {
                    case QuickPlotType::Histogram: plot_type = "hist"; break;
                    case QuickPlotType::Bar: plot_type = "bar"; break;
                    case QuickPlotType::Line: plot_type = "plot"; break;
                    case QuickPlotType::Scatter: plot_type = "scatter"; break;
                    case QuickPlotType::Box: plot_type = "boxplot"; break;
                    case QuickPlotType::Pie: plot_type = "pie"; break;
                    case QuickPlotType::Stairs: plot_type = "step"; break;
                    case QuickPlotType::Stem: plot_type = "stem"; break;
                    case QuickPlotType::Area: plot_type = "fill_between"; break;
                    default: plot_type = "plot"; break;
                }

                // Build data array string
                std::ostringstream data_ss;
                data_ss << "data = [";
                for (size_t i = 0; i < plot_popup_.x_data.size(); i++) {
                    if (i > 0) data_ss << ", ";
                    data_ss << plot_popup_.x_data[i];
                    if (i > 100) {
                        data_ss << ", ...";  // Truncate for large datasets
                        break;
                    }
                }
                data_ss << "]";

                std::ostringstream script;
                script << "import matplotlib.pyplot as plt\n";
                script << "import numpy as np\n\n";
                script << "# Data from Table Viewer\n";
                script << data_ss.str() << "\n\n";
                script << "plt.figure(figsize=(10, 6))\n";

                if (plot_type == "hist") {
                    script << "plt.hist(data, bins=30, edgecolor='black', alpha=0.7)\n";
                    script << "plt.xlabel('Value')\n";
                    script << "plt.ylabel('Frequency')\n";
                } else if (plot_type == "bar") {
                    script << "plt.bar(range(len(data)), data, alpha=0.7)\n";
                    script << "plt.xlabel('Index')\n";
                    script << "plt.ylabel('Value')\n";
                } else if (plot_type == "scatter" && !plot_popup_.y_data.empty()) {
                    std::ostringstream y_ss;
                    y_ss << "y_data = [";
                    for (size_t i = 0; i < plot_popup_.y_data.size() && i < 100; i++) {
                        if (i > 0) y_ss << ", ";
                        y_ss << plot_popup_.y_data[i];
                    }
                    y_ss << "]";
                    script << y_ss.str() << "\n";
                    script << "plt.scatter(data[:len(y_data)], y_data, alpha=0.7)\n";
                    script << "plt.xlabel('X')\n";
                    script << "plt.ylabel('Y')\n";
                } else if (plot_type == "boxplot") {
                    script << "plt.boxplot(data)\n";
                } else if (plot_type == "pie") {
                    script << "# Binning data for pie chart\n";
                    script << "counts, bins = np.histogram(data, bins=8)\n";
                    script << "labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(counts))]\n";
                    script << "plt.pie(counts, labels=labels, autopct='%1.1f%%')\n";
                } else if (plot_type == "step") {
                    script << "plt.step(range(len(data)), data, where='mid')\n";
                } else if (plot_type == "stem") {
                    script << "plt.stem(range(len(data)), data)\n";
                } else if (plot_type == "fill_between") {
                    script << "x = range(len(data))\n";
                    script << "plt.fill_between(x, data, alpha=0.5)\n";
                    script << "plt.plot(x, data)\n";
                } else {
                    script << "plt.plot(data)\n";
                }

                script << "plt.title('" << plot_popup_.title << "')\n";
                script << "plt.tight_layout()\n";
                script << "plt.show()\n";

                // Copy to clipboard
                ImGui::SetClipboardText(script.str().c_str());
                spdlog::info("Python matplotlib script copied to clipboard ({} bytes)", script.str().length());
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

