#include "descriptive_stats_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

DescriptiveStatsPanel::DescriptiveStatsPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

DescriptiveStatsPanel::~DescriptiveStatsPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void DescriptiveStatsPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CALCULATOR " Descriptive Statistics###DescStats", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else if (has_stats_) {
            if (ImGui::BeginTabBar("StatsTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_TABLE " Statistics")) {
                    RenderStatisticsTable();
                    ImGui::Spacing();
                    RenderPercentileTable();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Box Plot")) {
                    RenderBoxPlot();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Histogram")) {
                    RenderHistogram();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Select a data table and column to compute statistics");
        }
    }
    ImGui::End();
}

void DescriptiveStatsPanel::RenderToolbar() {
    RenderDataSelector();
    ImGui::SameLine();
    RenderColumnSelector();

    ImGui::SameLine();

    bool can_compute = current_table_ && selected_column_ >= 0;
    if (!can_compute) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
    }

    if (!can_compute) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!has_stats_) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportStats");
    }

    if (!has_stats_) ImGui::EndDisabled();

    RenderExportOptions();
}

void DescriptiveStatsPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(150);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                current_table_ = registry.GetTable(name);
                selected_column_ = -1;
                has_stats_ = false;

                // Find numeric columns
                numeric_columns_.clear();
                if (current_table_) {
                    const auto& headers = current_table_->GetHeaders();
                    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
                        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
                        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
                        }
                    }
                }
            }
        }
        ImGui::EndCombo();
    }
}

void DescriptiveStatsPanel::RenderColumnSelector() {
    ImGui::SetNextItemWidth(150);

    std::string preview = selected_column_ >= 0 && selected_column_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[selected_column_]
        : "Select column...";

    if (ImGui::BeginCombo("##ColSelect", preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            bool is_selected = (selected_column_ == static_cast<int>(i));
            if (ImGui::Selectable(numeric_columns_[i].c_str(), is_selected)) {
                selected_column_ = static_cast<int>(i);
                has_stats_ = false;
            }
        }
        ImGui::EndCombo();
    }
}

void DescriptiveStatsPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing statistics...", ICON_FA_SPINNER);
}

void DescriptiveStatsPanel::RenderStatisticsTable() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (ImGui::BeginTable("StatsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Statistic", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto AddRow = [](const char* name, double value, const char* format = "%.4f") {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text(format, value);
        };

        AddRow("Count", stats_.count, "%.0f");
        AddRow("Mean", stats_.mean);
        AddRow("Median", stats_.median);
        AddRow("Mode", stats_.mode);
        AddRow("Std. Deviation", stats_.std_dev);
        AddRow("Variance", stats_.variance);
        AddRow("Std. Error (SEM)", stats_.sem);
        AddRow("Minimum", stats_.min);
        AddRow("Maximum", stats_.max);
        AddRow("Range", stats_.range);
        AddRow("Sum", stats_.sum, "%.2f");
        AddRow("Q1 (25%)", stats_.q1);
        AddRow("Q3 (75%)", stats_.q3);
        AddRow("IQR", stats_.iqr);
        AddRow("Skewness", stats_.skewness);
        AddRow("Kurtosis", stats_.kurtosis);
        AddRow("CV (%)", stats_.cv, "%.2f%%");

        ImGui::EndTable();
    }
}

void DescriptiveStatsPanel::RenderPercentileTable() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (stats_.percentiles.size() != 7) return;

    ImGui::Text("Percentiles:");

    if (ImGui::BeginTable("PercentilesTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        const char* labels[] = {"5%", "10%", "25%", "50%", "75%", "90%", "95%"};
        for (const char* label : labels) {
            ImGui::TableSetupColumn(label, ImGuiTableColumnFlags_WidthFixed, 60);
        }
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        for (size_t i = 0; i < 7; i++) {
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", stats_.percentiles[i]);
        }

        ImGui::EndTable();
    }
}

void DescriptiveStatsPanel::RenderBoxPlot() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (column_data_.empty()) {
        ImGui::TextDisabled("No data available for box plot");
        return;
    }

    if (ImPlot::BeginPlot("Box Plot", ImVec2(-1, 300))) {
        ImPlot::SetupAxes(nullptr, "Value");

        // For box plot, we manually draw it
        double whisker_low = stats_.q1 - 1.5 * stats_.iqr;
        double whisker_high = stats_.q3 + 1.5 * stats_.iqr;

        // Find actual whisker extents (closest values within range)
        double actual_low = stats_.min;
        double actual_high = stats_.max;
        for (double v : column_data_) {
            if (v >= whisker_low && v < actual_low) actual_low = v;
            if (v <= whisker_high && v > actual_high) actual_high = v;
        }
        actual_low = std::max(actual_low, whisker_low);
        actual_high = std::min(actual_high, whisker_high);

        // Draw box
        double box_x[] = {0.7, 1.3, 1.3, 0.7, 0.7};
        double box_y[] = {stats_.q1, stats_.q1, stats_.q3, stats_.q3, stats_.q1};
        ImPlot::PlotLine("##Box", box_x, box_y, 5);

        // Median line
        double med_x[] = {0.7, 1.3};
        double med_y[] = {stats_.median, stats_.median};
        ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1), 2);
        ImPlot::PlotLine("Median", med_x, med_y, 2);

        // Whiskers
        double wh_x[] = {1.0, 1.0};
        double wh_low_y[] = {stats_.q1, actual_low};
        double wh_high_y[] = {stats_.q3, actual_high};
        ImPlot::PlotLine("##WLow", wh_x, wh_low_y, 2);
        ImPlot::PlotLine("##WHigh", wh_x, wh_high_y, 2);

        // Whisker caps
        double cap_x[] = {0.85, 1.15};
        double cap_low[] = {actual_low, actual_low};
        double cap_high[] = {actual_high, actual_high};
        ImPlot::PlotLine("##CapLow", cap_x, cap_low, 2);
        ImPlot::PlotLine("##CapHigh", cap_x, cap_high, 2);

        // Mean point
        double mean_x[] = {1.0};
        double mean_y[] = {stats_.mean};
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 6, ImVec4(0, 0.8f, 0, 1));
        ImPlot::PlotScatter("Mean", mean_x, mean_y, 1);

        // Outliers
        std::vector<double> outlier_x, outlier_y;
        for (double v : column_data_) {
            if (v < actual_low || v > actual_high) {
                outlier_x.push_back(1.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.2);
                outlier_y.push_back(v);
            }
        }
        if (!outlier_x.empty()) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(1, 0, 0, 0.7f));
            ImPlot::PlotScatter("Outliers", outlier_x.data(), outlier_y.data(), static_cast<int>(outlier_x.size()));
        }

        ImPlot::EndPlot();
    }

    // Legend
    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Orange line: Median");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0, 0.8f, 0, 1), "Green diamond: Mean");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Red dots: Outliers");
}

void DescriptiveStatsPanel::RenderHistogram() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (column_data_.empty()) {
        ImGui::TextDisabled("No data available for histogram");
        return;
    }

    static int num_bins = 20;
    ImGui::SliderInt("Bins", &num_bins, 5, 50);

    if (ImPlot::BeginPlot("Histogram", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Value", "Frequency");

        ImPlot::PlotHistogram("Data", column_data_.data(), static_cast<int>(column_data_.size()), num_bins);

        // Overlay normal curve
        if (stats_.std_dev > 0) {
            std::vector<double> x_curve, y_curve;
            double range = stats_.max - stats_.min;
            double bin_width = range / num_bins;
            double scale = static_cast<double>(column_data_.size()) * bin_width;

            for (int i = 0; i <= 100; i++) {
                double x = stats_.min + (range * i / 100.0);
                double z = (x - stats_.mean) / stats_.std_dev;
                double y = scale * (1.0 / (stats_.std_dev * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
                x_curve.push_back(x);
                y_curve.push_back(y);
            }

            ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1), 2);
            ImPlot::PlotLine("Normal Fit", x_curve.data(), y_curve.data(), static_cast<int>(x_curve.size()));
        }

        ImPlot::EndPlot();
    }

    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Orange curve: Normal distribution fit");
}

void DescriptiveStatsPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportStats")) {
        ImGui::Text("Export Statistics");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(stats_mutex_);

            std::ofstream file(export_path_);
            if (file) {
                file << "Statistic,Value\n";
                file << "Count," << stats_.count << "\n";
                file << "Mean," << stats_.mean << "\n";
                file << "Median," << stats_.median << "\n";
                file << "Mode," << stats_.mode << "\n";
                file << "Std. Deviation," << stats_.std_dev << "\n";
                file << "Variance," << stats_.variance << "\n";
                file << "SEM," << stats_.sem << "\n";
                file << "Minimum," << stats_.min << "\n";
                file << "Maximum," << stats_.max << "\n";
                file << "Range," << stats_.range << "\n";
                file << "Sum," << stats_.sum << "\n";
                file << "Q1," << stats_.q1 << "\n";
                file << "Q3," << stats_.q3 << "\n";
                file << "IQR," << stats_.iqr << "\n";
                file << "Skewness," << stats_.skewness << "\n";
                file << "Kurtosis," << stats_.kurtosis << "\n";
                file << "CV (%)," << stats_.cv << "\n";

                spdlog::info("Exported statistics to: {}", export_path_);
            } else {
                spdlog::error("Failed to export statistics to: {}", export_path_);
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void DescriptiveStatsPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void DescriptiveStatsPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    selected_column_ = -1;
    has_stats_ = false;

    // Find numeric columns
    numeric_columns_.clear();
    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < table->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
        }
    }
}

void DescriptiveStatsPanel::ComputeAsync() {
    if (is_computing_.load() || !current_table_ || selected_column_ < 0) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    // Find the actual column index
    const auto& headers = current_table_->GetHeaders();
    size_t col_idx = 0;
    int numeric_idx = 0;

    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            if (numeric_idx == selected_column_) {
                col_idx = i;
                break;
            }
            numeric_idx++;
        }
    }

    auto table = current_table_;
    size_t column_index = col_idx;

    compute_thread_ = std::make_unique<std::thread>([this, table, column_index]() {
        try {
            auto values = DataAnalyzer::GetNumericValues(*table, column_index);
            auto result = DataAnalyzer::ComputeDescriptiveStats(values);

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_ = std::move(result);
                column_data_ = std::move(values);
                has_stats_ = true;
            }

            spdlog::info("Descriptive statistics computed: n={}", stats_.count);

        } catch (const std::exception& e) {
            spdlog::error("Descriptive statistics error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
