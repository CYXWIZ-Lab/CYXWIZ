#include "distribution_fitter_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

DistributionFitterPanel::DistributionFitterPanel() = default;

DistributionFitterPanel::~DistributionFitterPanel() {
    if (fit_thread_ && fit_thread_->joinable()) {
        is_fitting_ = false;
        fit_thread_->join();
    }
}

void DistributionFitterPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_AREA " Distribution Fitter###DistFitter", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_fitting_.load()) {
            RenderLoadingIndicator();
        } else if (has_results_) {
            RenderFitResults();

            if (ImGui::BeginTabBar("DistTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Histogram + Fit")) {
                    RenderHistogramWithFit();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Q-Q Plot")) {
                    RenderQQPlot();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_TABLE " Parameters")) {
                    RenderParameterTable();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Select a data table and column to fit distributions");
        }
    }
    ImGui::End();
}

void DistributionFitterPanel::RenderToolbar() {
    RenderDataSelector();
    ImGui::SameLine();
    RenderColumnSelector();

    ImGui::SameLine();

    bool can_fit = current_table_ && selected_column_ >= 0;
    if (!can_fit) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Fit All")) {
        FitAsync();
    }

    if (!can_fit) ImGui::EndDisabled();
}

void DistributionFitterPanel::RenderDataSelector() {
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
                has_results_ = false;

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

void DistributionFitterPanel::RenderColumnSelector() {
    ImGui::SetNextItemWidth(150);

    std::string preview = selected_column_ >= 0 && selected_column_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[selected_column_]
        : "Select column...";

    if (ImGui::BeginCombo("##ColSelect", preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            bool is_selected = (selected_column_ == static_cast<int>(i));
            if (ImGui::Selectable(numeric_columns_[i].c_str(), is_selected)) {
                selected_column_ = static_cast<int>(i);
                has_results_ = false;
            }
        }
        ImGui::EndCombo();
    }
}

void DistributionFitterPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Fitting distributions...", ICON_FA_SPINNER);
}

void DistributionFitterPanel::RenderFitResults() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (fit_results_.empty()) {
        ImGui::TextDisabled("No distributions could be fitted");
        return;
    }

    ImGui::Text("%s Fit Results (ranked by AIC)", ICON_FA_STAR);
    ImGui::Spacing();

    if (ImGui::BeginTable("FitTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY, ImVec2(0, 150))) {

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 30);
        ImGui::TableSetupColumn("Distribution", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("AIC", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("BIC", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("KS Stat", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("KS p-value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < fit_results_.size(); i++) {
            const auto& fit = fit_results_[i];

            ImGui::TableNextRow();

            // Selection radio
            ImGui::TableNextColumn();
            bool selected = (selected_dist_ == static_cast<int>(i));
            if (ImGui::RadioButton(("##sel" + std::to_string(i)).c_str(), selected)) {
                selected_dist_ = static_cast<int>(i);
            }

            // Distribution name with fit indicator
            ImGui::TableNextColumn();
            if (fit.good_fit) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%s %s",
                                  ICON_FA_CIRCLE_CHECK, fit.name.c_str());
            } else {
                ImGui::Text("%s", fit.name.c_str());
            }

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", fit.aic);

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", fit.bic);

            ImGui::TableNextColumn();
            ImGui::Text("%.4f", fit.ks_statistic);

            ImGui::TableNextColumn();
            if (fit.ks_p_value > 0.05) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.4f", fit.ks_p_value);
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f), "%.4f", fit.ks_p_value);
            }
        }

        ImGui::EndTable();
    }

    ImGui::TextDisabled("Green check = good fit (KS p-value > 0.05)");
    ImGui::Spacing();
}

void DistributionFitterPanel::RenderQQPlot() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (column_data_.empty() || selected_dist_ >= static_cast<int>(fit_results_.size())) {
        ImGui::TextDisabled("No data for Q-Q plot");
        return;
    }

    const auto& fit = fit_results_[selected_dist_];

    // Get theoretical quantiles
    auto theoretical = DataAnalyzer::TheoreticalQuantiles(column_data_.size(), fit.type, fit.parameters);

    // Sort observed data
    std::vector<double> sorted_data = column_data_;
    std::sort(sorted_data.begin(), sorted_data.end());

    if (ImPlot::BeginPlot("Q-Q Plot", ImVec2(-1, 350))) {
        ImPlot::SetupAxes("Theoretical Quantiles", "Sample Quantiles");

        // Plot points
        ImPlot::PlotScatter("Data", theoretical.data(), sorted_data.data(), static_cast<int>(sorted_data.size()));

        // Reference line (y = x)
        double min_val = std::min(*std::min_element(theoretical.begin(), theoretical.end()),
                                  *std::min_element(sorted_data.begin(), sorted_data.end()));
        double max_val = std::max(*std::max_element(theoretical.begin(), theoretical.end()),
                                  *std::max_element(sorted_data.begin(), sorted_data.end()));
        double ref_x[] = {min_val, max_val};
        double ref_y[] = {min_val, max_val};
        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 1), 2);
        ImPlot::PlotLine("Reference", ref_x, ref_y, 2);

        ImPlot::EndPlot();
    }

    ImGui::TextWrapped("Points close to the red line indicate good fit. Deviations suggest the data"
                       " does not follow the %s distribution.", fit.name.c_str());
}

void DistributionFitterPanel::RenderHistogramWithFit() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (column_data_.empty() || selected_dist_ >= static_cast<int>(fit_results_.size())) {
        ImGui::TextDisabled("No data for histogram");
        return;
    }

    const auto& fit = fit_results_[selected_dist_];

    static int num_bins = 20;
    ImGui::SliderInt("Bins", &num_bins, 5, 50);

    if (ImPlot::BeginPlot("Histogram with Fitted Distribution", ImVec2(-1, 350))) {
        ImPlot::SetupAxes("Value", "Density");

        // Histogram as density (normalized)
        ImPlot::PlotHistogram("Data", column_data_.data(), static_cast<int>(column_data_.size()),
                              num_bins, 1.0, ImPlotRange(), ImPlotHistogramFlags_Density);

        // Fitted PDF overlay
        double min_val = *std::min_element(column_data_.begin(), column_data_.end());
        double max_val = *std::max_element(column_data_.begin(), column_data_.end());
        double range = max_val - min_val;

        std::vector<double> x_curve, y_curve;
        for (int i = 0; i <= 100; i++) {
            double x = min_val + (range * i / 100.0);
            double y = 0.0;

            switch (fit.type) {
                case DistributionType::Normal: {
                    double mu = fit.parameters.count("mu") ? fit.parameters.at("mu") : 0;
                    double sigma = fit.parameters.count("sigma") ? fit.parameters.at("sigma") : 1;
                    if (sigma > 0) {
                        double z = (x - mu) / sigma;
                        y = (1.0 / (sigma * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
                    }
                    break;
                }
                case DistributionType::Uniform: {
                    double a = fit.parameters.count("a") ? fit.parameters.at("a") : min_val;
                    double b = fit.parameters.count("b") ? fit.parameters.at("b") : max_val;
                    if (x >= a && x <= b && b > a) {
                        y = 1.0 / (b - a);
                    }
                    break;
                }
                case DistributionType::Exponential: {
                    double lambda = fit.parameters.count("lambda") ? fit.parameters.at("lambda") : 1;
                    if (x >= 0 && lambda > 0) {
                        y = lambda * std::exp(-lambda * x);
                    }
                    break;
                }
                case DistributionType::LogNormal: {
                    double mu = fit.parameters.count("mu") ? fit.parameters.at("mu") : 0;
                    double sigma = fit.parameters.count("sigma") ? fit.parameters.at("sigma") : 1;
                    if (x > 0 && sigma > 0) {
                        double z = (std::log(x) - mu) / sigma;
                        y = (1.0 / (x * sigma * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
                    }
                    break;
                }
                default:
                    break;
            }

            x_curve.push_back(x);
            y_curve.push_back(y);
        }

        ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1), 2);
        ImPlot::PlotLine(("Fitted " + fit.name).c_str(), x_curve.data(), y_curve.data(), static_cast<int>(x_curve.size()));

        ImPlot::EndPlot();
    }
}

void DistributionFitterPanel::RenderParameterTable() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (selected_dist_ >= static_cast<int>(fit_results_.size())) {
        ImGui::TextDisabled("No distribution selected");
        return;
    }

    const auto& fit = fit_results_[selected_dist_];

    ImGui::Text("%s Distribution: %s", ICON_FA_CHART_PIE, fit.name.c_str());
    ImGui::Spacing();

    if (ImGui::BeginTable("ParamsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& param : fit.parameters) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", param.first.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", param.second);
        }

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Text("Goodness of Fit:");

    if (ImGui::BeginTable("GoFTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto AddRow = [](const char* name, double value, const char* fmt = "%.4f") {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text(fmt, value);
        };

        AddRow("Log-Likelihood", fit.log_likelihood);
        AddRow("AIC", fit.aic, "%.2f");
        AddRow("BIC", fit.bic, "%.2f");
        AddRow("KS Statistic", fit.ks_statistic);
        AddRow("KS p-value", fit.ks_p_value);

        ImGui::EndTable();
    }

    ImGui::Spacing();

    // Interpretation
    if (fit.good_fit) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                          "%s Good fit: KS test p-value > 0.05", ICON_FA_CIRCLE_CHECK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f),
                          "%s Poor fit: KS test p-value <= 0.05", ICON_FA_CIRCLE_XMARK);
    }

    ImGui::TextWrapped("Note: Lower AIC/BIC values indicate better fit when comparing distributions.");
}

void DistributionFitterPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void DistributionFitterPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    selected_column_ = -1;
    has_results_ = false;

    numeric_columns_.clear();
    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < table->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
        }
    }
}

void DistributionFitterPanel::FitAsync() {
    if (is_fitting_.load() || !current_table_ || selected_column_ < 0) return;

    if (fit_thread_ && fit_thread_->joinable()) {
        fit_thread_->join();
    }

    is_fitting_ = true;

    // Find actual column index
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

    fit_thread_ = std::make_unique<std::thread>([this, table, column_index]() {
        try {
            auto values = DataAnalyzer::GetNumericValues(*table, column_index);
            auto results = DataAnalyzer::FitAllDistributions(values);

            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                fit_results_ = std::move(results);
                column_data_ = std::move(values);
                selected_dist_ = 0;  // Select best fit
                has_results_ = true;
            }

            spdlog::info("Distribution fitting complete: {} distributions fitted", fit_results_.size());

        } catch (const std::exception& e) {
            spdlog::error("Distribution fitting error: {}", e.what());
        }

        is_fitting_ = false;
    });
}

} // namespace cyxwiz
