#include "regression_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>

namespace cyxwiz {

RegressionPanel::RegressionPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

RegressionPanel::~RegressionPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void RegressionPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(650, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_LINE " Regression Analysis###Regression", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderConfiguration();

            if (has_result_) {
                ImGui::Separator();

                if (ImGui::BeginTabBar("RegTabs")) {
                    if (ImGui::BeginTabItem(ICON_FA_TABLE " Summary")) {
                        RenderModelSummary();
                        RenderCoefficientsTable();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Scatter + Fit")) {
                        RenderScatterPlot();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Residuals")) {
                        RenderResidualsPlot();
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }
    }
    ImGui::End();
}

void RegressionPanel::RenderToolbar() {
    RenderDataSelector();

    ImGui::SameLine();

    if (!has_result_) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportRegression");
    }

    if (!has_result_) ImGui::EndDisabled();

    RenderExportOptions();
}

void RegressionPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                current_table_ = registry.GetTable(name);
                x_column_ = -1;
                y_column_ = -1;
                has_result_ = false;

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

void RegressionPanel::RenderConfiguration() {
    if (!current_table_) {
        ImGui::TextDisabled("Select a data table to configure regression");
        return;
    }

    ImGui::Text("%s Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    // Regression type
    const char* types[] = {"Linear Regression", "Polynomial Regression"};
    ImGui::Text("Type:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::Combo("##RegType", &regression_type_, types, IM_ARRAYSIZE(types))) {
        has_result_ = false;
    }

    // Polynomial degree
    if (regression_type_ == 1) {
        ImGui::Text("Degree:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("##Degree", &poly_degree_, 2, 10)) {
            has_result_ = false;
        }
    }

    ImGui::Spacing();

    // X column
    ImGui::Text("Predictor (X):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string x_preview = x_column_ >= 0 && x_column_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[x_column_] : "Select...";

    if (ImGui::BeginCombo("##XCol", x_preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), x_column_ == static_cast<int>(i))) {
                x_column_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Y column
    ImGui::Text("Response (Y):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string y_preview = y_column_ >= 0 && y_column_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[y_column_] : "Select...";

    if (ImGui::BeginCombo("##YCol", y_preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), y_column_ == static_cast<int>(i))) {
                y_column_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();

    // Run button
    bool can_run = x_column_ >= 0 && y_column_ >= 0 && x_column_ != y_column_;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run Regression", ImVec2(150, 0))) {
        RunRegression();
    }

    if (!can_run) ImGui::EndDisabled();
}

void RegressionPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing regression...", ICON_FA_SPINNER);
}

void RegressionPanel::RenderModelSummary() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    ImGui::Text("%s Model Summary", ICON_FA_CHART_PIE);
    ImGui::Spacing();

    if (ImGui::BeginTable("SummaryTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 200);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto AddRow = [](const char* name, const char* value) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text("%s", value);
        };

        char buf[64];

        snprintf(buf, sizeof(buf), "%zu", result_.n);
        AddRow("Observations (n)", buf);

        snprintf(buf, sizeof(buf), "%.6f", result_.r_squared);
        AddRow("R-squared", buf);

        snprintf(buf, sizeof(buf), "%.6f", result_.adjusted_r_squared);
        AddRow("Adjusted R-squared", buf);

        snprintf(buf, sizeof(buf), "%.4f", result_.f_statistic);
        AddRow("F-statistic", buf);

        snprintf(buf, sizeof(buf), "%.6f", result_.f_p_value);
        AddRow("F-test p-value", buf);

        snprintf(buf, sizeof(buf), "%.4f", result_.rmse);
        AddRow("RMSE", buf);

        snprintf(buf, sizeof(buf), "%.4f", result_.mae);
        AddRow("MAE", buf);

        snprintf(buf, sizeof(buf), "%.4f", result_.mse);
        AddRow("MSE", buf);

        ImGui::EndTable();
    }

    // R-squared interpretation
    ImGui::Spacing();
    const char* interpretation = "Poor";
    ImVec4 color = ImVec4(0.8f, 0.4f, 0.4f, 1.0f);

    if (result_.r_squared >= 0.9) {
        interpretation = "Excellent";
        color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
    } else if (result_.r_squared >= 0.7) {
        interpretation = "Good";
        color = ImVec4(0.4f, 0.7f, 0.4f, 1.0f);
    } else if (result_.r_squared >= 0.5) {
        interpretation = "Moderate";
        color = ImVec4(0.8f, 0.7f, 0.2f, 1.0f);
    } else if (result_.r_squared >= 0.3) {
        interpretation = "Weak";
        color = ImVec4(0.8f, 0.5f, 0.2f, 1.0f);
    }

    ImGui::TextColored(color, "Model fit: %s (R^2 = %.1f%%)", interpretation, result_.r_squared * 100);
}

void RegressionPanel::RenderCoefficientsTable() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    ImGui::Spacing();
    ImGui::Text("%s Coefficients", ICON_FA_TABLE);
    ImGui::Spacing();

    if (ImGui::BeginTable("CoefTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable)) {

        ImGui::TableSetupColumn("Term", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Coefficient", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Std. Error", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("t-value", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("p-value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.coefficients.size(); i++) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.predictor_names[i].c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.coefficients[i]);

            ImGui::TableNextColumn();
            if (i < result_.std_errors.size()) {
                ImGui::Text("%.4f", result_.std_errors[i]);
            }

            ImGui::TableNextColumn();
            if (i < result_.t_values.size()) {
                ImGui::Text("%.3f", result_.t_values[i]);
            }

            ImGui::TableNextColumn();
            if (i < result_.p_values.size()) {
                double p = result_.p_values[i];
                if (p < 0.001) {
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "< 0.001 ***");
                } else if (p < 0.01) {
                    ImGui::TextColored(ImVec4(0.3f, 0.7f, 0.3f, 1.0f), "%.4f **", p);
                } else if (p < 0.05) {
                    ImGui::TextColored(ImVec4(0.5f, 0.7f, 0.3f, 1.0f), "%.4f *", p);
                } else if (p < 0.1) {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.3f, 1.0f), "%.4f .", p);
                } else {
                    ImGui::Text("%.4f", p);
                }
            }
        }

        ImGui::EndTable();
    }

    ImGui::TextDisabled("Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1");

    // Model equation
    ImGui::Spacing();
    ImGui::Text("Model Equation:");
    std::string equation = "y = ";
    for (size_t i = 0; i < result_.coefficients.size(); i++) {
        if (i == 0) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.4f", result_.coefficients[i]);
            equation += buf;
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), " %c %.4f*%s",
                    result_.coefficients[i] >= 0 ? '+' : '-',
                    std::abs(result_.coefficients[i]),
                    result_.predictor_names[i].c_str());
            equation += buf;
        }
    }
    ImGui::TextWrapped("%s", equation.c_str());
}

void RegressionPanel::RenderScatterPlot() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (x_data_.empty() || y_data_.empty()) {
        ImGui::TextDisabled("No data for scatter plot");
        return;
    }

    if (ImPlot::BeginPlot("Scatter Plot with Regression Line", ImVec2(-1, 400))) {
        std::string x_label = x_column_ >= 0 && x_column_ < static_cast<int>(numeric_columns_.size())
            ? numeric_columns_[x_column_] : "X";
        std::string y_label = y_column_ >= 0 && y_column_ < static_cast<int>(numeric_columns_.size())
            ? numeric_columns_[y_column_] : "Y";

        ImPlot::SetupAxes(x_label.c_str(), y_label.c_str());

        // Plot data points
        ImPlot::PlotScatter("Data", x_data_.data(), y_data_.data(), static_cast<int>(x_data_.size()));

        // Plot regression line
        if (!result_.predicted.empty()) {
            // Generate smooth curve
            double x_min = *std::min_element(x_data_.begin(), x_data_.end());
            double x_max = *std::max_element(x_data_.begin(), x_data_.end());

            std::vector<double> x_curve, y_curve;
            for (int i = 0; i <= 100; i++) {
                double x = x_min + (x_max - x_min) * i / 100.0;
                double y = result_.coefficients[0];  // Intercept

                if (regression_type_ == 0) {
                    // Linear
                    if (result_.coefficients.size() > 1) {
                        y += result_.coefficients[1] * x;
                    }
                } else {
                    // Polynomial
                    for (size_t j = 1; j < result_.coefficients.size(); j++) {
                        y += result_.coefficients[j] * std::pow(x, j);
                    }
                }

                x_curve.push_back(x);
                y_curve.push_back(y);
            }

            ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1), 2);
            ImPlot::PlotLine("Fit", x_curve.data(), y_curve.data(), static_cast<int>(x_curve.size()));
        }

        ImPlot::EndPlot();
    }
}

void RegressionPanel::RenderResidualsPlot() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (result_.residuals.empty() || result_.predicted.empty()) {
        ImGui::TextDisabled("No residuals data");
        return;
    }

    // Residuals vs Fitted
    if (ImPlot::BeginPlot("Residuals vs Fitted", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Fitted Values", "Residuals");

        ImPlot::PlotScatter("Residuals", result_.predicted.data(), result_.residuals.data(),
                           static_cast<int>(result_.residuals.size()));

        // Zero line
        double x_min = *std::min_element(result_.predicted.begin(), result_.predicted.end());
        double x_max = *std::max_element(result_.predicted.begin(), result_.predicted.end());
        double zero_x[] = {x_min, x_max};
        double zero_y[] = {0, 0};
        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 1), 1);
        ImPlot::PlotLine("##Zero", zero_x, zero_y, 2);

        ImPlot::EndPlot();
    }

    // Residuals histogram
    if (ImPlot::BeginPlot("Residuals Distribution", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Residual", "Frequency");
        ImPlot::PlotHistogram("Residuals", result_.residuals.data(),
                              static_cast<int>(result_.residuals.size()), 20);
        ImPlot::EndPlot();
    }

    // Residuals statistics
    double mean_resid = DataAnalyzer::Mean(result_.residuals);
    double std_resid = DataAnalyzer::StdDev(result_.residuals, mean_resid);

    ImGui::Text("Residuals: Mean = %.4f, Std = %.4f", mean_resid, std_resid);
}

void RegressionPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportRegression")) {
        ImGui::Text("Export Regression Results");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(result_mutex_);

            std::ofstream file(export_path_);
            if (file) {
                file << "Model Summary\n";
                file << "Metric,Value\n";
                file << "R-squared," << result_.r_squared << "\n";
                file << "Adjusted R-squared," << result_.adjusted_r_squared << "\n";
                file << "F-statistic," << result_.f_statistic << "\n";
                file << "F-test p-value," << result_.f_p_value << "\n";
                file << "RMSE," << result_.rmse << "\n";
                file << "MAE," << result_.mae << "\n";
                file << "MSE," << result_.mse << "\n";
                file << "n," << result_.n << "\n\n";

                file << "Coefficients\n";
                file << "Term,Coefficient,Std.Error,t-value,p-value\n";
                for (size_t i = 0; i < result_.coefficients.size(); i++) {
                    file << result_.predictor_names[i] << ","
                         << result_.coefficients[i] << ","
                         << (i < result_.std_errors.size() ? result_.std_errors[i] : 0) << ","
                         << (i < result_.t_values.size() ? result_.t_values[i] : 0) << ","
                         << (i < result_.p_values.size() ? result_.p_values[i] : 0) << "\n";
                }

                spdlog::info("Exported regression results to: {}", export_path_);
            } else {
                spdlog::error("Failed to export regression results to: {}", export_path_);
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void RegressionPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void RegressionPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    x_column_ = -1;
    y_column_ = -1;
    has_result_ = false;

    numeric_columns_.clear();
    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < table->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
        }
    }
}

void RegressionPanel::RunRegression() {
    if (is_computing_.load() || !current_table_ || x_column_ < 0 || y_column_ < 0) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    // Find actual column indices
    auto GetNumericColumnIndex = [&](int numeric_idx) -> size_t {
        int count = 0;
        for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
            auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
            if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                if (count == numeric_idx) return i;
                count++;
            }
        }
        return 0;
    };

    size_t x_idx = GetNumericColumnIndex(x_column_);
    size_t y_idx = GetNumericColumnIndex(y_column_);

    auto table = current_table_;
    int type = regression_type_;
    int degree = poly_degree_;

    compute_thread_ = std::make_unique<std::thread>([this, table, x_idx, y_idx, type, degree]() {
        try {
            auto x_vals = DataAnalyzer::GetNumericValues(*table, x_idx);
            auto y_vals = DataAnalyzer::GetNumericValues(*table, y_idx);

            // Align data (ensure same length)
            size_t n = std::min(x_vals.size(), y_vals.size());
            x_vals.resize(n);
            y_vals.resize(n);

            RegressionResult res;

            if (type == 0) {
                // Linear regression
                res = DataAnalyzer::LinearRegression(x_vals, y_vals);
            } else {
                // Polynomial regression
                res = DataAnalyzer::PolynomialRegression(x_vals, y_vals, degree);
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_ = std::move(res);
                x_data_ = std::move(x_vals);
                y_data_ = std::move(y_vals);
                has_result_ = true;
            }

            spdlog::info("Regression complete: R^2 = {}", result_.r_squared);

        } catch (const std::exception& e) {
            spdlog::error("Regression error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
