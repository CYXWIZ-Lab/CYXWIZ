#pragma once

#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

class DataTable;

/**
 * RegressionPanel - Regression analysis tool
 *
 * Features:
 * - Linear, Polynomial, and Multiple regression
 * - X/Y column selectors
 * - Polynomial degree slider
 * - Coefficients table with SE, t, p
 * - R-squared and model diagnostics
 * - Scatter plot with regression line
 * - Residuals plot
 * - Export coefficients
 */
class RegressionPanel {
public:
    RegressionPanel();
    ~RegressionPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    void AnalyzeTable(const std::string& table_name);
    void AnalyzeTable(std::shared_ptr<DataTable> table);

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderConfiguration();
    void RenderLoadingIndicator();
    void RenderModelSummary();
    void RenderCoefficientsTable();
    void RenderScatterPlot();
    void RenderResidualsPlot();
    void RenderExportOptions();

    void RunRegression();

    bool visible_ = false;
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    std::vector<std::string> numeric_columns_;

    // Configuration
    int regression_type_ = 0;  // 0=Linear, 1=Polynomial
    int x_column_ = -1;
    int y_column_ = -1;
    int poly_degree_ = 2;
    std::vector<int> multi_x_columns_;  // For multiple regression

    // Data
    std::vector<double> x_data_;
    std::vector<double> y_data_;

    RegressionResult result_;
    bool has_result_ = false;

    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
