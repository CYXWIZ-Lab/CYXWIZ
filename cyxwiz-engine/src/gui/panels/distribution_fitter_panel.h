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
 * DistributionFitterPanel - Fit distributions to data
 *
 * Features:
 * - Auto-fit multiple distributions (Normal, Uniform, Exponential, LogNormal)
 * - Ranked list by AIC/BIC
 * - QQ-plot for selected distribution
 * - Histogram with fitted PDF overlay
 * - Parameter estimates table
 * - Kolmogorov-Smirnov test results
 */
class DistributionFitterPanel {
public:
    DistributionFitterPanel();
    ~DistributionFitterPanel();

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
    void RenderColumnSelector();
    void RenderLoadingIndicator();
    void RenderFitResults();
    void RenderQQPlot();
    void RenderHistogramWithFit();
    void RenderParameterTable();

    void FitAsync();

    bool visible_ = false;
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    int selected_column_ = -1;
    std::vector<std::string> numeric_columns_;

    std::vector<DistributionFitResult> fit_results_;
    std::vector<double> column_data_;
    int selected_dist_ = 0;  // Index in fit_results_
    bool has_results_ = false;

    std::atomic<bool> is_fitting_{false};
    std::unique_ptr<std::thread> fit_thread_;
    std::mutex results_mutex_;
};

} // namespace cyxwiz
