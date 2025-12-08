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
 * HypothesisTestPanel - Statistical hypothesis testing wizard
 *
 * Features:
 * - Test type selector (t-tests, ANOVA, Chi-square)
 * - Column selectors for groups
 * - Parameter inputs (hypothesized mean, alpha level)
 * - Results display with p-value, CI, effect size
 * - Visual interpretation of results
 */
class HypothesisTestPanel {
public:
    HypothesisTestPanel();
    ~HypothesisTestPanel();

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
    void RenderTestTypeSelector();
    void RenderTestConfiguration();
    void RenderOneSampleConfig();
    void RenderTwoSampleConfig();
    void RenderPairedConfig();
    void RenderANOVAConfig();
    void RenderChiSquareConfig();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderPValueVisualization();
    void RenderDecision();
    void RenderEffectSizeInterpretation();

    void RunTest();

    bool visible_ = false;
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    std::vector<std::string> numeric_columns_;
    std::vector<std::string> all_columns_;

    // Test configuration
    int test_type_ = 0;  // 0=OneSample, 1=TwoSample, 2=Paired, 3=ANOVA, 4=ChiSquare
    int sample1_col_ = -1;
    int sample2_col_ = -1;
    int group_col_ = -1;  // For ANOVA grouping
    float hypothesized_mean_ = 0.0f;
    float alpha_ = 0.05f;
    bool equal_variance_ = true;

    // For chi-square
    std::vector<int> chi_square_cols_;

    HypothesisTestResult result_;
    bool has_result_ = false;

    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
