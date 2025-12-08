#pragma once

#include "../panel.h"
#include "../../core/test_executor.h"
#include "../icons.h"
#include <imgui.h>
#include <string>
#include <mutex>

namespace cyxwiz {

/**
 * TestResultsPanel - Display testing metrics and results
 * Shows accuracy, confusion matrix, per-class metrics, and predictions
 */
class TestResultsPanel : public Panel {
public:
    TestResultsPanel();
    ~TestResultsPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_GAUGE; }

    // Update with new results (thread-safe)
    void SetResults(const TestingMetrics& results);

    // Clear results
    void Clear();

    // Check if we have results
    bool HasResults() const { return has_results_; }

private:
    TestingMetrics results_;
    bool has_results_ = false;
    mutable std::mutex results_mutex_;

    // Tab state
    int selected_tab_ = 0;

    // Display options
    bool show_percentages_ = true;
    bool show_absolute_ = false;
    int selected_class_ = -1;
    char filter_text_[256] = {0};

    // Confusion matrix display options
    bool normalize_confusion_ = false;
    float cell_size_ = 40.0f;

    // Rendering helpers
    void RenderToolbar();
    void RenderOverviewTab();
    void RenderConfusionMatrixTab();
    void RenderPerClassTab();
    void RenderPredictionsTab();

    // Helpers
    ImVec4 GetAccuracyColor(float accuracy);
    ImVec4 GetMetricColor(float value);
    ImVec4 GetConfusionCellColor(int value, int max_value);
    void ExportToCSV();
    void ExportToJSON();
};

} // namespace cyxwiz
