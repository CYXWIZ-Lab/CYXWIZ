#pragma once

#include "../panel.h"
#include "../../core/test_executor.h"
#include "../../core/test_results_presentation.h"
#include "test_errors_by_value.h"
#include "../icons.h"
#include <imgui.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace arrow {
class Table;
}

namespace cyxwiz {

// A status tone's colour, for the panel and the views that repeat its headline
// (Data Studio > Results).
ImVec4 TestToneColour(test_report::Tone tone);

/**
 * TestResultsPanel - a test's results (TOFIX129 B5): the headline and
 * figures, the confusion matrix, the classes, the curves, the predictions and
 * the comparison with the run before. A thin renderer over
 * core/test_results_presentation; exports write what the panel shows.
 * TOFIX157: the one full view of a test - each prediction names its dataset
 * row, a regression lists its rows by error, and By column value slices the
 * error by a column of the tested table (Data Studio > Results links here).
 */
class TestResultsPanel : public Panel {
public:
    TestResultsPanel();
    ~TestResultsPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_GAUGE; }

    // A finished test (thread-safe). A failed run shows its message and keeps
    // the last successful results; it never becomes the run compared against.
    void SetResults(const TestingMetrics& results);

    // Forget the results and the run before them.
    void Clear();

    bool HasResults() const { return has_results_; }

    // Train > Test > Compare Test Results...: shows the panel on the
    // Comparison tab (latest run beside the previous one). TOFIX129 G3.
    void ShowComparison();
    bool HasPreviousResults() const { return has_previous_; }
    // Train > Test > Export Test Report...: the JSON report of the shown run.
    void ExportReport();

    // "Run test again" (Train > Test > Run Test).
    void SetRunTestCallback(std::function<void()> callback) { run_test_ = std::move(callback); }

private:
    TestingMetrics results_;
    bool has_results_ = false;
    TestingMetrics previous_results_;  // the successful run before results_
    bool has_previous_ = false;
    std::string failure_message_;      // the latest run failed: why
    bool select_comparison_tab_ = false;
    mutable std::mutex results_mutex_;

    test_report::View view_;           // rebuilt when the results change
    bool view_dirty_ = true;

    char filter_text_[256] = {0};
    test_report::ClassSort sort_by_ = test_report::ClassSort::Name;
    bool sort_ascending_ = true;
    int matrix_mode_ = 0;              // 0 counts, 1 share of each true class
    int predictions_mode_ = 0;         // 0 all, 1 only mistakes (regression: largest errors first)
    std::shared_ptr<arrow::Table> rows_;  // the shown run's per-row table (By column value)
    ErrorsByValueView by_value_;
    std::string export_note_;
    bool export_failed_ = false;
    std::function<void()> run_test_;

    void RenderHeader();
    void RenderOverviewTab();
    void RenderConfusionMatrixTab();
    void RenderClassesTab();
    void RenderCurvesTab();
    void RenderPredictionsTab();
    void RenderRegressionRowsTab();
    void RenderByValueTab();
    void RenderComparisonTab();
    void RenderFooter();
    void ExportToCSV();
    void ExportToJSON();
};

} // namespace cyxwiz
