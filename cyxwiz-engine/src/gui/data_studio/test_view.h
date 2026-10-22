#pragma once

// Data Studio > Results > Test (TOFIX129 D4, approved mockup boards 1-5): one
// view of a test, which was the Test Results window (TOFIX129 B5, TOFIX157)
// and the Confusion Matrix, ROC / AUC and PR Curve windows. It shows the
// latest test, a kept test (results/tests), or any table read as a test (a
// true column, a predicted column and the scores). The figures come from the
// test code every other place uses (test_predictions, MetricsFromColumns,
// FinishClassifierMetrics, the device ROC / PR) through
// core/test_results_presentation; this file only draws them.
//
// Sections: Overview, Confusion matrix, Classes, Curves, Threshold (the
// positive class by name, a typed threshold, Best F1), Predictions, Plots
// (regression), By column value, Comparison. Export CSV / JSON, Keep as node,
// Clear (asks first), Run test again.

#include "explore_tab.h"
#include "test_errors_by_value.h"
#include "../../core/test_executor.h"
#include "../../core/test_predictions_metrics.h"
#include "../../core/test_results_presentation.h"

#include <imgui.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// A status tone's colour (the headline, chips, changes).
ImVec4 TestToneColour(test_report::Tone tone);

class TestView {
public:
    // A finished test (any thread). A failed run shows its message and keeps
    // the last successful results; it never becomes the run compared against.
    void SetResults(const TestingMetrics& results);
    bool HasResults() const;
    // The latest successful test's main score: accuracy, or MAE (regression).
    std::optional<double> LatestScore() const;
    // What to show: a kept test (path), the latest test on its Comparison
    // section (Train > Test > Compare test results), or A table... (Tools >
    // Evaluation with no test yet).
    void ShowKept(const std::string& path);
    void ShowComparison();
    void ShowTable();
    // Train > Test > Export test report: the shown run as JSON (a Save dialog).
    // False when nothing is shown.
    bool ExportReport();
    // A line under the buttons (Keep as node's outcome).
    void SetNote(const std::string& text, bool failed) {
        std::lock_guard<std::mutex> lock(mutex_);
        note_ = text;
        note_failed_ = failed;
    }

    void Render();
    // A table being read as a test, or the error being sliced by a column.
    bool IsRunning() const { return table_job_.Running() || by_value_.IsRunning(); }

    std::function<void()> run_test;             // "Run test again"
    std::function<void()> keep;                 // "Keep as node..." (Results' dialog)
    std::function<std::string()> keep_blocker;  // why the latest test cannot be kept ("" when it can)

private:
    enum class Source { Latest, Kept, Table };
    struct TableAnswer {
        TestingMetrics metrics;
        int64_t left_out = 0;
        std::string error;
        bool ok() const { return error.empty(); }
    };

    const TestingMetrics* Shown() const;
    void LoadKept(const std::string& path);  // with mutex_ held
    void Rebuild();
    void RenderSource();
    void RenderTableChoices();
    void RenderHeader(const TestingMetrics& m);
    void RenderOverview();
    void RenderMatrix();
    void RenderClasses();
    void RenderPositiveClass(const TestingMetrics& m);
    bool EnsureCurves(const TestingMetrics& m);
    void RenderCurves(const TestingMetrics& m);
    void RenderThreshold(const TestingMetrics& m);
    void RenderPredictions(const TestingMetrics& m);
    void RenderRegressionRows(const TestingMetrics& m);
    void RenderRegressionPlots(const TestingMetrics& m);
    void RenderComparison();
    void RenderFooter(const TestingMetrics& m);
    void Export(bool json);

    mutable std::mutex mutex_;  // results_ / previous_ / failure_ (SetResults may come from another thread)
    TestingMetrics results_;
    bool has_results_ = false;
    TestingMetrics previous_;  // the successful run before results_
    bool has_previous_ = false;
    std::string failure_;      // the latest run failed: why
    bool dirty_ = true;

    Source source_ = Source::Latest;
    TestingMetrics kept_;
    std::string kept_name_, kept_path_, kept_error_;

    // A table read as a test.
    std::string table_name_;
    PredictionColumns columns_;
    std::string columns_for_;  // the table the columns were guessed for
    ExploreJob<TableAnswer> table_job_;
    std::string table_key_;
    const TableAnswer* table_answer_ = nullptr;

    // The shown run, drawn.
    uint64_t shown_version_ = 0;  // changes whenever the shown run does
    test_report::View view_;
    std::shared_ptr<arrow::Table> rows_;  // its per-row table (By column value)
    ErrorsByValueView by_value_;
    bool select_comparison_ = false;

    char filter_[256] = {0};
    test_report::ClassSort sort_by_ = test_report::ClassSort::Name;
    bool sort_ascending_ = true;
    int matrix_mode_ = 0;       // 0 counts, 1 share of each true class
    int predictions_mode_ = 0;  // 0 all, 1 only mistakes (regression: largest errors first)
    int regression_rows_mode_ = 0;
    int plot_ = 0;              // 0 predicted vs actual, 1 residuals vs predicted, 2 residual histogram

    // The positive class's curves and threshold.
    int positive_ = -1;  // -1: class 1 of two, class 0 of more
    double threshold_ = 0.5;
    std::string curves_key_;
    ClassCurves curves_;
    std::string curves_error_;

    // Regression plots (first target), rebuilt with the shown run.
    uint64_t plotted_version_ = UINT64_MAX;
    std::vector<double> truth_, predicted_, residuals_;
    std::vector<int64_t> bins_;
    double bins_lo_ = 0.0, bins_hi_ = 0.0;

    std::string note_;
    bool note_failed_ = false;
};

}  // namespace cyxwiz
