#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Language models: a training run's generation
// previews per epoch (question, evidence, what the model cited, book /
// chapter / verse, how it stopped, loops) with slices. A test run (TOFIX157):
// the latest test's headline, read back from its per-row predictions with the
// Test Results panel's own calculation and words, and a button to that panel,
// which shows the whole run (one view of a test, not two). Keep as node
// (approved mockup, TOFIX157) saves the test's rows once in the project
// (results/tests) and adds a Data Input reading them and a Test Evaluation
// node; the kept tests are listed under the run and open in Test Results.
// The model is core/data_studio_results.

#include "../../core/data_studio_results.h"
#include "../../core/kept_tests.h"
#include "../../core/test_results_presentation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ResultsView {
public:
    void Render();
    // Reads the runs again (the tab was opened).
    void Activate() { runs_time_ = -1.0; }
    // For the job's caption: runs that wrote previews, and the latest test's rows.
    size_t PreviewRunCount();
    int64_t TestRowCount() const;

    // "Open Test Results" (MainWindow shows the panel).
    std::function<void()> on_open_test_results;
    // A kept test's "Open" (MainWindow shows it in Test Results).
    std::function<void(const std::string& path)> on_open_kept;

    // Keep as node (the "In the graph" line's button): why the latest test
    // cannot be kept now ("" when it can); `graph_open` from Data Studio.
    std::string KeepBlocker(bool graph_open) const;
    // Opens the Keep dialog on the next frame.
    void RequestKeep() {
        keep_requested_ = true;
        source_ = 1;  // the dialog is drawn with the Test run view
        source_chosen_ = true;
    }
    // Adds the Data Input reading the kept file (project-relative) and its
    // Test Evaluation node (DataStudioPanel -> NodeEditor).
    std::function<void(const std::string& path, const std::string& name)> on_keep;
    // The open graph's name, for the kept test's default name.
    std::string graph_name;

private:
    void RefreshRuns();
    void LoadEpoch();
    void RenderPreviews();
    void RenderTest();
    void RenderKeepDialog();
    void RenderKeptTests();

    int source_ = 0;  // 0 language-model previews, 1 test run
    bool source_chosen_ = false;

    std::vector<studio::PreviewRun> runs_;
    double runs_time_ = -1.0;
    int run_ = 0;         // index into runs_
    int epoch_index_ = -1;  // index into runs_[run_].epochs (-1: the last)
    std::string loaded_key_;
    studio::PreviewEpoch epoch_;
    std::string epoch_error_;
    int selected_sample_ = -1;

    // The latest test run (rebuilt when test_predictions is replaced).
    std::string predictions_key_;
    test_report::View test_view_;
    std::string test_error_;

    // Keep as node and the project's kept tests (read again every 2 s).
    bool keep_requested_ = false;
    char keep_name_[128] = {0};
    std::string keep_note_;
    bool keep_failed_ = false;
    std::vector<KeptTest> kept_;
    double kept_time_ = -1.0;
};

}  // namespace cyxwiz
