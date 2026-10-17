#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Language models: a training run's generation
// previews per epoch (question, evidence, what the model cited, book /
// chapter / verse, how it stopped, loops) with slices. A test run (TOFIX157):
// the latest test's headline, read back from its per-row predictions with the
// Test Results panel's own calculation and words, and a button to that panel,
// which shows the whole run (one view of a test, not two). The model is
// core/data_studio_results.

#include "../../core/data_studio_results.h"
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

private:
    void RefreshRuns();
    void LoadEpoch();
    void RenderPreviews();
    void RenderTest();

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
};

}  // namespace cyxwiz
