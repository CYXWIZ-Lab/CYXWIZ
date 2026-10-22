#pragma once

// Data Studio > Results (TOFIX148 job 7; TOFIX129 D4 approved mockup): a
// model's results next to the data, as views picked by chips.
//  - Test: one view of a test (test_view.h): the latest test, a kept test or
//    any table read as a test. It replaced the Test Results window and the
//    Tools > Evaluation windows (TOFIX129 D4).
//  - Generations: a language model's training-run generation previews, the
//    ones the Training Dashboard also shows (gui/generation_preview_view,
//    TOFIX159 A4).
// Keep as node (TOFIX157) saves the latest test's rows once in the project
// (results/tests) and adds a Data Input reading them and a Test Evaluation
// node; kept tests open from Test's Showing picker.

#include "test_view.h"
#include "../generation_preview_view.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ResultsView {
public:
    enum class View { Test, Generations };

    ResultsView();
    void Render();
    // Reads the runs again (the tab was opened).
    void Activate() { runs_time_ = -1.0; }
    void Show(View view) {
        view_ = view;
        view_chosen_ = true;
    }
    TestView& Test() { return test_; }
    bool IsRunning() const { return test_.IsRunning(); }
    // For the job's caption: runs that wrote previews, and the latest test's rows.
    size_t PreviewRunCount();
    int64_t TestRowCount() const;

    // Keep as node: why the latest test cannot be kept now ("" when it can);
    // `graph_open` from Data Studio.
    std::string KeepBlocker(bool graph_open) const;
    // Opens the Keep dialog on the next frame.
    void RequestKeep() {
        keep_requested_ = true;
        Show(View::Test);
    }
    // Adds the Data Input reading the kept file (project-relative) and its
    // Test Evaluation node (DataStudioPanel -> NodeEditor).
    std::function<void(const std::string& path, const std::string& name)> on_keep;
    // The open graph's name (the kept test's default name), and whether a
    // graph is open (Keep needs one).
    std::string graph_name;
    bool graph_open = false;

private:
    void RefreshRuns();
    void RenderPreviews();
    void RenderKeepDialog();

    View view_ = View::Test;
    bool view_chosen_ = false;

    TestView test_;

    std::vector<GenerationPreviewRun> runs_;
    double runs_time_ = -1.0;
    int run_ = 0;  // index into runs_
    GenerationPreviewView preview_view_;

    bool keep_requested_ = false;
    char keep_name_[128] = {0};
};

}  // namespace cyxwiz
