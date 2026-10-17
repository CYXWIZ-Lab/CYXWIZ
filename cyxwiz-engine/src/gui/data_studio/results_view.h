#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Language models: a training run's generation
// previews per epoch (question, evidence, what the model cited, book /
// chapter / verse, how it stopped, loops) with slices. A test run: the latest
// test's per-row predictions - a classifier's accuracy, confusion per class,
// precision and recall and worst rows; a regression's MAE, RMSE, bias and
// largest errors - and (TOFIX157 B2) the error by the values of a column of
// the tested table, joined by row. The model is core/data_studio_results.

#include "../../core/data_studio_results.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

class ResultsView {
public:
    ~ResultsView();
    void Render();
    // Reads the runs again (the tab was opened).
    void Activate() { runs_time_ = -1.0; }
    // For the job's caption: runs that wrote previews, and the latest test's rows.
    size_t PreviewRunCount();
    int64_t TestRowCount() const;

private:
    void RefreshRuns();
    void LoadEpoch();
    void RenderPreviews();
    void RenderTest();
    void RenderClassifier();
    void RenderRegression();
    void RenderErrorsByValue();
    void SliceBy(const std::string& column);

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
    bool regression_ = false;
    studio::ClassifierResults classifier_;
    studio::RegressionResults regression_results_;
    studio::TestRowErrors row_errors_;

    // Error by column value: the tested table's columns, the one chosen and
    // its slices, read by a task (a table on disk is read a row group at a time).
    std::vector<std::string> slice_columns_;
    std::string slice_column_;
    studio::ErrorSlices slices_;
    std::string slices_error_;
    uint64_t slice_task_ = 0;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
