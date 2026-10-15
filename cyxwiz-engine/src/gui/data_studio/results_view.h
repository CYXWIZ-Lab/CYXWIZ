#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Language models: a training run's generation
// previews per epoch (question, evidence, what the model cited, book /
// chapter / verse, how it stopped, loops) with slices. Classifiers: the latest
// test run's per-row predictions (accuracy, confusion per class, precision and
// recall, worst rows). The model is core/data_studio_results.

#include "../../core/data_studio_results.h"

#include <string>
#include <vector>

namespace cyxwiz {

class ResultsView {
public:
    void Render();
    // Reads the runs again (the tab was opened).
    void Activate() { runs_time_ = -1.0; }

private:
    void RefreshRuns();
    void LoadEpoch();
    void RenderPreviews();
    void RenderClassifier();

    int source_ = 0;  // 0 language-model previews, 1 classifier test
    bool source_chosen_ = false;

    std::vector<studio::PreviewRun> runs_;
    double runs_time_ = -1.0;
    int run_ = 0;         // index into runs_
    int epoch_index_ = -1;  // index into runs_[run_].epochs (-1: the last)
    std::string loaded_key_;
    studio::PreviewEpoch epoch_;
    std::string epoch_error_;
    int selected_sample_ = -1;

    std::string predictions_key_;
    studio::ClassifierResults classifier_;
};

}  // namespace cyxwiz
