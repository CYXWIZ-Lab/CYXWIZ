#pragma once

// Data Studio > Results > Cross-validation and Learning curve (TOFIX129 D4,
// approved mockup boards 6-7). One class, two modes:
//  - Cross-validation: the graph trained k times, each fold scored once;
//  - Learning curve: the same over shares of each fold's training rows.
// The settings are typed (folds, seed, shares), Run asks MainWindow to start
// it through Train's own preparation (it compiles the graph, prepares its data
// and hands the table to core/cross_validation_runner on a task), and the
// outcome is drawn from core/cross_validation_presentation. Progress and the
// folds arrive from the task (any thread).

#include "../../core/cross_validation_presentation.h"
#include "../../core/cross_validation_runner.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class FoldsView {
public:
    enum class Mode { CrossValidation, LearningCurve };
    explicit FoldsView(Mode mode);

    void Render();
    bool IsRunning() const;

    // From the task (any thread).
    void Started();
    void Progress(const CrossValidationProgress& progress);
    void FoldDone(const CrossValidationRun& done);
    void Finished(CrossValidationOutcome outcome);

    // Run: MainWindow starts it on the open graph; Cancel stops it.
    std::function<void(const CrossValidationSettings& settings)> run;
    std::function<void()> cancel;
    // Why Run cannot start now ("" when it can): no graph, Train running.
    std::function<std::string()> run_blocker;
    // The latest test's main score (accuracy, or MAE), for the sentence.
    std::function<std::optional<double>()> single_test;
    // The graph's Cross-Validation / Learning Curves node, when it has one: its
    // settings are what Run uses (edited in the node), or why they are wrong.
    struct GraphNode {
        std::string name;
        CrossValidationSettings settings;
        std::string error;
    };
    std::function<std::optional<GraphNode>()> graph_node;
    // Add to the graph: the node with these settings after the Data Input; a
    // word on what happened.
    std::function<std::string(const CrossValidationSettings& settings)> add_to_graph;
    // Train > Cross-validate / Learning curve: Run on the next frame.
    void RequestRun() { run_requested_ = true; }
    bool IsLearningCurve() const { return mode_ == Mode::LearningCurve; }

private:
    CrossValidationSettings Settings();
    void RenderLocked();
    void RenderSettings(bool running);
    void RenderRunning();
    void RenderCrossValidation(const CrossValidationOutcome& o);
    void RenderLearningCurve(const CrossValidationOutcome& o);
    void Export();

    Mode mode_;
    // Settings (typed).
    int folds_ = 5;
    int stratified_ = 0;  // 0 yes, 1 no
    int seed_ = 42;
    char shares_text_[128] = "10%, 25%, 50%, 75%, 100%";
    std::string shares_error_;
    int score_ = 0;       // index into CurveScores(regression)

    mutable std::mutex mutex_;
    bool running_ = false;
    std::chrono::steady_clock::time_point started_;
    CrossValidationProgress progress_;
    std::vector<CrossValidationRun> live_;  // folds finished in the running job
    std::optional<CrossValidationOutcome> outcome_;
    uint64_t outcome_version_ = 0;

    // Drawn from the outcome (rebuilt when it or the score changes).
    uint64_t drawn_version_ = UINT64_MAX;
    int drawn_score_ = -1;
    folds_view::CrossValidationView cv_;
    folds_view::LearningCurveView curve_;

    std::string note_;
    bool note_failed_ = false;
    std::optional<CrossValidationSettings> pending_run_;  // pressed this frame (called after the lock)
    bool run_requested_ = false;                          // Train menu
    bool pending_cancel_ = false;
};

}  // namespace cyxwiz
