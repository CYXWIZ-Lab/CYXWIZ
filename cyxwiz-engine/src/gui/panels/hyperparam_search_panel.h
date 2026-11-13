#pragma once

#include "gui/panel.h"
#include "core/engine_config.h"
#include "core/hyperparam_search.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

// How to search: the strategy, trials, time limit and score of `config`
// (the panel's Setup and Preferences > Search). Grid shows its
// number of combinations when `show_combinations`. True when one changed.
bool SearchSettingsFields(SearchConfig& config, bool show_combinations);

/**
 * Hyperparameter Search (TOFIX178 T4 H1, approved mockup)
 *
 * Setup: what to vary, picked from the compiled graph's properties a trial
 * can change; how to search (Grid / Random / Bayesian), the trial count, the
 * time limit and the score. Start search goes through Train's checks and data
 * preparation (MainWindow); each trial then trains the graph the way Train
 * does with the values set (core/trial_runner).
 *
 * Trials: progress, the best trial so far (Apply it to the graph, Ctrl+Z
 * undoes; Export CSV), the score per trial and a row per trial.
 */
class HyperparamSearchPanel : public Panel {
public:
    HyperparamSearchPanel();
    ~HyperparamSearchPanel() override = default;

    void Render() override;

    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

    // Start search: Train's checks and preparation, ending in BeginTrials
    // or NotStarted. `launch_pending` is true while that launch waits (data
    // loads, the memory question).
    void SetStartCallback(std::function<void()> start, std::function<bool()> launch_pending) {
        start_ = std::move(start);
        launch_pending_ = std::move(launch_pending);
    }
    // What Train would train, prepared: the trials start (main thread).
    void BeginTrials(TrialSetup setup);
    // The launch stopped before the trials ("Not started: ...").
    void NotStarted(const std::string& why);

    // Preparing or running trials.
    bool IsSearching() const;

private:
    enum class Stage { Setup, Preparing, Running, Finished };

    void RenderHeader();
    void RenderSetup();
    void RenderParameters();
    void RenderPicker();
    void RenderHowToSearch();
    void RenderTrials();
    void RenderBest(const std::vector<TrialResult>& results, int best_id);
    void RenderScoreBars(const std::vector<TrialResult>& results, int best_id);
    void RenderTrialTable(const std::vector<TrialResult>& results, int best_id);

    void StartSearch();
    void ApplyBest();
    void ExportCsv();
    // Why Start search cannot run now ("" when it can).
    std::string StartBlocker() const;

    gui::NodeEditor* node_editor_ = nullptr;
    std::function<void()> start_;
    std::function<bool()> launch_pending_;

    std::unique_ptr<HyperparamSearch> search_engine_;
    SearchConfig config_;  // as set up; the search runs a copy

    Stage stage_ = Stage::Setup;
    bool show_trials_ = false;   // the Trials view, else Setup
    bool picker_open_ = false;
    std::string not_started_;    // why the last Start search did not start
    // The saved defaults last set on config_: when Preferences changes them,
    // the setup follows (not while a search runs).
    std::optional<core::HyperparamSearchDefaults> followed_defaults_;
    std::string applied_note_;   // after Apply best
};

} // namespace cyxwiz
