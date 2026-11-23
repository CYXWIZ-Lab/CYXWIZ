#pragma once

// What Hyperparameter Search shows (TOFIX178 T4 H1, approved mockup), as data:
// the properties a trial can change, read from the compiled graph; a picked
// property's starting range; the words for the kinds, the strategies and the
// trials; the best trial's changes applied to the graph. No ImGui.

#include "engine_config.h"
#include "hyperparam_search.h"

#include <string>
#include <vector>

namespace cyxwiz::search_presentation {

// A node of the trained model and the properties a trial can change on it.
struct TargetNode {
    int node_id = -1;
    std::string node_name;
    std::vector<std::string> properties;  // by name, as the node keeps them
    std::string note;                     // why there are none (a head)
};

// The layers in model order, then the optimizer, the scheduler or
// regularization and the DataLoader. A head (a layer with no trainable layer
// after it: it sets the outputs the loss expects) is listed with a note.
std::vector<TargetNode> SearchTargets(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                                      const TrainingConfiguration& compiled);

// "learning_rate" -> "learning rate".
std::string PropertyLabel(const std::string& property);
// The parameter's name in the search: "Adam learning rate".
std::string ParameterName(const gui::MLNode& node, const std::string& property);
// `parameter` added to `space`, its name made unique: a second node with the
// same name gets its node number ("Dense units #4"), since trial values are
// kept by name.
void AddParameter(SearchSpace& space, SearchParameter parameter);

// The range a picked property starts with, around its value on the graph:
// a rate on a log scale from a tenth to ten times, a count as whole numbers
// from a quarter to twice in quarter steps, the batch size as half / same /
// double, a fraction around its value.
SearchParameter DefaultParameter(const gui::MLNode& node, const std::string& property);

// "Log scale", "Whole number, step 32", "Choices", "Decimal".
std::string KindText(const SearchParameter& parameter);
// "0.0001 - 0.01", "32 - 256", "32, 64, 128".
std::string RangeText(const SearchParameter& parameter);
// One sentence on how the strategy picks the next values.
std::string StrategyText(SearchStrategy strategy);

struct Objective {
    const char* key;    // SearchConfig::objective
    const char* label;  // "Validation loss (lower)"
    bool minimize;
};
const std::vector<Objective>& Objectives();
// "Validation loss", "Validation accuracy", "Validation MAE".
std::string ObjectiveName(const std::string& key);

// The saved defaults (Preferences > Search) set on a search's
// strategy, trials, time limit and score; and read back from one.
void ApplyDefaults(SearchConfig& config, const core::HyperparamSearchDefaults& defaults);
core::HyperparamSearchDefaults DefaultsOf(const SearchConfig& config);

// The Trials view's status line.
struct RunStatus {
    std::string label;     // "Running", "Finished", "Stopped", "Time limit reached", "Three trials in a row failed"
    std::string headline;  // "Trial 7 of 20 · epoch 4 of 10"; when finished "20 trials, 1 failed", ended early "5 of 20 trials"
    std::string timing;    // "about 52 s per trial · 11 min left"
    float fraction = 0.0f; // of all the trials
};
// `trial` is the running trial (0-based), `epoch` / `epochs` its progress;
// `end` says why a search that is not running ended.
RunStatus Status(bool running, SearchEnd end, int trial, int total, int epoch, int epochs,
                 const std::vector<TrialResult>& results);

// "Validation loss 0.081 · accuracy 97.6 %" (MAE for numbers).
std::string ScoreText(const TrialResult& result);
// One parameter's value in a trial: "0.0021", "192", "64".
std::string ValueText(const SearchParameter& parameter, const TrialResult& result);
// "Adam learning rate 0.0021 · Dense 1 units 192".
std::string ValuesText(const SearchConfig& config, const TrialResult& result);

// The graph with the changes set (Apply best to the graph).
std::vector<gui::MLNode> WithChanges(std::vector<gui::MLNode> nodes, const std::vector<TrialChange>& changes);

}  // namespace cyxwiz::search_presentation
