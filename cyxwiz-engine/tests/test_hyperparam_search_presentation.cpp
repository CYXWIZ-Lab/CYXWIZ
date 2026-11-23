// Hyperparameter Search's presentation (TOFIX178 T4 H1): what can be varied,
// read from the compiled graph (the head is listed, not searched); a picked
// property's starting range and its words; the trials' status line; the best
// trial's values and its changes applied to the graph.

#include "core/hyperparam_search_presentation.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace sp = cyxwiz::search_presentation;
using gui::MLNode;
using gui::NodeLink;
using gui::NodeType;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

MLNode Node(int id, NodeType type, const char* name, std::map<std::string, std::string> parameters = {}) {
    MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.parameters = std::move(parameters);
    gui::NodePin in;
    in.id = id * 10;
    in.is_input = true;
    gui::NodePin out;
    out.id = id * 10 + 1;
    node.inputs.push_back(in);
    node.outputs.push_back(out);
    return node;
}

const SearchParameter* Find(const SearchConfig& config, const std::string& name) {
    for (const auto& p : config.space.parameters)
        if (p.name == name) return &p;
    return nullptr;
}

}  // namespace

int main() {
    // mnist_mlp's shape: Data Input -> Dense 1 (128) -> ReLU -> Dense 2 (10),
    // Adam, DataLoader (batch 64, 10 epochs).
    const std::vector<MLNode> nodes = {
        Node(1, NodeType::DataInput, "Data Input"),
        Node(2, NodeType::Dense, "Dense 1", {{"units", "128"}, {"use_bias", "true"}}),
        Node(3, NodeType::ReLU, "ReLU"),
        Node(4, NodeType::Dense, "Dense 2", {{"units", "10"}}),
        Node(5, NodeType::Adam, "Adam", {{"learning_rate", "0.001"}, {"beta1", "0.9"}, {"weight_decay", "0"}, {"epsilon", "1e-8"}}),
        Node(6, NodeType::DataLoader, "DataLoader", {{"batch_size", "64"}, {"epochs", "10"}, {"shuffle", "true"}}),
    };
    std::vector<NodeLink> links;
    for (int i = 1; i < 4; ++i) links.push_back({i, i, i * 10 + 1, i + 1, (i + 1) * 10});
    TrainingConfiguration compiled;
    for (int id : {2, 3, 4}) {
        CompiledLayer layer;
        layer.node_id = id;
        compiled.layers.push_back(layer);
    }
    compiled.optimizer_node_id = 5;

    const auto targets = sp::SearchTargets(nodes, links, compiled);
    Check(targets.size() == 4, "targets: Dense 1, Dense 2, Adam, DataLoader (ReLU has nothing), got " +
                                   std::to_string(targets.size()));
    if (targets.size() == 4) {
        Check(targets[0].node_name == "Dense 1" && targets[0].properties == std::vector<std::string>{"units"},
              "Dense 1 offers units (not use_bias)");
        Check(targets[1].node_name == "Dense 2" && targets[1].properties.empty() &&
                  targets[1].note == "sets the 10 outputs the loss expects: not searched",
              "the head is listed with its reason: " + targets[1].note);
        Check(targets[2].properties == std::vector<std::string>{"beta1", "learning_rate", "weight_decay"},
              "Adam offers beta1, learning rate, weight decay (not epsilon)");
        Check(targets[3].properties == std::vector<std::string>{"batch_size", "epochs"},
              "the DataLoader offers batch size and epochs");
    }

    // Starting ranges and their words (the mockup's three rows).
    const auto lr = sp::DefaultParameter(nodes[4], "learning_rate");
    Check(lr.type == ParamType::LogScale && lr.node_id == 5 && lr.property == "learning_rate", "learning rate: log scale");
    Check(lr.name == "Adam learning rate", "named after the node: " + lr.name);
    Check(sp::KindText(lr) == "Log scale" && sp::RangeText(lr) == "0.0001 - 0.01", "lr range " + sp::RangeText(lr));
    const auto units = sp::DefaultParameter(nodes[1], "units");
    Check(sp::KindText(units) == "Whole number, step 32" && sp::RangeText(units) == "32 - 256",
          "units " + sp::KindText(units) + " " + sp::RangeText(units));
    const auto batch = sp::DefaultParameter(nodes[5], "batch_size");
    Check(sp::KindText(batch) == "Choices" && sp::RangeText(batch) == "32, 64, 128", "batch " + sp::RangeText(batch));
    const auto beta1 = sp::DefaultParameter(nodes[4], "beta1");
    Check(beta1.type == ParamType::Continuous && sp::RangeText(beta1) == "0.8 - 0.95", "beta1 " + sp::RangeText(beta1));
    const auto decay = sp::DefaultParameter(nodes[4], "weight_decay");
    Check(decay.type == ParamType::LogScale && decay.min_value > 0.0f, "a weight decay of 0 still gets a log range");

    // The search space is valid and each value becomes the node's property.
    SearchConfig config;
    config.space.parameters = {lr, units, batch};
    Check(config.space.IsValid(), "the picked parameters make a valid space");
    TrialResult best;
    best.trial_id = 4;
    best.completed = true;
    best.val_loss = 0.081f;
    best.val_accuracy = 0.976f;
    best.parameters = {{"Adam learning rate", 0.0021f}, {"Dense 1 units", 192.0f}};
    best.categorical_params = {{"DataLoader batch size", "64"}};
    Check(sp::ScoreText(best) == "Validation loss 0.081 \xC2\xB7 accuracy 97.6 %", "score " + sp::ScoreText(best));
    Check(sp::ValuesText(config, best) ==
              "Adam learning rate 0.0021 \xC2\xB7 Dense 1 units 192 \xC2\xB7 DataLoader batch size 64",
          "values " + sp::ValuesText(config, best));
    const auto applied = sp::WithChanges(nodes, TrialChangesFor(config.space, best.parameters, best.categorical_params));
    Check(applied[4].parameters.at("learning_rate") == "0.0021" && applied[1].parameters.at("units") == "192" &&
              applied[5].parameters.at("batch_size") == "64" && applied[3].parameters.at("units") == "10",
          "Apply best sets the three properties and nothing else");
    Check(Find(config, "Dense 1 units") != nullptr && sp::ValueText(*Find(config, "Dense 1 units"), best) == "192",
          "a whole number shows without decimals");

    // Two nodes with one name: the second parameter gets its node number.
    {
        SearchSpace space;
        MLNode other = Node(9, NodeType::Dense, "Dense 1", {{"units", "64"}});
        sp::AddParameter(space, sp::DefaultParameter(nodes[1], "units"));
        sp::AddParameter(space, sp::DefaultParameter(other, "units"));
        Check(space.parameters.size() == 2 && space.parameters[1].name == "Dense 1 units #9",
              "a repeated name is made unique: " + space.parameters.back().name);
    }

    // The status line: trial 7 of 20, epoch 4 of 10, six done at 52 s each.
    std::vector<TrialResult> done(6);
    for (auto& r : done) {
        r.completed = true;
        r.duration_seconds = 52.0f;
    }
    done[3].completed = false;
    const auto running = sp::Status(true, SearchEnd::AllTrials, 6, 20, 4, 10, done);
    Check(running.headline == "Trial 7 of 20 \xC2\xB7 epoch 4 of 10", "headline " + running.headline);
    Check(running.timing == "about 52 s per trial \xC2\xB7 12 min left", "timing " + running.timing);
    Check(running.fraction > 0.31f && running.fraction < 0.33f, "fraction counts the epochs of the running trial");
    Check(running.label == "Running", "running label");
    const auto first = sp::Status(true, SearchEnd::AllTrials, 0, 20, 0, 0, {});
    Check(first.timing == "The time per trial shows after the first one.", "before the first trial ends");
    const auto finished = sp::Status(false, SearchEnd::AllTrials, 0, 6, 0, 0, done);
    Check(finished.label == "Finished" && finished.headline == "6 trials, 1 failed", "finished " + finished.headline);
    // Ended early: the label says why, the headline how far it got.
    const auto stopped = sp::Status(false, SearchEnd::Stopped, 0, 20, 0, 0, done);
    Check(stopped.label == "Stopped" && stopped.headline == "6 of 20 trials, 1 failed", "stopped " + stopped.headline);
    Check(sp::Status(false, SearchEnd::TimeLimit, 0, 20, 0, 0, done).label == "Time limit reached", "time limit");
    Check(sp::Status(false, SearchEnd::Failures, 0, 20, 0, 0, done).label == "Three trials in a row failed",
          "three failures");

    // The saved defaults (Preferences): set on a search, read back, saved and
    // reloaded; values out of range are clamped.
    {
        core::HyperparamSearchDefaults defaults;
        defaults.strategy = "bayesian";
        defaults.trials = 40;
        defaults.hours = 2.5;
        defaults.objective = "val_accuracy";
        SearchConfig search;
        sp::ApplyDefaults(search, defaults);
        Check(search.strategy == SearchStrategy::Bayesian && search.max_trials == 40 &&
                  search.max_time_seconds == 9000.0f && search.objective == "val_accuracy" && !search.minimize,
              "the defaults set the strategy, trials, hours and score");
        Check(sp::DefaultsOf(search) == defaults, "and read back the same");

        auto& engine_config = core::EngineConfig::Instance();
        const auto path = std::filesystem::temp_directory_path() / "cyxwiz_search_defaults_test.json";
        engine_config.SetSearchDefaults(defaults);
        Check(engine_config.Save(path), "the settings save");
        engine_config.SetSearchDefaults({});
        Check(engine_config.Load(path), "the settings load");
        Check(engine_config.GetSearchDefaults() == defaults, "the defaults survive a save and load");
        std::error_code error;
        std::filesystem::remove(path, error);

        core::HyperparamSearchDefaults wrong;
        wrong.strategy = "anneal";
        wrong.trials = 5000;
        wrong.hours = 0.0;
        wrong.objective = "f1";
        engine_config.SetSearchDefaults(wrong);
        const auto clamped = engine_config.GetSearchDefaults();
        Check(clamped.strategy == "random" && clamped.trials == core::kSearchMaxTrials &&
                  clamped.hours == core::kSearchMinHours && clamped.objective == "val_loss",
              "unknown or out-of-range defaults are clamped");
    }

    if (g_failures) {
        std::printf("%d search presentation checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("Hyperparameter search presentation passed\n");
    return EXIT_SUCCESS;
}
