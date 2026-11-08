#include "hyperparam_search_presentation.h"

#include "nas_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace cyxwiz::search_presentation {

namespace {

// The properties offered for a search, by how their values vary.
const std::set<std::string> kWhole = {"units",         "filters",   "out_channels", "hidden_size", "num_layers",
                                      "embedding_dim", "num_heads", "epochs",       "T_max",       "step_size"};
const std::set<std::string> kLog = {"learning_rate", "lr", "weight_decay", "eta_min"};
const std::set<std::string> kFraction = {"rate",     "dropout",     "beta1",        "beta2",
                                         "momentum", "warmup_ratio", "min_lr_ratio", "gamma"};

bool Offered(const std::string& property) {
    return property == "batch_size" || kWhole.count(property) || kLog.count(property) || kFraction.count(property);
}

double ValueOf(const gui::MLNode& node, const std::string& property, double fallback) {
    const auto it = node.parameters.find(property);
    if (it == node.parameters.end()) return fallback;
    try {
        size_t used = 0;
        const double v = std::stod(it->second, &used);
        return used > 0 && std::isfinite(v) ? v : fallback;
    } catch (...) {
        return fallback;
    }
}

std::string Number(double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.4g", v);
    return text;
}

const char* const kDot = " \xC2\xB7 ";  // " · "

}  // namespace

std::vector<TargetNode> SearchTargets(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                                      const TrainingConfiguration& compiled) {
    std::set<int> heads;
    {
        const std::vector<int> trainable = NASEvaluator::FindTrainableLayers(nodes);
        const std::vector<int> body = NASEvaluator::FindBodyLayers(nodes, links);
        for (int index : trainable)
            if (std::find(body.begin(), body.end(), index) == body.end()) heads.insert(nodes[index].id);
    }
    const auto node_of = [&nodes](int id) -> const gui::MLNode* {
        const auto it = std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
        return it == nodes.end() ? nullptr : &*it;
    };

    std::vector<TargetNode> targets;
    std::set<int> listed;
    const auto add = [&](const gui::MLNode* node, bool loop) {
        if (!node || !listed.insert(node->id).second) return;
        TargetNode target{node->id, node->name, {}, {}};
        if (heads.count(node->id)) {
            const double outputs = ValueOf(*node, "units", ValueOf(*node, "out_channels", ValueOf(*node, "filters", 0)));
            target.note = outputs > 0 ? "sets the " + Number(outputs) + " outputs the loss expects: not searched"
                                      : "sets the outputs the loss expects: not searched";
            targets.push_back(std::move(target));
            return;
        }
        for (const auto& [property, value] : node->parameters) {
            if (loop ? (property == "batch_size" || property == "epochs") : Offered(property))
                target.properties.push_back(property);
        }
        if (!target.properties.empty()) targets.push_back(std::move(target));
    };
    for (const auto& layer : compiled.layers) add(node_of(layer.node_id), false);
    add(node_of(compiled.optimizer_node_id), false);
    add(node_of(compiled.scheduler_node_id), false);
    add(node_of(compiled.regularization_node_id), false);
    for (const auto& node : nodes)
        if (node.type == gui::NodeType::DataLoader) add(&node, true);
    return targets;
}

std::string PropertyLabel(const std::string& property) {
    std::string label = property;
    std::replace(label.begin(), label.end(), '_', ' ');
    return label;
}

std::string ParameterName(const gui::MLNode& node, const std::string& property) {
    return node.name + " " + PropertyLabel(property);
}

void AddParameter(SearchSpace& space, SearchParameter parameter) {
    const auto taken = [&space](const std::string& name) {
        return std::any_of(space.parameters.begin(), space.parameters.end(),
                           [&name](const SearchParameter& p) { return p.name == name; });
    };
    if (taken(parameter.name)) parameter.name += " #" + std::to_string(parameter.node_id);
    space.parameters.push_back(std::move(parameter));
}

SearchParameter DefaultParameter(const gui::MLNode& node, const std::string& property) {
    const std::string name = ParameterName(node, property);
    SearchParameter p;
    if (property == "batch_size") {
        const int v = std::max(1, static_cast<int>(std::lround(ValueOf(node, property, 32))));
        std::vector<std::string> options;
        if (v / 2 >= 1) options.push_back(std::to_string(v / 2));
        options.push_back(std::to_string(v));
        options.push_back(std::to_string(v * 2));
        p = SearchParameter::Categorical(name, options);
    } else if (kWhole.count(property)) {
        const int v = std::max(1, static_cast<int>(std::lround(ValueOf(node, property, 64))));
        const int step = std::max(1, v / 4);
        p = SearchParameter::Discrete(name, step, std::max(v * 2, step + 1), step);
    } else if (kLog.count(property)) {
        const double v = ValueOf(node, property, 0.0);
        p = v > 0.0 ? SearchParameter::LogScale(name, static_cast<float>(v / 10.0), static_cast<float>(v * 10.0))
                    : SearchParameter::LogScale(name, 1e-6f, 1e-2f);
    } else {
        // A fraction: near 1 (beta, momentum) it varies in its distance to 1.
        const double v = std::clamp(ValueOf(node, property, 0.0), 0.0, 1.0);
        if (v >= 0.8) {
            p = SearchParameter::Continuous(name, static_cast<float>(1.0 - (1.0 - v) * 2.0),
                                            static_cast<float>(1.0 - (1.0 - v) / 2.0));
        } else if (v > 0.0) {
            p = SearchParameter::Continuous(name, static_cast<float>(v / 2.0), static_cast<float>(std::min(0.95, v * 2.0)));
        } else {
            p = SearchParameter::Continuous(name, 0.0f, 0.5f);
        }
    }
    p.node_id = node.id;
    p.property = property;
    return p;
}

std::string KindText(const SearchParameter& parameter) {
    switch (parameter.type) {
    case ParamType::LogScale: return "Log scale";
    case ParamType::Discrete: return "Whole number, step " + Number(parameter.step);
    case ParamType::Categorical: return "Choices";
    case ParamType::Continuous: return "Decimal";
    }
    return {};
}

std::string RangeText(const SearchParameter& parameter) {
    if (parameter.type == ParamType::Categorical) {
        std::string text;
        for (const auto& option : parameter.options) text += (text.empty() ? "" : ", ") + option;
        return text;
    }
    return Number(parameter.min_value) + " - " + Number(parameter.max_value);
}

std::string StrategyText(SearchStrategy strategy) {
    switch (strategy) {
    case SearchStrategy::Grid:
        return "Every combination of the values (a range is cut into 10 steps). Best for 1 or 2 properties with few "
               "values.";
    case SearchStrategy::Random: return "Values drawn at random within the ranges. A good start for many properties.";
    case SearchStrategy::Bayesian:
        return "A few random trials first, then each next trial goes where the results so far say the score should "
               "improve most (Gaussian process). Finds good values in fewer trials.";
    }
    return {};
}

const std::vector<Objective>& Objectives() {
    static const std::vector<Objective> objectives = {
        {"val_loss", "Validation loss (lower)", true},
        {"val_accuracy", "Validation accuracy (higher)", false},
        {"val_mae", "Validation MAE (numbers)", true},
    };
    return objectives;
}

std::string ObjectiveName(const std::string& key) {
    if (key == "val_accuracy") return "Validation accuracy";
    if (key == "val_mae") return "Validation MAE";
    return "Validation loss";
}

RunStatus Status(bool running, int trial, int total, int epoch, int epochs, const std::vector<TrialResult>& results) {
    RunStatus status;
    total = std::max(total, 1);
    const int finished = static_cast<int>(results.size());
    int failed = 0;
    double seconds = 0.0;
    for (const auto& r : results) {
        if (!r.completed) ++failed;
        seconds += r.duration_seconds;
    }
    if (running) {
        status.headline = "Trial " + std::to_string(trial + 1) + " of " + std::to_string(total);
        if (epochs > 0) status.headline += kDot + std::string("epoch ") + std::to_string(epoch) + " of " + std::to_string(epochs);
        const float within = epochs > 0 ? static_cast<float>(epoch) / static_cast<float>(epochs) : 0.0f;
        status.fraction = std::clamp((static_cast<float>(finished) + within) / static_cast<float>(total), 0.0f, 1.0f);
    } else {
        status.headline = std::to_string(finished) + (finished == 1 ? " trial" : " trials");
        if (failed > 0) status.headline += ", " + std::to_string(failed) + " failed";
        status.fraction = finished > 0 ? 1.0f : 0.0f;
    }
    if (finished > 0) {
        const double per_trial = seconds / finished;
        status.timing = "about " + (per_trial < 90.0 ? Number(std::round(per_trial)) + " s"
                                                     : Number(std::round(per_trial / 60.0)) + " min") +
                        " per trial";
        if (running) {
            const double left = per_trial * std::max(0, total - finished) / 60.0;
            status.timing += kDot + (left < 1.0 ? std::string("under a minute left")
                                                : Number(std::round(left)) + " min left");
        }
    } else if (running) {
        status.timing = "The time per trial shows after the first one.";
    }
    return status;
}

std::string ScoreText(const TrialResult& result) {
    if (!result.completed) return result.error_message;
    char text[96];
    if (result.regression) {
        std::snprintf(text, sizeof(text), "Validation loss %.4g%sMAE %.4g", result.val_loss, kDot, result.val_mae);
    } else {
        std::snprintf(text, sizeof(text), "Validation loss %.4g%saccuracy %.1f %%", result.val_loss, kDot,
                      result.val_accuracy * 100.0f);
    }
    return text;
}

std::string ValueText(const SearchParameter& parameter, const TrialResult& result) {
    if (parameter.type == ParamType::Categorical) {
        const auto it = result.categorical_params.find(parameter.name);
        return it == result.categorical_params.end() ? "-" : it->second;
    }
    const auto it = result.parameters.find(parameter.name);
    if (it == result.parameters.end()) return "-";
    if (parameter.type == ParamType::Discrete) return std::to_string(std::lround(it->second));
    return Number(it->second);
}

std::string ValuesText(const SearchConfig& config, const TrialResult& result) {
    std::string text;
    for (const auto& p : config.space.parameters)
        text += (text.empty() ? "" : kDot) + p.name + " " + ValueText(p, result);
    return text;
}

std::vector<gui::MLNode> WithChanges(std::vector<gui::MLNode> nodes, const std::vector<TrialChange>& changes) {
    for (const auto& change : changes) {
        for (auto& node : nodes)
            if (node.id == change.node_id) node.parameters[change.property] = change.value;
    }
    return nodes;
}

}  // namespace cyxwiz::search_presentation
