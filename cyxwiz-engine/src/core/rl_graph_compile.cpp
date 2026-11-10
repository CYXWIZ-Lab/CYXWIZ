#include "rl_graph_compile.h"

#include "rl_graph_presentation.h"

#include <cyxwiz/error_codes.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>

namespace cyxwiz {

namespace {

std::mutex g_lookup_mutex;
RLEnvironmentLookup g_lookup;

bool LookupEnvironment(const std::string& key, plugin::PluginEnvironmentInfo& info) {
    std::lock_guard<std::mutex> lock(g_lookup_mutex);
    return g_lookup && g_lookup(key, info);
}

std::string Param(const gui::MLNode& node, const char* key, const char* fallback) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() || it->second.empty() ? fallback : it->second;
}

const gui::NodeLink* LinkInto(const std::vector<gui::NodeLink>& links, const gui::MLNode& node, size_t input) {
    if (input >= node.inputs.size()) return nullptr;
    for (const auto& l : links)
        if (l.to_node == node.id && l.to_pin == node.inputs[input].id) return &l;
    return nullptr;
}

}  // namespace

void SetRLEnvironmentLookup(RLEnvironmentLookup lookup) {
    std::lock_guard<std::mutex> lock(g_lookup_mutex);
    g_lookup = std::move(lookup);
}

std::vector<int> HiddenUnits(const std::vector<CompiledLayer>& layers) {
    std::vector<int> units;
    for (const auto& layer : layers)
        if (layer.units > 0) units.push_back(layer.units);
    return units;
}

bool IsRLGraphNode(gui::NodeType type) {
    return type == gui::NodeType::RLEnvironment || type == gui::NodeType::PolicyHead ||
           type == gui::NodeType::ValueHead || type == gui::NodeType::PPO;
}

TrainingConfiguration GraphCompiler::CompileReinforcementLearning(const std::vector<gui::MLNode>& nodes,
                                                                  const std::vector<gui::NodeLink>& links) {
    TrainingConfiguration config;
    auto rl = std::make_shared<RLTrainingConfiguration>();
    const auto issue = [&](IssueLevel level, const std::string& message, const gui::MLNode* node,
                           const char* code) {
        ValidationIssue i;
        i.level = level;
        i.message = message;
        i.node_id = node ? node->id : -1;
        i.node_name = node ? node->name : "";
        i.error_code = code;
        config.issues.push_back(std::move(i));
    };
    const auto error = [&](const std::string& message, const gui::MLNode* node,
                           const char* code = errors::Compiler::InvalidConnectivity) {
        issue(IssueLevel::Error, message, node, code);
    };

    // ---- One PPO, and nothing of a supervised graph
    std::vector<const gui::MLNode*> ppos;
    for (const auto& n : nodes) {
        if (n.type == gui::NodeType::PPO) ppos.push_back(&n);
        else if (n.type == gui::NodeType::DataInput || n.type == gui::NodeType::DatasetInput ||
                 std::any_of(n.outputs.begin(), n.outputs.end(), [](const gui::NodePin& pin) {
                     return pin.type == gui::PinType::Loss || pin.type == gui::PinType::Optimizer;
                 }))
            error(n.name + " belongs to a model trained on data; a graph trains either that or a policy on an "
                           "environment, not both",
                  &n, errors::Compiler::UnsupportedTrainingNode);
    }
    if (ppos.size() != 1) {
        error(ppos.empty() ? "A reinforcement learning graph needs a PPO node: connect its Policy, Environment and "
                             "Value"
                           : "A graph trains one policy: keep one PPO node",
              ppos.empty() ? nullptr : ppos[1], errors::Compiler::MissingTrainingPathNode);
    }
    const gui::MLNode* ppo = ppos.empty() ? nullptr : ppos[0];

    // ---- PPO's three inputs
    const gui::MLNode* env = nullptr;
    const gui::MLNode* policy_head = nullptr;
    const gui::MLNode* value_head = nullptr;
    if (ppo) {
        const auto from = [&](size_t input, gui::NodeType want, const char* what) -> const gui::MLNode* {
            const gui::NodeLink* l = LinkInto(links, *ppo, input);
            const gui::MLNode* n = l ? FindNodeById(l->from_node, nodes) : nullptr;
            if (!n || n->type != want) {
                error(std::string("Connect ") + what + " to PPO's " + ppo->inputs[input].name, ppo,
                      errors::Compiler::MissingTrainingPathNode);
                return nullptr;
            }
            return n;
        };
        policy_head = from(0, gui::NodeType::PolicyHead, "a Policy Head");
        env = from(1, gui::NodeType::RLEnvironment, "an Environment's Environment pin");
        value_head = from(2, gui::NodeType::ValueHead, "a Value Head");
        const gui::NodeLink* env_link = LinkInto(links, *ppo, 1);
        if (env && env_link && env->outputs.size() > 1 && env_link->from_pin != env->outputs[1].id) {
            error("PPO's Environment takes the Environment's Environment pin, not its Observation", ppo);
            env = nullptr;
        }
    }

    // ---- The environment
    if (env) {
        rl->environment_node_id = env->id;
        rl->environment_key = Param(*env, "environment", "");
        rl->environment_settings = rl_graph::StoredSettings(env->parameters);
        try {
            rl->copies = std::stoi(Param(*env, "copies", "1"));
        } catch (...) {
            rl->copies = 0;
        }
        if (rl->copies < 1 || rl->copies > 1024)
            error("Copies must be a whole number from 1 to 1024", env, errors::Compiler::InvalidParameter);
        if (!LookupEnvironment(rl->environment_key, rl->environment)) {
            const size_t colon = rl->environment_key.find(':');
            const std::string id = colon == std::string::npos ? rl->environment_key : rl->environment_key.substr(colon + 1);
            const std::string provider = colon == std::string::npos ? "" : rl->environment_key.substr(0, colon);
            error(rl->environment_key.empty()
                      ? std::string("Pick an environment")
                      : id + " comes from the plugin " + provider + ", which is not loaded: load it in Plugin Manager",
                  env, errors::Compiler::MissingTrainingPathNode);
            env = nullptr;
        } else {
            rl->observation_size = static_cast<size_t>(rl->environment.observation_size);
            rl->policy_outputs = static_cast<size_t>(rl->environment.discrete_actions ? rl->environment.action_count
                                                                                     : rl->environment.action_size);
        }
    }

    // ---- Each network: one chain of layers from the Observation to its head
    const auto chain = [&](const gui::MLNode* head, std::vector<CompiledLayer>& layers) {
        if (!head || !env) return;
        std::vector<const gui::MLNode*> path;
        const gui::MLNode* at = head;
        std::set<int> seen;
        while (true) {
            const gui::NodeLink* l = LinkInto(links, *at, 0);
            const gui::MLNode* prev = l ? FindNodeById(l->from_node, nodes) : nullptr;
            if (!prev || !seen.insert(prev->id).second) {
                error("Nothing reaches the " + head->name + " from the Environment's Observation" +
                          (at == head ? "" : ": " + at->name + "'s input is not connected"),
                      head);
                return;
            }
            if (prev->type == gui::NodeType::RLEnvironment) {
                if (prev->id != env->id) {
                    error(head->name + " is reached from another Environment than the one PPO trains in", head);
                    return;
                }
                if (prev->outputs.empty() || l->from_pin != prev->outputs[0].id) {
                    error("Connect the Environment's Observation, not its Environment pin, to " + at->name, at);
                    return;
                }
                break;
            }
            const bool allowed = prev->type == gui::NodeType::Dense || prev->type == gui::NodeType::LayerNorm ||
                                 prev->type == gui::NodeType::Dropout || IsActivation(prev->type);
            if (!allowed) {
                error(prev->name + " cannot be in a policy or value network yet: use Dense, activations, Layer Norm "
                                   "or Dropout",
                      prev, errors::Compiler::UnsupportedTrainingNode);
                return;
            }
            path.push_back(prev);
            at = prev;
        }
        std::reverse(path.begin(), path.end());
        for (const auto* n : path) layers.push_back(ExtractLayerConfig(*n));
        if (!path.empty() && path.back()->type == gui::NodeType::Softmax && head->type == gui::NodeType::PolicyHead)
            issue(IssueLevel::Warning,
                  "A Softmax right before the Policy Head: the head already turns its input into a distribution, "
                  "so remove it unless you mean it",
                  path.back(), errors::Compiler::GenericIssue);
    };
    if (policy_head) rl->policy_head_node_id = policy_head->id;
    if (value_head) rl->value_head_node_id = value_head->id;
    chain(policy_head, rl->policy_layers);
    chain(value_head, rl->value_layers);

    // ---- PPO's settings
    if (ppo) {
        rl->ppo_node_id = ppo->id;
        auto& c = rl->ppo;
        const auto number = [&](const char* key, const char* fallback, double& out) {
            const std::string text = Param(*ppo, key, fallback);
            char* end = nullptr;
            out = std::strtod(text.c_str(), &end);
            if (!end || *end != '\0' || !std::isfinite(out)) {
                error("PPO's '" + std::string(key) + "' is not a number: " + text, ppo, errors::Compiler::InvalidParameter);
                return false;
            }
            return true;
        };
        double v = 0.0;
        if (number("total_timesteps", "100000", v)) c.total_timesteps = static_cast<int64_t>(v);
        if (number("n_steps", "2048", v)) c.n_steps = static_cast<int>(v);
        if (number("batch_size", "64", v)) c.batch_size = static_cast<int>(v);
        if (number("n_epochs", "10", v)) c.n_epochs = static_cast<int>(v);
        if (number("learning_rate", "0.0003", v)) c.learning_rate = static_cast<float>(v);
        if (number("gamma", "0.99", v)) c.gamma = static_cast<float>(v);
        if (number("gae_lambda", "0.95", v)) c.gae_lambda = static_cast<float>(v);
        if (number("max_grad_norm", "0.5", v)) c.max_grad_norm = static_cast<float>(v);
        if (number("clip_range", "0.2", v)) c.loss.clip_range = static_cast<float>(v);
        if (number("clip_range_vf", "0", v)) c.loss.clip_range_vf = static_cast<float>(v);
        if (number("ent_coef", "0", v)) c.loss.ent_coef = static_cast<float>(v);
        if (number("vf_coef", "0.5", v)) c.loss.vf_coef = static_cast<float>(v);
        if (number("seed", "1", v)) c.seed = static_cast<uint64_t>(std::max(v, 0.0));
        c.loss.normalize_advantage = Param(*ppo, "normalize_advantage", "true") == "true";
        c.normalize_observations = Param(*ppo, "normalize_observations", "false") == "true";
        c.normalize_rewards = Param(*ppo, "normalize_rewards", "false") == "true";
        if (Param(*ppo, "stop_at_return", "false") == "true" && number("target_return", "475", v))
            c.target_return = static_cast<float>(v);
        // Policy Head's own setting
        if (policy_head) {
            const std::string text = Param(*policy_head, "log_std_init", "0");
            char* end = nullptr;
            const double s = std::strtod(text.c_str(), &end);
            if (end && *end == '\0' && std::isfinite(s)) c.log_std_init = static_cast<float>(s);
            else error("Log std at start is not a number: " + text, policy_head, errors::Compiler::InvalidParameter);
        }
        if (c.total_timesteps < 1 || c.n_steps < 1 || c.batch_size < 1 || c.n_epochs < 1)
            error("Total steps, steps per copy, minibatch size and epochs must each be at least 1", ppo,
                  errors::Compiler::InvalidParameter);
        if (!(c.learning_rate > 0.0f))
            error("The learning rate must be above 0", ppo, errors::Compiler::InvalidParameter);
        if (!(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.gae_lambda >= 0.0f && c.gae_lambda <= 1.0f))
            error("The discount must be in (0, 1] and the GAE lambda in [0, 1]", ppo, errors::Compiler::InvalidParameter);
        const long long rollout = static_cast<long long>(rl->copies) * c.n_steps;
        if (rl->copies >= 1 && c.n_steps >= 1 && c.batch_size > rollout)
            error("Minibatch size " + std::to_string(c.batch_size) + " is larger than one rollout (" +
                      std::to_string(rl->copies) + (rl->copies == 1 ? " copy" : " copies") + " x " +
                      std::to_string(c.n_steps) + " steps = " + std::to_string(rollout) +
                      "). Lower it, or raise Steps per copy or Copies",
                  ppo, errors::Compiler::InvalidParameter);
    }

    // ---- Every RL node takes part
    for (const auto& n : nodes) {
        if (!IsRLGraphNode(n.type)) continue;
        if (n.id != rl->environment_node_id && n.id != rl->policy_head_node_id && n.id != rl->value_head_node_id &&
            n.id != rl->ppo_node_id)
            issue(IssueLevel::Warning, n.name + " is not connected to the PPO that trains", &n,
                  errors::Compiler::InvalidConnectivity);
    }

    config.is_valid = !config.HasErrors();
    std::string joined;
    for (const auto& i : config.issues) {
        if (i.level != IssueLevel::Error) continue;
        if (!joined.empty()) joined += "\n";
        joined += (i.node_name.empty() ? "" : i.node_name + ": ") + i.message;
    }
    config.error_message = joined;
    if (config.is_valid) config.rl = std::move(rl);
    return config;
}

}  // namespace cyxwiz
