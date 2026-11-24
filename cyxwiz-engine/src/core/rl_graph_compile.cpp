#include "rl_graph_compile.h"

#include "rl_graph_presentation.h"

#include <cyxwiz/error_codes.h>
#include <cyxwiz/neural_provider.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>

namespace cyxwiz {

namespace {

std::mutex g_lookup_mutex;
RLEnvironmentLookup g_lookup;

bool LookupEnvironment(const std::string& key, const std::map<std::string, std::string>& settings,
                       plugin::PluginEnvironmentInfo& info, std::string& error) {
    std::lock_guard<std::mutex> lock(g_lookup_mutex);
    return g_lookup && g_lookup(key, settings, info, error);
}

std::string Param(const gui::MLNode& node, const char* key, const char* fallback) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() || it->second.empty() ? fallback : it->second;
}

// Continuous actions with a finite low and high bound each (SAC and TD3 squash into them)
bool BoundedActions(const plugin::PluginEnvironmentInfo& info) {
    const auto size = static_cast<size_t>(std::max(info.action_size, 0));
    if (info.action_low.size() != size || info.action_high.size() != size) return false;
    for (size_t i = 0; i < size; ++i)
        if (!std::isfinite(info.action_low[i]) || !std::isfinite(info.action_high[i])) return false;
    return true;
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

std::pair<std::vector<int>, std::vector<int>> RLNetworkUnits(const RLTrainingConfiguration& c) {
    switch (c.algorithm) {
        case RLAlgorithm::SAC:
        case RLAlgorithm::TD3:
            return {HiddenUnits(c.policy_layers), HiddenUnits(c.q_layers)};
        case RLAlgorithm::DQN:
            return {HiddenUnits(c.q_layers), HiddenUnits(c.q_layers)};
        case RLAlgorithm::PPO:
            break;
    }
    return {HiddenUnits(c.policy_layers), HiddenUnits(c.value_layers)};
}

namespace {

// A network's chain as an MLP request (the hidden layers, then its output
// layer), or false when a layer has no fused form
bool MlpChain(const std::vector<CompiledLayer>& layers, size_t inputs, size_t outputs, size_t& count,
              size_t (&widths)[NeuralOpRequest::kMaxMlpLayers + 1],
              NeuralActivation (&acts)[NeuralOpRequest::kMaxMlpLayers], float& slope, float& alpha) {
    count = 0;
    bool has_slope = false, has_alpha = false;  // one of each per network, as FusedMlp takes them
    widths[0] = inputs;
    for (const auto& layer : layers) {
        if (layer.type == gui::NodeType::Dense) {
            if (count == NeuralOpRequest::kMaxMlpLayers) return false;
            widths[++count] = static_cast<size_t>(std::max(layer.units, 0));
            acts[count - 1] = NeuralActivation::None;
            continue;
        }
        if (count == 0 || acts[count - 1] != NeuralActivation::None) return false;  // an activation after a Dense
        switch (layer.type) {
            case gui::NodeType::ReLU: acts[count - 1] = NeuralActivation::Relu; break;
            case gui::NodeType::Tanh: acts[count - 1] = NeuralActivation::Tanh; break;
            case gui::NodeType::Sigmoid: acts[count - 1] = NeuralActivation::Sigmoid; break;
            case gui::NodeType::GELU: acts[count - 1] = NeuralActivation::Gelu; break;
            case gui::NodeType::Swish: acts[count - 1] = NeuralActivation::Swish; break;
            case gui::NodeType::SELU: acts[count - 1] = NeuralActivation::Selu; break;
            case gui::NodeType::ELU:
                if (has_alpha && layer.alpha != alpha) return false;
                acts[count - 1] = NeuralActivation::Elu;
                alpha = layer.alpha;
                has_alpha = true;
                break;
            case gui::NodeType::LeakyReLU:
                if (has_slope && layer.negative_slope != slope) return false;
                acts[count - 1] = NeuralActivation::LeakyRelu;
                slope = layer.negative_slope;
                has_slope = true;
                break;
            default: return false;
        }
    }
    if (count == NeuralOpRequest::kMaxMlpLayers) return false;
    widths[++count] = outputs;  // the head's output layer
    acts[count - 1] = NeuralActivation::None;
    return true;
}

}  // namespace

const char* RLAlgorithmName(RLAlgorithm algorithm) {
    switch (algorithm) {
        case RLAlgorithm::SAC:
            return "SAC";
        case RLAlgorithm::TD3:
            return "TD3";
        case RLAlgorithm::DQN:
            return "DQN";
        case RLAlgorithm::PPO:
            break;
    }
    return "PPO";
}

bool UsesTwinQ(RLAlgorithm algorithm) { return algorithm == RLAlgorithm::SAC || algorithm == RLAlgorithm::TD3; }

bool IsOffPolicy(RLAlgorithm algorithm) { return algorithm != RLAlgorithm::PPO; }

uint64_t RLSeed(const RLTrainingConfiguration& c) {
    switch (c.algorithm) {
        case RLAlgorithm::SAC:
            return c.sac.seed;
        case RLAlgorithm::TD3:
            return c.td3.seed;
        case RLAlgorithm::DQN:
            return c.dqn.seed;
        case RLAlgorithm::PPO:
            break;
    }
    return c.ppo.seed;
}

namespace {

// The minibatch of a gradient step (PPO's minibatch, the off-policy learners' batch)
int RLBatchSize(const RLTrainingConfiguration& c) {
    switch (c.algorithm) {
        case RLAlgorithm::SAC:
            return c.sac.batch_size;
        case RLAlgorithm::TD3:
            return c.td3.batch_size;
        case RLAlgorithm::DQN:
            return c.dqn.batch_size;
        case RLAlgorithm::PPO:
            break;
    }
    return c.ppo.batch_size;
}

}  // namespace

bool FusedNetworksExpected(const RLTrainingConfiguration& config, DeviceType platform, int device_id) {
    if (platform == DeviceType::CPU) return false;
    const bool sac = config.algorithm == RLAlgorithm::SAC;
    const size_t observations = std::max<size_t>(config.observation_size, 1);
    const size_t actions = std::max<size_t>(config.policy_outputs, 1);
    // PPO: the policy (logits or means) and the value network. SAC: the actor
    // (mean and log std per action); TD3: the actor (one action each); both
    // with the Q network on [observation | action] (its twin and the targets
    // are the same chain). DQN: the Q network, one value per choice (its
    // target the same chain).
    struct Network {
        const std::vector<CompiledLayer>* layers;
        size_t inputs, outputs;
    };
    std::vector<Network> networks;
    if (config.algorithm == RLAlgorithm::DQN) {
        networks.push_back({&config.q_layers, observations, actions});
    } else {
        networks.push_back({&config.policy_layers, observations, sac ? 2 * actions : actions});
        networks.push_back(UsesTwinQ(config.algorithm) ? Network{&config.q_layers, observations + actions, 1}
                                                       : Network{&config.value_layers, observations, 1});
    }
    const int batch = RLBatchSize(config);
    for (const auto& n : networks) {
        NeuralOpRequest r;
        r.op = NeuralOp::MlpForward;
        r.target.platform = platform;
        r.target.device_id = device_id;
        r.device_resident = true;
        r.dtype = DataType::Float32;
        r.batch = 1;
        if (!MlpChain(*n.layers, n.inputs, n.outputs, r.mlp_layers, r.mlp_widths, r.mlp_activations,
                      r.mlp_negative_slope, r.mlp_elu_alpha))
            return false;
        if (!NeuralProviderRegistry::Instance().FindSupporting(r)) return false;
        r.op = NeuralOp::MlpBackward;  // the minibatch's backward, with its row limit
        r.batch = static_cast<size_t>(std::max(batch, 1));
        if (!NeuralProviderRegistry::Instance().FindSupporting(r)) return false;
    }
    return true;
}

bool IsRLLearnerNode(gui::NodeType type) {
    return type == gui::NodeType::PPO || type == gui::NodeType::SAC || type == gui::NodeType::TD3 ||
           type == gui::NodeType::DQN;
}

bool IsRLGraphNode(gui::NodeType type) {
    return type == gui::NodeType::RLEnvironment || type == gui::NodeType::PolicyHead ||
           type == gui::NodeType::ValueHead || type == gui::NodeType::QHead || IsRLLearnerNode(type);
}

RLAlgorithm RLAlgorithmOf(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::SAC:
            return RLAlgorithm::SAC;
        case gui::NodeType::TD3:
            return RLAlgorithm::TD3;
        case gui::NodeType::DQN:
            return RLAlgorithm::DQN;
        default:
            return RLAlgorithm::PPO;
    }
}

const char* RLLearnerName(gui::NodeType type) { return RLAlgorithmName(RLAlgorithmOf(type)); }

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

    // ---- One learner (PPO, SAC, TD3 or DQN), and nothing of a supervised graph
    std::vector<const gui::MLNode*> learners;
    for (const auto& n : nodes) {
        if (IsRLLearnerNode(n.type))
            learners.push_back(&n);
        else if (n.type == gui::NodeType::DataInput || n.type == gui::NodeType::DatasetInput ||
                 std::any_of(n.outputs.begin(), n.outputs.end(), [](const gui::NodePin& pin) {
                     return pin.type == gui::PinType::Loss || pin.type == gui::PinType::Optimizer;
                 }))
            error(n.name + " belongs to a model trained on data; a graph trains either that or a policy on an "
                           "environment, not both",
                  &n, errors::Compiler::UnsupportedTrainingNode);
    }
    if (learners.size() != 1) {
        error(learners.empty() ? "A reinforcement learning graph needs a learner: a PPO node (connect its Policy, "
                                 "Environment and Value), a SAC or TD3 node (Policy, Environment and Q) or a DQN "
                                 "node (Q and Environment)"
                               : "A graph trains one policy: keep one PPO, SAC, TD3 or DQN node",
              learners.empty() ? nullptr : learners[1], errors::Compiler::MissingTrainingPathNode);
    }
    const gui::MLNode* learner = learners.empty() ? nullptr : learners[0];
    rl->algorithm = learner ? RLAlgorithmOf(learner->type) : RLAlgorithm::PPO;
    const bool sac = rl->algorithm == RLAlgorithm::SAC;
    const bool td3 = rl->algorithm == RLAlgorithm::TD3;
    const bool dqn = rl->algorithm == RLAlgorithm::DQN;
    const bool twin_q = UsesTwinQ(rl->algorithm);
    const std::string learner_name = RLAlgorithmName(rl->algorithm);

    // ---- The learner's inputs: Policy, Environment, then Value (PPO) or Q (SAC,
    // TD3); DQN's Q, then Environment
    const gui::MLNode* env = nullptr;
    const gui::MLNode* policy_head = nullptr;
    const gui::MLNode* value_head = nullptr;
    const gui::MLNode* q_head = nullptr;
    if (learner) {
        const auto from = [&](size_t input, gui::NodeType want, const char* what) -> const gui::MLNode* {
            const gui::NodeLink* l = LinkInto(links, *learner, input);
            const gui::MLNode* n = l ? FindNodeById(l->from_node, nodes) : nullptr;
            if (!n || n->type != want) {
                error(std::string("Connect ") + what + " to " + learner_name + "'s " +
                          (input < learner->inputs.size() ? learner->inputs[input].name : std::string("inputs")),
                      learner, errors::Compiler::MissingTrainingPathNode);
                return nullptr;
            }
            return n;
        };
        if (dqn) {
            q_head = from(0, gui::NodeType::QHead, "a Q Head");
            env = from(1, gui::NodeType::RLEnvironment, "an Environment's Environment pin");
        } else {
            policy_head = from(0, gui::NodeType::PolicyHead, "a Policy Head");
            env = from(1, gui::NodeType::RLEnvironment, "an Environment's Environment pin");
            if (twin_q) q_head = from(2, gui::NodeType::QHead, "a Q Head");
            else value_head = from(2, gui::NodeType::ValueHead, "a Value Head");
        }
        const gui::NodeLink* env_link = LinkInto(links, *learner, 1);
        if (env && env_link && env->outputs.size() > 1 && env_link->from_pin != env->outputs[1].id) {
            error(learner_name + "'s Environment takes the Environment's Environment pin, not its Observation",
                  learner);
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
        std::string settings_error;
        const bool found =
            LookupEnvironment(rl->environment_key, rl->environment_settings, rl->environment, settings_error);
        if (!found && !settings_error.empty()) {  // known, but its settings are not valid
            error(settings_error, env, errors::Compiler::InvalidParameter);
            env = nullptr;
        } else if (!found) {
            const size_t colon = rl->environment_key.find(':');
            const std::string id = colon == std::string::npos ? rl->environment_key : rl->environment_key.substr(colon + 1);
            const std::string provider = colon == std::string::npos ? "" : rl->environment_key.substr(0, colon);
            error(rl->environment_key.empty()
                      ? std::string("Pick an environment")
                      : id + " comes from the plugin " + provider + ", which is not loaded: load it in Plugin Manager",
                  env, errors::Compiler::MissingTrainingPathNode);
            env = nullptr;
        } else if (twin_q && rl->environment.discrete_actions) {
            error(learner_name + " learns continuous actions, and " + rl->environment.id +
                      " has a choice of actions: train it with PPO or DQN",
                  learner, errors::Compiler::UnsupportedTrainingNode);
            env = nullptr;
        } else if (twin_q && !BoundedActions(rl->environment)) {
            error(learner_name + " needs bounded actions, and " + rl->environment.id +
                      " has actions without finite bounds: give its actuators a control range, or train it with PPO",
                  learner, errors::Compiler::UnsupportedTrainingNode);
            env = nullptr;
        } else if (dqn && !rl->environment.discrete_actions) {
            error("DQN chooses among actions, and " + rl->environment.id +
                      " has continuous actions: train it with SAC, TD3 or PPO",
                  learner, errors::Compiler::UnsupportedTrainingNode);
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
                    error(head->name + " is reached from another Environment than the one " + learner_name +
                              " trains in",
                          head);
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
                error(prev->name + " cannot be in a policy, value or Q network yet: use Dense, activations, Layer "
                                   "Norm or Dropout",
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
    if (q_head) rl->q_head_node_id = q_head->id;
    chain(policy_head, rl->policy_layers);
    chain(value_head, rl->value_layers);
    chain(q_head, rl->q_layers);
    if (q_head && twin_q && !rl->q_layers.empty() && rl->q_layers.front().type != gui::NodeType::Dense)
        error("Start the Q layers with a Dense: the action joins the observation at its input", q_head);

    // ---- The learner's settings
    const auto number = [&](const char* key, const char* fallback, double& out) {
        const std::string text = Param(*learner, key, fallback);
        char* end = nullptr;
        out = std::strtod(text.c_str(), &end);
        if (!end || *end != '\0' || !std::isfinite(out)) {
            error(learner_name + "'s '" + std::string(key) + "' is not a number: " + text, learner,
                  errors::Compiler::InvalidParameter);
            return false;
        }
        return true;
    };
    // Counts clamped before the cast (a typed 1e30 is not undefined behaviour);
    // CheckSetup refuses what is still out of range
    const auto count = [](double x, double most) { return std::clamp(x, -1.0, most); };
    double v = 0.0;
    if (learner) rl->learner_node_id = learner->id;
    // ---- What SAC, TD3 and DQN share (the loop's settings), each learner
    // with its own defaults; every branch reads and checks in its node's order
    struct OffPolicyDefaults {
        const char* batch_size;
        const char* learning_rate;
        const char* tau;
        const char* train_freq;
    };
    const auto read_loop_settings = [&](auto& c, const OffPolicyDefaults& d) {
        if (number("total_timesteps", "1000000", v)) c.total_timesteps = static_cast<int64_t>(count(v, 1e15));
        if (number("buffer_size", "1000000", v)) c.buffer_size = static_cast<int64_t>(count(v, 1e15));
        if (number("learning_starts", "100", v)) c.learning_starts = static_cast<int64_t>(count(v, 1e15));
        if (number("batch_size", d.batch_size, v)) c.batch_size = static_cast<int>(count(v, 1e9));
        if (number("learning_rate", d.learning_rate, v)) c.learning_rate = static_cast<float>(v);
        if (number("gamma", "0.99", v)) c.gamma = static_cast<float>(v);
        if (number("tau", d.tau, v)) c.tau = static_cast<float>(v);
        if (number("train_freq", d.train_freq, v)) c.train_freq = static_cast<int>(count(v, 1e9));
        if (number("gradient_steps", "1", v)) c.gradient_steps = static_cast<int>(count(v, 1e9));
    };
    const auto read_seed = [&](auto& c) {
        if (number("seed", "1", v)) c.seed = static_cast<uint64_t>(count(std::max(v, 0.0), 2147483647.0));
    };
    // The observation statistics and the early stop
    const auto read_stop = [&](auto& c, const char* target_return) {
        c.normalize_observations = Param(*learner, "normalize_observations", "false") == "true";
        if (Param(*learner, "stop_at_return", "false") == "true" && number("target_return", target_return, v))
            c.target_return = static_cast<float>(v);
    };
    // own_count: the learner's own count ("target every", "policy every") and whether it is at least 1
    const auto check_counts = [&](const auto& c, bool own_count_ok, const char* own_count) {
        if (c.total_timesteps < 1 || c.buffer_size < 1 || c.learning_starts < 0 || c.batch_size < 1 ||
            c.train_freq < 1 || c.gradient_steps < 1 || !own_count_ok)
            error(std::string("Total steps, buffer size, batch size, train every, gradient steps and ") + own_count +
                      " must each be at least 1, and the random steps first at least 0",
                  learner, errors::Compiler::InvalidParameter);
    };
    const auto check_discount = [&](const auto& c) {
        if (!(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.tau > 0.0f && c.tau <= 1.0f))
            error("The discount and the target update must each be in (0, 1]", learner,
                  errors::Compiler::InvalidParameter);
    };
    const auto check_batch = [&](const auto& c) {
        if (c.batch_size >= 1 && c.buffer_size >= 1 && c.batch_size > c.buffer_size)
            error("Batch size " + std::to_string(c.batch_size) + " is larger than the replay buffer (" +
                      std::to_string(c.buffer_size) + " steps). Lower it, or raise Buffer size",
                  learner, errors::Compiler::InvalidParameter);
    };
    if (learner && sac) {
        auto& c = rl->sac;
        read_loop_settings(c, {"256", "0.0003", "0.005", "1"});
        if (number("target_update_interval", "1", v)) c.target_update_interval = static_cast<int>(count(v, 1e9));
        if (number("ent_coef", "1.0", v)) c.ent_coef = static_cast<float>(v);
        read_seed(c);
        c.ent_coef_auto = Param(*learner, "ent_coef_mode", "learned") != "fixed";
        c.target_entropy_auto = Param(*learner, "target_entropy_mode", "auto") != "typed";
        if (!c.target_entropy_auto && number("target_entropy", "-1", v)) c.target_entropy = static_cast<float>(v);
        if (c.target_entropy_auto) c.target_entropy = -static_cast<float>(rl->policy_outputs);
        read_stop(c, "4000");
        check_counts(c, c.target_update_interval >= 1, "target every");
        if (!(c.learning_rate > 0.0f))
            error("The learning rate must be above 0", learner, errors::Compiler::InvalidParameter);
        check_discount(c);
        if (!(c.ent_coef > 0.0f))
            error("The entropy weight must be above 0", learner, errors::Compiler::InvalidParameter);
        check_batch(c);
    }
    if (learner && td3) {
        auto& c = rl->td3;
        read_loop_settings(c, {"256", "0.001", "0.005", "1"});
        if (number("policy_delay", "2", v)) c.policy_delay = static_cast<int>(count(v, 1e9));
        if (number("action_noise_std", "0.1", v)) c.action_noise_std = static_cast<float>(v);
        if (number("target_policy_noise", "0.2", v)) c.target_policy_noise = static_cast<float>(v);
        if (number("target_noise_clip", "0.5", v)) c.target_noise_clip = static_cast<float>(v);
        read_seed(c);
        c.action_noise = Param(*learner, "action_noise", "normal") != "none";
        read_stop(c, "4000");
        check_counts(c, c.policy_delay >= 1, "policy every");
        if (!(c.learning_rate > 0.0f))
            error("The learning rate must be above 0", learner, errors::Compiler::InvalidParameter);
        check_discount(c);
        if ((c.action_noise && !(c.action_noise_std > 0.0f)) || !(c.target_policy_noise >= 0.0f) ||
            !(c.target_noise_clip >= 0.0f))
            error("The noise std must be above 0, the target smoothing noise and its clip at least 0", learner,
                  errors::Compiler::InvalidParameter);
        check_batch(c);
    }
    if (learner && dqn) {
        auto& c = rl->dqn;
        read_loop_settings(c, {"32", "0.0001", "1.0", "4"});
        if (number("target_update_interval", "10000", v))
            c.target_update_interval = static_cast<int64_t>(count(v, 1e15));
        if (number("exploration_fraction", "0.1", v)) c.exploration_fraction = static_cast<float>(v);
        if (number("exploration_initial_eps", "1.0", v)) c.exploration_initial_eps = static_cast<float>(v);
        if (number("exploration_final_eps", "0.05", v)) c.exploration_final_eps = static_cast<float>(v);
        if (number("max_grad_norm", "10", v)) c.max_grad_norm = static_cast<float>(v);
        read_seed(c);
        read_stop(c, "475");
        check_counts(c, c.target_update_interval >= 1, "target every");
        if (!(c.learning_rate > 0.0f) || !(c.max_grad_norm > 0.0f))
            error("The learning rate and the largest gradient norm must each be above 0", learner,
                  errors::Compiler::InvalidParameter);
        check_discount(c);
        const auto share = [](float x) { return x >= 0.0f && x <= 1.0f; };
        if (!share(c.exploration_fraction) || !share(c.exploration_initial_eps) || !share(c.exploration_final_eps))
            error("The exploration's start, end and share of the run must each be from 0 to 1", learner,
                  errors::Compiler::InvalidParameter);
        check_batch(c);
    }
    if (learner && rl->algorithm == RLAlgorithm::PPO) {
        const gui::MLNode* ppo = learner;
        auto& c = rl->ppo;
        if (number("total_timesteps", "100000", v)) c.total_timesteps = static_cast<int64_t>(count(v, 1e15));
        if (number("n_steps", "2048", v)) c.n_steps = static_cast<int>(count(v, 1e9));
        if (number("batch_size", "64", v)) c.batch_size = static_cast<int>(count(v, 1e9));
        if (number("n_epochs", "10", v)) c.n_epochs = static_cast<int>(count(v, 1e9));
        if (number("learning_rate", "0.0003", v)) c.learning_rate = static_cast<float>(v);
        if (number("gamma", "0.99", v)) c.gamma = static_cast<float>(v);
        if (number("gae_lambda", "0.95", v)) c.gae_lambda = static_cast<float>(v);
        if (number("max_grad_norm", "0.5", v)) c.max_grad_norm = static_cast<float>(v);
        if (number("clip_range", "0.2", v)) c.loss.clip_range = static_cast<float>(v);
        if (number("clip_range_vf", "0", v)) c.loss.clip_range_vf = static_cast<float>(v);
        if (number("ent_coef", "0", v)) c.loss.ent_coef = static_cast<float>(v);
        if (number("vf_coef", "0.5", v)) c.loss.vf_coef = static_cast<float>(v);
        if (number("seed", "1", v)) c.seed = static_cast<uint64_t>(count(std::max(v, 0.0), 2147483647.0));
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
            n.id != rl->q_head_node_id && n.id != rl->learner_node_id)
            issue(IssueLevel::Warning, n.name + " is not connected to the " + learner_name + " that trains", &n,
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
