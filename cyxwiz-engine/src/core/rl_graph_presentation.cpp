#include "rl_graph_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::rl_graph {

namespace {

std::string Number(double value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", value);
    return buf;
}

std::string Plural(int n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

}  // namespace

std::vector<EnvironmentChoice> EnvironmentChoices(const std::vector<ListedEnvironment>& environments,
                                                  const std::string& current_key) {
    std::vector<EnvironmentChoice> out;
    bool current_listed = current_key.empty();
    for (const auto& e : environments) {
        out.push_back({e.key, e.info.id + " · " + e.provider_name});
        current_listed = current_listed || e.key == current_key;
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.label < b.label; });
    if (!current_listed) out.insert(out.begin(), {current_key, current_key + " · not loaded"});
    return out;
}

std::vector<Fact> EnvironmentFacts(const plugin::PluginEnvironmentInfo& info) {
    std::vector<Fact> facts;
    facts.push_back({"Observation", Plural(info.observation_size, "value", "values")});
    if (info.discrete_actions) {
        facts.push_back({"Actions", Plural(info.action_count, "choice", "choices")});
    } else {
        std::string actions = Plural(info.action_size, "value", "values");
        const bool same = !info.action_low.empty() &&
                          std::all_of(info.action_low.begin(), info.action_low.end(),
                                      [&](float v) { return v == info.action_low[0]; }) &&
                          std::all_of(info.action_high.begin(), info.action_high.end(),
                                      [&](float v) { return v == info.action_high[0]; });
        if (same) actions += " in [" + Number(info.action_low[0]) + ", " + Number(info.action_high[0]) + "]";
        facts.push_back({"Actions", actions});
    }
    facts.push_back({"Episode limit", info.max_episode_steps > 0 ? Plural(info.max_episode_steps, "step", "steps")
                                                                 : std::string("none")});
    if (info.step_seconds > 0.0) facts.push_back({"One step", Number(info.step_seconds) + " s"});
    return facts;
}

std::vector<Fact> PolicyHeadFacts(const plugin::PluginEnvironmentInfo& info, size_t features, gui::NodeType learner) {
    const std::string from = features > 0 ? std::to_string(features) + " -> " : std::string();
    if (learner == gui::NodeType::TD3 && !info.discrete_actions)  // SB3 TD3's actor: one layer, tanh
        return {{"Action", "deterministic: tanh of the layer, " + Plural(info.action_size, "action", "actions") +
                               ", scaled to the action bounds"},
                {"Output layer", from + Plural(info.action_size, "action", "actions") + " (added)"},
                {"Target copy", "kept by TD3 for the Q targets (Polyak, Target update)"},
                {"Exploration", "the TD3 node's noise (no log std)"}};
    if (learner == gui::NodeType::SAC && !info.discrete_actions)  // SB3 SAC's actor: mean and log std layers, log std clipped to [-20, 2]
        return {{"Distribution", "Gaussian squashed by tanh, " + Plural(info.action_size, "action", "actions")},
                {"Output layers", from + Plural(info.action_size, "mean", "means") + " and " + from +
                                      Plural(info.action_size, "log std", "log stds") + " (added)"},
                {"Log std", "from the state, clipped to [-20, 2]"}};
    if (info.discrete_actions)
        return {{"Distribution", "Categorical, " + Plural(info.action_count, "choice", "choices")},
                {"Output layer", from + Plural(info.action_count, "logit", "logits") + " (added)"},
                {"Output init gain", "0.01"}};
    return {{"Distribution", "Gaussian, " + Plural(info.action_size, "action", "actions")},
            {"Output layer", from + Plural(info.action_size, "mean", "means") + " (added)"},
            {"Output init gain", "0.01"}};
}

std::vector<Fact> ValueHeadFacts(size_t features) {
    return {{"Output layer", (features > 0 ? std::to_string(features) + " -> " : std::string()) + "1 (added)"},
            {"Output init gain", "1"}};
}

std::vector<Fact> QHeadFacts(size_t features, size_t actions, gui::NodeType learner) {
    std::vector<Fact> facts;
    facts.push_back({"Output layer", (features > 0 ? std::to_string(features) + " -> " : std::string()) + "1 (added)"});
    facts.push_back({"Action", actions > 0 ? "joins the first Q layer's input (" + std::to_string(actions) +
                                                 (actions == 1 ? " value)" : " values)")
                                           : std::string("joins the first Q layer's input")});
    const std::string by = learner == gui::NodeType::TD3 ? "TD3" : "SAC";
    facts.push_back({"Networks", "2 Q networks and a target copy of each (built by " + by + ")"});
    if (learner == gui::NodeType::TD3)
        facts.push_back({"Policy", "learns from Q network 1; both learn toward the smaller target"});
    return facts;
}

std::string ExplorationNoiseText(bool gaussian, const std::string& std_text) {
    if (!gaussian) return "None: the policy's own actions after the random steps";
    return "Gaussian noise, std " + std_text + " of the action range, after the random steps";
}

std::string ReplayMemoryText(uint64_t bytes) {
    const double mb = static_cast<double>(bytes) / 1e6;  // decimal, as device memory is sold
    if (mb >= 1000.0) return "about " + Number(std::round(mb / 100.0) / 10.0) + " GB";
    return "about " + Number(std::max(1.0, std::round(mb))) + " MB";
}

ReplayFit ReplayFitText(uint64_t bytes, const std::string& device_name, uint64_t device_bytes) {
    ReplayFit fit;
    if (device_bytes == 0) {
        fit.text = ReplayMemoryText(bytes) + " on the training device";
        return fit;
    }
    // The device's size in the same words, without "about"
    const std::string size = ReplayMemoryText(device_bytes).substr(6);
    const std::string device = (device_name.empty() ? std::string("the training device") : "the " + device_name) +
                               "'s " + size;
    const double share = static_cast<double>(bytes) / static_cast<double>(device_bytes);
    if (share > 1.0) {
        fit.text = ReplayMemoryText(bytes) + ", more than " + device;
        fit.advice = "The replay buffer does not fit the training device: lower Buffer size.";
        return fit;
    }
    char percent[16];
    std::snprintf(percent, sizeof(percent), "%.0f %%", std::max(1.0, std::round(100.0 * share)));
    fit.text = ReplayMemoryText(bytes) + ", " + percent + " of " + device;
    if (share > 0.75)
        fit.advice = "The networks and their updates need room on the device too: a smaller Buffer size is safer.";
    return fit;
}

std::string AutoTargetEntropyText(size_t actions) {
    return "-" + std::to_string(actions) + " (minus the " + Plural(static_cast<int>(actions), "action", "actions") + ")";
}

std::map<std::string, std::string> StoredSettings(const std::map<std::string, std::string>& parameters) {
    std::map<std::string, std::string> out;
    const std::string prefix = kSettingPrefix;
    for (const auto& [key, value] : parameters)
        if (key.rfind(prefix, 0) == 0 && key.size() > prefix.size() && !value.empty())
            out[key.substr(prefix.size())] = value;
    return out;
}

namespace {

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

std::vector<int> Inputs(const std::vector<gui::NodeLink>& links, int node_id) {
    std::vector<int> from;
    for (const auto& l : links)
        if (l.to_node == node_id) from.push_back(l.from_node);
    return from;
}

}  // namespace

const gui::MLNode* UpstreamEnvironment(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                                       int node_id) {
    std::vector<int> stack = Inputs(links, node_id), seen;
    while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
        seen.push_back(id);
        const gui::MLNode* n = FindNode(nodes, id);
        if (!n) continue;
        if (n->type == gui::NodeType::RLEnvironment) return n;
        for (const int from : Inputs(links, id)) stack.push_back(from);
    }
    return nullptr;
}

size_t UpstreamUnits(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links, int node_id) {
    std::vector<int> from = Inputs(links, node_id);
    for (int hops = 0; from.size() == 1 && hops < 64; ++hops) {
        const gui::MLNode* n = FindNode(nodes, from[0]);
        if (!n) return 0;
        if (n->type == gui::NodeType::Dense) {
            const auto it = n->parameters.find("units");
            try {
                return it == n->parameters.end() ? 64 : static_cast<size_t>(std::stoul(it->second));
            } catch (...) {
                return 0;
            }
        }
        if (n->type == gui::NodeType::RLEnvironment) return 0;
        from = Inputs(links, n->id);
    }
    return 0;
}

gui::NodeType LearnerFed(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links, int node_id) {
    for (const auto& l : links) {
        if (l.from_node != node_id) continue;
        const gui::MLNode* to = FindNode(nodes, l.to_node);
        if (to && (to->type == gui::NodeType::PPO || to->type == gui::NodeType::SAC || to->type == gui::NodeType::TD3))
            return to->type;
    }
    return gui::NodeType::Unknown;
}

std::optional<TunedPreset> TunedPpoSettings(const std::string& environment_id) {
    TunedPreset p;
    if (environment_id == "CartPole-v1") {
        // RL Zoo ppo.yml CartPole-v1 (its linear learning-rate and clip schedules as constants)
        p.source = "RL Zoo (rl-baselines3-zoo) PPO settings for CartPole-v1";
        p.environment = {{"copies", "8"}};
        p.learner = {{"total_timesteps", "100000"}, {"n_steps", "32"},        {"batch_size", "256"},
                 {"n_epochs", "20"},           {"learning_rate", "0.001"}, {"gamma", "0.98"},
                 {"gae_lambda", "0.8"},        {"clip_range", "0.2"},     {"ent_coef", "0"},
                 {"vf_coef", "0.5"},           {"max_grad_norm", "0.5"},  {"stop_at_return", "true"},
                 {"target_return", "475"},     {"normalize_observations", "false"},
                 {"normalize_rewards", "false"}};
        p.policy_head = {{"log_std_init", "0"}};
        return p;
    }
    if (environment_id == "InvertedPendulum-v5") {
        p.source = "the Engine's tested InvertedPendulum settings (Stable-Baselines3 defaults otherwise)";
        p.environment = {{"copies", "4"}};
        p.learner = {{"total_timesteps", "300000"}, {"n_steps", "512"},        {"batch_size", "64"},
                 {"n_epochs", "10"},           {"learning_rate", "0.0003"}, {"gamma", "0.999"},
                 {"gae_lambda", "0.9"},        {"clip_range", "0.2"},      {"ent_coef", "0"},
                 {"vf_coef", "0.5"},           {"max_grad_norm", "0.5"},   {"stop_at_return", "true"},
                 {"target_return", "950"},     {"normalize_observations", "false"},
                 {"normalize_rewards", "false"}};
        p.policy_head = {{"log_std_init", "0"}};
        return p;
    }
    if (environment_id == "HalfCheetah-v5") {
        // RL Zoo ppo.yml HalfCheetah-v4: normalize: true, n_envs 1, policy log_std_init -2
        p.source = "RL Zoo (rl-baselines3-zoo) PPO settings for HalfCheetah";
        p.environment = {{"copies", "1"}};
        p.learner = {{"total_timesteps", "1000000"}, {"n_steps", "512"},           {"batch_size", "64"},
                 {"n_epochs", "20"},            {"learning_rate", "2.0633e-05"}, {"gamma", "0.98"},
                 {"gae_lambda", "0.92"},        {"clip_range", "0.1"},          {"ent_coef", "0.000401762"},
                 {"vf_coef", "0.58096"},        {"max_grad_norm", "0.8"},       {"stop_at_return", "false"},
                 {"target_return", "4800"},     {"normalize_observations", "true"},
                 {"normalize_rewards", "true"}};
        p.policy_head = {{"log_std_init", "-2"}};
        p.layers_note = "RL Zoo trains it with two layers of 256 and ReLU; the preset does not change your layers.";
        return p;
    }
    return std::nullopt;
}

std::optional<TunedPreset> TunedSacSettings(const std::string& environment_id) {
    TunedPreset p;
    if (environment_id == "HalfCheetah-v5") {
        // RL Zoo sac.yml HalfCheetah-v4: n_timesteps 1e6, learning_starts 10000, SB3 defaults otherwise
        p.source = "RL Zoo (rl-baselines3-zoo) SAC settings for HalfCheetah";
        p.environment = {{"copies", "1"}};
        p.learner = {{"total_timesteps", "1000000"}, {"buffer_size", "1000000"}, {"learning_starts", "10000"},
                     {"batch_size", "256"},          {"learning_rate", "0.0003"}, {"gamma", "0.99"},
                     {"tau", "0.005"},               {"train_freq", "1"},         {"gradient_steps", "1"},
                     {"target_update_interval", "1"}, {"ent_coef_mode", "learned"}, {"ent_coef", "1.0"},
                     {"target_entropy_mode", "auto"}, {"stop_at_return", "false"}, {"normalize_observations", "false"}};
        p.layers_note = "RL Zoo trains it with SAC's default two layers of 256 and ReLU in the policy and the Q "
                        "networks; the preset does not change your layers.";
        return p;
    }
    return std::nullopt;
}

std::optional<TunedPreset> TunedTd3Settings(const std::string& environment_id) {
    TunedPreset p;
    if (environment_id == "HalfCheetah-v5") {
        // RL Zoo td3.yml HalfCheetah-v4 (its mujoco defaults): n_timesteps 1e6, learning_starts 10000, noise
        // normal 0.1, learning_rate 1e-3, batch_size 256, train_freq 1, gradient_steps 1, SB3 defaults otherwise
        p.source = "RL Zoo (rl-baselines3-zoo) TD3 settings for HalfCheetah";
        p.environment = {{"copies", "1"}};
        p.learner = {{"total_timesteps", "1000000"}, {"buffer_size", "1000000"}, {"learning_starts", "10000"},
                     {"batch_size", "256"},          {"learning_rate", "0.001"},  {"gamma", "0.99"},
                     {"tau", "0.005"},               {"train_freq", "1"},         {"gradient_steps", "1"},
                     {"policy_delay", "2"},          {"action_noise", "normal"},  {"action_noise_std", "0.1"},
                     {"target_policy_noise", "0.2"}, {"target_noise_clip", "0.5"}, {"stop_at_return", "false"},
                     {"normalize_observations", "false"}};
        p.layers_note = "RL Zoo trains it with TD3's default layers of 400 and 300 and ReLU in the policy and the Q "
                        "networks; the preset does not change your layers.";
        return p;
    }
    return std::nullopt;
}

}  // namespace cyxwiz::rl_graph
