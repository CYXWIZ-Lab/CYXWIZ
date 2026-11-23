#include "rl_graph_presentation.h"

#include <algorithm>
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

std::vector<Fact> PolicyHeadFacts(const plugin::PluginEnvironmentInfo& info, size_t features) {
    const std::string from = features > 0 ? std::to_string(features) + " -> " : std::string();
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

std::optional<PpoPreset> TunedPpoSettings(const std::string& environment_id) {
    PpoPreset p;
    if (environment_id == "CartPole-v1") {
        // RL Zoo ppo.yml CartPole-v1 (its linear learning-rate and clip schedules as constants)
        p.source = "RL Zoo (rl-baselines3-zoo) PPO settings for CartPole-v1";
        p.environment = {{"copies", "8"}};
        p.ppo = {{"total_timesteps", "100000"}, {"n_steps", "32"},        {"batch_size", "256"},
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
        p.ppo = {{"total_timesteps", "300000"}, {"n_steps", "512"},        {"batch_size", "64"},
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
        p.ppo = {{"total_timesteps", "1000000"}, {"n_steps", "512"},           {"batch_size", "64"},
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

}  // namespace cyxwiz::rl_graph
