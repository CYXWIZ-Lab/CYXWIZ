#include "rl_reference.h"

#include "../rl_graph_compile.h"
#include "../user_folders.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

namespace cyxwiz::rl {

namespace {

// SB3's default for a key the recorded settings leave out
double Setting(const nlohmann::json& settings, const char* key, double fallback) {
    return settings.contains(key) ? settings.at(key).get<double>() : fallback;
}

// the run's settings are floats
bool Same(double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(std::abs(a), std::abs(b)); }

// A recorded net_arch: a list of widths, or one width for two layers (the PPO
// generator's "policy"); the fallback when the settings leave it out
std::vector<int> Layers(const nlohmann::json& settings, const char* key, std::vector<int> fallback) {
    if (!settings.is_object() || !settings.contains(key)) return fallback;
    const auto& v = settings.at(key);
    if (v.is_number_integer()) return {v.get<int>(), v.get<int>()};
    return v.get<std::vector<int>>();
}

// The first recorded task of this environment and copy count whose settings
// match, as a reference; layers(task): the hidden widths it was recorded with
std::optional<RLReference> Find(const std::string& json_text, const std::string& environment_id, int copies,
                                const std::function<bool(const nlohmann::json& task)>& matches,
                                const std::function<std::vector<int>(const nlohmann::json& task)>& layers) {
    if (json_text.empty()) return std::nullopt;
    try {
        const auto doc = nlohmann::json::parse(json_text);
        const auto& columns = doc.at("columns");
        const auto column = [&](const char* name) {
            for (size_t i = 0; i < columns.size(); ++i)
                if (columns[i].get<std::string>() == name) return static_cast<int>(i);
            return -1;
        };
        const int steps_col = column("time/total_timesteps");
        const int return_col = column("rollout/ep_rew_mean");
        if (steps_col < 0 || return_col < 0) return std::nullopt;
        for (const auto& task : doc.at("tasks")) {
            if (task.at("id").get<std::string>() != environment_id || task.at("copies").get<int>() != copies) continue;
            if (!matches(task)) continue;
            RLReference ref;
            ref.layers = layers(task);
            for (const auto& run : task.at("runs")) {
                std::vector<std::pair<double, double>> curve;
                for (const auto& row : run.at("updates")) {
                    const auto& r = row.at(return_col);
                    if (!r.is_number() || !std::isfinite(r.get<double>())) continue;
                    curve.emplace_back(row.at(steps_col).get<double>(), r.get<double>());
                }
                if (!curve.empty()) ref.seeds.push_back(std::move(curve));
            }
            if (ref.seeds.empty()) continue;
            std::string version = "?";
            if (doc.contains("versions") && doc.at("versions").contains("stable_baselines3"))
                version = doc.at("versions").at("stable_baselines3").get<std::string>();
            ref.label = "SB3 " + version + ", " + std::to_string(ref.seeds.size()) +
                        (ref.seeds.size() == 1 ? " seed" : " seeds");
            return ref;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace

std::optional<double> RLReference::At(double steps) const {
    if (seeds.empty()) return std::nullopt;
    double sum = 0.0;
    for (const auto& s : seeds) {
        if (s.empty() || steps < s.front().first || steps > s.back().first) return std::nullopt;
        const auto hi = std::lower_bound(s.begin(), s.end(), steps,
                                         [](const auto& p, double x) { return p.first < x; });
        if (hi->first == steps || hi == s.begin()) {
            sum += hi->second;
            continue;
        }
        const auto lo = hi - 1;
        const double f = (steps - lo->first) / (hi->first - lo->first);
        sum += lo->second + f * (hi->second - lo->second);
    }
    return sum / static_cast<double>(seeds.size());
}

std::optional<RLReference> FindPpoReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const PpoConfig& c) {
    return Find(json_text, environment_id, copies, [&c](const nlohmann::json& task) {
        const auto& p = task.at("ppo");
        const bool same = Same(Setting(p, "n_steps", 2048), c.n_steps) &&
                          Same(Setting(p, "batch_size", 64), c.batch_size) &&
                          Same(Setting(p, "n_epochs", 10), c.n_epochs) &&
                          Same(Setting(p, "learning_rate", 3e-4), c.learning_rate) &&
                          Same(Setting(p, "gamma", 0.99), c.gamma) &&
                          Same(Setting(p, "gae_lambda", 0.95), c.gae_lambda) &&
                          Same(Setting(p, "clip_range", 0.2), c.loss.clip_range) &&
                          Same(Setting(p, "ent_coef", 0.0), c.loss.ent_coef) &&
                          Same(Setting(p, "vf_coef", 0.5), c.loss.vf_coef) &&
                          Same(Setting(p, "max_grad_norm", 0.5), c.max_grad_norm);
        // The reference's VecNormalize (observations and rewards together), as the run's
        const bool normalize = task.value("normalize", false);
        return same && c.normalize_observations == normalize && c.normalize_rewards == normalize;
    }, [](const nlohmann::json& task) {  // SB3's MlpPolicy: 64-64; RL Zoo's HalfCheetah: "policy" 256
        return Layers(task.value("policy", nlohmann::json::object()), "net_arch", {64, 64});
    });
}

std::optional<RLReference> FindSacReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const SacConfig& c) {
    return Find(json_text, environment_id, copies, [&c](const nlohmann::json& task) {
        const auto& s = task.at("sac");
        // The recorded runs keep SB3's entropy weight learned from 1 with its
        // "auto" target entropy, and normalise nothing
        if (s.contains("ent_coef") || s.contains("target_entropy") || task.value("normalize", false)) return false;
        return Same(Setting(s, "learning_rate", 3e-4), c.learning_rate) &&
               Same(Setting(s, "buffer_size", 1e6), static_cast<double>(c.buffer_size)) &&
               Same(Setting(s, "learning_starts", 100), static_cast<double>(c.learning_starts)) &&
               Same(Setting(s, "batch_size", 256), c.batch_size) && Same(Setting(s, "tau", 0.005), c.tau) &&
               Same(Setting(s, "gamma", 0.99), c.gamma) && Same(Setting(s, "train_freq", 1), c.train_freq) &&
               Same(Setting(s, "gradient_steps", 1), c.gradient_steps) &&
               Same(Setting(s, "target_update_interval", 1), c.target_update_interval) && c.ent_coef_auto &&
               Same(c.ent_coef, 1.0) && c.target_entropy_auto && !c.normalize_observations;
    }, [](const nlohmann::json& task) { return Layers(task.at("sac"), "net_arch", {256, 256}); });  // SB3 SAC's
}

std::optional<RLReference> FindTd3Reference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const Td3Config& c) {
    return Find(json_text, environment_id, copies, [&c](const nlohmann::json& task) {
        const auto& s = task.at("td3");
        // The recorded runs explore with Gaussian noise of "noise_std" (none
        // when absent) and normalise nothing
        if (task.value("normalize", false) || c.normalize_observations) return false;
        const bool noise = s.contains("noise_std");
        if (noise != c.action_noise || (noise && !Same(s.at("noise_std").get<double>(), c.action_noise_std)))
            return false;
        return Same(Setting(s, "learning_rate", 1e-3), c.learning_rate) &&
               Same(Setting(s, "buffer_size", 1e6), static_cast<double>(c.buffer_size)) &&
               Same(Setting(s, "learning_starts", 100), static_cast<double>(c.learning_starts)) &&
               Same(Setting(s, "batch_size", 256), c.batch_size) && Same(Setting(s, "tau", 0.005), c.tau) &&
               Same(Setting(s, "gamma", 0.99), c.gamma) && Same(Setting(s, "train_freq", 1), c.train_freq) &&
               Same(Setting(s, "gradient_steps", 1), c.gradient_steps) &&
               Same(Setting(s, "policy_delay", 2), c.policy_delay) &&
               Same(Setting(s, "target_policy_noise", 0.2), c.target_policy_noise) &&
               Same(Setting(s, "target_noise_clip", 0.5), c.target_noise_clip);
    }, [](const nlohmann::json& task) { return Layers(task.at("td3"), "net_arch", {400, 300}); });  // SB3 TD3's
}

std::optional<RLReference> FindDqnReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const DqnConfig& c) {
    return Find(json_text, environment_id, copies, [&c](const nlohmann::json& task) {
        const auto& s = task.at("dqn");
        // The recorded runs normalise nothing
        if (task.value("normalize", false) || c.normalize_observations) return false;
        return Same(Setting(s, "learning_rate", 1e-4), c.learning_rate) &&
               Same(Setting(s, "buffer_size", 1e6), static_cast<double>(c.buffer_size)) &&
               Same(Setting(s, "learning_starts", 100), static_cast<double>(c.learning_starts)) &&
               Same(Setting(s, "batch_size", 32), c.batch_size) && Same(Setting(s, "tau", 1.0), c.tau) &&
               Same(Setting(s, "gamma", 0.99), c.gamma) && Same(Setting(s, "train_freq", 4), c.train_freq) &&
               Same(Setting(s, "gradient_steps", 1), c.gradient_steps) &&
               Same(Setting(s, "target_update_interval", 10000), static_cast<double>(c.target_update_interval)) &&
               Same(Setting(s, "exploration_fraction", 0.1), c.exploration_fraction) &&
               Same(Setting(s, "exploration_initial_eps", 1.0), c.exploration_initial_eps) &&
               Same(Setting(s, "exploration_final_eps", 0.05), c.exploration_final_eps) &&
               Same(Setting(s, "max_grad_norm", 10), c.max_grad_norm);
    }, [](const nlohmann::json& task) { return Layers(task.at("dqn"), "net_arch", {64, 64}); });  // SB3 DQN's
}

bool SameLayers(const RLReference& reference, const RLTrainingConfiguration& config) {
    const auto [first, second] = RLNetworkUnits(config);
    return first == reference.layers && second == reference.layers;
}

std::optional<RLReference> FindRLReference(const RLTrainingConfiguration& c) {
    // The recorded runs use the environment's defaults (no grounds, pushes or throws)
    if (!DefaultEnvironmentSettings(c.environment, c.environment_settings)) return std::nullopt;
    std::optional<RLReference> ref;
    switch (c.algorithm) {
        case RLAlgorithm::SAC:
            ref = FindSacReference(ReadReferenceFile("sac_sb3_curves.json"), c.environment.id, c.copies, c.sac);
            break;
        case RLAlgorithm::TD3:
            ref = FindTd3Reference(ReadReferenceFile("td3_sb3_curves.json"), c.environment.id, c.copies, c.td3);
            break;
        case RLAlgorithm::DQN:
            ref = FindDqnReference(ReadReferenceFile("dqn_sb3_curves.json"), c.environment.id, c.copies, c.dqn);
            break;
        case RLAlgorithm::PPO:
            ref = FindPpoReference(ReadReferenceFile("ppo_sb3_curves.json"), c.environment.id, c.copies, c.ppo);
            break;
    }
    if (ref && !SameLayers(*ref, c)) return std::nullopt;
    return ref;
}

bool DefaultEnvironmentSettings(const plugin::PluginEnvironmentInfo& environment,
                                const std::map<std::string, std::string>& settings) {
    const auto number = [](const std::string& text, double& out) {
        char* end = nullptr;
        out = std::strtod(text.c_str(), &end);
        return !text.empty() && end && *end == '\0';
    };
    for (const auto& [key, value] : settings) {
        const auto declared = std::find_if(environment.settings.begin(), environment.settings.end(),
                                           [&key](const plugin::PluginEnvironmentSetting& s) { return s.key == key; });
        if (declared == environment.settings.end()) return false;
        double a = 0.0, b = 0.0;
        const bool same = number(value, a) && number(declared->default_value, b) ? a == b : value == declared->default_value;
        if (!same) return false;
    }
    return true;
}

std::vector<std::filesystem::path> ReferenceFileFolders() {
    // Beside the executable, as the pattern library looks for its resources
    // (a macOS app bundle, a build tree); never the working folder.
    const auto exe_dir = core::ExecutableDirectory();
    std::vector<std::filesystem::path> folders;
    for (const auto& dir : {exe_dir / "resources", exe_dir / ".." / "Resources", exe_dir / ".." / "resources",
                            exe_dir / ".." / ".." / "resources",
                            exe_dir / ".." / ".." / ".." / "cyxwiz-engine" / "resources"})
        folders.push_back((dir / "rl_reference").lexically_normal());
    return folders;
}

std::string ReadReferenceFile(const std::string& name) {
    for (const auto& folder : ReferenceFileFolders()) {
        std::ifstream in(folder / name, std::ios::binary);
        if (!in) continue;
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }
    return {};
}

}  // namespace cyxwiz::rl
