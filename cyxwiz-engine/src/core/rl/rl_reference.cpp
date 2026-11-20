#include "rl_reference.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
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

// The first recorded task of this environment and copy count whose settings
// match, as a reference
std::optional<RLReference> Find(const std::string& json_text, const std::string& environment_id, int copies,
                                const std::function<bool(const nlohmann::json& task)>& matches) {
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
    });
}

std::string ReadReferenceFile(const std::string& name) {
    // where the pattern library looks for its resources
    for (const char* dir : {"resources/rl_reference", "cyxwiz-engine/resources/rl_reference", "../resources/rl_reference",
                            "../../resources/rl_reference", "../../../cyxwiz-engine/resources/rl_reference"}) {
        std::ifstream in(std::filesystem::path(dir) / name, std::ios::binary);
        if (!in) continue;
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }
    return {};
}

}  // namespace cyxwiz::rl
