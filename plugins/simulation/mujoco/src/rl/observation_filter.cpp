#include "observation_filter.h"
#include <algorithm>

namespace cyxwiz::plugin::rl {

ObsFilterConfig ObsFilterConfig::FromParameters(const std::map<std::string, std::string>& params) {
    ObsFilterConfig cfg;
    auto get = [&](const std::string& key, const std::string& def) -> std::string {
        auto it = params.find(key);
        return it != params.end() ? it->second : def;
    };

    cfg.include_qpos = (get("include_qpos", "true") == "true");
    cfg.include_qvel = (get("include_qvel", "true") == "true");
    cfg.include_sensors = (get("include_sensors", "false") == "true");
    cfg.normalize = (get("normalize", "true") == "true");

    return cfg;
}

ObservationFilter::ObservationFilter(const ObsFilterConfig& config)
    : config_(config) {}

std::vector<float> ObservationFilter::Filter(
    const std::vector<float>& qpos,
    const std::vector<float>& qvel,
    const std::vector<float>& sensors
) {
    std::vector<float> obs;

    if (config_.include_qpos) {
        obs.insert(obs.end(), qpos.begin(), qpos.end());
    }
    if (config_.include_qvel) {
        obs.insert(obs.end(), qvel.begin(), qvel.end());
    }
    if (config_.include_sensors) {
        obs.insert(obs.end(), sensors.begin(), sensors.end());
    }

    obs_dim_ = static_cast<int>(obs.size());

    if (config_.normalize && !obs.empty()) {
        UpdateStats(obs);
        return Normalize(obs);
    }

    return obs;
}

void ObservationFilter::UpdateStats(const std::vector<float>& obs) {
    if (running_mean_.size() != obs.size()) {
        running_mean_.assign(obs.size(), 0.0);
        running_var_.assign(obs.size(), 1.0);
        count_ = 0;
    }

    count_++;
    // Welford's online algorithm for running mean/variance
    for (size_t i = 0; i < obs.size(); ++i) {
        double delta = static_cast<double>(obs[i]) - running_mean_[i];
        running_mean_[i] += delta / count_;
        double delta2 = static_cast<double>(obs[i]) - running_mean_[i];
        running_var_[i] += delta * delta2;
    }
}

std::vector<float> ObservationFilter::Normalize(const std::vector<float>& obs) const {
    if (count_ < 2) return obs;

    std::vector<float> normalized(obs.size());
    for (size_t i = 0; i < obs.size(); ++i) {
        double variance = running_var_[i] / (count_ - 1);
        double std_dev = std::sqrt(variance + 1e-8);
        normalized[i] = static_cast<float>((obs[i] - running_mean_[i]) / std_dev);
        // Clip to prevent extreme values
        normalized[i] = std::clamp(normalized[i], -10.0f, 10.0f);
    }
    return normalized;
}

void ObservationFilter::Reset() {
    running_mean_.clear();
    running_var_.clear();
    count_ = 0;
    obs_dim_ = 0;
}

} // namespace cyxwiz::plugin::rl
