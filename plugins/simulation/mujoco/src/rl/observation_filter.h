#pragma once

#include <vector>
#include <string>
#include <map>
#include <cmath>

namespace cyxwiz::plugin::rl {

struct ObsFilterConfig {
    bool include_qpos = true;
    bool include_qvel = true;
    bool include_sensors = false;
    bool normalize = true;

    static ObsFilterConfig FromParameters(const std::map<std::string, std::string>& params);
};

/**
 * ObservationFilter - Selects and normalizes environment observations for policy input.
 *
 * Concatenates selected state components (qpos, qvel, sensors) into a flat vector.
 * Optionally applies running mean/std normalization.
 */
class ObservationFilter {
public:
    explicit ObservationFilter(const ObsFilterConfig& config = {});

    void SetConfig(const ObsFilterConfig& config) { config_ = config; }
    const ObsFilterConfig& GetConfig() const { return config_; }

    // Build observation vector from raw state components
    std::vector<float> Filter(
        const std::vector<float>& qpos,
        const std::vector<float>& qvel,
        const std::vector<float>& sensors
    );

    // Get the expected observation dimension (after last Filter call)
    int GetObsDim() const { return obs_dim_; }

    // Reset normalization statistics
    void Reset();

private:
    ObsFilterConfig config_;
    int obs_dim_ = 0;

    // Running normalization stats
    std::vector<double> running_mean_;
    std::vector<double> running_var_;
    int64_t count_ = 0;

    void UpdateStats(const std::vector<float>& obs);
    std::vector<float> Normalize(const std::vector<float>& obs) const;
};

} // namespace cyxwiz::plugin::rl
