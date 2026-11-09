#pragma once

// Observation and reward normalisation for N environment copies, as
// Stable-Baselines3's VecNormalize does it (TOFIX195 S4.3a): running mean and
// variance of the observations, rewards divided by the running deviation of
// the discounted return, both clipped. Works in place on the step buffers.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cyxwiz::rl {

// SB3's RunningMeanStd: parallel-algorithm mean and population variance in
// double, starting from mean 0, variance 1 and a count of 1e-4.
struct RunningMeanStd {
    explicit RunningMeanStd(size_t width = 1);
    void Update(const double* rows, size_t count);  // count rows of width values

    std::vector<double> mean;
    std::vector<double> var;
    double count = 1e-4;
};

struct VecNormalizeSettings {
    bool observations = false;
    bool rewards = false;
    float clip_observations = 10.0f;
    float clip_rewards = 10.0f;
    float gamma = 0.99f;    // discount of the return whose deviation scales the rewards
    float epsilon = 1e-8f;
};

class VecNormalize {
public:
    VecNormalize(size_t copies, size_t observation_size, VecNormalizeSettings settings);

    // After the environments' reset: learns from and normalises the first observations
    void Reset(float* observations);
    // After a step: learns from and normalises the observations, scales the
    // rewards and normalises the ended copies' final observations
    void Step(float* observations, float* rewards, const uint8_t* terminated, const uint8_t* truncated,
              float* final_observations);
    // With the statistics as they are (playback of a trained policy)
    void NormalizeObservations(float* observations, size_t rows) const;

    const VecNormalizeSettings& Settings() const { return settings_; }
    const RunningMeanStd& ObservationStats() const { return observation_stats_; }
    const RunningMeanStd& ReturnStats() const { return return_stats_; }
    // A saved policy's observation statistics (false when the width differs)
    bool RestoreObservationStats(const RunningMeanStd& stats);

private:
    void LearnObservations(const float* observations);

    size_t copies_;
    size_t observation_size_;
    VecNormalizeSettings settings_;
    RunningMeanStd observation_stats_;
    RunningMeanStd return_stats_;
    std::vector<double> returns_;
};

}  // namespace cyxwiz::rl
