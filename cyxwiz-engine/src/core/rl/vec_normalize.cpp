#include "vec_normalize.h"

#include <algorithm>
#include <cmath>

namespace cyxwiz::rl {

RunningMeanStd::RunningMeanStd(size_t width) : mean(width, 0.0), var(width, 1.0) {}

void RunningMeanStd::Update(const double* rows, size_t n) {
    if (n == 0) return;
    const size_t width = mean.size();
    const double total = count + static_cast<double>(n);
    for (size_t j = 0; j < width; ++j) {
        double batch_mean = 0.0;
        for (size_t i = 0; i < n; ++i) batch_mean += rows[i * width + j];
        batch_mean /= static_cast<double>(n);
        double batch_var = 0.0;
        for (size_t i = 0; i < n; ++i) batch_var += (rows[i * width + j] - batch_mean) * (rows[i * width + j] - batch_mean);
        batch_var /= static_cast<double>(n);
        const double delta = batch_mean - mean[j];
        const double m2 = var[j] * count + batch_var * static_cast<double>(n) +
                          delta * delta * count * static_cast<double>(n) / total;
        mean[j] += delta * static_cast<double>(n) / total;
        var[j] = m2 / total;
    }
    count = total;
}

VecNormalize::VecNormalize(size_t copies, size_t observation_size, VecNormalizeSettings settings)
    : copies_(copies), observation_size_(observation_size), settings_(settings),
      observation_stats_(observation_size), return_stats_(1), returns_(copies, 0.0) {}

void VecNormalize::LearnObservations(const float* observations) {
    if (!settings_.observations) return;
    const std::vector<double> rows(observations, observations + copies_ * observation_size_);
    observation_stats_.Update(rows.data(), copies_);
}

void VecNormalize::NormalizeObservations(float* observations, size_t rows) const {
    if (!settings_.observations) return;
    const double clip = settings_.clip_observations;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < observation_size_; ++j) {
            float& x = observations[i * observation_size_ + j];
            const double z = (x - observation_stats_.mean[j]) / std::sqrt(observation_stats_.var[j] + settings_.epsilon);
            x = static_cast<float>(std::clamp(z, -clip, clip));
        }
}

void VecNormalize::Reset(float* observations) {
    std::fill(returns_.begin(), returns_.end(), 0.0);
    LearnObservations(observations);
    NormalizeObservations(observations, copies_);
}

void VecNormalize::Step(float* observations, float* rewards, const uint8_t* terminated, const uint8_t* truncated,
                        float* final_observations) {
    LearnObservations(observations);
    NormalizeObservations(observations, copies_);
    for (size_t i = 0; i < copies_; ++i) returns_[i] = returns_[i] * settings_.gamma + rewards[i];
    return_stats_.Update(returns_.data(), copies_);
    if (settings_.rewards) {
        const double scale = std::sqrt(return_stats_.var[0] + settings_.epsilon);
        const double clip = settings_.clip_rewards;
        for (size_t i = 0; i < copies_; ++i) rewards[i] = static_cast<float>(std::clamp(rewards[i] / scale, -clip, clip));
    }
    for (size_t i = 0; i < copies_; ++i) {
        if (!terminated[i] && !truncated[i]) continue;
        NormalizeObservations(final_observations + i * observation_size_, 1);
        returns_[i] = 0.0;
    }
}

}  // namespace cyxwiz::rl
