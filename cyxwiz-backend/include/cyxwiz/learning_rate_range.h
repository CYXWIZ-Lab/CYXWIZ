// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// The learning rate range test (Smith, "Cyclical Learning Rates", 2017;
// fastai's lr_find), TOFIX129 D11: a model trains while the learning rate
// rises exponentially every optimizer step; the smoothed loss shows where
// training works and where it breaks. This header holds the test's rules,
// apart from training: the rate of each step, the smoothing, when it stops,
// and what it suggests. The Engine's Model window (through its training
// executor) and pycyxwiz.model.lr_range_test use it; checked against
// PyTorch and numpy (the Engine's learning_rate_range truth test).

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct CYXWIZ_API LearningRateRangeSettings {
    bool enabled = false;
    double start = 1e-7;      // the first step's rate
    double end = 10.0;        // the last step's rate
    int steps = 100;          // optimizer steps from start to end
    double smoothing = 0.05;  // weight of the newest loss in the running average
    double stop_factor = 4.0; // stop once the smoothed loss passes this x its lowest
};

struct CYXWIZ_API LearningRateRangePoint {
    double learning_rate = 0.0;
    double loss = 0.0;      // the batch's loss at that rate
    double smoothed = 0.0;  // running average of the losses so far
};

class CYXWIZ_API LearningRateRangeTest {
public:
    explicit LearningRateRangeTest(LearningRateRangeSettings settings);

    // start x (end / start)^(step / (steps - 1)).
    double LearningRateAt(int step) const;
    // The rate for the next optimizer step.
    double CurrentLearningRate() const { return LearningRateAt(static_cast<int>(points_.size())); }
    // The loss of the step just taken at CurrentLearningRate(). True when the
    // test is over (StopReason says why).
    bool Record(double loss);

    bool Finished() const { return finished_; }
    const std::string& StopReason() const { return stop_reason_; }
    const std::vector<LearningRateRangePoint>& Points() const { return points_; }
    const LearningRateRangeSettings& Settings() const { return settings_; }

private:
    LearningRateRangeSettings settings_;
    std::vector<LearningRateRangePoint> points_;
    double average_ = 0.0;
    double best_ = 0.0;
    bool finished_ = false;
    std::string stop_reason_;
};

// Where the curve says to train. The steepest fall: the most negative slope
// of the smoothed loss against log10(rate), up to the lowest point (numpy's
// gradient: central differences inside, one-sided at the ends). Lowest / 10:
// a tenth of the rate with the lowest smoothed loss.
struct CYXWIZ_API LearningRateRangeAnalysis {
    bool ok = false;
    std::string error;
    size_t lowest = 0, steepest = 0;  // indexes into the points
    double lowest_rate = 0.0, lowest_loss = 0.0;
    double steepest_rate = 0.0;
    double lowest_over_ten = 0.0;
    std::vector<double> slopes;  // d smoothed / d log10(rate) at each point
};

CYXWIZ_API LearningRateRangeAnalysis AnalyseLearningRateRange(const std::vector<LearningRateRangePoint>& points);

}  // namespace cyxwiz
