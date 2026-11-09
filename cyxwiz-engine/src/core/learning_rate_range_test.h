#pragma once

// The Model window's words for a learning rate range test (TOFIX129 D11).
// The test's rules (the rate of each step, the smoothing, when it stops, what
// it suggests) are the SDK's, cyxwiz/learning_rate_range.h; TrainingExecutor
// asks it for the next rate after every step (learning_rate_range in
// TrainingConfiguration).

#include <cyxwiz/learning_rate_range.h>

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

// The Model window's words for a finished test: the suggested rate by the
// chosen rule, why, and the button that puts it into the optimizer node.
struct LearningRateRangeWords {
    double suggested = 0.0;
    std::string suggested_text;  // "0.0148"
    std::string headline;        // "Suggested learning rate: 0.0148"
    std::string why;
    std::string use_label;       // "Use 0.0148 in Adam"
};
LearningRateRangeWords DescribeLearningRateRange(const LearningRateRangeAnalysis& analysis, bool steepest_rule,
                                                 const std::string& optimizer, double current_rate);

// A rate as people type it: three significant digits ("0.0148", "1e-07").
std::string FormatLearningRate(double rate);
// The curve as text: step, learning_rate, loss, smoothed (separator a comma or a tab).
std::string LearningRateRangeText(const std::vector<LearningRateRangePoint>& points, char separator);

}  // namespace cyxwiz
