#include "learning_rate_range_test.h"

#include <cstdio>

namespace cyxwiz {

std::string FormatLearningRate(double rate) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", rate);
    return text;
}

LearningRateRangeWords DescribeLearningRateRange(const LearningRateRangeAnalysis& analysis, bool steepest_rule,
                                                 const std::string& optimizer, double current_rate) {
    LearningRateRangeWords words;
    if (!analysis.ok) {
        words.headline = "No suggestion";
        words.why = analysis.error;
        return words;
    }
    words.suggested = steepest_rule ? analysis.steepest_rate : analysis.lowest_over_ten;
    words.suggested_text = FormatLearningRate(words.suggested);
    words.headline = "Suggested learning rate: " + words.suggested_text;
    const std::string lowest = FormatLearningRate(analysis.lowest_rate);
    if (steepest_rule) {
        words.why = "The loss falls fastest there (Smith's range test); it is lowest at " + lowest + ".";
        if (analysis.steepest == analysis.lowest)
            words.why = "The loss fell fastest at its lowest point, " + lowest + ": try Lowest / 10 as well.";
    } else {
        words.why = "A tenth of the rate with the lowest loss (" + lowest + "), a safer pick than the steepest fall (" +
                    FormatLearningRate(analysis.steepest_rate) + ").";
    }
    if (current_rate > 0.0)
        words.why += " The graph's " + (optimizer.empty() ? std::string("optimizer") : optimizer) + " uses " +
                     FormatLearningRate(current_rate) + " now.";
    words.use_label = "Use " + words.suggested_text + " in " + (optimizer.empty() ? std::string("the optimizer") : optimizer);
    return words;
}

std::string LearningRateRangeText(const std::vector<LearningRateRangePoint>& points, char separator) {
    std::string text = std::string("step") + separator + "learning_rate" + separator + "loss" + separator + "smoothed\n";
    char line[160];
    for (size_t i = 0; i < points.size(); ++i) {
        std::snprintf(line, sizeof(line), "%zu%c%.9g%c%.9g%c%.9g\n", i + 1, separator, points[i].learning_rate,
                      separator, points[i].loss, separator, points[i].smoothed);
        text += line;
    }
    return text;
}

}  // namespace cyxwiz
