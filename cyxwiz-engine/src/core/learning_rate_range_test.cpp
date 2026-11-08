#include "learning_rate_range_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace cyxwiz {

LearningRateRangeTest::LearningRateRangeTest(LearningRateRangeSettings settings) : settings_(settings) {
    settings_.steps = std::max(2, settings_.steps);
    best_ = std::numeric_limits<double>::infinity();
}

double LearningRateRangeTest::LearningRateAt(int step) const {
    const double fraction = static_cast<double>(std::min(step, settings_.steps - 1)) / (settings_.steps - 1);
    return settings_.start * std::pow(settings_.end / settings_.start, fraction);
}

bool LearningRateRangeTest::Record(double loss) {
    if (finished_) return true;
    const double rate = CurrentLearningRate();
    average_ = points_.empty() ? loss : settings_.smoothing * loss + (1.0 - settings_.smoothing) * average_;
    points_.push_back({rate, loss, average_});
    if (!std::isfinite(loss)) {
        finished_ = true;
        stop_reason_ = "The loss stopped being a number: the rate is far too high.";
    } else if (average_ > settings_.stop_factor * best_) {
        finished_ = true;
        char text[96];
        std::snprintf(text, sizeof(text), "Stopped early: the smoothed loss passed %g x its lowest.",
                      settings_.stop_factor);
        stop_reason_ = text;
    } else {
        best_ = std::min(best_, average_);
        if (static_cast<int>(points_.size()) >= settings_.steps) {
            finished_ = true;
            stop_reason_ = "All " + std::to_string(settings_.steps) + " steps ran.";
        }
    }
    return finished_;
}

LearningRateRangeAnalysis AnalyseLearningRateRange(const std::vector<LearningRateRangePoint>& points) {
    LearningRateRangeAnalysis analysis;
    // The last point of a stopped test is the one that broke it; keep only finite losses.
    std::vector<double> x, y;
    for (const auto& point : points) {
        if (!std::isfinite(point.smoothed) || point.learning_rate <= 0.0) break;
        x.push_back(std::log10(point.learning_rate));
        y.push_back(point.smoothed);
    }
    if (x.size() < 3) {
        analysis.error = "Too few steps to read the curve: run more steps or a wider range.";
        return analysis;
    }
    const size_t n = x.size();
    analysis.slopes.resize(n);
    // numpy.gradient(y, x), edge_order 1.
    analysis.slopes[0] = (y[1] - y[0]) / (x[1] - x[0]);
    analysis.slopes[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    for (size_t i = 1; i + 1 < n; ++i) {
        const double hs = x[i] - x[i - 1], hd = x[i + 1] - x[i];
        analysis.slopes[i] = (hs * hs * y[i + 1] + (hd * hd - hs * hs) * y[i] - hd * hd * y[i - 1]) / (hs * hd * (hd + hs));
    }
    analysis.lowest = static_cast<size_t>(std::min_element(y.begin(), y.end()) - y.begin());
    analysis.steepest = analysis.lowest == 0
        ? 0
        : static_cast<size_t>(std::min_element(analysis.slopes.begin(), analysis.slopes.begin() + analysis.lowest + 1) -
                              analysis.slopes.begin());
    analysis.lowest_rate = points[analysis.lowest].learning_rate;
    analysis.lowest_loss = y[analysis.lowest];
    analysis.steepest_rate = points[analysis.steepest].learning_rate;
    analysis.lowest_over_ten = analysis.lowest_rate / 10.0;
    analysis.ok = true;
    return analysis;
}

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
