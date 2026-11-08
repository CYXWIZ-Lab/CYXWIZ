#include "threshold_calibration.h"

#include "class_words.h"
#include "test_results_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace cyxwiz::test_report {

namespace {

size_t Points(const ROCCurveData& roc) { return std::min({roc.fpr.size(), roc.tpr.size(), roc.thresholds.size()}); }

std::string Fixed(double v, int digits) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.*f", digits, v);
    return text;
}

double Sigmoid(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }

}  // namespace

// ---- Threshold ------------------------------------------------------------------

ThresholdSweep SweepThresholds(const ROCCurveData& roc, int positives, int negatives) {
    ThresholdSweep sweep;
    for (size_t i = Points(roc); i-- > 1;) {
        const ThresholdFigures f = AtThreshold(roc, positives, negatives, roc.thresholds[i]);
        if (!f.ok || !std::isfinite(roc.thresholds[i])) continue;
        sweep.thresholds.push_back(roc.thresholds[i]);
        sweep.precision.push_back(f.precision);
        sweep.recall.push_back(f.recall);
        sweep.f1.push_back(f.f1);
    }
    return sweep;
}

double ThresholdForRecall(const ROCCurveData& roc, int positives, int negatives, double target) {
    // Recall falls as the threshold rises: the highest threshold still at the target.
    double best = NAN;
    for (size_t i = 1; i < Points(roc); ++i) {
        const ThresholdFigures f = AtThreshold(roc, positives, negatives, roc.thresholds[i]);
        if (f.ok && f.recall >= target - 1e-12 && std::isfinite(roc.thresholds[i]) &&
            (std::isnan(best) || roc.thresholds[i] > best))
            best = roc.thresholds[i];
    }
    return best;
}

double LowestCostThreshold(const ROCCurveData& roc, int positives, int negatives, double cost_fp, double cost_fn) {
    double best = NAN, best_cost = std::numeric_limits<double>::infinity();
    for (size_t i = Points(roc); i-- > 1;) {  // lowest score first: a tie keeps it
        const ThresholdFigures f = AtThreshold(roc, positives, negatives, roc.thresholds[i]);
        if (!f.ok || !std::isfinite(roc.thresholds[i])) continue;
        const double cost = cost_fp * f.fp + cost_fn * f.fn;
        if (cost < best_cost - 1e-12) {
            best_cost = cost;
            best = roc.thresholds[i];
        }
    }
    return best;
}

FloorFigures AtConfidenceFloor(const TestingMetrics& m, double floor) {
    FloorFigures f;
    f.floor = floor;
    const size_t n = std::min({m.predictions.size(), m.ground_truth.size(), m.confidences.size()});
    f.total = static_cast<int>(n);
    int right = 0;
    for (size_t i = 0; i < n; ++i) {
        if (m.confidences[i] < floor) continue;
        ++f.answered;
        right += m.predictions[i] == m.ground_truth[i];
    }
    f.answered_share = n ? static_cast<double>(f.answered) / static_cast<double>(n) : 0.0;
    f.accuracy = f.answered ? static_cast<double>(right) / f.answered : 0.0;
    return f;
}

std::vector<FloorFigures> ConfidenceFloorCurve(const TestingMetrics& m) {
    std::vector<FloorFigures> curve;
    for (int k = 0; k < 100; ++k) curve.push_back(AtConfidenceFloor(m, k / 100.0));
    return curve;
}

// ---- Calibration ----------------------------------------------------------------

Reliability ReliabilityOf(const std::vector<double>& confidence, const std::vector<int>& correct, int bins) {
    Reliability r;
    bins = std::max(1, bins);
    r.bins.resize(static_cast<size_t>(bins));
    std::vector<double> conf_sum(bins, 0.0), right(bins, 0.0);
    const size_t n = std::min(confidence.size(), correct.size());
    for (int b = 0; b < bins; ++b) {
        r.bins[b].low = static_cast<double>(b) / bins;
        r.bins[b].high = static_cast<double>(b + 1) / bins;
    }
    for (size_t i = 0; i < n; ++i) {
        const int b = std::min(bins - 1, std::max(0, static_cast<int>(confidence[i] * bins)));
        ++r.bins[b].count;
        conf_sum[b] += confidence[i];
        right[b] += correct[i] ? 1.0 : 0.0;
    }
    r.rows = static_cast<int>(n);
    for (int b = 0; b < bins; ++b) {
        auto& bin = r.bins[b];
        if (bin.count == 0) continue;
        bin.confidence = conf_sum[b] / bin.count;
        bin.accuracy = right[b] / bin.count;
        r.ece += static_cast<double>(bin.count) / static_cast<double>(n) * std::fabs(bin.accuracy - bin.confidence);
    }
    return r;
}

std::vector<double> WithTemperature(const float* probabilities, int classes, double temperature) {
    std::vector<double> z(classes);
    double top = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < classes; ++c) {
        z[c] = std::log(std::max(1e-12, static_cast<double>(probabilities[c]))) / temperature;
        top = std::max(top, z[c]);
    }
    double total = 0.0;
    for (auto& v : z) total += (v = std::exp(v - top));
    for (auto& v : z) v /= total;
    return z;
}

TemperatureFit FitTemperature(const std::vector<float>& probabilities, int classes, const std::vector<int>& truth,
                              const std::vector<size_t>& rows) {
    const auto nll = [&](double t) {
        double sum = 0.0;
        for (size_t r : rows) {
            const auto p = WithTemperature(&probabilities[r * classes], classes, t);
            sum -= std::log(std::max(1e-300, p[static_cast<size_t>(truth[r])]));
        }
        return rows.empty() ? 0.0 : sum / static_cast<double>(rows.size());
    };
    // Golden-section search on log T in [log 0.05, log 20].
    double lo = std::log(0.05), hi = std::log(20.0);
    const double g = (std::sqrt(5.0) - 1.0) / 2.0;
    double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
    double f1 = nll(std::exp(x1)), f2 = nll(std::exp(x2));
    while (hi - lo > 1e-9) {
        if (f1 < f2) {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = nll(std::exp(x1));
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = nll(std::exp(x2));
        }
    }
    TemperatureFit fit;
    fit.temperature = std::exp((lo + hi) / 2.0);
    fit.nll_before = nll(1.0);
    fit.nll_after = nll(fit.temperature);
    return fit;
}

PlattFit FitPlatt(const std::vector<double>& scores, const std::vector<int>& labels) {
    const size_t n = std::min(scores.size(), labels.size());
    double negatives = 0.0;
    for (size_t i = 0; i < n; ++i) negatives += labels[i] <= 0;
    const double positives = static_cast<double>(n) - negatives;
    // Platt's targets and start (scikit-learn's _sigmoid_calibration).
    const double hi = (positives + 1.0) / (positives + 2.0), lo = 1.0 / (negatives + 2.0);
    double a = 0.0, b = std::log((negatives + 1.0) / (positives + 1.0));
    for (int iteration = 0; iteration < 100; ++iteration) {
        // p = 1 / (1 + exp(a s + b)); loss = -sum t log p + (1 - t) log(1 - p).
        double ga = 0.0, gb = 0.0, haa = 0.0, hab = 0.0, hbb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double t = labels[i] > 0 ? hi : lo;
            const double p = Sigmoid(-(a * scores[i] + b));
            const double d = t - p;  // d loss / d (a s + b)
            ga += d * scores[i];
            gb += d;
            const double w = p * (1.0 - p);
            haa += w * scores[i] * scores[i];
            hab += w * scores[i];
            hbb += w;
        }
        haa += 1e-12;
        hbb += 1e-12;
        const double det = haa * hbb - hab * hab;
        if (std::fabs(det) < 1e-300) break;
        const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
        a -= da;
        b -= db;
        if (std::fabs(da) < 1e-12 && std::fabs(db) < 1e-12) break;
    }
    return {a, b};
}

double ApplyPlatt(const PlattFit& fit, double score) { return Sigmoid(-(fit.a * score + fit.b)); }

IsotonicFit FitIsotonic(const std::vector<double>& scores, const std::vector<int>& labels) {
    const size_t n = std::min(scores.size(), labels.size());
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return scores[i] < scores[j]; });
    // Tied scores: one point, the mean label, weighted by the count.
    std::vector<double> x, y, w;
    for (size_t k = 0; k < n; ++k) {
        const double s = scores[order[k]], v = labels[order[k]] > 0 ? 1.0 : 0.0;
        if (!x.empty() && x.back() == s) {
            y.back() = (y.back() * w.back() + v) / (w.back() + 1.0);
            w.back() += 1.0;
        } else {
            x.push_back(s);
            y.push_back(v);
            w.push_back(1.0);
        }
    }
    // Pool adjacent violators.
    std::vector<double> block_y, block_w;
    std::vector<size_t> block_end;
    for (size_t i = 0; i < x.size(); ++i) {
        block_y.push_back(y[i]);
        block_w.push_back(w[i]);
        block_end.push_back(i);
        while (block_y.size() >= 2 && block_y[block_y.size() - 2] > block_y.back()) {
            const double wsum = block_w[block_w.size() - 2] + block_w.back();
            const double merged = (block_y[block_y.size() - 2] * block_w[block_w.size() - 2] + block_y.back() * block_w.back()) / wsum;
            block_y.pop_back();
            block_w.pop_back();
            const size_t end = block_end.back();
            block_end.pop_back();
            block_y.back() = merged;
            block_w.back() = wsum;
            block_end.back() = end;
        }
    }
    std::vector<double> fitted(x.size());
    size_t start = 0;
    for (size_t b = 0; b < block_y.size(); ++b) {
        for (size_t i = start; i <= block_end[b]; ++i) fitted[i] = block_y[b];
        start = block_end[b] + 1;
    }
    // Keep only the points where the fitted value changes (the ends of each run).
    IsotonicFit fit;
    for (size_t i = 0; i < x.size(); ++i) {
        const bool keep = i == 0 || i + 1 == x.size() || fitted[i - 1] != fitted[i] || fitted[i] != fitted[i + 1];
        if (!keep) continue;
        fit.x.push_back(x[i]);
        fit.y.push_back(fitted[i]);
    }
    return fit;
}

double ApplyIsotonic(const IsotonicFit& fit, double score) {
    if (fit.x.empty()) return score;
    if (score <= fit.x.front()) return fit.y.front();
    if (score >= fit.x.back()) return fit.y.back();
    const size_t hi = static_cast<size_t>(std::upper_bound(fit.x.begin(), fit.x.end(), score) - fit.x.begin());
    const size_t lo = hi - 1;
    const double t = (score - fit.x[lo]) / (fit.x[hi] - fit.x[lo]);
    return fit.y[lo] + t * (fit.y[hi] - fit.y[lo]);
}

CalibrationReport Calibrate(const TestingMetrics& m, CalibrationMethod method, int positive) {
    CalibrationReport report;
    const int classes = m.probability_classes;
    const size_t n = std::min(m.predictions.size(), m.ground_truth.size());
    if (classes < 2 || m.class_probabilities.size() != n * static_cast<size_t>(classes) || n < 20) {
        report.error = classes < 2 || m.class_probabilities.empty()
                           ? "This test kept no class probabilities (a test of a table of predictions keeps only its columns)."
                           : "Calibration needs at least 20 test rows.";
        return report;
    }
    std::vector<size_t> fit_rows, check_rows;
    for (size_t i = 0; i < n; ++i) (i % 2 == 0 ? fit_rows : check_rows).push_back(i);
    report.fit_rows = static_cast<int>(fit_rows.size());
    report.check_rows = static_cast<int>(check_rows.size());
    report.yes_no = classes == 2;
    report.method = report.yes_no ? method : CalibrationMethod::Temperature;
    std::string fixed_by;

    if (report.method == CalibrationMethod::Temperature) {
        const auto fit = FitTemperature(m.class_probabilities, classes, m.ground_truth, fit_rows);
        report.temperature = fit.temperature;
        std::vector<double> before_conf, after_conf;
        std::vector<int> right;
        for (size_t r : check_rows) {
            const float* p = &m.class_probabilities[r * classes];
            const auto top = std::max_element(p, p + classes);
            before_conf.push_back(*top);
            const auto q = WithTemperature(p, classes, fit.temperature);
            after_conf.push_back(*std::max_element(q.begin(), q.end()));
            right.push_back(static_cast<int>(top - p) == m.ground_truth[r]);
        }
        report.before = ReliabilityOf(before_conf, right);
        report.after = ReliabilityOf(after_conf, right);
        fixed_by = "One temperature (" + Fixed(fit.temperature, 3) + ")";
    } else {
        const auto score_of = [&](size_t r) { return static_cast<double>(m.class_probabilities[r * classes + positive]); };
        const auto label_of = [&](size_t r) { return m.ground_truth[r] == positive ? 1 : 0; };
        std::vector<double> fit_scores;
        std::vector<int> fit_labels;
        for (size_t r : fit_rows) {
            fit_scores.push_back(score_of(r));
            fit_labels.push_back(label_of(r));
        }
        std::vector<double> before_p, after_p;
        std::vector<int> labels;
        if (report.method == CalibrationMethod::Platt) report.platt = FitPlatt(fit_scores, fit_labels);
        else report.isotonic = FitIsotonic(fit_scores, fit_labels);
        for (size_t r : check_rows) {
            before_p.push_back(score_of(r));
            after_p.push_back(report.method == CalibrationMethod::Platt ? ApplyPlatt(report.platt, score_of(r))
                                                                        : ApplyIsotonic(report.isotonic, score_of(r)));
            labels.push_back(label_of(r));
        }
        report.before = ReliabilityOf(before_p, labels);
        report.after = ReliabilityOf(after_p, labels);
        fixed_by = report.method == CalibrationMethod::Platt ? "Platt's sigmoid" : "Isotonic regression";
    }

    // The band that is furthest off, among those with enough rows.
    const ReliabilityBin* worst = nullptr;
    const int enough = std::max(5, report.before.rows / 20);
    for (const auto& bin : report.before.bins)
        if (bin.count >= enough && (!worst || std::fabs(bin.accuracy - bin.confidence) > std::fabs(worst->accuracy - worst->confidence)))
            worst = &bin;
    const std::string what = report.yes_no ? "is a " + ClassName(m, positive) : "is right";
    if (worst) {
        report.headline = "When the model says " + Fixed(worst->confidence * 100.0, 0) + " %, it " + what + " " +
                          Fixed(worst->accuracy * 100.0, 0) + " % of the time" +
                          (std::fabs(worst->accuracy - worst->confidence) < 0.02
                               ? ": close to its word."
                               : worst->accuracy > worst->confidence ? ": it undersells itself." : ": it oversells itself.");
    }
    report.detail = "Calibration error " + Fixed(report.before.ece, 4) + ". " + fixed_by + ", fitted on half the test rows (" +
                    std::to_string(report.fit_rows) + "), brings it to " + Fixed(report.after.ece, 4) +
                    " on the other half (" + std::to_string(report.check_rows) + ")" +
                    (report.method == CalibrationMethod::Temperature ? " without changing a single prediction."
                     : report.method == CalibrationMethod::Platt     ? "; the rows keep their order."
                                                                     : "; tied scores can share a value.");
    report.ok = true;
    return report;
}

}  // namespace cyxwiz::test_report
