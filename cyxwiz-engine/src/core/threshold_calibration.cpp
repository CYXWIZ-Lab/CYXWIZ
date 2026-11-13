#include "threshold_calibration.h"

#include "class_words.h"
#include "test_results_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace cyxwiz::test_report {

namespace {

size_t Points(const ROCCurveData& roc) { return std::min({roc.fpr.size(), roc.tpr.size(), roc.thresholds.size()}); }

std::string Fixed(double v, int digits) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.*f", digits, v);
    return text;
}

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
    const std::string already = m.decision_calibrated ? "These probabilities already include the model's kept calibration. " : "";
    if (report.method == CalibrationMethod::Temperature && std::fabs(report.temperature - 1.0) < 0.02) {
        report.detail = already + "Calibration error " + Fixed(report.before.ece, 4) + ". A temperature fitted on half the test rows (" +
                        std::to_string(report.fit_rows) + ") comes out at " + Fixed(report.temperature, 3) +
                        ": the probabilities need no fix.";
        report.ok = true;
        return report;
    }
    report.detail = already + "Calibration error " + Fixed(report.before.ece, 4) + ". " + fixed_by + ", fitted on half the test rows (" +
                    std::to_string(report.fit_rows) + "), brings it to " + Fixed(report.after.ece, 4) +
                    " on the other half (" + std::to_string(report.check_rows) + ")" +
                    (report.method == CalibrationMethod::Temperature ? " without changing a single prediction."
                     : report.method == CalibrationMethod::Platt     ? "; the rows keep their order."
                                                                     : "; tied scores can share a value.");
    report.ok = true;
    return report;
}

}  // namespace cyxwiz::test_report
