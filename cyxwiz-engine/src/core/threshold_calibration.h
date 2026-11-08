#pragma once

// Results > Test > Threshold and Calibration (TOFIX129 D11): pure maths over
// a test's scores, checked against scikit-learn / scipy
// (generate_threshold_calibration_fixtures.py).
//
// Threshold (a yes / no model, or one class against the rest): precision,
// recall and F1 at every threshold, the threshold that keeps recall at a
// target, the one with the lowest cost of errors. Several classes: a
// confidence floor, below which a row is left for a person.
//
// Calibration: how often the model is right when it says p (reliability
// bins, expected calibration error), and a fix that does not change any
// prediction's order: one temperature for several classes (softmax of
// log p / T), Platt's sigmoid or isotonic regression for a yes / no score.
// Fitted on half the test rows (the even ones) and checked on the other
// half, so the "after" figures are on rows the fit did not see.

#include "test_executor.h"

#include <string>
#include <vector>

namespace cyxwiz::test_report {

// ---- Threshold ------------------------------------------------------------------

// Every distinct score of the class's ROC curve, ascending, with the
// figures when that score is the threshold.
struct ThresholdSweep {
    std::vector<double> thresholds, precision, recall, f1;
};
ThresholdSweep SweepThresholds(const ROCCurveData& roc, int positives, int negatives);

// The highest threshold whose recall is at least `target`; NaN when none.
double ThresholdForRecall(const ROCCurveData& roc, int positives, int negatives, double target);

// The threshold with the lowest cost_fp x false positives + cost_fn x false
// negatives (the lowest score on a tie); NaN when the curve has no point.
double LowestCostThreshold(const ROCCurveData& roc, int positives, int negatives, double cost_fp, double cost_fn);

// Several classes: answer only when the confidence (the top probability)
// is at least `floor`.
struct FloorFigures {
    double floor = 0.0;
    int answered = 0, total = 0;
    double answered_share = 0.0;
    double accuracy = 0.0;  // of the answered rows
};
FloorFigures AtConfidenceFloor(const TestingMetrics& m, double floor);
// Floors 0.00, 0.01, ... 0.99.
std::vector<FloorFigures> ConfidenceFloorCurve(const TestingMetrics& m);

// ---- Calibration ----------------------------------------------------------------

struct ReliabilityBin {
    double low = 0.0, high = 0.0;
    int count = 0;
    double confidence = 0.0;  // mean confidence of its rows
    double accuracy = 0.0;    // share of its rows the model got right
};
struct Reliability {
    std::vector<ReliabilityBin> bins;  // `bins` equal bands of [0, 1]; 1.0 in the last
    double ece = 0.0;                  // sum over bins of count / n x |accuracy - confidence|
    int rows = 0;
};
Reliability ReliabilityOf(const std::vector<double>& confidence, const std::vector<int>& correct, int bins = 10);

// Several classes: T minimizing the mean negative log-likelihood of
// softmax(log p / T) over `rows` (p clipped at 1e-12), on log T.
struct TemperatureFit {
    double temperature = 1.0;
    double nll_before = 0.0, nll_after = 0.0;
};
TemperatureFit FitTemperature(const std::vector<float>& probabilities, int classes, const std::vector<int>& truth,
                              const std::vector<size_t>& rows);
// One row's probabilities after a temperature.
std::vector<double> WithTemperature(const float* probabilities, int classes, double temperature);

// Yes / no: Platt's sigmoid 1 / (1 + exp(a s + b)) of the score s, fitted
// as scikit-learn's _sigmoid_calibration (Platt's smoothed targets, Newton
// steps on the log-loss).
struct PlattFit {
    double a = 0.0, b = 0.0;
};
PlattFit FitPlatt(const std::vector<double>& scores, const std::vector<int>& labels);
double ApplyPlatt(const PlattFit& fit, double score);

// Yes / no: isotonic regression of the label on the score (non-decreasing,
// 0..1), as scikit-learn's IsotonicRegression(out_of_bounds="clip"): ties of
// the score averaged, pool adjacent violators, linear between the fitted
// points, clipped outside them.
struct IsotonicFit {
    std::vector<double> x, y;  // the fitted points, x ascending
};
IsotonicFit FitIsotonic(const std::vector<double>& scores, const std::vector<int>& labels);
double ApplyIsotonic(const IsotonicFit& fit, double score);

// The Calibration section's figures: before and after on the check half.
enum class CalibrationMethod { Temperature, Platt, Isotonic };
struct CalibrationReport {
    bool ok = false;
    std::string error;
    CalibrationMethod method = CalibrationMethod::Temperature;
    bool yes_no = false;
    int fit_rows = 0, check_rows = 0;
    Reliability before, after;          // on the check rows
    double temperature = 1.0;
    PlattFit platt;
    IsotonicFit isotonic;
    std::string headline;               // "When the model says 75 %, it is right 85 % of the time: ..."
    std::string detail;
};
// Yes / no: Platt or isotonic of P(positive class); several classes:
// temperature (Platt / isotonic fall back to it).
CalibrationReport Calibrate(const TestingMetrics& m, CalibrationMethod method, int positive = 1);

}  // namespace cyxwiz::test_report
