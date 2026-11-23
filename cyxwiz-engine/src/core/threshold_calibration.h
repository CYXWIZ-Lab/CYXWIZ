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
// Calibration: how often the model is right when it says p, and a fix that
// does not change any prediction's order (the fits are the SDK's,
// cyxwiz/calibration.h). Fitted on half the test rows (the even ones) and
// checked on the other half, so the "after" figures are on rows the fit did
// not see.

#include "test_executor.h"

#include <cyxwiz/calibration.h>

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
