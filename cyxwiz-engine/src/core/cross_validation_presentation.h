#pragma once

// Results > Cross-validation and Learning curve (TOFIX129 D4, approved mockup
// boards 6-7): what a cross-validation outcome says, in words worked out from
// its numbers, for the views and their tests. Pure data in and out.

#include "cross_validation_runner.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::folds_view {

struct FoldRow {
    std::string fold, train_rows, val_rows;
    std::vector<std::string> scores;  // classes: accuracy, macro F1, loss; numbers: MAE, RMSE, R-squared
    bool flagged = false;             // a score far from the others (see note)
};

struct CrossValidationView {
    bool regression = false;
    std::string headline;               // "Accuracy 94.6% +- 1.7 over 5 folds"
    std::vector<std::string> pills;     // other scores, range, time
    std::string sentence;               // what to expect, and the single test beside it
    std::string note;                   // a fold that stands out
    std::vector<std::string> columns;   // the fold table's score columns
    std::vector<FoldRow> rows;
    FoldRow mean;
    std::vector<double> bars;           // the main score per fold
    double bars_mean = 0.0;
    std::string bars_label;             // "accuracy", "MAE"
    bool percent = false;               // bars are fractions shown as %
};

// `single_test` is the latest test's main score (accuracy, or MAE), when known.
CrossValidationView BuildCrossValidationView(const CrossValidationOutcome& outcome, std::optional<double> single_test = std::nullopt);

enum class CurveScore { Accuracy, MacroF1, Loss, Mae, Rmse, R2 };
// The scores a learning curve can follow: classes or numbers.
std::vector<CurveScore> CurveScores(bool regression);
const char* CurveScoreName(CurveScore score);

struct CurvePoint {
    double share = 0.0, rows = 0.0;  // rows: mean training rows over the folds
    double train_mean = 0.0, train_sd = 0.0, val_mean = 0.0, val_sd = 0.0;
};

struct LearningCurveView {
    std::string score_name;
    bool percent = false;           // a fraction shown as %
    bool higher_is_better = true;
    std::vector<CurvePoint> points;
    std::vector<std::vector<std::string>> table;  // rows, training, validation, gap
    std::string sentence;           // where validation levels off, or that it still rises
    std::string advice;             // the gap at full size and what would help
};

LearningCurveView BuildLearningCurveView(const CrossValidationOutcome& outcome, CurveScore score);

// "Fold 3 of 5 (training rows 50%) - epoch 22 of 40 - about 1 min left".
std::string ProgressLine(const CrossValidationProgress& progress, double elapsed_seconds);

// "10%, 25%, 50%, 100%" or "0.1 0.25 0.5 1": the training shares, sorted; false
// with `error` when a value is not above 0 and at most 100% (1).
bool ParseShares(const std::string& text, std::vector<double>& shares, std::string& error);
std::string SharesText(const std::vector<double>& shares);

// Export CSV: one row per training (fold, share, rows, validation scores, and
// for a learning curve the training rows' scores), then the settings.
std::string OutcomeCsv(const CrossValidationOutcome& outcome);

}  // namespace cyxwiz::folds_view
