#pragma once

// Results > Test (TOFIX129 B5, D4): the headline, figures, classes, confusion
// matrix, mistakes and the comparison with the run before, from a test's
// TestingMetrics. Pure data in and out (no ImGui), so the panel and its tests
// say the same words.

#include "test_executor.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::test_report {

enum class Tone { Ok, Warning, Error, Info, Neutral };

struct Chip {
    std::string text;
    Tone tone = Tone::Neutral;
};

struct Tile {
    std::string value;
    std::string label;
};

struct ClassRow {
    int index = 0;
    std::string name;
    float precision = 0.0f, recall = 0.0f, f1 = 0.0f;
    int found = 0;          // true positives
    int false_alarms = 0;   // false positives
    int support = 0;
    bool never_predicted = false;
};

enum class ClassSort { Name, Precision, Recall, F1, Found, FalseAlarms, Support };

struct MatrixView {
    std::vector<std::string> names;
    std::vector<std::vector<int>> counts;   // [actual][predicted]
    std::vector<int> row_totals;            // support of each actual class
    int max_count = 0;
    std::string largest_confusion;          // "true Depression predicted as Suicidal (641)"
};

struct ComparisonRow {
    std::string metric;
    std::string previous, latest, change;
    Tone change_tone = Tone::Neutral;
};

struct View {
    std::string kind;              // "classification", "regression", "causal_lm", "metric_learning"
    bool classification = false;   // the class tabs apply
    std::string headline;          // "Accuracy 61.04%", "MAE 0.1234", "Token accuracy 41.20%"
    Tone headline_tone = Tone::Neutral;
    std::string subtitle;          // "5,311 test samples, 1.50 s"
    std::string note;              // quick test, 1-NN explanation
    std::vector<Chip> chips;
    std::vector<Tile> tiles;
    std::vector<std::pair<std::string, std::string>> facts;  // best / worst classes, pair and retrieval metrics, regression bias
    std::vector<ClassRow> classes;
    MatrixView matrix;
    size_t correct = 0;
    std::vector<int> mistakes;     // indices of misclassified test samples, most confident first
    std::vector<int> largest_errors;  // regression: every kept row, largest mean |error| first
    std::vector<ComparisonRow> comparison;
    std::string comparison_note;
};

// The name of class `index`: the run's own name, else "Class N".
std::string ClassName(const TestingMetrics& m, int index);

View BuildView(const TestingMetrics& latest, const TestingMetrics* previous);

// Classes filtered by name (case-insensitive, empty = all) and sorted.
std::vector<ClassRow> SortClasses(std::vector<ClassRow> rows, ClassSort by, bool ascending,
                                  const std::string& filter);

// A failed test shows only its message; it never becomes the run Comparison
// compares against.
bool IsFailedRun(const TestingMetrics& m);

// Two classes, or one class against the rest (TOFIX129 D4): the figures at a
// threshold (the class is predicted when its score is at least the threshold),
// read off that class's ROC curve, which holds the true and false positives at
// every distinct score (TP = TPR x positives, FP = FPR x negatives). Moving the
// threshold does no new pass over the rows.
struct ThresholdFigures {
    bool ok = false;
    double threshold = 0.5;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    double precision = 0.0, recall = 0.0, specificity = 0.0, f1 = 0.0, balanced_accuracy = 0.0, mcc = 0.0,
           accuracy = 0.0;
};
ThresholdFigures AtThreshold(const ROCCurveData& roc, int positives, int negatives, double threshold);
// The score with the highest F1 (the lowest such score on a tie, the
// classification fixtures' threshold contract); NaN when the curve has no
// point.
double BestF1Threshold(const ROCCurveData& roc, int positives, int negatives);

// A regression test's R-squared: 1 - SSE / SST per target, averaged over the
// targets (scikit-learn's r2_score); NaN when no rows were kept or a target
// is constant.
double RSquared(const TestingMetrics& m);

}  // namespace cyxwiz::test_report
