#pragma once

// Test Results (TOFIX129 B5): the headline, figures, classes, confusion
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
    std::vector<std::pair<std::string, std::string>> facts;  // best / worst classes, pair and retrieval metrics
    std::vector<ClassRow> classes;
    MatrixView matrix;
    size_t correct = 0;
    std::vector<int> mistakes;     // indices of misclassified test samples
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

}  // namespace cyxwiz::test_report
