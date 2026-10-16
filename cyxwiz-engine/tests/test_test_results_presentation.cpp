// Test Results wording and exports (TOFIX129 B5), on the sentiment LSTM's
// real test (1 epoch, 5,311 samples, 61.04%): headline, classes with names,
// best and worst classes, the largest confusion, mistakes, sorting, the
// comparison (a failed run never compared against), and the CSV / JSON
// exports of the shown results.

#include "../src/core/test_results_export.h"
#include "../src/core/test_results_presentation.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using namespace cyxwiz;
using namespace cyxwiz::test_report;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

// The 7-class matrix of the run (rows actual, columns predicted).
const std::vector<std::vector<int>> kMatrix = {
    {173, 60, 143, 6, 0, 8, 0},   {7, 1532, 91, 6, 0, 0, 0},   {15, 79, 801, 641, 0, 5, 0},
    {2, 39, 299, 726, 0, 0, 0},   {44, 28, 184, 6, 0, 6, 0},   {91, 18, 168, 2, 0, 10, 0},
    {9, 15, 90, 5, 0, 2, 0}};
const std::vector<std::string> kNames = {"Anxiety", "Normal", "Depression", "Suicidal", "Stress", "Bipolar",
                                         "Personality disorder"};

TestingMetrics SentimentRun(bool with_names) {
    TestingMetrics m;
    m.is_complete = true;
    m.total_samples = 5311;
    m.test_loss = 1.0698f;
    m.test_accuracy = 0.6104f;
    m.macro_precision = 0.3811f;
    m.macro_recall = 0.3736f;
    m.macro_f1 = 0.3584f;
    m.weighted_f1 = 0.5739f;
    m.total_time_seconds = 1.5f;
    m.samples_per_second = 3552.0f;
    m.confusion_matrix.num_classes = 7;
    m.confusion_matrix.matrix = kMatrix;
    for (int c = 0; c < 7; ++c) {
        ClassMetrics cm;
        cm.class_id = c;
        if (with_names) cm.class_name = kNames[static_cast<size_t>(c)];
        int column = 0, row = 0;
        for (int i = 0; i < 7; ++i) column += kMatrix[static_cast<size_t>(i)][static_cast<size_t>(c)];
        for (int j = 0; j < 7; ++j) row += kMatrix[static_cast<size_t>(c)][static_cast<size_t>(j)];
        cm.true_positives = kMatrix[static_cast<size_t>(c)][static_cast<size_t>(c)];
        cm.false_positives = column - cm.true_positives;
        cm.false_negatives = row - cm.true_positives;
        cm.support = row;
        cm.precision = column ? static_cast<float>(cm.true_positives) / column : 0.0f;
        cm.recall = row ? static_cast<float>(cm.true_positives) / row : 0.0f;
        cm.f1_score = cm.precision + cm.recall > 0 ? 2 * cm.precision * cm.recall / (cm.precision + cm.recall) : 0.0f;
        m.per_class_metrics.push_back(cm);
    }
    // Four predictions: two right, two wrong.
    m.predictions = {1, 2, 0, 1};
    m.ground_truth = {0, 0, 0, 1};
    m.confidences = {0.93f, 0.38f, 0.34f, 0.9f};
    return m;
}

void CheckOverview() {
    const View v = BuildView(SentimentRun(true), nullptr);
    Check(v.headline == "Accuracy 61.04%" && v.headline_tone == Tone::Warning, "headline: " + v.headline);
    Check(v.subtitle == "5,311 test samples, 1.50 s", "subtitle: " + v.subtitle);
    Check(v.chips.size() == 4 && v.chips[0].text == "loss 1.0698" && v.chips[3].text == "3,552 samples/s", "chips");
    Check(v.tiles.size() == 4 && v.tiles[2].value == "0.3584", "tiles");
    Check(v.facts[0].first == "Best class" && v.facts[0].second == "Normal, F1 0.8993 (1,532 of 1,636 found)",
          "best class: " + v.facts[0].second);
    Check(v.facts[1].first == "Worst classes" &&
              v.facts[1].second == "Stress, Personality disorder, F1 0.0000: never predicted",
          "worst classes: " + v.facts[1].second);
    Check(v.matrix.largest_confusion == "The largest confusion: true Depression predicted as Suicidal (641).",
          "largest confusion: " + v.matrix.largest_confusion);
    Check(v.matrix.row_totals[1] == 1636 && v.matrix.row_totals[6] == 121, "row totals are the supports");
    Check(v.classes[4].never_predicted && !v.classes[1].never_predicted, "never predicted");
    Check(v.correct == 2 && v.mistakes == std::vector<int>({0, 1}), "mistakes are the wrong predictions");
    Check(v.comparison.empty(), "no comparison for one run");

    const View unnamed = BuildView(SentimentRun(false), nullptr);
    Check(unnamed.classes[1].name == "Class 1" && Contains(unnamed.facts[0].second, "Class 1"),
          "without names the classes are numbered");
}

void CheckSorting() {
    const View v = BuildView(SentimentRun(true), nullptr);
    const auto by_f1 = SortClasses(v.classes, ClassSort::F1, false, "");
    Check(by_f1.front().name == "Normal" && by_f1[1].name == "Suicidal", "sorted by F1, highest first");
    const auto by_support = SortClasses(v.classes, ClassSort::Support, true, "");
    Check(by_support.front().name == "Personality disorder", "sorted by support, smallest first");
    const auto filtered = SortClasses(v.classes, ClassSort::Name, true, "DIS");
    Check(filtered.size() == 1 && filtered[0].name == "Personality disorder", "filter ignores case");
}

void CheckComparison() {
    TestingMetrics before = SentimentRun(true);
    before.test_accuracy = 0.55f;
    before.partial = true;
    const View v = BuildView(SentimentRun(true), &before);
    Check(v.comparison.size() == 8 && v.comparison[1].metric == "Accuracy" && v.comparison[1].previous == "55.00%" &&
              v.comparison[1].latest == "61.04%" && v.comparison[1].change == "+6.04%" &&
              v.comparison[1].change_tone == Tone::Ok,
          "accuracy compared");
    Check(v.comparison_note == "The previous run was a quick test (first batches only).", "quick-test note");
    TestingMetrics failed;
    failed.status_message = "Model has no layers";
    Check(IsFailedRun(failed) && BuildView(SentimentRun(true), &failed).comparison.empty(),
          "a failed run is never compared against");
}

void CheckExports() {
    const TestingMetrics m = SentimentRun(true);
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_test_results_export";
    std::filesystem::create_directories(dir);
    std::string error;
    Check(WriteTestResultsCsv(m, (dir / "r.csv").string(), &error), "csv: " + error);
    std::stringstream csv;
    {
        std::ifstream csv_in(dir / "r.csv");
        csv << csv_in.rdbuf();
    }
    Check(Contains(csv.str(), "Normal,0.86") && Contains(csv.str(), "Found (TP),False alarms (FP),Missed (FN)") &&
              Contains(csv.str(), ",Anxiety,Normal,Depression") && Contains(csv.str(), "0,Normal,Anxiety,no,0.93"),
          "csv has names, found / false alarms / missed, matrix and predictions");
    Check(WriteTestResultsJson(m, (dir / "r.json").string(), &error), "json: " + error);
    nlohmann::json j;
    {
        std::ifstream json_in(dir / "r.json");
        j = nlohmann::json::parse(json_in);
    }
    Check(j["objective"] == "classification" && j["quick_test"] == false && j["predictions"].size() == 4 &&
              j["confusion_matrix"]["class_names"][2] == "Depression" &&
              j["per_class_metrics"][1]["class_name"] == "Normal",
          "json has the shown run");
    Check(!WriteTestResultsCsv(m, (dir / "missing" / "r.csv").string(), &error) && !error.empty(),
          "a path that cannot be written says why");
    std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
    CheckOverview();
    CheckSorting();
    CheckComparison();
    CheckExports();
    std::cout << "test results presentation: overview, classes, sorting, comparison, exports. OK\n";
    return 0;
}
