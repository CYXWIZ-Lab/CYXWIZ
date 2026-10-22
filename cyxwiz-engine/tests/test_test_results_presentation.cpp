// Test Results wording and exports (TOFIX129 B5), on the sentiment LSTM's
// real test (1 epoch, 5,311 samples, 61.04%): headline, classes with names,
// best and worst classes, the largest confusion, mistakes, sorting, the
// comparison (a failed run never compared against), and the CSV / JSON
// exports of the shown results.

#include "../src/core/test_evaluation_tables.h"
#include "../src/core/test_predictions_metrics.h"
#include "../src/core/test_results_export.h"
#include "../src/core/test_results_presentation.h"
#include "../src/core/model_export_presentation.h"
#include "../src/core/model_open_presentation.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
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

    // The most confident mistake first (TOFIX157).
    TestingMetrics swapped = SentimentRun(true);
    swapped.confidences = {0.40f, 0.95f, 0.34f, 0.9f};
    Check(BuildView(swapped, nullptr).mistakes == std::vector<int>({1, 0}), "mistakes: most confident first");
}

// The Test Evaluation node's tables (TOFIX157): the panel's figures and order.
void CheckEvaluationTables() {
    TestingMetrics m = SentimentRun(true);
    m.row_ids = {106, 319, 410, 6};
    TestEvaluationTables tables;
    std::string error;
    Check(BuildTestEvaluationTables(m, 20, tables, error), "tables: " + error);
    const auto text = [](const std::shared_ptr<arrow::Table>& table, const char* column, int row) {
        const auto scalar = table->GetColumnByName(column)->GetScalar(row);
        return scalar.ok() && (*scalar)->is_valid ? (*scalar)->ToString() : std::string("(null)");
    };
    Check(tables.metrics->num_rows() == 7 && text(tables.metrics, "metric", 0) == "accuracy" &&
              Contains(text(tables.metrics, "value", 0), "0.6104") && text(tables.metrics, "metric", 6) == "rows",
          "metrics: accuracy first, rows last");
    Check(text(tables.confusion, "true", 0) == "Anxiety" && text(tables.confusion, "predicted", 0) == "Anxiety" &&
              text(tables.confusion, "count", 0) == "173",
          "confusion pairs by name, empty cells left out");
    Check(tables.per_class->num_rows() == 7 && text(tables.per_class, "class", 1) == "Normal" &&
              text(tables.per_class, "rows", 1) == "1636",
          "per class");
    Check(tables.worst_rows->num_rows() == 2 && text(tables.worst_rows, "dataset_row", 0) == "107" &&
              text(tables.worst_rows, "predicted", 0) == "Normal" && text(tables.worst_rows, "dataset_row", 1) == "320",
          "worst rows: the most confident mistake first, by dataset row");
    Check(BuildTestEvaluationTables(m, 1, tables, error) && tables.worst_rows->num_rows() == 1, "worst_rows caps the rows");
    m.row_ids.clear();
    Check(BuildTestEvaluationTables(m, 20, tables, error) && text(tables.worst_rows, "dataset_row", 0) == "(null)",
          "rows not recorded: no dataset row");

    TestingMetrics r;
    r.is_complete = true;
    r.regression_mode = true;
    r.test_mae = 1.0f;
    r.regression_width = 1;
    r.regression_truth = {1.0f, 2.0f, 3.0f};
    r.regression_predictions = {1.5f, 5.0f, 3.0f};
    r.row_ids = {4, 9, 2};
    Check(BuildTestEvaluationTables(r, 20, tables, error), "regression tables: " + error);
    Check(text(tables.metrics, "metric", 0) == "mae" && text(tables.metrics, "metric", 2) == "bias",
          "regression metrics: mae, rmse, bias");
    Check(tables.confusion->num_rows() == 0 && tables.per_class->num_rows() == 0, "regression: no class tables");
    Check(text(tables.worst_rows, "dataset_row", 0) == "10" && Contains(text(tables.worst_rows, "error", 0), "3"),
          "regression worst rows: the largest error first");
}

// A regression run's kept rows: bias and the largest errors (TOFIX157).
void CheckRegressionRows() {
    TestingMetrics m;
    m.is_complete = true;
    m.regression_mode = true;
    m.total_samples = 3;
    m.test_mae = 1.0f;
    m.regression_width = 2;
    m.regression_truth = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    m.regression_predictions = {1.5f, 1.5f, 5.0f, 2.0f, 2.0f, 3.0f};  // row errors 0.5, 1.5, 0.5
    const View v = BuildView(m, nullptr);
    Check(v.largest_errors == std::vector<int>({1, 0, 2}), "largest mean |error| first, ties in test order");
    Check(!v.facts.empty() && v.facts.back().first == "Bias" && v.facts.back().second == "+0.5000 (predicts high)",
          "bias: " + (v.facts.empty() ? std::string() : v.facts.back().second));
    m.regression_truth.clear();
    m.regression_predictions.clear();
    Check(BuildView(m, nullptr).largest_errors.empty(), "no kept rows: no largest errors");
}

// ---- Results > Test (TOFIX129 D4), against scikit-learn (generate_evaluation_fixtures.py) ----

nlohmann::json EvaluationFixture() {
    std::ifstream in(CYXWIZ_EVALUATION_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_EVALUATION_FIXTURE);
    return nlohmann::json::parse(in);
}

std::shared_ptr<arrow::Array> TextArray(const nlohmann::json& values) {
    arrow::StringBuilder b;
    for (const auto& v : values) Check(b.Append(v.get<std::string>()).ok(), "append text");
    std::shared_ptr<arrow::Array> out;
    Check(b.Finish(&out).ok(), "finish text");
    return out;
}

std::shared_ptr<arrow::Array> NumberArray(const nlohmann::json& values) {
    arrow::DoubleBuilder b;
    for (const auto& v : values) Check(b.Append(v.get<double>()).ok(), "append number");
    std::shared_ptr<arrow::Array> out;
    Check(b.Finish(&out).ok(), "finish number");
    return out;
}

bool Near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

std::string Num(double v) {
    char text[40];
    std::snprintf(text, sizeof(text), "%.9g", v);
    return text;
}

// Two classes read from a table (text labels, one score): the curves for each
// positive class, the figures at a threshold and the best-F1 threshold.
void CheckThresholds() {
    const auto j = EvaluationFixture()["binary"];
    const auto table = arrow::Table::Make(
        arrow::schema({arrow::field("actual", arrow::utf8()), arrow::field("predicted", arrow::utf8()),
                       arrow::field("p_male", arrow::float64())}),
        {TextArray(j["truth"]), TextArray(j["predicted"]), NumberArray(j["score_male"])});
    PredictionColumns columns;
    columns.truth = "actual";
    columns.predicted = "predicted";
    columns.scores = {"p_male"};
    columns.positive = "MALE";
    TestingMetrics m;
    int64_t left_out = -1;
    std::string error;
    Check(MetricsFromColumns(*table, columns, m, left_out, error), "two classes read as a test: " + error);
    Check(left_out == 0 && m.total_samples == static_cast<int>(j["truth"].size()), "every row read");
    Check(Near(m.test_accuracy, j["accuracy"].get<double>(), 1e-6), "accuracy " + Num(m.test_accuracy));
    Check(ClassName(m, 0) == "FEMALE" && ClassName(m, 1) == "MALE", "classes in the label encoder's order");

    ClassCurves male;
    Check(CurvesForClass(m, 1, male, error), "MALE curves: " + error);
    Check(Near(male.roc.auc, j["auc_male"].get<double>(), 1e-9), "AUC (MALE) " + Num(male.roc.auc));
    Check(Near(male.pr.average_precision, j["ap_male"].get<double>(), 1e-9), "AP (MALE) " + Num(male.pr.average_precision));
    for (const auto& at : j["at"]) {
        const double t = at["threshold"].get<double>();
        const ThresholdFigures f = AtThreshold(male.roc, male.positives, male.negatives, t);
        const std::string where = " at " + Num(t);
        Check(f.ok && f.tp == at["tp"].get<int>() && f.fp == at["fp"].get<int>() && f.tn == at["tn"].get<int>() &&
                  f.fn == at["fn"].get<int>(),
              "counts" + where + ": " + std::to_string(f.tp) + " " + std::to_string(f.fp) + " " + std::to_string(f.tn) + " " +
                  std::to_string(f.fn));
        Check(Near(f.precision, at["precision"].get<double>(), 1e-9) && Near(f.recall, at["recall"].get<double>(), 1e-9) &&
                  Near(f.specificity, at["specificity"].get<double>(), 1e-9) && Near(f.f1, at["f1"].get<double>(), 1e-9) &&
                  Near(f.balanced_accuracy, at["balanced"].get<double>(), 1e-9) && Near(f.mcc, at["mcc"].get<double>(), 1e-9),
              "figures" + where);
    }
    const double best = BestF1Threshold(male.roc, male.positives, male.negatives);
    Check(Near(best, j["best_threshold"].get<double>(), 1e-6), "best-F1 threshold " + Num(best));
    Check(Near(AtThreshold(male.roc, male.positives, male.negatives, best).f1, j["best_f1"].get<double>(), 1e-9), "best F1");

    ClassCurves female;  // class 0 of two: worked out on the device
    Check(CurvesForClass(m, 0, female, error), "FEMALE curves: " + error);
    Check(Near(female.roc.auc, j["auc_female"].get<double>(), 1e-6), "AUC (FEMALE) " + Num(female.roc.auc));
    Check(female.positives == male.negatives && female.negatives == male.positives, "FEMALE counts");
    Check(!AtThreshold(male.roc, 0, 10, 0.5).ok, "no positives: no figures");
}

// Three classes and a regression read from tables: the same figures as
// scikit-learn, score_<class> columns taken in class order whatever their
// order in the table, and the first guess at the columns.
void CheckTableAsTest() {
    const auto fixture = EvaluationFixture();
    const auto& j = fixture["multiclass"];
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("label", arrow::utf8()),
                                                         arrow::field("prediction", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> arrays = {TextArray(j["truth"]), TextArray(j["predicted"])};
    std::vector<std::string> classes = j["classes"].get<std::vector<std::string>>();
    for (auto c = classes.rbegin(); c != classes.rend(); ++c) {  // reversed on purpose
        fields.push_back(arrow::field("score_" + *c, arrow::float64()));
        arrays.push_back(NumberArray(j["scores"][*c]));
    }
    const auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    const PredictionColumns guess = GuessPredictionColumns(*table);
    Check(guess.truth == "label" && guess.predicted == "prediction" && guess.scores.size() == 3 && !guess.numbers,
          "guess: " + guess.truth + " / " + guess.predicted);
    TestingMetrics m;
    int64_t left_out = 0;
    std::string error;
    Check(MetricsFromColumns(*table, guess, m, left_out, error), "three classes read as a test: " + error);
    Check(Near(m.test_accuracy, j["accuracy"].get<double>(), 1e-6) && Near(m.macro_f1, j["macro_f1"].get<double>(), 1e-6) &&
              Near(m.weighted_f1, j["weighted_f1"].get<double>(), 1e-6),
          "accuracy / F1 " + Num(m.test_accuracy) + " " + Num(m.macro_f1));
    for (int c = 0; c < 3; ++c) {
        ClassCurves curves;
        Check(CurvesForClass(m, c, curves, error), "curves " + classes[static_cast<size_t>(c)] + ": " + error);
        Check(ClassName(m, c) == classes[static_cast<size_t>(c)] &&
                  Near(curves.roc.auc, j["auc"][classes[static_cast<size_t>(c)]].get<double>(), 1e-6),
              "AUC " + classes[static_cast<size_t>(c)] + " " + Num(curves.roc.auc));
    }
    PredictionColumns one = guess;
    one.scores = {"score_Gentoo"};
    Check(!MetricsFromColumns(*table, one, m, left_out, error) && error.find("one score column fits two classes") == 0,
          "one score for three classes is refused: " + error);

    const auto& r = fixture["regression"];
    const auto numbers = arrow::Table::Make(
        arrow::schema({arrow::field("body_mass_g", arrow::float64()), arrow::field("predicted_body_mass_g", arrow::float64())}),
        {NumberArray(r["truth"]), NumberArray(r["predicted"])});
    const PredictionColumns n = GuessPredictionColumns(*numbers);
    Check(n.predicted == "predicted_body_mass_g" && n.truth.empty(), "no true-looking name: the user picks it");
    PredictionColumns regression;
    regression.truth = "body_mass_g";
    regression.predicted = "predicted_body_mass_g";
    regression.numbers = true;
    Check(MetricsFromColumns(*numbers, regression, m, left_out, error), "regression read as a test: " + error);
    Check(m.regression_mode && Near(m.test_mae, r["mae"].get<double>(), 1e-3) && Near(m.test_rmse, r["rmse"].get<double>(), 1e-3),
          "MAE / RMSE " + Num(m.test_mae) + " " + Num(m.test_rmse));
    Check(Near(RSquared(m), r["r2"].get<double>(), 1e-6), "R-squared " + Num(RSquared(m)));
    const View v = BuildView(m, nullptr);
    Check(v.chips.size() > 1 && v.chips[1].text.rfind("R\xC2\xB2 0.70", 0) == 0, "R-squared chip: " + v.chips[1].text);
    for (const auto& chip : v.chips) Check(chip.text.rfind("loss", 0) != 0, "a table has no loss to show: " + chip.text);
    Check(v.subtitle == "67 test samples", "nor a duration: " + v.subtitle);
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

    // With the rows of the tested table (TOFIX157): each prediction names its
    // dataset row (1-based); a regression writes its rows.
    TestingMetrics rows = SentimentRun(true);
    rows.row_ids = {106, 319, 410, 6};
    rows.source_dataset = "blobs";
    Check(WriteTestResultsCsv(rows, (dir / "rows.csv").string(), &error), "csv with rows: " + error);
    std::stringstream rows_csv;
    {
        std::ifstream in(dir / "rows.csv");
        rows_csv << in.rdbuf();
    }
    Check(Contains(rows_csv.str(), "Tested table,blobs") && Contains(rows_csv.str(), "Index,Dataset row,Predicted") &&
              Contains(rows_csv.str(), "0,107,Normal,Anxiety,no,0.93"),
          "csv predictions carry the dataset row");
    Check(WriteTestResultsJson(rows, (dir / "rows.json").string(), &error), "json with rows: " + error);
    {
        std::ifstream in(dir / "rows.json");
        const auto rj = nlohmann::json::parse(in);
        Check(rj["tested_table"] == "blobs" && rj["predictions"][3]["dataset_row"] == 7, "json predictions carry the dataset row");
    }
    TestingMetrics regression;
    regression.is_complete = true;
    regression.regression_mode = true;
    regression.regression_width = 1;
    regression.regression_truth = {1.0f, 2.0f};
    regression.regression_predictions = {1.5f, 1.0f};
    regression.row_ids = {4, 9};
    Check(WriteTestResultsCsv(regression, (dir / "reg.csv").string(), &error), "regression csv: " + error);
    std::stringstream reg_csv;
    {
        std::ifstream in(dir / "reg.csv");
        reg_csv << in.rdbuf();
    }
    Check(Contains(reg_csv.str(), "Index,Dataset row,True,Predicted,Error") && Contains(reg_csv.str(), "1,10,2,1,-1"),
          "regression csv rows");
    std::filesystem::remove_all(dir);
}

// Export Model: the formats of this build, the default file and its renaming.
void CheckExportModel() {
    using namespace cyxwiz::model_export;
    const auto without_onnx = AvailableFormats(false);
    Check(without_onnx.size() == 2 && without_onnx[0].format == cyxwiz::ModelFormat::CyxModel &&
              without_onnx[1].format == cyxwiz::ModelFormat::Safetensors,
          "without the ONNX exporter: package and safetensors");
    Check(AvailableFormats(true).size() == 3 && !IsAvailable(cyxwiz::ModelFormat::GGUF, true),
          "ONNX when built; GGUF never offered");
    const std::string path = DefaultPath("D:/tmp/project", "D:/tmp/project/cyxgraph/sentiment_lstm.cyxgraph",
                                         cyxwiz::ModelFormat::CyxModel);
    Check(path == (std::filesystem::path("D:/tmp/project") / "exports" / "sentiment_lstm.cyxmodel").string(),
          "default file in the project's exports folder: " + path);
    Check(DefaultPath("", "x.cyxgraph", cyxwiz::ModelFormat::CyxModel).empty(), "no project: no default file");
    Check(std::filesystem::path(WithExtension(path, cyxwiz::ModelFormat::Safetensors)).filename() ==
              "sentiment_lstm.safetensors",
          "switching format renames the extension");
    Check(DefaultName("", "D:/p/cyxgraph/sentiment_lstm.cyxgraph") == "sentiment_lstm" &&
              DefaultName("trained name", "x.cyxgraph") == "trained name",
          "the name: the trained one, else the graph's");
    cyxwiz::ExportResult result;
    result.output_path = "D:/x.cyxmodel";
    result.file_size_bytes = 2 * 1024 * 1024;
    result.num_parameters = 519719;
    result.num_layers = 6;
    result.export_time_ms = 120;
    const auto rows = ResultRows(result);
    Check(rows.size() == 4 && rows[1].second == "2.00 MB" && rows[2].second == "519,719 in 6 layers",
          "result rows");
}


// Open Model File: the exported sentiment package, a weights-only file and an
// unreadable file.
void CheckOpenModel() {
    cyxwiz::ProbeResult probe;
    probe.valid = true;
    probe.format = cyxwiz::ModelFormat::CyxModel;
    probe.format_version = "CYXW v3";
    probe.model_name = "sentiment_analysis_lstm_word_vectors";
    probe.num_parameters = 519719;
    probe.num_layers = 6;
    probe.file_size = 2174719;
    probe.epochs_trained = 1;
    probe.final_accuracy = 0.6104f;
    probe.has_graph = true;
    probe.has_training_history = true;
    probe.has_optimizer_state = true;
    probe.layer_names = {"embedding.weight", "dense.bias"};
    probe.layer_shapes["embedding.weight"] = {5000, 100};
    const model_open::View v = model_open::BuildView(probe);
    Check(v.readable && v.format_line == "CyxWiz package (CYXW v3)", "format line: " + v.format_line);
    Check(v.rows.size() == 5 && v.rows[1].second == "519,719 in 6 layers" &&
              v.rows[2].second == "1 epoch, accuracy 61.04%" &&
              v.rows[3].second == "graph, weights, training history and optimizer state" &&
              v.rows[4].second == "2.07 MB",
          "rows: name, parameters, trained, contains, size");
    Check(v.layers == "embedding.weight [5000, 100], dense.bias", "layers: " + v.layers);
    Check(v.can_open_graph && Contains(v.note, "without its weights"), "a package opens its graph");
    Check(model_open::LayersLine(probe, 1) == "embedding.weight [5000, 100], and 1 more", "layers are capped");

    cyxwiz::ProbeResult weights;
    weights.valid = true;
    weights.format = cyxwiz::ModelFormat::Safetensors;
    weights.format_version = "safetensors";
    const model_open::View w = model_open::BuildView(weights);
    Check(w.format_line == "Weights only (safetensors)" && w.rows[0].second == "Not named" && !w.can_open_graph &&
              w.open_disabled_reason == "This file has no graph",
          "weights only: nothing to open");

    cyxwiz::ProbeResult bad;
    bad.error_message = "Not a model file";
    const model_open::View b = model_open::BuildView(bad);
    Check(!b.readable && b.error == "Not a model file" && !b.can_open_graph, "an unreadable file says why");
}

}  // namespace

int main() {
    CheckExportModel();
    CheckOpenModel();
    CheckOverview();
    CheckSorting();
    CheckRegressionRows();
    CheckThresholds();
    CheckTableAsTest();
    CheckEvaluationTables();
    CheckComparison();
    CheckExports();
    std::cout << "test results presentation: overview, classes, sorting, regression rows, thresholds and tables against scikit-learn, evaluation tables, comparison, exports, export model, open "
                 "model. OK\n";
    return 0;
}
