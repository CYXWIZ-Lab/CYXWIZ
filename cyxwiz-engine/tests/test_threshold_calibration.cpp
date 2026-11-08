// Results > Test > Threshold and Calibration (TOFIX129 D11) against
// scikit-learn / scipy (generate_threshold_calibration_fixtures.py, on the
// penguins predictions of fixtures/evaluation_sklearn.json): the threshold
// sweep, the recall target and lowest-cost thresholds, the confidence floor,
// reliability bins and ECE, temperature scaling, Platt's sigmoid and
// isotonic regression, and the Calibration section's words.
#include "../src/core/test_results_presentation.h"
#include "../src/core/threshold_calibration.h"

#include <cyxwiz/model_evaluation.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
using namespace cyxwiz::test_report;

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

bool Near(double a, double b, double tolerance = 1e-9) {
    return std::abs(a - b) <= tolerance * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

std::string Num(double v) {
    char text[40];
    std::snprintf(text, sizeof(text), "%.10g", v);
    return text;
}

// The binary test as Test keeps it: P(FEMALE), P(MALE) per row, MALE = 1.
TestingMetrics BinaryRun(const nlohmann::json& source) {
    const auto& b = source.at("binary");
    TestingMetrics m;
    const auto truth = b.at("truth").get<std::vector<std::string>>();
    const auto scores = b.at("score_male").get<std::vector<double>>();
    m.probability_classes = 2;
    for (size_t i = 0; i < truth.size(); ++i) {
        const int t = truth[i] == "MALE" ? 1 : 0;
        m.ground_truth.push_back(t);
        m.predictions.push_back(scores[i] >= 0.5 ? 1 : 0);
        m.confidences.push_back(static_cast<float>(std::max(scores[i], 1.0 - scores[i])));
        m.class_probabilities.push_back(static_cast<float>(1.0 - scores[i]));
        m.class_probabilities.push_back(static_cast<float>(scores[i]));
    }
    return m;
}

TestingMetrics SpeciesRun(const nlohmann::json& truth_fixture) {
    TestingMetrics m;
    const auto probabilities = truth_fixture.at("multiclass_probabilities").get<std::vector<std::vector<double>>>();
    m.ground_truth = truth_fixture.at("multiclass_truth").get<std::vector<int>>();
    m.probability_classes = static_cast<int>(probabilities.front().size());
    for (const auto& row : probabilities) {
        const auto top = std::max_element(row.begin(), row.end());
        m.predictions.push_back(static_cast<int>(top - row.begin()));
        m.confidences.push_back(static_cast<float>(*top));
        for (double p : row) m.class_probabilities.push_back(static_cast<float>(p));
    }
    for (const auto& name : truth_fixture.at("classes")) {
        ClassMetrics c;
        c.class_name = name.get<std::string>();
        m.per_class_metrics.push_back(c);
    }
    return m;
}

void CheckThreshold(const nlohmann::json& source, const nlohmann::json& f) {
    const auto& b = source.at("binary");
    std::vector<int> truth;
    for (const auto& t : b.at("truth")) truth.push_back(t.get<std::string>() == "MALE" ? 1 : 0);
    const auto scores = b.at("score_male").get<std::vector<double>>();
    const int positives = static_cast<int>(std::count(truth.begin(), truth.end(), 1));
    const int negatives = static_cast<int>(truth.size()) - positives;
    const auto roc = ModelEvaluation::ComputeROC(truth, scores);
    const auto sweep = SweepThresholds(roc, positives, negatives);
    Check(!sweep.thresholds.empty() && std::is_sorted(sweep.thresholds.begin(), sweep.thresholds.end()),
          "the sweep runs up the scores");
    int matched = 0;
    for (const auto& want : f.at("sweep")) {
        const double t = want.at("threshold").get<double>();
        for (size_t i = 0; i < sweep.thresholds.size(); ++i) {
            if (!Near(sweep.thresholds[i], t, 1e-12)) continue;
            ++matched;
            Check(Near(sweep.precision[i], want.at("precision").get<double>()) &&
                      Near(sweep.recall[i], want.at("recall").get<double>()) && Near(sweep.f1[i], want.at("f1").get<double>()),
                  "sweep figures at " + Num(t));
        }
    }
    Check(matched == static_cast<int>(sweep.thresholds.size()), "every sweep point is one of scikit-learn's: " +
                                                                   std::to_string(matched) + " of " +
                                                                   std::to_string(sweep.thresholds.size()));
    const double recall = ThresholdForRecall(roc, positives, negatives, 0.9);
    Check(Near(recall, f.at("recall_0_9").get<double>(), 1e-12), "recall >= 0.9 at " + Num(recall));
    const double cost = LowestCostThreshold(roc, positives, negatives, 1.0, 5.0);
    Check(Near(cost, f.at("cost_1_5").get<double>(), 1e-12), "lowest cost of 1 x FP + 5 x FN at " + Num(cost));
}

void CheckFloor(const nlohmann::json& f) {
    const TestingMetrics m = SpeciesRun(f);
    for (const auto& want : f.at("floor")) {
        const auto at = AtConfidenceFloor(m, want.at("floor").get<double>());
        Check(at.answered == want.at("answered").get<int>() && Near(at.accuracy, want.at("accuracy").get<double>(), 1e-9),
              "confidence floor " + Num(at.floor) + ": " + std::to_string(at.answered) + " answered");
    }
    Check(ConfidenceFloorCurve(m).size() == 100, "the floor curve, 0.00 to 0.99");
}

void CheckReliability(const char* what, const Reliability& r, const nlohmann::json& want) {
    Check(Near(r.ece, want.at("ece").get<double>(), 1e-6), std::string(what) + ": ECE " + Num(r.ece));
    for (size_t k = 0; k < r.bins.size() && k < want.at("bins").size(); ++k) {
        const auto& w = want.at("bins").at(k);
        Check(r.bins[k].count == w.at("count").get<int>() && Near(r.bins[k].confidence, w.at("confidence").get<double>(), 1e-6) &&
                  Near(r.bins[k].accuracy, w.at("accuracy").get<double>(), 1e-9),
              std::string(what) + ": bin " + std::to_string(k));
    }
}

void CheckCalibration(const nlohmann::json& source, const nlohmann::json& f) {
    const TestingMetrics binary = BinaryRun(source);
    // Reliability of P(MALE) against "is MALE", all rows.
    std::vector<double> p;
    std::vector<int> male;
    for (size_t i = 0; i < binary.ground_truth.size(); ++i) {
        p.push_back(binary.class_probabilities[2 * i + 1]);
        male.push_back(binary.ground_truth[i]);
    }
    CheckReliability("yes / no reliability", ReliabilityOf(p, male), f.at("binary_reliability_all"));

    const auto platt = Calibrate(binary, CalibrationMethod::Platt, 1);
    Check(platt.ok && platt.yes_no && platt.fit_rows == 34 && platt.check_rows == 33, "Platt on the even rows: " + platt.error);
    Check(Near(platt.platt.a, f.at("platt").at("a").get<double>(), 1e-5) && Near(platt.platt.b, f.at("platt").at("b").get<double>(), 1e-5),
          "Platt a = " + Num(platt.platt.a) + ", b = " + Num(platt.platt.b));
    const auto platt_check = f.at("platt").at("check").get<std::vector<double>>();
    for (size_t k = 0; k < platt_check.size(); ++k)
        Check(Near(ApplyPlatt(platt.platt, binary.class_probabilities[2 * (2 * k + 1) + 1]), platt_check[k], 1e-5),
              "Platt check row " + std::to_string(k));
    Check(Near(platt.before.ece, f.at("platt").at("ece_before").get<double>(), 1e-6) &&
              Near(platt.after.ece, f.at("platt").at("ece_after").get<double>(), 1e-4),
          "Platt ECE " + Num(platt.before.ece) + " -> " + Num(platt.after.ece));

    const auto iso = Calibrate(binary, CalibrationMethod::Isotonic, 1);
    const auto iso_check = f.at("isotonic").at("check").get<std::vector<double>>();
    for (size_t k = 0; k < iso_check.size(); ++k)
        Check(Near(ApplyIsotonic(iso.isotonic, binary.class_probabilities[2 * (2 * k + 1) + 1]), iso_check[k], 1e-6),
              "isotonic check row " + std::to_string(k));
    Check(Near(iso.after.ece, f.at("isotonic").at("ece_after").get<double>(), 1e-6), "isotonic ECE after " + Num(iso.after.ece));

    const TestingMetrics species = SpeciesRun(f);
    std::vector<double> top;
    std::vector<int> right;
    for (size_t i = 0; i < species.predictions.size(); ++i) {
        top.push_back(species.confidences[i]);
        right.push_back(species.predictions[i] == species.ground_truth[i]);
    }
    CheckReliability("several-class reliability", ReliabilityOf(top, right), f.at("multiclass_reliability_all"));
    const auto temperature = Calibrate(species, CalibrationMethod::Temperature);
    const auto& t = f.at("temperature");
    Check(temperature.ok && !temperature.yes_no && Near(temperature.temperature, t.at("T").get<double>(), 1e-5),
          "temperature " + Num(temperature.temperature) + ", scipy " + Num(t.at("T").get<double>()));
    Check(Near(temperature.before.ece, t.at("ece_before").get<double>(), 1e-6) &&
              Near(temperature.after.ece, t.at("ece_after").get<double>(), 1e-5),
          "temperature ECE " + Num(temperature.before.ece) + " -> " + Num(temperature.after.ece));
    std::vector<size_t> fit_rows;
    for (size_t i = 0; i < species.ground_truth.size(); i += 2) fit_rows.push_back(i);
    const auto nll = FitTemperature(species.class_probabilities, species.probability_classes, species.ground_truth, fit_rows);
    Check(Near(nll.nll_before, t.at("nll_before").get<double>(), 1e-6) && Near(nll.nll_after, t.at("nll_after").get<double>(), 1e-6),
          "NLL before and after the temperature");
    Check(temperature.detail.find("One temperature (") != std::string::npos &&
              temperature.detail.find("without changing a single prediction.") != std::string::npos,
          "temperature words: " + temperature.detail);
    // Platt / isotonic on several classes fall back to the temperature.
    Check(Calibrate(species, CalibrationMethod::Isotonic).method == CalibrationMethod::Temperature,
          "several classes: temperature scaling");
    TestingMetrics bare;
    bare.predictions = {0, 1};
    bare.ground_truth = {0, 1};
    Check(!Calibrate(bare, CalibrationMethod::Temperature).ok, "no probabilities: no calibration, with a reason");
}

}  // namespace

int main() {
    std::ifstream source_in(CYXWIZ_EVALUATION_FIXTURE), truth_in(CYXWIZ_THRESHOLD_CALIBRATION_FIXTURE);
    Check(source_in.good() && truth_in.good(), "the fixtures open");
    if (!source_in.good() || !truth_in.good()) return 1;
    const auto source = nlohmann::json::parse(source_in);
    const auto truth = nlohmann::json::parse(truth_in);
    CheckThreshold(source, truth);
    CheckFloor(truth);
    CheckCalibration(source, truth);
    if (failures > 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "threshold and calibration: " << checks << " checks passed against scikit-learn / scipy\n";
    return 0;
}
