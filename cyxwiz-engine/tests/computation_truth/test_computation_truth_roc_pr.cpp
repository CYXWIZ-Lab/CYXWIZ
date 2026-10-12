// ROC and precision-recall curves (ModelEvaluation, on the ArrayFire device)
// against scikit-learn (TOFIX140): fixtures/roc_pr_sklearn.json from
// reference/generate_roc_pr_fixtures.py. Binary curves point by point with
// AUC / AP, and one-vs-rest AUC / AP per class with the macro AP. Runs on the
// active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu).
#include "test_device_selection.h"

#include <cyxwiz/model_evaluation.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::ModelEvaluation;

// Scores are float32 on a device without float64; the fixture's two-decimal
// and softmax scores keep their order and ties in float32.
constexpr double kTolerance = 1e-6;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what) {
    Check(std::abs(actual - expected) <= kTolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", sklearn " + std::to_string(expected));
}

void CheckSeries(const std::vector<double>& actual, const json& expected, const std::string& what) {
    const auto values = expected.get<std::vector<double>>();
    Check(actual.size() == values.size(),
          what + ": " + std::to_string(actual.size()) + " points, sklearn " + std::to_string(values.size()));
    for (size_t i = 0; i < values.size(); ++i) CheckValue(actual[i], values[i], what + " [" + std::to_string(i) + "]");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside =
        std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "roc_pr_sklearn.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_ROC_PR_FIXTURE);
}

void RunBinary(const json& c, const std::string& name) {
    const auto y = c.at("y_true").get<std::vector<int>>();
    const auto s = c.at("y_scores").get<std::vector<double>>();

    const auto roc = ModelEvaluation::ComputeROC(y, s);
    Check(roc.success, name + " roc: " + roc.error_message);
    const json& er = c.at("roc");
    CheckSeries(roc.fpr, er.at("fpr"), name + " fpr");
    CheckSeries(roc.tpr, er.at("tpr"), name + " tpr");
    Check(std::isinf(roc.thresholds.front()), name + " first ROC threshold is +inf");
    CheckSeries(std::vector<double>(roc.thresholds.begin() + 1, roc.thresholds.end()), er.at("thresholds"),
                name + " roc thresholds");
    CheckValue(roc.auc, er.at("auc").get<double>(), name + " AUC");

    const auto pr = ModelEvaluation::ComputePRCurve(y, s);
    Check(pr.success, name + " pr: " + pr.error_message);
    const json& ep = c.at("pr");
    CheckSeries(pr.precision, ep.at("precision"), name + " precision");
    CheckSeries(pr.recall, ep.at("recall"), name + " recall");
    CheckSeries(pr.thresholds, ep.at("thresholds"), name + " pr thresholds");
    CheckValue(pr.average_precision, ep.at("ap").get<double>(), name + " AP");
}

void RunMulticlass(const json& c, const std::string& name) {
    const auto y = c.at("y_true").get<std::vector<int>>();
    const auto s = c.at("y_scores").get<std::vector<std::vector<double>>>();

    const auto roc = ModelEvaluation::ComputeMulticlassROC(y, s);
    Check(roc.success, name + " roc: " + roc.error_message);
    CheckSeries(roc.class_auc, c.at("class_auc"), name + " class AUC");

    const auto pr = ModelEvaluation::ComputeMulticlassPRCurve(y, s);
    Check(pr.success, name + " pr: " + pr.error_message);
    CheckSeries(pr.class_ap, c.at("class_ap"), name + " class AP");
    CheckValue(pr.average_precision, c.at("macro_ap").get<double>(), name + " macro AP");
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the ROC / PR fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("op").get<std::string>() + " " + c.at("label").get<std::string>();
        if (c.at("op") == "binary") {
            RunBinary(c, name);
        } else {
            RunMulticlass(c, name);
        }
        ++passed;
    }
    std::cout << "ROC / PR curves match scikit-learn: " << passed << " cases" << std::endl;
    return 0;
}
