// Classical supervised models (Group C classic ML nodes) against scikit-learn
// 1.8 (TOFIX145, classic ML on the device).
//
// fixtures/classic_ml_sklearn.json (generate_classic_ml_fixtures.py). Runs on the
// active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu).
#include "test_device_selection.h"

#include <cyxwiz/classic_ml.h>

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
using Matrix = std::vector<std::vector<double>>;
using cyxwiz::ClassicML;

constexpr double kTolerance = 1e-8;  // float64 on the device

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what, double tolerance = kTolerance) {
    Check(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", sklearn " + std::to_string(expected));
}

void CheckMatrix(const Matrix& actual, const json& expected, const std::string& what, double tolerance = kTolerance) {
    const auto rows = expected.get<Matrix>();
    Check(actual.size() == rows.size(), what + ": rows " + std::to_string(actual.size()));
    for (size_t r = 0; r < rows.size(); ++r) {
        Check(actual[r].size() == rows[r].size(), what + ": cols");
        for (size_t c = 0; c < rows[r].size(); ++c)
            CheckValue(actual[r][c], rows[r][c], what + "(" + std::to_string(r) + "," + std::to_string(c) + ")", tolerance);
    }
}

void CheckLabels(const std::vector<int>& actual, const json& expected, const std::string& what) {
    const auto labels = expected.get<std::vector<int>>();
    Check(actual.size() == labels.size(), what + ": count");
    for (size_t i = 0; i < labels.size(); ++i)
        Check(actual[i] == labels[i], what + " [" + std::to_string(i) + "]: " + std::to_string(actual[i]) + ", sklearn " +
                                          std::to_string(labels[i]));
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside =
        std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "classic_ml_sklearn.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_CLASSIC_ML_FIXTURE);
}

void RunGaussianNB(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    const auto y = c.at("y").get<std::vector<int>>();
    const auto x_test = c.at("x_test").get<Matrix>();
    const auto model = ClassicML::FitGaussianNB(x, y, c.at("n_classes").get<int>(), c.at("var_smoothing").get<double>());
    Check(model.success, name + " fit: " + model.error_message);
    CheckMatrix(model.theta, c.at("theta"), name + " theta");
    CheckMatrix(model.var, c.at("var"), name + " var");
    const auto prior = c.at("class_prior").get<std::vector<double>>();
    for (size_t i = 0; i < prior.size(); ++i) CheckValue(model.class_prior[i], prior[i], name + " prior");
    CheckValue(model.epsilon, c.at("epsilon").get<double>(), name + " epsilon");

    const auto train = ClassicML::PredictGaussianNB(model, x);
    Check(train.success, name + " predict: " + train.error_message);
    CheckLabels(train.labels, c.at("predict"), name + " predict");
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", 1e-7);
    const auto held_out = ClassicML::PredictGaussianNB(model, x_test);
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test", 1e-7);
}

void RunKNN(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    const auto model = ClassicML::FitKNN(x, c.at("y").get<std::vector<int>>(), c.at("n_classes").get<int>(),
                                         c.at("n_neighbors").get<int>(), c.at("weights").get<std::string>(),
                                         c.at("metric").get<std::string>());
    Check(model.success, name + " fit: " + model.error_message);
    const auto train = ClassicML::PredictKNN(model, x);
    Check(train.success, name + " predict: " + train.error_message);
    CheckLabels(train.labels, c.at("predict"), name + " predict");
    // Distance weighting on the training rows: sklearn's euclidean expansion
    // (|a|^2 + |b|^2 - 2ab) leaves a row's distance to itself near 1e-8, not 0,
    // so its share is 1 - ~5e-8 where the exact zero distance gives 1.
    const double train_tolerance = c.at("weights") == "distance" ? 1e-6 : kTolerance;
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", train_tolerance);
    const auto held_out = ClassicML::PredictKNN(model, c.at("x_test").get<Matrix>());
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test");
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the classic ML fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string op = c.at("op").get<std::string>();
        const std::string name = op + " " + c.at("label").get<std::string>();
        if (op == "gaussian_nb") RunGaussianNB(c, name);
        else if (op == "knn") RunKNN(c, name);
        else Check(false, "unknown case " + op);
        ++passed;
    }

    // Refusals: a class with no rows, a label out of range, a feature-count mismatch.
    Check(!ClassicML::FitGaussianNB({{0.0}, {1.0}}, {0, 0}, 2).success, "an empty class is refused");
    Check(!ClassicML::FitGaussianNB({{0.0}, {1.0}}, {0, 3}, 2).success, "a label out of range is refused");
    const auto fitted = ClassicML::FitGaussianNB({{0.0, 1.0}, {1.0, 0.0}}, {0, 1}, 2);
    Check(fitted.success && !ClassicML::PredictGaussianNB(fitted, {{1.0}}).success, "a feature-count mismatch is refused");
    Check(!ClassicML::FitKNN({{0.0}, {1.0}}, {0, 1}, 2, 3).success, "more neighbours than rows is refused");
    Check(!ClassicML::FitKNN({{0.0}, {1.0}}, {0, 1}, 2, 1, "median").success, "an unknown weighting is refused");
    Check(!ClassicML::FitKNN({{0.0}, {1.0}}, {0, 1}, 2, 1, "uniform", "chebyshev").success, "an unknown metric is refused");

    std::cout << "classic ML matches scikit-learn: " << passed << " cases" << std::endl;
    return 0;
}
