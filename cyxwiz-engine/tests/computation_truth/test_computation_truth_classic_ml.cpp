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
using cyxwiz::test::DeviceTolerance;

// Closed-form values: float64 rounding, or float32 rounding (worst measured on
// the Intel Iris Xe, TOFIX145 item 7: 1.4e-7).
double Tolerance() { return DeviceTolerance(1e-8, 5e-5); }
// Values that went through a solver or a distance expansion.
double SolvedTolerance() { return DeviceTolerance(1e-6, 1e-3); }

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what, double tolerance = Tolerance()) {
    Check(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", sklearn " + std::to_string(expected));
}

void CheckMatrix(const Matrix& actual, const json& expected, const std::string& what, double tolerance = Tolerance()) {
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
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", DeviceTolerance(1e-7, 5e-5));
    const auto held_out = ClassicML::PredictGaussianNB(model, x_test);
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test",
                DeviceTolerance(1e-7, 5e-5));
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
    const double train_tolerance = c.at("weights") == "distance" ? SolvedTolerance() : Tolerance();
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", train_tolerance);
    const auto held_out = ClassicML::PredictKNN(model, c.at("x_test").get<Matrix>());
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test");
}

void RunLogistic(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    // The reference is converged to 1e-12; so is the fit here (1e-6 in float32).
    const auto model = ClassicML::FitLogistic(x, c.at("y").get<std::vector<int>>(), c.at("n_classes").get<int>(),
                                              c.at("C").get<double>(), 100000,
                                              DeviceTolerance(1e-12, 1e-6));
    Check(model.success, name + " fit: " + model.error_message);
    CheckMatrix(model.coef, c.at("coef"), name + " coef", SolvedTolerance());
    const auto intercept = c.at("intercept").get<std::vector<double>>();
    Check(model.intercept.size() == intercept.size(), name + " intercept count");
    for (size_t i = 0; i < intercept.size(); ++i) CheckValue(model.intercept[i], intercept[i], name + " intercept", SolvedTolerance());
    const auto train = ClassicML::PredictLogistic(model, x);
    Check(train.success, name + " predict: " + train.error_message);
    CheckLabels(train.labels, c.at("predict"), name + " predict");
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", SolvedTolerance());
    const auto held_out = ClassicML::PredictLogistic(model, c.at("x_test").get<Matrix>());
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test", SolvedTolerance());
}

double SvmGamma(const json& c, const Matrix& x) {
    const auto& params = c.at("params");
    std::string gamma = "scale";
    if (params.contains("gamma")) gamma = params.at("gamma").is_string() ? params.at("gamma").get<std::string>()
                                                                          : std::to_string(params.at("gamma").get<double>());
    const double value = ClassicML::ResolveSVMGamma(x, gamma);
    CheckValue(value, c.at("gamma_value").get<double>(), "gamma " + gamma);
    return value;
}

std::string SvmKernel(const json& c) { return c.at("params").value("kernel", std::string("rbf")); }
int SvmDegree(const json& c) { return c.at("params").value("degree", 3); }
double SvmCoef0(const json& c) { return c.at("params").value("coef0", 0.0); }

void RunSVC(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    const auto model = ClassicML::FitSVC(x, c.at("y").get<std::vector<int>>(), c.at("n_classes").get<int>(),
                                         c.at("params").value("C", 1.0), SvmKernel(c), SvmGamma(c, x), SvmDegree(c),
                                         SvmCoef0(c), DeviceTolerance(1e-9, 1e-4));
    Check(model.success && model.converged, name + " fit: " + model.error_message);
    // Support vectors per class, as sklearn's n_support_.
    const auto n_support = c.at("n_support").get<std::vector<int>>();
    const auto labels = c.at("y").get<std::vector<int>>();
    std::vector<int> per_class(n_support.size(), 0);
    for (const auto& row : model.support_vectors) {
        for (size_t r = 0; r < x.size(); ++r) {
            if (x[r] == row) {
                ++per_class[static_cast<size_t>(labels[r])];
                break;
            }
        }
    }
    for (size_t k = 0; k < n_support.size(); ++k)
        Check(per_class[k] == n_support[k], name + " support vectors of class " + std::to_string(k) + ": " +
                                                std::to_string(per_class[k]) + ", sklearn " + std::to_string(n_support[k]));
    const auto train = ClassicML::PredictSVC(model, x);
    Check(train.success, name + " predict: " + train.error_message);
    CheckLabels(train.labels, c.at("predict"), name + " predict");
    const auto x_test = c.at("x_test").get<Matrix>();
    const auto held_out = ClassicML::PredictSVC(model, x_test);
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckLabels(held_out.labels, c.at("predict_test"), name + " predict test");
    std::string error;
    CheckMatrix(ClassicML::SVMDecision(model, x_test, &error), c.at("decision_test"), name + " decision " + error, SolvedTolerance());
}

// SVC(probability=True): Platt sigmoids per class pair from 5-fold decision
// values, the folds drawn by libsvm's generator from scikit-learn's seed.
void RunSVCProba(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    const auto labels = c.at("y").get<std::vector<int>>();
    const double C = c.at("params").value("C", 1.0);
    const double tol = DeviceTolerance(1e-9, 1e-4);
    auto model = ClassicML::FitSVC(x, labels, c.at("n_classes").get<int>(), C, SvmKernel(c), SvmGamma(c, x),
                                   SvmDegree(c), SvmCoef0(c), tol);
    Check(model.success, name + " fit: " + model.error_message);
    const unsigned seed = ClassicML::SVCProbabilitySeed(c.at("random_state").get<unsigned>());
    Check(seed == c.at("libsvm_seed").get<unsigned>(),
          name + " libsvm seed " + std::to_string(seed) + ", sklearn " + c.at("libsvm_seed").dump());
    std::string error;
    Check(ClassicML::FitSVCProbability(model, x, labels, C, tol, seed, &error), name + " probability fit: " + error);
    const auto prob_a = c.at("prob_a").get<std::vector<double>>(), prob_b = c.at("prob_b").get<std::vector<double>>();
    Check(model.prob_a.size() == prob_a.size() && model.prob_b.size() == prob_b.size(), name + " sigmoid count");
    for (size_t p = 0; p < prob_a.size(); ++p) {
        CheckValue(model.prob_a[p], prob_a[p], name + " probA", SolvedTolerance());
        CheckValue(model.prob_b[p], prob_b[p], name + " probB", SolvedTolerance());
    }
    const auto train = ClassicML::PredictSVC(model, x);
    Check(train.success, name + " predict: " + train.error_message);
    CheckLabels(train.labels, c.at("predict"), name + " predict (votes, not the probabilities)");
    CheckMatrix(train.probabilities, c.at("predict_proba"), name + " predict_proba", SolvedTolerance());
    const auto held_out = ClassicML::PredictSVC(model, c.at("x_test").get<Matrix>());
    Check(held_out.success, name + " predict test: " + held_out.error_message);
    CheckMatrix(held_out.probabilities, c.at("predict_proba_test"), name + " predict_proba test", SolvedTolerance());
}

void RunSVR(const json& c, const std::string& name) {
    const auto x = c.at("x").get<Matrix>();
    const auto& params = c.at("params");
    const auto model = ClassicML::FitSVR(x, c.at("t").get<std::vector<double>>(), params.value("C", 1.0),
                                         params.value("epsilon", 0.1), SvmKernel(c), SvmGamma(c, x), SvmDegree(c),
                                         SvmCoef0(c), DeviceTolerance(1e-9, 1e-4));
    Check(model.success && model.converged, name + " fit: " + model.error_message);
    Check(static_cast<int>(model.support_vectors.size()) == c.at("n_support").get<int>(),
          name + " support vectors: " + std::to_string(model.support_vectors.size()));
    CheckValue(-model.rho.front(), c.at("intercept").get<double>(), name + " intercept", SolvedTolerance());
    std::string error;
    const auto train = ClassicML::PredictSVR(model, x, &error);
    Check(!train.empty(), name + " predict: " + error);
    const auto expected = c.at("predict").get<std::vector<double>>();
    for (size_t i = 0; i < expected.size(); ++i) CheckValue(train[i], expected[i], name + " predict", SolvedTolerance());
    const auto held_out = ClassicML::PredictSVR(model, c.at("x_test").get<Matrix>(), &error);
    const auto expected_test = c.at("predict_test").get<std::vector<double>>();
    Check(held_out.size() == expected_test.size(), name + " predict test: " + error);
    for (size_t i = 0; i < expected_test.size(); ++i)
        CheckValue(held_out[i], expected_test[i], name + " predict test", SolvedTolerance());
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
        else if (op == "logistic") RunLogistic(c, name);
        else if (op == "svc") RunSVC(c, name);
        else if (op == "svc_proba") RunSVCProba(c, name);
        else if (op == "svr") RunSVR(c, name);
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
    Check(!ClassicML::FitLogistic({{0.0}, {1.0}}, {0, 0}, 1).success, "one class is refused");
    Check(!ClassicML::FitLogistic({{0.0}, {1.0}}, {0, 1}, 2, 0.0).success, "C = 0 is refused");
    const auto short_fit = ClassicML::FitLogistic({{0.0}, {0.4}, {0.6}, {1.0}}, {0, 1, 0, 1}, 2, 1.0, 1, 1e-12);
    Check(short_fit.success && !short_fit.converged && short_fit.n_iter == 1, "max_iter stops the fit unconverged");
    Check(!ClassicML::FitSVC({{0.0}, {1.0}}, {0, 1}, 2, 1.0, "laplacian").success, "an unknown kernel is refused");
    Check(!ClassicML::FitSVC({{0.0}, {1.0}}, {0, 1}, 2, 1.0, "rbf", 0.0).success, "gamma = 0 is refused");
    Check(!ClassicML::FitSVR({{0.0}, {1.0}}, {0.0, 1.0}, 1.0, -0.1).success, "a negative epsilon is refused");
    std::string probability_error;
    cyxwiz::SVMModel unfitted;
    Check(!ClassicML::FitSVCProbability(unfitted, {{0.0}}, {0}, 1.0, 1e-3, 0, &probability_error) &&
              !probability_error.empty(),
          "probabilities need a fitted SVC");
    Check(ClassicML::ResolveSVMGamma({{1.0, 2.0}}, "auto") == 0.5 &&
              ClassicML::ResolveSVMGamma({{1.0, 2.0}}, "0.25") == 0.25 &&
              ClassicML::ResolveSVMGamma({{1.0, 2.0}}, "fast") == 0.0 &&
              ClassicML::ResolveSVMGamma({{3.0, 3.0}}, "scale") == 1.0,
          "gamma auto, a number, an unreadable word, and scale on constant data");

    std::cout << "classic ML matches scikit-learn: " << passed << " cases" << std::endl;
    return 0;
}
