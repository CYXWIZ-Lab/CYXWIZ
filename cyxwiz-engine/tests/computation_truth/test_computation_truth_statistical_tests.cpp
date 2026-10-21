// cyxwiz::StatisticalTests (the Explore Tests and Regression tabs, the
// regression nodes; TOFIX129 D3) against scipy and statsmodels.
//
// fixtures/statistical_tests_scipy.json (generate_statistical_tests_fixtures.py).
// Runs on the active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu).
#include "test_device_selection.h"

#include <cyxwiz/statistical_tests.h>

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
using Vec = std::vector<double>;
using Groups = std::vector<Vec>;
using cyxwiz::StatisticalTests;
using cyxwiz::test::DeviceTolerance;

double Tolerance() { return DeviceTolerance(1e-8, 5e-5); }

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

// Relative for values above 1, absolute below; a p-value is compared on its
// own scale (tiny p must match in relative terms, not just be "near 0").
void CheckValue(double actual, double expected, const std::string& what, double tolerance = Tolerance()) {
    Check(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", reference " + std::to_string(expected));
}

void CheckP(double actual, double expected, const std::string& what, double tolerance = Tolerance()) {
    if (expected < 1e-6) {
        Check(expected == 0.0 ? actual < 1e-300 : std::abs(actual - expected) <= 1e3 * tolerance * expected,
              what + ": p " + std::to_string(actual) + ", reference " + std::to_string(expected) + " (relative)");
        return;
    }
    CheckValue(actual, expected, what, tolerance);
}

void CheckVec(const Vec& actual, const json& expected, const std::string& what, bool p = false) {
    const Vec want = expected.get<Vec>();
    Check(actual.size() == want.size(), what + ": length " + std::to_string(actual.size()));
    for (size_t i = 0; i < want.size(); ++i) {
        const std::string label = what + "[" + std::to_string(i) + "]";
        if (p) CheckP(actual[i], want[i], label);
        else CheckValue(actual[i], want[i], label);
    }
}

void CheckTest(const cyxwiz::TestResult& r, const json& c, const std::string& name) {
    Check(r.ok, name + ": " + r.error);
    if (c.contains("statistic")) CheckValue(r.statistic, c.at("statistic").get<double>(), name + " statistic");
    if (c.contains("df")) CheckValue(r.df, c.at("df").get<double>(), name + " df");
    if (c.contains("df2")) CheckValue(r.df2, c.at("df2").get<double>(), name + " df2");
    if (c.contains("p")) CheckP(r.p, c.at("p").get<double>(), name + " p");
    if (c.contains("estimate")) CheckValue(r.estimate, c.at("estimate").get<double>(), name + " estimate");
    if (c.contains("ci")) {
        CheckValue(r.ci_low, c.at("ci")[0].get<double>(), name + " ci low");
        CheckValue(r.ci_high, c.at("ci")[1].get<double>(), name + " ci high");
    }
    if (c.contains("effect")) CheckValue(r.effect, c.at("effect").get<double>(), name + " effect");
}

void CheckRefused(const std::string& error, const json& c, const std::string& name) {
    const std::string message = c.at("message").get<std::string>();
    Check(error.find(message) != std::string::npos, name + ": expected a refusal containing '" + message + "', got '" + error + "'");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "statistical_tests_scipy.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_STATISTICAL_TESTS_FIXTURE);
}

void RunCase(const json& c, const std::string& name) {
    const std::string op = c.at("op").get<std::string>();
    const auto arg = [&](size_t i) { return c.at("args")[i].get<double>(); };
    // The distributions are host scalar functions: float64 everywhere.
    if (op == "t_sf") return CheckP(StatisticalTests::StudentTSf(arg(0), arg(1)), c.at("want"), name, 1e-9);
    if (op == "t_ppf") return CheckValue(StatisticalTests::StudentTPpf(arg(0), arg(1)), c.at("want"), name, 1e-9);
    if (op == "f_sf") return CheckP(StatisticalTests::FSf(arg(0), arg(1), arg(2)), c.at("want"), name, 1e-9);
    if (op == "chi2_sf") return CheckP(StatisticalTests::ChiSquareSf(arg(0), arg(1)), c.at("want"), name, 1e-9);
    if (op == "normal_sf") return CheckP(StatisticalTests::NormalSf(arg(0)), c.at("want"), name, 1e-12);
    // R's ptukey is accurate to about 1e-7; scipy integrates its own way.
    if (op == "tukey_sf") return CheckValue(StatisticalTests::StudentizedRangeSf(arg(0), arg(1), arg(2)), c.at("want"), name, 2e-6);

    if (op == "one_sample_t") {
        const Vec x = c.at("x").get<Vec>();
        CheckTest(StatisticalTests::OneSampleT(x, c.at("mu").get<double>()), c, name);
        const auto r90 = StatisticalTests::OneSampleT(x, c.at("mu").get<double>(), 0.90);
        CheckValue(r90.ci_low, c.at("ci90")[0].get<double>(), name + " ci90 low");
        return CheckValue(r90.ci_high, c.at("ci90")[1].get<double>(), name + " ci90 high");
    }
    if (op == "two_sample_t")
        return CheckTest(StatisticalTests::TwoSampleT(c.at("a").get<Vec>(), c.at("b").get<Vec>(), c.at("equal_var").get<bool>()), c, name);
    if (op == "paired_t") return CheckTest(StatisticalTests::PairedT(c.at("a").get<Vec>(), c.at("b").get<Vec>()), c, name);
    if (op == "anova")
        return CheckTest(StatisticalTests::OneWayAnova(c.at("groups").get<Groups>(), c.at("welch").get<bool>()), c, name);
    if (op == "levene") return CheckTest(StatisticalTests::Levene(c.at("groups").get<Groups>()), c, name);
    if (op == "kruskal") return CheckTest(StatisticalTests::KruskalWallis(c.at("groups").get<Groups>()), c, name);
    if (op == "mann_whitney") return CheckTest(StatisticalTests::MannWhitneyU(c.at("a").get<Vec>(), c.at("b").get<Vec>()), c, name);
    if (op == "tukey") {
        std::string error;
        const auto pairs = StatisticalTests::TukeyHsd(c.at("groups").get<Groups>(), 0.95, &error);
        const auto& want = c.at("pairs");
        Check(pairs.size() == want.size(), name + ": pair count (" + error + ")");
        for (size_t i = 0; i < pairs.size(); ++i) {
            const std::string pn = name + " pair " + std::to_string(i);
            Check(pairs[i].a == want[i][0].get<size_t>() && pairs[i].b == want[i][1].get<size_t>(), pn + ": order");
            CheckValue(pairs[i].difference, want[i][2].get<double>(), pn + " difference");
            CheckValue(pairs[i].ci_low, want[i][3].get<double>(), pn + " ci low", 2e-6);
            CheckValue(pairs[i].ci_high, want[i][4].get<double>(), pn + " ci high", 2e-6);
            CheckValue(pairs[i].p, want[i][5].get<double>(), pn + " p", 2e-6);
        }
        return;
    }
    if (op == "chi_square") {
        const auto r = StatisticalTests::ChiSquareIndependence(c.at("table").get<Groups>());
        CheckTest(r.test, c, name);
        CheckValue(r.min_expected, c.at("min_expected").get<double>(), name + " min expected");
        const Groups e = c.at("expected").get<Groups>();
        for (size_t i = 0; i < e.size(); ++i)
            for (size_t j = 0; j < e[i].size(); ++j) CheckValue(r.expected[i][j], e[i][j], name + " expected");
        return;
    }
    if (op == "ols" || op == "poly") {
        const auto fit = op == "ols"
            ? StatisticalTests::FitLinear(c.at("columns").get<Groups>(), c.at("y").get<Vec>(), c.at("names").get<std::vector<std::string>>())
            : StatisticalTests::FitPolynomial(c.at("x").get<Vec>(), c.at("y").get<Vec>(), c.at("degree").get<int>(), "x");
        Check(fit.ok, name + ": " + fit.error);
        CheckVec(fit.coef, c.at("coef"), name + " coef");
        CheckVec(fit.se, c.at("se"), name + " se");
        CheckVec(fit.p, c.at("p"), name + " p", true);
        CheckVec(fit.ci_low, c.at("ci_low"), name + " ci low");
        CheckVec(fit.ci_high, c.at("ci_high"), name + " ci high");
        CheckValue(fit.r2, c.at("r2").get<double>(), name + " r2");
        CheckValue(fit.aic, c.at("aic").get<double>(), name + " aic");
        if (op == "ols") {
            CheckVec(fit.t, c.at("t"), name + " t");
            CheckValue(fit.adj_r2, c.at("adj_r2").get<double>(), name + " adj r2");
            CheckValue(fit.f, c.at("f").get<double>(), name + " F");
            CheckP(fit.f_p, c.at("f_p").get<double>(), name + " F p");
            CheckValue(fit.rse, c.at("rse").get<double>(), name + " rse");
            CheckValue(fit.rmse, c.at("rmse").get<double>(), name + " rmse");
            CheckValue(fit.mae, c.at("mae").get<double>(), name + " mae");
            CheckVec(fit.residuals, c.at("residuals"), name + " residuals");
        }
        return;
    }
    if (op == "refuse_one_sample") return CheckRefused(StatisticalTests::OneSampleT(c.at("x").get<Vec>(), 0.0).error, c, name);
    if (op == "refuse_ols")
        return CheckRefused(StatisticalTests::FitLinear(c.at("columns").get<Groups>(), c.at("y").get<Vec>(),
                                                        c.at("names").get<std::vector<std::string>>()).error, c, name);
    if (op == "refuse_poly")
        return CheckRefused(StatisticalTests::FitPolynomial(c.at("x").get<Vec>(), c.at("y").get<Vec>(), c.at("degree").get<int>(), "x").error,
                            c, name);
    Check(false, name + ": unknown op " + op);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the statistical tests fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("op").get<std::string>() + " " + c.at("label").get<std::string>();
        try {
            RunCase(c, name);
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
        ++passed;
    }
    std::cout << "statistical tests match scipy / statsmodels: " << passed << " cases" << std::endl;
    return 0;
}
