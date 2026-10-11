// Distribution functions behind the hypothesis tests and regression p-values
// (Student t, F, chi-square CDFs, the t critical value) and the one-sample
// t-test, against scipy (tests/fixtures/generate_distribution_fixtures.py,
// scipy 1.17.1).
#include "../src/core/data_analyzer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using cyxwiz::DataAnalyzer;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

void Near(double got, double want, double tol, const std::string& what) {
    Check(std::fabs(got - want) <= tol, what + ": got " + std::to_string(got) + ", scipy " + std::to_string(want));
}

struct Case {
    double args[3];
    double want;
};

const Case kTCdf[] = {
    {{0.5, 3.0}, 0.67427601757592448},
    {{1.96, 10.0}, 0.96078187987615016},
    {{2.5, 30.0}, 0.99094217546596663},
    {{-1.2, 5.0}, 0.14194552835305124},
    {{4.0, 2.0}, 0.97140452079103168},
    {{0.0, 7.0}, 0.5},
    {{-3.5, 120.0}, 0.00032690386993093818},
};
const Case kFCdf[] = {
    {{2.5, 3.0, 20.0}, 0.91115624806231066},
    {{0.8, 5.0, 10.0}, 0.42579902202690123},
    {{10.0, 1.0, 246.0}, 0.99823796107092844},
    {{1.0, 12.0, 4.0}, 0.4449462890625},
    {{4.2, 2.0, 60.0}, 0.98037297571184578},
};
const Case kChi2Cdf[] = {
    {{3.84, 1.0}, 0.94995647875129485},
    {{10.0, 4.0}, 0.95957231800548726},
    {{30.0, 10.0}, 0.99914335878922467},
    {{250.0, 200.0}, 0.99062086833117391},
    {{0.5, 3.0}, 0.081108588345324181},
    {{180.0, 150.0}, 0.95217543085657064},
};
// {alpha, df}: two-tailed critical value t.ppf(1 - alpha / 2, df).
const Case kTInv[] = {
    {{0.05, 1.0}, 12.706204736174694},
    {{0.05, 5.0}, 2.570581835636315},
    {{0.01, 30.0}, 2.7499956535672254},
    {{0.05, 247.0}, 1.9696147551832692},
    {{0.1, 12.0}, 1.782287555649319},
};

std::string Args(const Case& c, int n) {
    std::string s = "(";
    for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string(c.args[i]);
    return s + ")";
}

}  // namespace

int main() {
    constexpr double kTol = 1e-10;
    for (const Case& c : kTCdf) Near(DataAnalyzer::StudentTCDF(c.args[0], c.args[1]), c.want, kTol, "t cdf" + Args(c, 2));
    for (const Case& c : kFCdf) Near(DataAnalyzer::FCDF(c.args[0], c.args[1], c.args[2]), c.want, kTol, "f cdf" + Args(c, 3));
    for (const Case& c : kChi2Cdf) Near(DataAnalyzer::ChiSquareCDF(c.args[0], c.args[1]), c.want, kTol, "chi2 cdf" + Args(c, 2));
    for (const Case& c : kTInv) Near(DataAnalyzer::TInv(c.args[0], c.args[1]), c.want, 1e-8, "t inv" + Args(c, 2));

    // One-sample t-test: statistic, p-value and the 95% interval of the mean.
    const std::vector<double> sample = {4.2, 5.1, 3.9, 6.3, 4.8, 5.5, 4.1, 5.9, 4.4, 5.0, 3.7, 6.1};
    const auto r = DataAnalyzer::OneSampleTTest(sample, 4.0, 0.05);
    Near(r.test_statistic, 3.5836478399792639, 1e-10, "one-sample t statistic");
    Near(r.p_value, 0.0042911305189685198, 1e-10, "one-sample p-value");
    Near(r.confidence_interval_low, 4.3536733648204464, 1e-8, "one-sample 95% interval low");
    Near(r.confidence_interval_high, 5.4796599685128875, 1e-8, "one-sample 95% interval high");
    Check(r.reject_null, "one-sample t-test rejects H0 at 0.05");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("distributions: t / F / chi-square CDFs, t critical value and one-sample t-test match scipy. OK\n");
    return 0;
}
