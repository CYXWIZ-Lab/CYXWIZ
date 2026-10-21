// The normal CDF and the Distribution tab's fits (maximum-likelihood
// parameters, AIC, BIC, Kolmogorov-Smirnov), against scipy. The t, F and
// chi-square functions and the tests are cyxwiz::StatisticalTests
// (test_computation_truth_statistical_tests)
// (tests/fixtures/generate_distribution_fixtures.py, scipy 1.17.1).
#include "../src/core/data_analyzer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using cyxwiz::DataAnalyzer;
using cyxwiz::DistributionFitResult;
using cyxwiz::DistributionType;

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

const Case kNormalCdf[] = {
    {{-1.5, 0.0, 1.0}, 0.066807201268858057},
    {{0.3, 0.0, 1.0}, 0.61791142218895256},
    {{2.0, 1.0, 0.5}, 0.97724986805182079},
    {{-8.0, 0.0, 1.0}, 6.2209605742717405e-16},
};

// A fixed log-normal draw (seed 7). Parameters: Normal {mu, sigma}, Uniform
// {a, b}, Exponential {lambda}, LogNormal {mu, sigma} of log(x). The KS p-value
// is the asymptotic Kolmogorov distribution with Stephens' correction
// (scipy.stats.kstwobign.sf), not scipy kstest's exact small-sample p.
const std::vector<double> kFitSample = {
    33.152, 43.331, 25.875, 14.857, 21.995, 13.565, 34.957, 110.63, 21.264, 18.946,
    51.463, 45.659, 36.411, 14.333, 32.255, 61.916, 9.877, 21.936, 5.983, 10.375,
    6.312, 26.8, 10.584, 42.273, 38.133, 27.988, 3.438, 20.393, 31.701, 36.671,
    8.355, 21.542, 13.727, 15.991, 86.04, 16.01, 32.16, 73.402, 19.585, 29.948};
struct FitCase {
    DistributionType type;
    std::vector<std::pair<const char*, double>> parameters;
    double aic, bic, ks_statistic, ks_p_value;
};
const FitCase kFits[] = {
    {DistributionType::Normal, {{"mu", 29.745825000000004}, {"sigma", 21.958193665335383}},
     364.64631126287384, 368.0240701711017, 0.15124512293343895, 0.29508826858531151},
    {DistributionType::Uniform, {{"a", 3.4380000000000002}, {"b", 110.63}},
     377.96972951890456, 381.34748842713242, 0.48111799388014032, 8.0458473661599757e-09},
    {DistributionType::Exponential, {{"lambda", 0.033618163221225163}},
     353.41510288073187, 355.1039823348458, 0.191205420469309, 0.094403950310889287},
    {DistributionType::LogNormal, {{"mu", 3.144333931352937}, {"sigma", 0.72938408765084473}},
     343.8174118434265, 347.19517075165436, 0.065593536651172513, 0.99383960774434921},
};

DistributionFitResult Fit(DistributionType type) {
    switch (type) {
        case DistributionType::Normal: return DataAnalyzer::FitNormal(kFitSample);
        case DistributionType::Uniform: return DataAnalyzer::FitUniform(kFitSample);
        case DistributionType::Exponential: return DataAnalyzer::FitExponential(kFitSample);
        default: return DataAnalyzer::FitLogNormal(kFitSample);
    }
}

std::string Args(const Case& c, int n) {
    std::string s = "(";
    for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string(c.args[i]);
    return s + ")";
}

}  // namespace

int main() {
    for (const Case& c : kNormalCdf) {
        Near(DataAnalyzer::NormalCDF(c.args[0], c.args[1], c.args[2]), c.want, 1e-15, "normal cdf" + Args(c, 3));
    }

    for (const FitCase& c : kFits) {
        const DistributionFitResult r = Fit(c.type);
        for (const auto& [name, value] : c.parameters) {
            const auto it = r.parameters.find(name);
            Check(it != r.parameters.end(), r.name + " has parameter " + name);
            if (it != r.parameters.end()) Near(it->second, value, 1e-10 * (1.0 + std::fabs(value)), r.name + " " + name);
        }
        Near(r.aic, c.aic, 1e-8, r.name + " AIC");
        Near(r.bic, c.bic, 1e-8, r.name + " BIC");
        Near(r.ks_statistic, c.ks_statistic, 1e-12, r.name + " KS statistic");
        Near(r.ks_p_value, c.ks_p_value, 1e-12, r.name + " KS p-value");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("distributions: the normal CDF and the distribution fits match scipy. OK\n");
    return 0;
}
