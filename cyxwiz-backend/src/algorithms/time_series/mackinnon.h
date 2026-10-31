#pragma once

// MacKinnon's response-surface tables for the (augmented) Dickey-Fuller test,
// as statsmodels.tsa.adfvalues uses them: the approximate p-value of a tau
// statistic (MacKinnon 1994, Tables 2 and 3 for N = 1) and the finite-sample
// critical values at 1%, 5% and 10% (MacKinnon 2010, N = 1).

#include <cmath>
#include <string>

#include "cyxwiz/statistical_tests.h"

namespace cyxwiz::mackinnon {

inline int RegressionIndex(const std::string& regression) {
    if (regression == "n") return 0;
    if (regression == "c") return 1;
    if (regression == "ct") return 2;
    return 3;  // "ctt"
}

// The tau beyond which p = 1, below which p = 0, and the split between the
// small-p and large-p polynomials (N = 1), for n, c, ct, ctt.
constexpr double kTauMax[4] = {INFINITY, 2.74, 0.7, 0.54};
constexpr double kTauMin[4] = {-19.04, -18.83, -16.18, -17.17};
constexpr double kTauStar[4] = {-1.04, -1.61, -2.89, -3.21};
// p = Phi(c0 + c1 tau + c2 tau^2) below tau*, Phi(d0 + d1 tau + d2 tau^2 + d3 tau^3) above.
constexpr double kSmallP[4][3] = {{0.6344, 1.2378, 3.2496e-2}, {2.1659, 1.4412, 3.8269e-2}, {3.2512, 1.6047, 4.9588e-2}, {4.0003, 1.658, 4.8288e-2}};
constexpr double kLargeP[4][4] = {{0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2},
                                  {1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2},
                                  {2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2},
                                  {3.0778, 4.9529e-1, -4.1477e-1, -5.9359e-2}};
// Critical values (N = 1): c0 + c1 / n + c2 / n^2 + c3 / n^3 at 1%, 5%, 10%.
constexpr double kCrit[4][3][4] = {
    {{-2.56574, -2.2358, -3.627, 0}, {-1.94100, -0.2686, -3.365, 31.223}, {-1.61682, 0.2656, -2.714, 25.364}},
    {{-3.43035, -6.5393, -16.786, -79.433}, {-2.86154, -2.8903, -4.234, -40.040}, {-2.56677, -1.5384, -2.809, 0}},
    {{-3.95877, -9.0531, -28.428, -134.155}, {-3.41049, -4.3904, -9.036, -45.374}, {-3.12705, -2.5856, -3.925, -22.380}},
    {{-4.37113, -11.5882, -35.819, -334.047}, {-3.83239, -5.9057, -12.490, -118.284}, {-3.55326, -3.6596, -5.293, -63.559}}};

inline double PValue(double tau, const std::string& regression) {
    const int r = RegressionIndex(regression);
    if (tau > kTauMax[r]) return 1.0;
    if (tau < kTauMin[r]) return 0.0;
    double z;
    if (tau <= kTauStar[r]) {
        const double* c = kSmallP[r];
        z = c[0] + c[1] * tau + c[2] * tau * tau;
    } else {
        const double* c = kLargeP[r];
        z = c[0] + c[1] * tau + c[2] * tau * tau + c[3] * tau * tau * tau;
    }
    return StatisticalTests::NormalCdf(z);
}

// The 1%, 5% and 10% critical values for `nobs` observations (the asymptotic
// ones when nobs is 0).
inline void CriticalValues(const std::string& regression, int nobs, double out[3]) {
    const int r = RegressionIndex(regression);
    const double inv = nobs > 0 ? 1.0 / static_cast<double>(nobs) : 0.0;
    for (int level = 0; level < 3; ++level) {
        const double* c = kCrit[r][level];
        out[level] = c[0] + c[1] * inv + c[2] * inv * inv + c[3] * inv * inv * inv;
    }
}

}  // namespace cyxwiz::mackinnon
