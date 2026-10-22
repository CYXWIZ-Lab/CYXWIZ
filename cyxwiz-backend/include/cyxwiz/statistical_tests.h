#pragma once

// Hypothesis tests and linear models (TOFIX129 D3): the sums, variances,
// ranks and least-squares solves run on ArrayFire (the selected device);
// the t / F / chi-square / studentized-range distributions are scalar
// functions on the host. Every function follows scipy (statsmodels for the
// linear model) and is checked against it in the computation-truth tests.
// p-values come from upper-tail functions computed directly, so a p of
// 1e-80 is exact rather than 1 - (1 - 1e-80) = 0.

#include "api_export.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

// One test's answer. `ok` false: `error` says why, in words for the user.
struct CYXWIZ_API TestResult {
    bool ok = false;
    std::string error;
    double statistic = NAN;  // t, F, chi-square, U, H, W
    double df = NAN;         // degrees of freedom (the first for F)
    double df2 = NAN;        // the second for F
    double p = NAN;          // two-sided where it applies
    double estimate = NAN;   // a mean, or a difference of means
    double ci_low = NAN, ci_high = NAN;  // of `estimate`, at the confidence asked
    double effect = NAN;     // Cohen's d (d_z for paired), eta squared, Cramer's V
};

// One pair of groups in Tukey's HSD (indices into the groups given).
struct CYXWIZ_API TukeyPair {
    size_t a = 0, b = 0;
    double difference = NAN;  // mean(a) - mean(b)
    double ci_low = NAN, ci_high = NAN;
    double p = NAN;
};

struct CYXWIZ_API ChiSquareResult {
    TestResult test;  // statistic, df, p, effect = Cramer's V
    std::vector<std::vector<double>> expected;  // rows x columns
    double min_expected = NAN;
};

// An ordinary least-squares fit (statsmodels OLS).
struct CYXWIZ_API LinearFit {
    bool ok = false;
    std::string error;
    std::vector<std::string> names;  // "intercept" first, then the predictors
    std::vector<double> coef, se, t, p, ci_low, ci_high;
    double r2 = NAN, adj_r2 = NAN, f = NAN, f_p = NAN;
    double rse = NAN;   // residual standard error, sqrt(SSE / df_resid)
    double rmse = NAN;  // sqrt(SSE / n)
    double mae = NAN;
    double aic = NAN;   // -2 log-likelihood + 2 (parameters)
    int64_t n = 0;
    int df_model = 0, df_resid = 0;
    std::vector<double> fitted, residuals;
};

class CYXWIZ_API StatisticalTests {
public:
    // ========== Distributions (host, scalar) ==========
    static double NormalCdf(double z);
    static double NormalSf(double z);  // 1 - NormalCdf(z), direct
    static double StudentTCdf(double t, double df);
    static double StudentTSf(double t, double df);  // P(T > t), direct
    static double StudentTPpf(double q, double df);  // the t with P(T <= t) = q
    static double FSf(double f, double df1, double df2);  // P(F > f), direct
    static double ChiSquareSf(double x, double df);  // P(X > x), direct
    // P(Q > q) for the studentized range of k means with df degrees of freedom (R's ptukey).
    static double StudentizedRangeSf(double q, double k, double df);

    // ========== Tests (data on the device) ==========
    // Missing values (NaN) must be removed by the caller; paired data must
    // keep its rows together (pass only rows where both are present).
    static TestResult OneSampleT(const std::vector<double>& x, double mu0, double confidence = 0.95);
    // Student (equal_var) or Welch; effect = Cohen's d with the pooled SD.
    static TestResult TwoSampleT(const std::vector<double>& a, const std::vector<double>& b, bool equal_var,
                                 double confidence = 0.95);
    // On the differences a - b; effect = d_z.
    static TestResult PairedT(const std::vector<double>& a, const std::vector<double>& b, double confidence = 0.95);
    // Classic one-way ANOVA, or Welch's (welch); effect = eta squared from the classic sums of squares.
    static TestResult OneWayAnova(const std::vector<std::vector<double>>& groups, bool welch);
    // Empty when a group has fewer than 2 values, or the device fails (`error` says why when given).
    static std::vector<TukeyPair> TukeyHsd(const std::vector<std::vector<double>>& groups, double confidence = 0.95,
                                           std::string* error = nullptr);
    // Brown-Forsythe / Levene centred on the median (scipy's default).
    static TestResult Levene(const std::vector<std::vector<double>>& groups);
    // Two-sided, normal approximation with continuity and tie correction (scipy method="asymptotic").
    static TestResult MannWhitneyU(const std::vector<double>& a, const std::vector<double>& b);
    static TestResult KruskalWallis(const std::vector<std::vector<double>>& groups);
    // Pearson chi-square of independence, no continuity correction.
    static ChiSquareResult ChiSquareIndependence(const std::vector<std::vector<double>>& counts);

    // ========== Linear models (device QR) ==========
    // y on the given columns, with an intercept unless `intercept` is false
    // (then R2 is uncentred and F has as many model degrees of freedom as
    // columns, as statsmodels does); names label the columns.
    static LinearFit FitLinear(const std::vector<std::vector<double>>& columns, const std::vector<double>& y,
                               const std::vector<std::string>& names, bool intercept = true, double confidence = 0.95);
    // y on x, x^2 .. x^degree (degree 1 .. 6), fitted on centred, scaled x and
    // reported in the original units.
    static LinearFit FitPolynomial(const std::vector<double>& x, const std::vector<double>& y, int degree,
                                   const std::string& name, double confidence = 0.95);
};

}  // namespace cyxwiz
