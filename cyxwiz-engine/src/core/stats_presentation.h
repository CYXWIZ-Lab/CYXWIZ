#pragma once

// Words of Data Studio's Explore tabs Tests and Regression (TOFIX129 D3,
// approved mockup d3_statistics_mockup.html): the answer as a sentence, the
// checks behind it, effect sizes in words, and the CSV exports. Answers in,
// text out, so the tabs and their test agree. No ImGui, no queries.

#include "stats_explore.h"

#include <string>
#include <vector>

namespace cyxwiz::stats_view {

// A p-value in two significant digits: 4.8e-16, 0.23, 0.0064; "< 1e-300" below that.
std::string P(double p);
// A figure in a few digits (explore_view::Figure).
std::string Num(double v);
// "526.2 .. 840.6".
std::string Range(double lo, double hi);
// The confidence as a percent: 0.95 -> "95%".
std::string Level(double confidence);

const char* QuestionLabel(stats::Question q);

// "large", "medium", "small", "negligible" (Cohen's cut-offs).
const char* DWords(double d);         // Cohen's d: 0.2 / 0.5 / 0.8
const char* EtaWords(double eta2);    // eta squared: 0.01 / 0.06 / 0.14
const char* VWords(double v);         // Cramer's V: "weak" / "moderate" / "strong" at 0.1 / 0.3 / 0.5

struct Sentence {
    std::string text;
    bool significant = false;  // the answer is "yes, there is a difference / relation"
};

// The answer in one or two sentences.
Sentence Answer(const stats::TestAnswer& a);
// The checks behind it (variances, normality, expected counts, the rank test).
std::vector<std::string> Checks(const stats::TestAnswer& a);
// A Tukey pair's verdict: "differ" / "same".
const char* PairVerdict(const TukeyPair& pair, double confidence);
// "11 rows have no sex and are left out." ("" when none).
std::string LeftOut(const stats::TestAnswer& a);

// Regression: what the coefficients say, and how the fit compares.
std::string RegressionSentence(const stats::RegressionAnswer& a);
std::string ComparisonSentence(const stats::RegressionAnswer& a);  // polynomial vs line, or the reduced model
std::string ResidualSentence(const stats::RegressionAnswer& a);
// "body_mass_g = -6424.8 + 50.27 x flipper_length_mm + ...".
std::string Equation(const stats::RegressionAnswer& a);

// CSV (Export CSV): the test's figures; the fit's coefficients, then a row
// per row used with its values, the prediction and the residual.
std::string TestCsv(const stats::TestAnswer& a);
std::string RegressionCsv(const stats::RegressionAnswer& a);

}  // namespace cyxwiz::stats_view
