#include "stats_presentation.h"

#include "explore_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::stats_view {

using stats::Question;

namespace {

double Alpha(const stats::TestAnswer& a) { return 1.0 - a.request.confidence; }

std::string AlphaText(double alpha) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", alpha);
    return buf;
}

std::string Count(int64_t n) { return explore_view::Number(static_cast<size_t>(std::max<int64_t>(0, n))); }

std::string Csv(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (const char c : s) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

std::string Full(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return buf;
}

}  // namespace

std::string P(double p) {
    if (!std::isfinite(p)) return "-";
    if (p < 1e-300) return "< 1e-300";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2g", p);
    return buf;
}

std::string PairP(double p) { return std::isfinite(p) && p < 1e-15 ? std::string("< 1e-15") : P(p); }

std::string Num(double v) { return explore_view::Figure(v); }

std::string Range(double lo, double hi) { return Num(lo) + " .. " + Num(hi); }

std::string Aic(double v) {
    if (!std::isfinite(v)) return "-";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}

std::string Level(double confidence) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%g%%", confidence * 100.0);
    return buf;
}

const char* QuestionLabel(Question q) {
    switch (q) {
        case Question::TwoGroups: return "Two groups";
        case Question::Paired: return "Paired columns";
        case Question::SeveralGroups: return "Several groups";
        case Question::OneValue: return "One column vs a value";
        case Question::TwoCategories: return "Two categories";
    }
    return "";
}

std::string TestName(const stats::TestRequest& r) {
    switch (r.question) {
        case Question::TwoGroups: return r.welch ? "Welch two-sample t" : "Student two-sample t";
        case Question::Paired: return "paired t";
        case Question::SeveralGroups: return r.welch ? "Welch one-way ANOVA" : "classic one-way ANOVA";
        case Question::OneValue: return "one-sample t";
        case Question::TwoCategories: return "chi-square of independence";
    }
    return {};
}

const char* EffectName(Question q) {
    switch (q) {
        case Question::Paired: return "d (paired)";
        case Question::SeveralGroups: return "eta squared";
        case Question::TwoCategories: return "Cramer's V";
        default: return "Cohen's d";
    }
}

const char* DWords(double d) {
    const double x = std::fabs(d);
    return x >= 0.8 ? "large" : x >= 0.5 ? "medium" : x >= 0.2 ? "small" : "negligible";
}
const char* EtaWords(double eta2) { return eta2 >= 0.14 ? "large" : eta2 >= 0.06 ? "medium" : eta2 >= 0.01 ? "small" : "negligible"; }
const char* VWords(double v) { return v >= 0.5 ? "strong" : v >= 0.3 ? "moderate" : v >= 0.1 ? "weak" : "very weak"; }

const char* PairVerdict(const TukeyPair& pair, double confidence) { return pair.p < 1.0 - confidence ? "differ" : "same"; }

std::string LeftOut(const stats::TestAnswer& a) {
    if (a.left_out <= 0) return {};
    const auto& r = a.request;
    const std::string rows = a.left_out == 1 ? "1 row has" : Count(a.left_out) + " rows have";
    switch (r.question) {
        case Question::TwoGroups:
            if (r.two_columns) return Count(a.left_out) + " missing values are left out.";
            return rows + " no " + r.values + " or no " + r.groups + " and are left out.";
        case Question::SeveralGroups: return rows + " no " + r.values + " or no " + r.groups + " and are left out.";
        case Question::Paired: return rows + " only one of the two values and are left out (pairs stay together).";
        case Question::OneValue: return rows + " no " + r.values + " and are left out.";
        case Question::TwoCategories: return rows + " no " + r.groups + " or no " + r.second + " and are left out.";
    }
    return {};
}

Sentence Answer(const stats::TestAnswer& a) {
    Sentence s;
    if (!a.ok()) return {a.error, false};
    const auto& r = a.request;
    const double alpha = Alpha(a), p = a.main.p;
    s.significant = p < alpha;
    const std::string at = " at alpha " + AlphaText(alpha) + " (p " + P(p) + ")";
    switch (r.question) {
        case Question::TwoGroups: {
            const auto& g0 = a.groups[0];
            const auto& g1 = a.groups[1];
            const double diff = a.main.estimate;
            const bool first = diff >= 0.0;
            const std::string hi = first ? g0.name : g1.name, lo = first ? g1.name : g0.name;
            const double clo = first ? a.main.ci_low : -a.main.ci_high, chi = first ? a.main.ci_high : -a.main.ci_low;
            s.text = (r.two_columns ? hi + " is higher than " + lo : r.values + " is higher for " + hi + " than for " + lo) + " by " +
                     Num(std::fabs(diff)) + " on average (" + Level(r.confidence) + " CI " + Num(clo) + " to " + Num(chi) + "). ";
            s.text += s.significant ? "The difference is significant" + at + "."
                                    : "The difference is not significant" + at + "; the CI includes 0.";
            break;
        }
        case Question::Paired: {
            const double d = a.main.estimate;
            s.text = r.values + " is " + (d >= 0 ? "higher" : "lower") + " than " + r.second + " by " + Num(std::fabs(d)) +
                     " on average in the same rows (" + Level(r.confidence) + " CI " + Num(a.main.ci_low) + " to " + Num(a.main.ci_high) + "). ";
            s.text += s.significant ? "The change is significant" + at + "." : "The change is not significant" + at + "; the CI includes 0.";
            break;
        }
        case Question::SeveralGroups: {
            s.text = s.significant ? r.values + " differs between the " + r.groups + " values (p " + P(p) + ")"
                                   : r.values + " does not differ significantly between the " + r.groups + " values" + at;
            if (s.significant && !a.pairs.empty()) {
                std::string differ, same;
                for (const auto& pr : a.pairs) {
                    const std::string pair = a.groups[pr.a].name + " and " + a.groups[pr.b].name;
                    if (pr.p < alpha) differ += (differ.empty() ? "" : "; ") + pair;
                    else same += (same.empty() ? "" : "; ") + pair + " (p " + P(pr.p) + ")";
                }
                s.text += ": " + (differ.empty() ? std::string("no single pair differs") : "these pairs differ: " + differ);
                if (!same.empty()) s.text += ". These do not: " + same;
            }
            s.text += ".";
            break;
        }
        case Question::OneValue: {
            const auto& g = a.groups[0];
            const bool inside = a.main.ci_low <= r.value && r.value <= a.main.ci_high;
            s.text = "The mean " + r.values + " (" + Num(g.mean) + ") is " + (s.significant ? "" : "not ") +
                     "significantly different from " + Num(r.value) + at + "; " + Num(r.value) + " is " + (inside ? "inside" : "outside") +
                     " the " + Level(r.confidence) + " CI of the mean.";
            break;
        }
        case Question::TwoCategories: {
            s.text = r.groups + " and " + r.second + (s.significant ? " are related" : " are not significantly related") + " (p " +
                     P(p) + "; Cramer's V " + Num(a.main.effect) + ", " + VWords(a.main.effect) + ")";
            // Values of the rows seen with only one value of the columns.
            std::string only;
            for (size_t i = 0; i < a.counts.size(); ++i) {
                size_t seen = 0, at_col = 0;
                for (size_t j = 0; j < a.counts[i].size(); ++j)
                    if (a.counts[i][j] > 0) {
                        ++seen;
                        at_col = j;
                    }
                if (seen == 1) only += (only.empty() ? "" : "; ") + a.row_names[i] + " only with " + a.col_names[at_col];
            }
            s.text += only.empty() ? "." : ": " + only + ".";
            break;
        }
    }
    return s;
}

std::vector<std::string> Checks(const stats::TestAnswer& a) {
    std::vector<std::string> out;
    if (!a.ok()) return out;
    const auto& r = a.request;
    const double alpha = Alpha(a);
    const bool groups = r.question == Question::TwoGroups || r.question == Question::SeveralGroups;
    if (groups && a.variances.ok) {
        if (a.variances.p < alpha)
            out.push_back("Variances differ (Levene p " + P(a.variances.p) + "): " +
                          (r.welch ? std::string("Welch's test, the default, does not assume they are equal.")
                                   : std::string("switch to Unequal (Welch); the equal-variance test assumes they are equal.")));
        else
            out.push_back("Variances look alike (Levene p " + P(a.variances.p) + ").");
    }
    if (r.question != Question::TwoCategories) {
        size_t not_normal = 0, small = 0, tested = 0;
        int64_t smallest = INT64_MAX;
        const size_t count = r.question == Question::Paired ? 1 : a.groups.size();
        for (size_t i = 0; i < count; ++i) {
            const auto& g = r.question == Question::Paired ? a.groups.back() : a.groups[i];
            smallest = std::min(smallest, g.n);
            if (!std::isfinite(g.normal_p)) ++small;
            else {
                ++tested;
                if (g.normal_p < alpha) ++not_normal;
            }
        }
        if (small > 0) out.push_back("Fewer than 20 values: normality cannot be checked; the t-test assumes it.");
        else if (not_normal > 0) {
            std::string line = (tested == 1 ? std::string("The values do not look normal") : std::string("Not every group looks normal"));
            if (smallest >= 30) line += ", but with " + Count(smallest) + "+ values each the test is still reliable";
            if (a.rank.ok) {
                const bool agree = (a.rank.p < alpha) == (a.main.p < alpha);
                line += agree ? "; the rank test agrees: p " + P(a.rank.p)
                              : "; the rank test disagrees (p " + P(a.rank.p) + "): trust it over the t-test";
            }
            out.push_back(line + ".");
        } else {
            out.push_back(tested == 1 ? "The values look normal." : "Every group looks normal.");
        }
        if (a.rank.ok && not_normal == 0 && small == 0)
            out.push_back(std::string(r.question == Question::SeveralGroups ? "Kruskal-Wallis" : "Mann-Whitney U") + " agrees: p " + P(a.rank.p) + ".");
    }
    if (r.question == Question::SeveralGroups && r.welch && a.classic.ok)
        out.push_back("Classic F " + Num(a.classic.statistic) + " (" + Num(a.classic.df) + ", " + Num(a.classic.df2) + "), p " +
                      P(a.classic.p) + ", for comparison.");
    if (r.question == Question::TwoCategories) {
        if (a.min_expected < 5.0)
            out.push_back("Some expected counts are below 5 (smallest " + Num(a.min_expected) +
                          "): the p-value is rough; merge rare values or collect more rows.");
        else
            out.push_back("Every expected count is 5 or more (smallest " + Num(a.min_expected) + "), so the test holds.");
    }
    return out;
}

std::string RegressionSentence(const stats::RegressionAnswer& a) {
    if (!a.ok()) return a.error;
    const LinearFit& f = a.fit;
    if (a.request.polynomial)
        return a.request.response + " follows a degree " + std::to_string(a.request.degree) + " curve in " + a.request.predictors[0] +
               " (R² " + Num(f.r2) + ").";
    // The strongest predictor (largest |t|) and what the CI says of the others.
    size_t best = 1;
    for (size_t j = 1; j < f.t.size(); ++j)
        if (std::fabs(f.t[j]) > std::fabs(f.t[best])) best = j;
    std::string s = f.names[best] + " carries the model: each extra 1 " + (f.coef[best] >= 0 ? "adds " : "takes away ") +
                    Num(std::fabs(f.coef[best])) + " to " + a.request.response + ".";
    if (f.names.size() == 2) {
        const bool zero = f.ci_low[1] <= 0.0 && f.ci_high[1] >= 0.0;
        return a.request.predictors[0] + (zero ? " does not clearly predict " : " predicts ") + a.request.response +
               ": each extra 1 " + (f.coef[1] >= 0 ? "adds " : "takes away ") + Num(std::fabs(f.coef[1])) +
               (zero ? " (its CI includes 0)." : " (R² " + Num(f.r2) + ").");
    }
    std::string little;
    for (size_t j = 1; j < f.names.size(); ++j)
        if (j != best && f.ci_low[j] <= 0.0 && f.ci_high[j] >= 0.0) little += (little.empty() ? "" : " and ") + f.names[j];
    if (!little.empty()) s += " " + little + " add little once the others are known (their CIs include 0).";
    return s;
}

std::string ComparisonSentence(const stats::RegressionAnswer& a) {
    if (!a.ok()) return {};
    if (a.request.polynomial && a.line.ok)
        return std::string(a.fit.aic < a.line.aic ? "The curve fits better than the line (lower AIC: " : "The curve fits no better than the line (AIC ") +
               Aic(a.fit.aic) + " vs " + Aic(a.line.aic) + "; R² " + Num(a.fit.r2) + " vs " + Num(a.line.r2) + ").";
    if (a.reduced.ok) {
        std::string names;
        for (const auto& n : a.reduced_names) names += (names.empty() ? "" : ", ") + n;
        return names + " alone gives R² " + Num(a.reduced.r2) + " (AIC " + Aic(a.reduced.aic) + " vs " + Aic(a.fit.aic) + ")" +
               (a.reduced.aic <= a.fit.aic ? ": the simpler model is as good." : ".");
    }
    return {};
}

std::string ResidualSentence(const stats::RegressionAnswer& a) {
    if (!a.ok()) return {};
    if (!std::isfinite(a.residual_normal_p)) return "Fewer than 20 rows: the residuals' normality cannot be checked.";
    return a.residual_normal_p >= 0.05 ? "The residuals look normal enough (p " + P(a.residual_normal_p) + ")."
                                       : "The residuals do not look normal (p " + P(a.residual_normal_p) +
                                             "): the CIs and p-values are rough; look at the Q-Q plot.";
}

std::string Equation(const stats::RegressionAnswer& a) {
    if (!a.ok()) return {};
    const LinearFit& f = a.fit;
    std::string s = a.request.response + " = " + Num(f.coef[0]);
    for (size_t j = 1; j < f.coef.size(); ++j)
        s += std::string(f.coef[j] < 0 ? " - " : " + ") + Num(std::fabs(f.coef[j])) + " x " + f.names[j];
    return s;
}

std::string TestCsv(const stats::TestAnswer& a) {
    std::string s = "question," + Csv(QuestionLabel(a.request.question)) + "\n";
    s += "statistic," + Full(a.main.statistic) + "\ndf," + Full(a.main.df) + "\n";
    if (std::isfinite(a.main.df2)) s += "df2," + Full(a.main.df2) + "\n";
    s += "p," + Full(a.main.p) + "\n";
    if (std::isfinite(a.main.estimate))
        s += "estimate," + Full(a.main.estimate) + "\nci_low," + Full(a.main.ci_low) + "\nci_high," + Full(a.main.ci_high) +
             "\nconfidence," + Full(a.request.confidence) + "\n";
    if (std::isfinite(a.main.effect)) s += "effect," + Full(a.main.effect) + "\n";
    if (!a.groups.empty()) {
        s += "\ngroup,n,mean,sd,normal_p\n";
        for (const auto& g : a.groups)
            s += Csv(g.name) + "," + std::to_string(g.n) + "," + Full(g.mean) + "," + Full(g.sd) + "," + Full(g.normal_p) + "\n";
    }
    if (!a.pairs.empty()) {
        s += "\npair_a,pair_b,difference,ci_low,ci_high,p\n";
        for (const auto& pr : a.pairs)
            s += Csv(a.groups[pr.a].name) + "," + Csv(a.groups[pr.b].name) + "," + Full(pr.difference) + "," + Full(pr.ci_low) + "," +
                 Full(pr.ci_high) + "," + Full(pr.p) + "\n";
    }
    if (!a.counts.empty()) {
        s += "\n" + Csv(a.request.groups);
        for (const auto& c : a.col_names) s += "," + Csv(c);
        s += "\n";
        for (size_t i = 0; i < a.counts.size(); ++i) {
            s += Csv(a.row_names[i]);
            for (const double v : a.counts[i]) s += "," + Full(v);
            s += "\n";
        }
    }
    return s;
}

std::string RegressionCsv(const stats::RegressionAnswer& a) {
    const LinearFit& f = a.fit;
    std::string s = "term,estimate,se,t,p,ci_low,ci_high\n";
    for (size_t j = 0; j < f.coef.size(); ++j)
        s += Csv(f.names[j]) + "," + Full(f.coef[j]) + "," + Full(f.se[j]) + "," + Full(f.t[j]) + "," + Full(f.p[j]) + "," +
             Full(f.ci_low[j]) + "," + Full(f.ci_high[j]) + "\n";
    s += "\nr2," + Full(f.r2) + "\nadj_r2," + Full(f.adj_r2) + "\nF," + Full(f.f) + "\nF_p," + Full(f.f_p) + "\nrse," + Full(f.rse) +
         "\nrmse," + Full(f.rmse) + "\nmae," + Full(f.mae) + "\naic," + Full(f.aic) + "\nn," + std::to_string(f.n) + "\n";
    s += "\n" + Csv(a.request.response);
    for (const auto& p : a.request.predictors) s += "," + Csv(p);
    s += ",predicted,residual\n";
    for (size_t i = 0; i < a.y.size() && i < f.fitted.size(); ++i) {
        s += Full(a.y[i]);
        for (const auto& c : a.columns) s += "," + Full(c[i]);
        s += "," + Full(f.fitted[i]) + "," + Full(f.residuals[i]) + "\n";
    }
    return s;
}

}  // namespace cyxwiz::stats_view
