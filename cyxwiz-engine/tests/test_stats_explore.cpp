// Data Studio's Explore tabs Tests and Regression (TOFIX129 D3): reading the
// columns (missing: NULL, not a number, or a project missing text), grouping
// and pairing rows, the left-out counts and refusals, the answers equal to
// cyxwiz::StatisticalTests on the same values (itself checked against scipy),
// and the words.

#include "../src/core/session_query_engine.h"
#include "../src/core/stats_explore.h"
#include "../src/core/stats_presentation.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Same(double a, double b, const std::string& what) {
    Check((std::isnan(a) && std::isnan(b)) || std::fabs(a - b) <= 1e-12 * (1.0 + std::fabs(b)),
          what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

constexpr int kRows = 40;

// v: numbers (row 7 NULL); w: numbers as text (row 9 "x", row 11 "N/A");
// g: A / B alternating (row 3 NULL, row 5 "N/A"); k: three groups; c, d: categories.
std::shared_ptr<arrow::Table> Table(std::vector<double>& v, std::vector<double>& w) {
    arrow::DoubleBuilder vb;
    arrow::StringBuilder wb, gb, kb, cb, db;
    for (int i = 0; i < kRows; ++i) {
        const double vi = 10.0 + 0.37 * i + (i % 3 == 0 ? 2.5 : -1.0) + (i % 2 ? 1.8 : 0.0);
        const double wi = vi + 0.9 + 0.11 * (i % 5) - (i % 7 == 0 ? 1.4 : 0.0);
        v.push_back(i == 7 ? NAN : vi);
        w.push_back(i == 9 || i == 11 ? NAN : wi);
        if (i == 7) (void)vb.AppendNull();
        else (void)vb.Append(vi);
        (void)wb.Append(i == 9 ? std::string("x") : i == 11 ? std::string("N/A") : std::to_string(wi));
        if (i == 3) (void)gb.AppendNull();
        else (void)gb.Append(i == 5 ? "N/A" : (i % 2 ? "B" : "A"));
        (void)kb.Append(i % 3 == 0 ? "low" : i % 3 == 1 ? "mid" : "high");
        (void)cb.Append(i < 20 ? "north" : "south");
        (void)db.Append(i % 4 == 0 ? "yes" : "no");
    }
    std::shared_ptr<arrow::Array> va, wa, ga, ka, ca, da;
    (void)vb.Finish(&va);
    (void)wb.Finish(&wa);
    (void)gb.Finish(&ga);
    (void)kb.Finish(&ka);
    (void)cb.Finish(&ca);
    (void)db.Finish(&da);
    return arrow::Table::Make(arrow::schema({arrow::field("v", arrow::float64()), arrow::field("w", arrow::utf8()),
                                             arrow::field("g", arrow::utf8()), arrow::field("k", arrow::utf8()),
                                             arrow::field("c", arrow::utf8()), arrow::field("d", arrow::utf8())}),
                              {va, wa, ga, ka, ca, da});
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_stats_explore";
    std::error_code ec;
    fs::remove_all(root, ec);
    {
        std::vector<double> v, w;
        SessionQueryEngine engine((root / "attach").string());
        std::string error;
        Check(engine.AttachArrow("t", 1, Table(v, w), &error), "attach: " + error);
        const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
        explore::MissingTexts texts;
        texts["g"] = {"N/A"};
        texts["w"] = {"N/A"};

        // Two groups, split by g: rows 3 (no g), 5 (N/A) and 7 (no v) are left out.
        stats::TestRequest two;
        two.values = "v";
        two.groups = "g";
        const auto a = stats::RunTest("t", two, run, texts);
        Check(a.ok(), "two groups: " + a.error);
        Check(a.rows == kRows && a.left_out == 3, "two groups left out 3: " + std::to_string(a.left_out));
        Check(a.groups.size() == 2 && a.groups[0].name == "A" && a.groups[1].name == "B", "groups A, B in order of appearance");
        std::vector<double> ga, gb;
        for (int i = 0; i < kRows; ++i) {
            if (i == 3 || i == 5 || i == 7) continue;
            (i % 2 ? gb : ga).push_back(v[static_cast<size_t>(i)]);
        }
        const auto welch = StatisticalTests::TwoSampleT(ga, gb, false);
        Same(a.main.statistic, welch.statistic, "Welch t");
        Same(a.main.p, welch.p, "Welch p");
        Same(a.main.df, welch.df, "Welch df");
        Same(a.rank.p, StatisticalTests::MannWhitneyU(ga, gb).p, "Mann-Whitney p");
        Same(a.variances.p, StatisticalTests::Levene({ga, gb}).p, "Levene p");
        Check(a.groups[0].n == static_cast<int64_t>(ga.size()) && std::isfinite(a.groups[0].normal_p), "group A: n and normality");
        two.welch = false;
        Same(stats::RunTest("t", two, run, texts).main.df, static_cast<double>(ga.size() + gb.size() - 2), "Student df");

        // Paired: v and w row by row; rows 7 (no v), 9 (not a number) and 11 (N/A) are left out.
        stats::TestRequest paired;
        paired.question = stats::Question::Paired;
        paired.values = "v";
        paired.second = "w";
        const auto p = stats::RunTest("t", paired, run, texts);
        Check(p.ok() && p.left_out == 3 && p.groups.size() == 3, "paired: 3 left out, 3 figure rows: " + p.error);
        std::vector<double> pv, pw;
        for (int i = 0; i < kRows; ++i)
            if (i != 7 && i != 9 && i != 11) {
                pv.push_back(v[static_cast<size_t>(i)]);
                pw.push_back(std::stod(std::to_string(w[static_cast<size_t>(i)])));
            }
        Same(p.main.statistic, StatisticalTests::PairedT(pv, pw).statistic, "paired t (rows kept together)");

        // Several groups by k; and two groups on k is refused (3 values).
        stats::TestRequest several;
        several.question = stats::Question::SeveralGroups;
        several.values = "v";
        several.groups = "k";
        const auto s = stats::RunTest("t", several, run, texts);
        Check(s.ok() && s.groups.size() == 3 && s.pairs.size() == 3 && s.classic.ok && s.rank.ok, "several groups: " + s.error);
        Check(s.main.df2 != s.classic.df2, "Welch's ANOVA has its own denominator df");
        two.groups = "k";
        Check(stats::RunTest("t", two, run, texts).error.find("Several groups") != std::string::npos, "two groups refuses 3 values");
        several.groups = "v";  // a number column: one group per value
        Check(stats::RunTest("t", several, run, texts).error.find("number column") != std::string::npos, "too many groups refused");

        // One value.
        stats::TestRequest one;
        one.question = stats::Question::OneValue;
        one.values = "v";
        one.value = 17.0;
        const auto o = stats::RunTest("t", one, run, texts);
        Check(o.ok() && o.left_out == 1 && o.groups[0].n == kRows - 1, "one value: " + o.error);

        // Two categories: c by d.
        stats::TestRequest cat;
        cat.question = stats::Question::TwoCategories;
        cat.groups = "c";
        cat.second = "d";
        const auto c = stats::RunTest("t", cat, run, texts);
        Check(c.ok() && c.row_names == std::vector<std::string>{"north", "south"} &&
                  c.col_names == std::vector<std::string>{"yes", "no"} && c.counts[0][0] == 5 && c.counts[1][1] == 15,
              "contingency table: " + c.error);

        // Regression: v on w (rows 7, 9, 11 left out); one constant column refused in words.
        stats::RegressionRequest reg;
        reg.response = "v";
        reg.predictors = {"w"};
        const auto r = stats::RunRegression("t", reg, run, texts);
        Check(r.ok() && r.used == kRows - 3 && r.x.size() == static_cast<size_t>(r.used), "regression: " + r.error);
        Same(r.fit.coef[1], StatisticalTests::FitLinear({pw}, pv, {"w"}).coef[1], "regression slope");
        reg.polynomial = true;
        reg.degree = 2;
        const auto poly = stats::RunRegression("t", reg, run, texts);
        Check(poly.ok() && poly.line.ok && poly.fit.coef.size() == 3, "polynomial with its line: " + poly.error);
        reg.predictors = {"w", "v"};
        Check(stats::RunRegression("t", reg, run, texts).error.find("one column") != std::string::npos, "polynomial takes one column");
        reg.polynomial = false;
        reg.predictors = {"v"};
        Check(stats::RunRegression("t", reg, run, texts).error.find("cannot predict itself") != std::string::npos, "no self prediction");
    }
    fs::remove_all(root, ec);

    // ---------------------------------------------------------------- words (penguins figures, scipy)
    stats::TestAnswer w;
    w.request.values = "body_mass_g";
    w.request.groups = "sex";
    w.rows = 344;
    w.left_out = 11;
    stats::GroupFigures male, female;
    male.name = "MALE";
    male.n = 168;
    male.mean = 4545.68;
    male.normal_p = 1.4e-17;
    female.name = "FEMALE";
    female.n = 165;
    female.mean = 3862.27;
    female.normal_p = 3.7e-12;
    w.groups = {male, female};
    w.main.ok = true;
    w.main.p = 4.793891255051459e-16;
    w.main.estimate = 683.4118;
    w.main.ci_low = 526.2453;
    w.main.ci_high = 840.5783;
    w.rank.ok = true;
    w.rank.p = 1.8133e-15;
    w.variances.ok = true;
    w.variances.p = 0.01434820670043522;
    const auto sentence = stats_view::Answer(w);
    Check(sentence.significant &&
              sentence.text == "body_mass_g is higher for MALE than for FEMALE by 683.4 on average (95% CI 526.2 to 840.6). "
                               "The difference is significant at alpha 0.05 (p 4.8e-16).",
          sentence.text);
    const auto checks = stats_view::Checks(w);
    Check(checks.size() == 2 && checks[0] == "Variances differ (Levene p 0.014): Welch's test, the default, does not assume they are equal." &&
              checks[1] == "Not every group looks normal, but with 165+ values each the test is still reliable; the rank test agrees: p 1.8e-15.",
          "checks: " + (checks.empty() ? std::string() : checks[0] + " | " + checks.back()));
    Check(stats_view::LeftOut(w) == "11 rows have no body_mass_g or no sex and are left out.", stats_view::LeftOut(w));
    // The other direction reads from the higher group, with the CI flipped.
    w.main.estimate = -683.4118;
    w.main.ci_low = -840.5783;
    w.main.ci_high = -526.2453;
    Check(stats_view::Answer(w).text.find("higher for FEMALE than for MALE by 683.4 on average (95% CI 526.2 to 840.6)") != std::string::npos,
          stats_view::Answer(w).text);

    stats::TestAnswer one;
    one.request.question = stats::Question::OneValue;
    one.request.values = "flipper_length_mm";
    one.request.value = 200;
    stats::GroupFigures flipper;
    flipper.name = "flipper_length_mm";
    flipper.n = 342;
    flipper.mean = 200.9152;
    one.groups = {flipper};
    one.main.ok = true;
    one.main.p = 0.2295674765105476;
    one.main.ci_low = 199.4196;
    one.main.ci_high = 202.4108;
    Check(stats_view::Answer(one).text == "The mean flipper_length_mm (200.9) is not significantly different from 200 at alpha 0.05 (p 0.23); "
                                          "200 is inside the 95% CI of the mean.",
          stats_view::Answer(one).text);

    stats::TestAnswer cat;
    cat.request.question = stats::Question::TwoCategories;
    cat.request.groups = "species";
    cat.request.second = "island";
    cat.row_names = {"Adelie", "Chinstrap", "Gentoo"};
    cat.col_names = {"Torgersen", "Biscoe", "Dream"};
    cat.counts = {{52, 44, 56}, {0, 0, 68}, {0, 124, 0}};
    cat.main.ok = true;
    cat.main.p = 1.354573829719252e-63;
    cat.main.effect = 0.6598431008795325;
    cat.min_expected = 10.279;
    Check(stats_view::Answer(cat).text == "species and island are related (p 1.4e-63; Cramer's V 0.6598, strong): Chinstrap only with Dream; "
                                          "Gentoo only with Biscoe.",
          stats_view::Answer(cat).text);
    Check(stats_view::Checks(cat).front().find("5 or more (smallest 10.28)") != std::string::npos, stats_view::Checks(cat).front());

    Check(stats_view::P(2.9e-82) == "2.9e-82" && stats_view::P(0.0064) == "0.0064" && stats_view::P(0.0) == "< 1e-300" &&
              stats_view::PairP(0.0) == "< 1e-15" && stats_view::PairP(0.0089) == "0.0089",
          "p words");
    Check(std::string(stats_view::DWords(0.94)) == "large" && std::string(stats_view::EtaWords(0.67)) == "large" &&
              std::string(stats_view::VWords(0.2)) == "weak",
          "effect words");

    // Regression words: penguins, three predictors (statsmodels).
    stats::RegressionAnswer r;
    r.request.response = "body_mass_g";
    r.request.predictors = {"flipper_length_mm", "bill_length_mm", "bill_depth_mm"};
    r.fit.ok = true;
    r.fit.names = {"intercept", "flipper_length_mm", "bill_length_mm", "bill_depth_mm"};
    r.fit.coef = {-6424.76, 50.2692, 4.1618, 20.0495};
    r.fit.t = {-11.44, 20.29, 0.781, 1.464};
    r.fit.ci_low = {-7529.2, 45.397, -6.3205, -6.8865};
    r.fit.ci_high = {-5320.4, 55.142, 14.644, 46.986};
    r.fit.aic = 5061.32;
    r.reduced.ok = true;
    r.reduced.r2 = 0.75899;
    r.reduced.aic = 5060.85;
    r.reduced_names = {"flipper_length_mm"};
    Check(stats_view::RegressionSentence(r) == "flipper_length_mm carries the model: each extra 1 adds 50.27 to body_mass_g. "
                                               "bill_length_mm and bill_depth_mm add little once the others are known (their CIs include 0).",
          stats_view::RegressionSentence(r));
    Check(stats_view::ComparisonSentence(r) == "flipper_length_mm alone gives R² 0.759 (AIC 5061 vs 5061): the simpler model is as good.",
          stats_view::ComparisonSentence(r));
    Check(stats_view::Equation(r) == "body_mass_g = -6425 + 50.27 x flipper_length_mm + 4.162 x bill_length_mm + 20.05 x bill_depth_mm",
          stats_view::Equation(r));

    std::cout << "stats explore: columns read, groups and pairs, refusals, answers and words. OK\n";
    return 0;
}
