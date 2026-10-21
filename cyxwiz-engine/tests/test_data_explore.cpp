// Data Studio Explore queries (TOFIX129 D1): incomplete rows, missing
// patterns, missing rows, outliers by three rules, correlation matrix, on a
// small table whose answers are worked out by hand, through the real query
// engine.

#include "../src/core/data_explore.h"

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

bool Near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

// 12 rows (1-based row numbers below):
//   a: 1..10, NULL at 11, 100 at 12 (the outlier)
//   b: 2 x row, NULL at 3 and 11
//   s: "x", "N/A" at 5 (a missing text), NULL at 11
//   c: 7 everywhere (no spread)
std::shared_ptr<arrow::Table> Table() {
    arrow::DoubleBuilder a;
    arrow::Int64Builder b, c;
    arrow::StringBuilder s;
    for (int row = 1; row <= 12; ++row) {
        if (row == 11) (void)a.AppendNull();
        else (void)a.Append(row == 12 ? 100.0 : row);
        if (row == 3 || row == 11) (void)b.AppendNull();
        else (void)b.Append(2 * row);
        if (row == 11) (void)s.AppendNull();
        else (void)s.Append(row == 5 ? "N/A" : "x");
        (void)c.Append(7);
    }
    std::shared_ptr<arrow::Array> aa, bb, cc, ss;
    (void)a.Finish(&aa);
    (void)b.Finish(&bb);
    (void)c.Finish(&cc);
    (void)s.Finish(&ss);
    auto schema = arrow::schema({arrow::field("a", arrow::float64()), arrow::field("b", arrow::int64()),
                                 arrow::field("s", arrow::utf8()), arrow::field("c", arrow::int64())});
    return arrow::Table::Make(schema, {aa, bb, ss, cc});
}

std::vector<int64_t> RowNumbers(const explore::RowPage& page) {
    std::vector<int64_t> out;
    auto col = page.rows ? page.rows->GetColumnByName("row") : nullptr;
    for (int64_t i = 0; col && i < col->length(); ++i) {
        auto s = col->GetScalar(i);
        auto d = (*s)->CastTo(arrow::int64());
        out.push_back(std::static_pointer_cast<arrow::Int64Scalar>(*d)->value);
    }
    return out;
}

double Score(const explore::RowPage& page, int64_t i) {
    auto s = page.rows->GetColumnByName("score")->GetScalar(i);
    return std::static_pointer_cast<arrow::DoubleScalar>(*(*s)->CastTo(arrow::float64()))->value;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_data_explore";
    std::error_code ec;
    fs::remove_all(root, ec);
    {
        SessionQueryEngine engine((root / "attach").string());
        std::string error;
        Check(engine.AttachArrow("t", 1, Table(), &error), "attach: " + error);
        const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
        explore::MissingTexts texts;
        texts["s"] = {"N/A"};
        ProfileOptions options;
        options.missing_text = texts;
        const DatasetProfile p = ProfileTable("t", run, options);
        Check(p.ok() && p.rows == 12, "profiled: " + p.error);

        // Missing: rows 3 (b), 5 (s, as text) and 11 (a, b, s) are incomplete.
        const explore::MissingSummary m = explore::SummariseMissing(p, texts, run);
        Check(m.ok(), "missing summary: " + m.error);
        Check(m.incomplete_rows == 3 && m.complete_rows == 9, "3 incomplete rows, 9 complete");
        Check(m.patterns.size() == 3, "three patterns");
        Check(m.patterns[0].columns == std::vector<std::string>{"a", "b", "s"} && m.patterns[0].rows == 1, "a, b and s together once");
        Check(m.patterns[1].columns == std::vector<std::string>{"b"} && m.patterns[2].columns == std::vector<std::string>{"s"},
              "b alone, s alone");

        Check(RowNumbers(explore::MissingRows("t", "b", {}, 0, run)) == std::vector<int64_t>{3, 11}, "b missing in rows 3 and 11");
        Check(RowNumbers(explore::MissingRows("t", "s", texts["s"], 0, run)) == std::vector<int64_t>{5, 11}, "s missing in 5 (N/A) and 11");
        Check(RowNumbers(explore::MissingRows("t", "b", {}, 1, run, 1)) == std::vector<int64_t>{11}, "second page of one row");

        // Complete table: no query, every row complete.
        DatasetProfile none = p;
        for (auto& c : none.columns) c.missing = 0;
        const explore::MissingSummary all = explore::SummariseMissing(none, {}, run);
        Check(all.complete_rows == 12 && all.incomplete_rows == 0 && all.patterns.empty(), "no missing values: all complete");

        // Outliers in a = 1..10, 100 (11 values).
        const ProfiledColumn* a = p.Find("a");
        Check(Near(a->q1, 3.5) && Near(a->q3, 8.5) && Near(a->median, 6.0), "quartiles 3.5 / 6 / 8.5");
        const auto iqr = explore::BoundsFor(*a, explore::OutlierRule::Iqr, 1.5);
        Check(iqr.ok() && Near(iqr.lo, -4.0) && Near(iqr.hi, 16.0), "IQR fences -4 .. 16");
        const auto iqr_n = explore::CountOutliers("t", *a, iqr, run);
        Check(iqr_n.error.empty() && iqr_n.low == 0 && iqr_n.high == 1, "IQR: one high outlier");
        const auto iqr_rows = explore::OutlierRows("t", *a, explore::OutlierRule::Iqr, iqr, 0, run);
        Check(iqr_rows.ok() && RowNumbers(iqr_rows) == std::vector<int64_t>{12}, "IQR outlier in row 12");
        Check(Near(Score(iqr_rows, 0), (100.0 - 8.5) / 5.0), "IQR score: 18.3 IQRs past the box");

        double mean = 0, ss = 0;
        const std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100};
        for (double v : values) mean += v / values.size();
        for (double v : values) ss += (v - mean) * (v - mean);
        const double sd = std::sqrt(ss / (values.size() - 1));
        Check(Near(a->mean, mean) && Near(a->std, sd), "mean and sample sd");
        const auto z = explore::BoundsFor(*a, explore::OutlierRule::ZScore, 2.0);
        Check(Near(z.hi, mean + 2 * sd), "Z bounds mean +- 2 sd");
        const auto z_rows = explore::OutlierRows("t", *a, explore::OutlierRule::ZScore, z, 0, run);
        Check(RowNumbers(z_rows) == std::vector<int64_t>{12} && Near(Score(z_rows, 0), (100 - mean) / sd), "Z: row 12, z score");

        const double mad = explore::MedianAbsoluteDeviation("t", *a, run, &error);
        Check(Near(mad, 3.0), "MAD of a is 3: " + error);
        const auto mz = explore::BoundsFor(*a, explore::OutlierRule::ModifiedZ, 3.5, mad);
        Check(mz.ok() && Near(mz.hi, 6.0 + 3.5 * 3.0 / 0.6745), "Modified Z bounds");
        const auto mz_rows = explore::OutlierRows("t", *a, explore::OutlierRule::ModifiedZ, mz, 0, run);
        Check(RowNumbers(mz_rows) == std::vector<int64_t>{12} && Near(Score(mz_rows, 0), 0.6745 * 94.0 / 3.0), "Modified Z: row 12");

        // A constant column has no spread for any rule (old window: everything flagged when MAD was 0).
        const ProfiledColumn* c = p.Find("c");
        Check(!explore::BoundsFor(*c, explore::OutlierRule::Iqr, 1.5).ok(), "constant: IQR unavailable");
        Check(!explore::BoundsFor(*c, explore::OutlierRule::ZScore, 3).ok(), "constant: Z unavailable");
        const double mad_c = explore::MedianAbsoluteDeviation("t", *c, run);
        Check(!explore::BoundsFor(*c, explore::OutlierRule::ModifiedZ, 3.5, mad_c).ok(), "constant: Modified Z unavailable");

        // Correlation: a and b over the rows where both are present (1, 2, 4..10, 12).
        Check(p.correlation.ok() && p.correlation.columns == std::vector<std::string>{"a", "b"}, "profile keeps the matrix of a, b (c has no spread)");
        std::vector<double> xs, ys;
        for (int row = 1; row <= 12; ++row) {
            if (row == 3 || row == 11) continue;
            xs.push_back(row == 12 ? 100.0 : row);
            ys.push_back(2.0 * row);
        }
        double mx = 0, my = 0;
        for (size_t i = 0; i < xs.size(); ++i) mx += xs[i] / xs.size(), my += ys[i] / ys.size();
        double sxy = 0, sxx = 0, syy = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            sxy += (xs[i] - mx) * (ys[i] - my);
            sxx += (xs[i] - mx) * (xs[i] - mx);
            syy += (ys[i] - my) * (ys[i] - my);
        }
        const double r = sxy / std::sqrt(sxx * syy);
        Check(Near(p.correlation.At(0, 1), r, 1e-9) && Near(p.correlation.At(1, 0), r, 1e-9), "r paired by row, not by position");
        Check(p.correlation.PairRows(0, 1) == 10, "10 rows have both a and b");
        Check(p.correlation.At(0, 0) == 1.0, "diagonal 1");
        Check(p.correlations.size() == 1 && Near(std::get<2>(p.correlations[0]), r, 1e-9), "strongest pairs from the matrix");
        const CorrelationMatrix chosen = Correlate("t", {"b", "a", "c"}, run, false);
        Check(chosen.ok() && Near(chosen.At(0, 1), r, 1e-9) && std::isnan(chosen.At(0, 2)), "chosen columns; no spread is NaN");
        Check(std::string(CorrelationStrength(0.87)) == "very strong" && std::string(CorrelationStrength(-0.47)) == "moderate",
              "strength words");
    }
    fs::remove_all(root, ec);
    std::cout << "data explore: all checks passed\n";
    return 0;
}
