#include "stats_explore.h"

#include "explore_read.h"

#include <cyxwiz/data_transform.h>

#include <algorithm>
#include <map>
#include <optional>

namespace cyxwiz::stats {

namespace {

using explore_read::LabelExpr;
using explore_read::Labels;
using explore_read::NumberExpr;
using explore_read::Numbers;
using explore_read::Read;
using explore_read::Select;

GroupFigures Figures(std::string name, std::vector<double> values) {
    GroupFigures g;
    g.name = std::move(name);
    g.n = static_cast<int64_t>(values.size());
    if (!values.empty()) {
        const ColumnStats s = DataTransform::ComputeColumnStats(values);
        g.mean = s.mean;
        g.min = s.min;
        g.q1 = s.q1;
        g.median = s.median;
        g.q3 = s.q3;
        g.max = s.max;
        // ComputeColumnStats gives the population SD; the tests use the sample SD.
        g.sd = values.size() > 1 ? s.std_dev * std::sqrt(static_cast<double>(values.size()) / (values.size() - 1.0)) : NAN;
        const NormalityTestResult t = DataTransform::DAgostinoPearsonTest(values);
        g.normal_p = t.computed ? t.p_value : NAN;
    }
    g.values = std::move(values);
    return g;
}

// Values split by label, labels in order of first appearance.
std::vector<GroupFigures> Split(const std::vector<std::optional<double>>& v, const std::vector<std::optional<std::string>>& g,
                                int64_t& used) {
    std::vector<std::string> order;
    std::map<std::string, std::vector<double>> by;
    used = 0;
    for (size_t i = 0; i < v.size() && i < g.size(); ++i) {
        if (!v[i] || !g[i]) continue;
        auto [it, added] = by.try_emplace(*g[i]);
        if (added) order.push_back(*g[i]);
        it->second.push_back(*v[i]);
        ++used;
    }
    std::vector<GroupFigures> out;
    for (const auto& name : order) out.push_back(Figures(name, std::move(by[name])));
    return out;
}

std::vector<std::vector<double>> ValuesOf(const std::vector<GroupFigures>& groups) {
    std::vector<std::vector<double>> out;
    for (const auto& g : groups) out.push_back(g.values);
    return out;
}

}  // namespace

TestAnswer RunTest(const std::string& table, const TestRequest& req, const QueryRunner& run, const explore::MissingTexts& texts) {
    TestAnswer a;
    a.request = req;
    std::vector<QueryParam> params;
    std::vector<std::string> exprs;
    const bool labels = req.question == Question::TwoCategories;
    const bool grouped = (req.question == Question::TwoGroups && !req.two_columns) || req.question == Question::SeveralGroups;
    const bool two_numbers = req.question == Question::Paired || (req.question == Question::TwoGroups && req.two_columns);
    if (labels) {
        if (req.groups.empty() || req.second.empty()) return a.error = "Pick the two columns.", a;
        if (req.groups == req.second) return a.error = "Pick two different columns.", a;
        exprs = {LabelExpr(req.groups, texts, params, "r"), LabelExpr(req.second, texts, params, "c")};
    } else {
        if (req.values.empty()) return a.error = "Pick the column of values.", a;
        exprs.push_back(NumberExpr(req.values, texts, params, "v"));
        if (grouped) {
            if (req.groups.empty()) return a.error = "Pick the column that splits the rows into groups.", a;
            exprs.push_back(LabelExpr(req.groups, texts, params, "g"));
        }
        if (two_numbers) {
            if (req.second.empty()) return a.error = "Pick the second column.", a;
            if (req.second == req.values) return a.error = "Pick two different columns.", a;
            exprs.push_back(NumberExpr(req.second, texts, params, "w"));
        }
    }
    const Read read = Select(table, exprs, std::move(params), run, "Tests: read the columns");
    if (!read.table) return a.error = read.error, a;
    a.rows = read.table->num_rows();
    const double confidence = req.confidence;

    switch (req.question) {
        case Question::OneValue: {
            std::vector<double> x;
            for (const auto& v : Numbers(*read.table, "v"))
                if (v) x.push_back(*v);
            a.left_out = a.rows - static_cast<int64_t>(x.size());
            a.groups.push_back(Figures(req.values, x));
            a.main = StatisticalTests::OneSampleT(a.groups[0].values, req.value, confidence);
            break;
        }
        case Question::Paired: {
            const auto v = Numbers(*read.table, "v"), w = Numbers(*read.table, "w");
            std::vector<double> x, y, d;
            for (size_t i = 0; i < v.size() && i < w.size(); ++i)
                if (v[i] && w[i]) {  // the rows stay together
                    x.push_back(*v[i]);
                    y.push_back(*w[i]);
                    d.push_back(*v[i] - *w[i]);
                }
            a.left_out = a.rows - static_cast<int64_t>(x.size());
            a.groups = {Figures(req.values, x), Figures(req.second, y), Figures(req.values + " - " + req.second, d)};
            a.main = StatisticalTests::PairedT(a.groups[0].values, a.groups[1].values, confidence);
            break;
        }
        case Question::TwoGroups:
        case Question::SeveralGroups: {
            if (req.two_columns) {
                std::vector<double> x, y;
                for (const auto& v : Numbers(*read.table, "v"))
                    if (v) x.push_back(*v);
                for (const auto& v : Numbers(*read.table, "w"))
                    if (v) y.push_back(*v);
                a.left_out = 2 * a.rows - static_cast<int64_t>(x.size() + y.size());
                a.groups = {Figures(req.values, x), Figures(req.second, y)};
            } else {
                int64_t used = 0;
                a.groups = Split(Numbers(*read.table, "v"), Labels(*read.table, "g"), used);
                a.left_out = a.rows - used;
            }
            const size_t k = a.groups.size();
            if (req.question == Question::TwoGroups && k != 2)
                return a.error = req.groups + " has " + std::to_string(k) + " values in these rows; Two groups needs exactly 2" +
                                 (k > 2 ? " (Several groups compares more)." : "."),
                       a;
            if (req.question == Question::SeveralGroups && (k < 2 || k > kMaxGroups))
                return a.error = k < 2 ? req.groups + " has only " + std::to_string(k) + " value in these rows; groups need at least 2."
                                       : req.groups + " has " + std::to_string(k) + " values; Several groups takes up to " +
                                             std::to_string(kMaxGroups) + " (is it a number column?).",
                       a;
            const auto values = ValuesOf(a.groups);
            if (k == 2) {
                a.main = StatisticalTests::TwoSampleT(values[0], values[1], !req.welch, confidence);
                a.rank = StatisticalTests::MannWhitneyU(values[0], values[1]);
            } else {
                a.main = StatisticalTests::OneWayAnova(values, req.welch);
                if (req.welch) a.classic = StatisticalTests::OneWayAnova(values, false);
                a.rank = StatisticalTests::KruskalWallis(values);
                std::string tukey_error;
                a.pairs = StatisticalTests::TukeyHsd(values, confidence, &tukey_error);
            }
            a.variances = StatisticalTests::Levene(values);
            break;
        }
        case Question::TwoCategories: {
            const auto r = Labels(*read.table, "r"), c = Labels(*read.table, "c");
            std::map<std::string, size_t> ri, ci;
            std::vector<std::pair<size_t, size_t>> cells;
            for (size_t i = 0; i < r.size() && i < c.size(); ++i) {
                if (!r[i] || !c[i]) continue;
                auto [rit, radd] = ri.try_emplace(*r[i], a.row_names.size());
                if (radd) a.row_names.push_back(*r[i]);
                auto [cit, cadd] = ci.try_emplace(*c[i], a.col_names.size());
                if (cadd) a.col_names.push_back(*c[i]);
                cells.emplace_back(rit->second, cit->second);
            }
            a.left_out = a.rows - static_cast<int64_t>(cells.size());
            if (a.row_names.size() > kMaxGroups || a.col_names.size() > kMaxGroups)
                return a.error = (a.row_names.size() > kMaxGroups ? req.groups : req.second) +
                                 " has more than " + std::to_string(kMaxGroups) + " values (is it a number column?).",
                       a;
            a.counts.assign(a.row_names.size(), std::vector<double>(a.col_names.size(), 0.0));
            for (const auto& [i, j] : cells) a.counts[i][j] += 1.0;
            const ChiSquareResult chi = StatisticalTests::ChiSquareIndependence(a.counts);
            a.main = chi.test;
            a.expected = chi.expected;
            a.min_expected = chi.min_expected;
            break;
        }
    }
    if (!a.main.ok) a.error = a.main.error;
    return a;
}

RegressionAnswer RunRegression(const std::string& table, const RegressionRequest& req, const QueryRunner& run,
                               const explore::MissingTexts& texts) {
    RegressionAnswer a;
    a.request = req;
    if (req.response.empty()) return a.error = "Pick the column to predict.", a;
    if (req.predictors.empty()) return a.error = "Add a column to predict from.", a;
    if (req.polynomial && req.predictors.size() != 1) return a.error = "A polynomial uses one column; keep one.", a;
    for (const auto& p : req.predictors)
        if (p == req.response) return a.error = req.response + " cannot predict itself.", a;
    std::vector<QueryParam> params;
    std::vector<std::string> exprs{NumberExpr(req.response, texts, params, "y")};
    for (size_t i = 0; i < req.predictors.size(); ++i)
        exprs.push_back(NumberExpr(req.predictors[i], texts, params, "x" + std::to_string(i)));
    const Read read = Select(table, exprs, std::move(params), run, "Regression: read the columns");
    if (!read.table) return a.error = read.error, a;
    a.rows = read.table->num_rows();
    const auto y = Numbers(*read.table, "y");
    std::vector<std::vector<std::optional<double>>> xs;
    for (size_t i = 0; i < req.predictors.size(); ++i) xs.push_back(Numbers(*read.table, "x" + std::to_string(i)));
    a.columns.assign(req.predictors.size(), {});
    for (size_t r = 0; r < y.size(); ++r) {
        if (!y[r]) continue;
        bool all = true;
        for (const auto& x : xs) all = all && r < x.size() && x[r];
        if (!all) continue;
        a.y.push_back(*y[r]);
        for (size_t i = 0; i < xs.size(); ++i) a.columns[i].push_back(*xs[i][r]);
    }
    a.used = static_cast<int64_t>(a.y.size());
    if (req.predictors.size() == 1) a.x = a.columns[0];
    if (req.polynomial) {
        a.fit = StatisticalTests::FitPolynomial(a.x, a.y, req.degree, req.predictors[0], req.confidence);
        if (a.fit.ok && req.degree > 1) a.line = StatisticalTests::FitLinear(a.columns, a.y, req.predictors, true, req.confidence);
    } else {
        a.fit = StatisticalTests::FitLinear(a.columns, a.y, req.predictors, true, req.confidence);
        if (a.fit.ok && req.predictors.size() > 1) {
            // The predictors whose CI excludes 0; refit on them when some do not.
            std::vector<std::vector<double>> kept;
            for (size_t i = 0; i < req.predictors.size(); ++i) {
                const size_t j = i + 1;  // after the intercept
                if (a.fit.ci_low[j] > 0.0 || a.fit.ci_high[j] < 0.0) {
                    kept.push_back(a.columns[i]);
                    a.reduced_names.push_back(req.predictors[i]);
                }
            }
            if (!kept.empty() && kept.size() < req.predictors.size())
                a.reduced = StatisticalTests::FitLinear(kept, a.y, a.reduced_names, true, req.confidence);
            else
                a.reduced_names.clear();
        }
    }
    if (!a.fit.ok) return a.error = a.fit.error, a;
    const NormalityTestResult normal = DataTransform::DAgostinoPearsonTest(a.fit.residuals);
    a.residual_normal_p = normal.computed ? normal.p_value : NAN;
    return a;
}

}  // namespace cyxwiz::stats
