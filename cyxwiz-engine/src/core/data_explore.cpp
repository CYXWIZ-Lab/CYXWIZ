#include "data_explore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace cyxwiz::explore {

namespace {

using Q = SessionQueryEngine;

std::string Num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

QueryResult Run(const QueryRunner& run, const std::string& table, const std::string& sql, std::vector<QueryParam> params,
                const char* label, bool cache = true) {
    QueryRequest r;
    r.sql = sql;
    r.params = std::move(params);
    r.inputs = {table};
    r.label = label;
    r.cache = cache;
    return run(r);
}

std::string ErrorOf(const QueryResult& r) { return r.cancelled ? "Cancelled." : r.error; }

double Value(const std::shared_ptr<arrow::Table>& t, const std::string& name, int64_t row = 0) {
    if (!t) return NAN;
    auto col = t->GetColumnByName(name);
    if (!col || row >= col->length()) return NAN;
    auto s = col->GetScalar(row);
    if (!s.ok() || !(*s)->is_valid) return NAN;
    auto d = (*s)->CastTo(arrow::float64());
    return d.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*d)->value : NAN;
}

size_t Count(const std::shared_ptr<arrow::Table>& t, const std::string& name, int64_t row = 0) {
    const double d = Value(t, name, row);
    return std::isfinite(d) && d > 0 ? static_cast<size_t>(std::llround(d)) : 0;
}

std::string Text(const std::shared_ptr<arrow::Table>& t, const std::string& name, int64_t row) {
    if (!t) return {};
    auto col = t->GetColumnByName(name);
    if (!col || row >= col->length()) return {};
    auto s = col->GetScalar(row);
    if (!s.ok() || !(*s)->is_valid) return {};
    if ((*s)->type->id() == arrow::Type::STRING) return std::static_pointer_cast<arrow::StringScalar>(*s)->ToString();
    return (*s)->ToString();
}

const std::vector<std::string>& TextsFor(const MissingTexts& texts, const std::string& column) {
    static const std::vector<std::string> kNone;
    auto it = texts.find(column);
    return it == texts.end() ? kNone : it->second;
}

// The table with its 1-based row number first.
std::string Numbered(const std::string& table) {
    return "(SELECT row_number() OVER () AS \"row\", * FROM " + Q::QuoteIdentifier(table) + ")";
}

}  // namespace

std::string MissingCondition(const std::string& column, const std::vector<std::string>& texts,
                             std::vector<QueryParam>& params) {
    const std::string col = Q::QuoteIdentifier(column);
    if (texts.empty()) return col + " IS NULL";
    std::string in;
    for (const auto& text : texts) {
        in += in.empty() ? "?" : ", ?";
        params.push_back(QueryParam::Of(text));
    }
    return "(" + col + " IS NULL OR CAST(" + col + " AS VARCHAR) IN (" + in + "))";
}

MissingSummary SummariseMissing(const DatasetProfile& profile, const MissingTexts& texts, const QueryRunner& run) {
    MissingSummary s;
    std::vector<const ProfiledColumn*> missing;
    for (const auto& c : profile.columns)
        if (c.missing > 0) missing.push_back(&c);
    if (missing.empty()) {
        s.complete_rows = profile.rows;
        return s;
    }
    // Incomplete rows: any column missing.
    std::vector<QueryParam> params;
    std::string any;
    for (const auto* c : missing) any += (any.empty() ? "" : " OR ") + MissingCondition(c->facts.name, TextsFor(texts, c->facts.name), params);
    QueryResult n = Run(run, profile.table, "SELECT count(*) AS n FROM " + Q::QuoteIdentifier(profile.table) + " WHERE " + any,
                        params, "Missing values");
    if (!n.ok) {
        s.error = ErrorOf(n);
        return s;
    }
    s.incomplete_rows = Count(n.table, "n");
    s.complete_rows = profile.rows - std::min(profile.rows, s.incomplete_rows);

    // Patterns: which of the first kPatternColumns such columns are missing in each incomplete row.
    const size_t told = std::min(missing.size(), kPatternColumns);
    std::vector<QueryParam> pparams;
    std::string key, where;
    for (size_t i = 0; i < told; ++i) {
        std::vector<QueryParam> one;
        const std::string cond = MissingCondition(missing[i]->facts.name, TextsFor(texts, missing[i]->facts.name), one);
        key += ", CASE WHEN " + cond + " THEN '" + std::to_string(i) + "' END";
        pparams.insert(pparams.end(), one.begin(), one.end());
    }
    for (size_t i = 0; i < told; ++i)
        where += (where.empty() ? "" : " OR ") + MissingCondition(missing[i]->facts.name, TextsFor(texts, missing[i]->facts.name), pparams);
    QueryResult p = Run(run, profile.table,
                        "SELECT concat_ws(','" + key + ") AS p, count(*) AS n FROM " + Q::QuoteIdentifier(profile.table) +
                            " WHERE " + where + " GROUP BY 1 ORDER BY n DESC, p LIMIT " + std::to_string(kPatterns),
                        pparams, "Missing values");
    if (!p.ok) {
        s.error = ErrorOf(p);
        return s;
    }
    for (int64_t row = 0; p.table && row < p.table->num_rows(); ++row) {
        MissingPattern pattern;
        pattern.rows = Count(p.table, "n", row);
        const std::string ids = Text(p.table, "p", row);
        size_t at = 0;
        while (at < ids.size()) {
            size_t end = ids.find(',', at);
            if (end == std::string::npos) end = ids.size();
            const size_t i = static_cast<size_t>(std::stoul(ids.substr(at, end - at)));
            if (i < told) pattern.columns.push_back(missing[i]->facts.name);
            at = end + 1;
        }
        s.patterns.push_back(std::move(pattern));
    }
    return s;
}

RowPage MissingRows(const std::string& table, const std::string& column, const std::vector<std::string>& texts,
                    size_t offset, const QueryRunner& run, size_t limit) {
    RowPage page;
    page.offset = offset;
    std::vector<QueryParam> params;
    const std::string cond = MissingCondition(column, texts, params);
    QueryResult r = Run(run, table,
                        "SELECT * FROM " + Numbered(table) + " WHERE " + cond + " ORDER BY \"row\" LIMIT " + std::to_string(limit) +
                            " OFFSET " + std::to_string(offset),
                        std::move(params), "Missing rows");
    if (!r.ok) page.error = ErrorOf(r);
    else page.rows = r.table;
    return page;
}

const char* OutlierRuleName(OutlierRule rule) {
    switch (rule) {
        case OutlierRule::Iqr: return "IQR";
        case OutlierRule::ZScore: return "Z-score";
        case OutlierRule::ModifiedZ: return "Modified Z";
    }
    return "";
}

const char* OutlierScoreName(OutlierRule rule) {
    switch (rule) {
        case OutlierRule::Iqr: return "IQRs past the box";
        case OutlierRule::ZScore: return "z";
        case OutlierRule::ModifiedZ: return "modified z";
    }
    return "";
}

const char* OutlierRuleExplain(OutlierRule rule) {
    switch (rule) {
        case OutlierRule::Iqr: return "IQR: outside Q1 - k x IQR and Q3 + k x IQR. Score: how many IQRs past the box.";
        case OutlierRule::ZScore: return "Z-score: outside mean +- k x std. Score: z.";
        case OutlierRule::ModifiedZ:
            return "Modified Z: outside median +- k x MAD / 0.6745, robust to the outliers themselves. Score: 0.6745 x (x - median) / MAD.";
    }
    return "";
}

double DefaultOutlierFactor(OutlierRule rule) {
    switch (rule) {
        case OutlierRule::Iqr: return 1.5;
        case OutlierRule::ZScore: return 3.0;
        case OutlierRule::ModifiedZ: return 3.5;
    }
    return 1.5;
}

OutlierBounds BoundsFor(const ProfiledColumn& c, OutlierRule rule, double factor, double mad) {
    OutlierBounds b;
    b.mad = mad;
    switch (rule) {
        case OutlierRule::Iqr: {
            const double iqr = c.q3 - c.q1;
            if (!std::isfinite(iqr) || iqr <= 0) {
                b.unavailable = "Half the values or more are the same (Q1 = Q3), so IQR finds no spread to measure against.";
                break;
            }
            b.lo = c.q1 - factor * iqr;
            b.hi = c.q3 + factor * iqr;
            break;
        }
        case OutlierRule::ZScore:
            if (!std::isfinite(c.std) || c.std <= 0) {
                b.unavailable = "Every value is the same, so there is no spread to measure against.";
                break;
            }
            b.lo = c.mean - factor * c.std;
            b.hi = c.mean + factor * c.std;
            break;
        case OutlierRule::ModifiedZ:
            if (!std::isfinite(mad) || mad <= 0) {
                b.unavailable = "Half the values or more equal the median (MAD = 0), so Modified Z finds no spread to measure against.";
                break;
            }
            b.lo = c.median - factor * mad / 0.6745;
            b.hi = c.median + factor * mad / 0.6745;
            break;
    }
    return b;
}

double MedianAbsoluteDeviation(const std::string& table, const ProfiledColumn& c, const QueryRunner& run, std::string* error) {
    const std::string col = "CAST(" + Q::QuoteIdentifier(c.facts.name) + " AS DOUBLE)";
    QueryResult r = Run(run, table,
                        "SELECT CAST(quantile_cont(abs(" + col + " - m), 0.5) AS DOUBLE) AS mad FROM " + Q::QuoteIdentifier(table) +
                            ", (SELECT quantile_cont(" + col + ", 0.5) AS m FROM " + Q::QuoteIdentifier(table) + ") WHERE " +
                            Q::QuoteIdentifier(c.facts.name) + " IS NOT NULL",
                        {}, "Outliers");
    if (!r.ok) {
        if (error) *error = ErrorOf(r);
        return NAN;
    }
    return Value(r.table, "mad");
}

OutlierCounts CountOutliers(const std::string& table, const ProfiledColumn& c, const OutlierBounds& b, const QueryRunner& run) {
    OutlierCounts n;
    const std::string col = Q::QuoteIdentifier(c.facts.name);
    QueryResult r = Run(run, table,
                        "SELECT count_if(" + col + " < " + Num(b.lo) + ") AS lo, count_if(" + col + " > " + Num(b.hi) + ") AS hi FROM " +
                            Q::QuoteIdentifier(table),
                        {}, "Outliers");
    if (!r.ok) {
        n.error = ErrorOf(r);
        return n;
    }
    n.low = Count(r.table, "lo");
    n.high = Count(r.table, "hi");
    return n;
}

RowPage OutlierRows(const std::string& table, const ProfiledColumn& c, OutlierRule rule, const OutlierBounds& b, size_t offset,
                    const QueryRunner& run, size_t limit) {
    RowPage page;
    page.offset = offset;
    const std::string col = Q::QuoteIdentifier(c.facts.name);
    const std::string x = "CAST(" + col + " AS DOUBLE)";
    std::string score;
    switch (rule) {
        case OutlierRule::Iqr: {
            const double iqr = c.q3 - c.q1;
            score = "CASE WHEN " + x + " > " + Num(c.q3) + " THEN (" + x + " - " + Num(c.q3) + ") / " + Num(iqr) + " ELSE (" + x +
                    " - " + Num(c.q1) + ") / " + Num(iqr) + " END";
            break;
        }
        case OutlierRule::ZScore: score = "(" + x + " - " + Num(c.mean) + ") / " + Num(c.std); break;
        case OutlierRule::ModifiedZ: score = "0.6745 * (" + x + " - " + Num(c.median) + ") / " + Num(b.mad); break;
    }
    QueryResult r = Run(run, table,
                        "SELECT \"row\", " + score + " AS score, * EXCLUDE (\"row\") FROM " + Numbered(table) + " WHERE " + col + " < " +
                            Num(b.lo) + " OR " + col + " > " + Num(b.hi) + " ORDER BY abs(score) DESC, \"row\" LIMIT " +
                            std::to_string(limit) + " OFFSET " + std::to_string(offset),
                        {}, "Outlier rows");
    if (!r.ok) page.error = ErrorOf(r);
    else page.rows = r.table;
    return page;
}

Bins CountBins(const std::string& table, const std::string& column, double min, double max, int bins, const QueryRunner& run) {
    Bins b;
    if (bins < 1 || !std::isfinite(min) || !std::isfinite(max) || max <= min) {
        b.error = "No range to count in.";
        return b;
    }
    const double width = (max - min) / bins;
    for (int i = 0; i <= bins; ++i) b.edges.push_back(min + width * i);
    b.counts.assign(static_cast<size_t>(bins), 0);
    const std::string col = Q::QuoteIdentifier(column);
    QueryResult r = Run(run, table,
                        "SELECT least(greatest(CAST(floor((CAST(" + col + " AS DOUBLE) - " + Num(min) + ") / " + Num(width) +
                            ") AS BIGINT), 0), " + std::to_string(bins - 1) + ") AS b, count(*) AS n FROM " +
                            Q::QuoteIdentifier(table) + " WHERE " + col + " IS NOT NULL GROUP BY 1",
                        {}, "Histogram");
    if (!r.ok) {
        b.error = ErrorOf(r);
        return b;
    }
    for (int64_t row = 0; r.table && row < r.table->num_rows(); ++row) {
        const double i = Value(r.table, "b", row);
        if (std::isfinite(i) && i >= 0 && i < bins) b.counts[static_cast<size_t>(i)] += Count(r.table, "n", row);
    }
    return b;
}

ColumnFigures Figures(const std::string& table, const std::string& column, bool exact, const QueryRunner& run) {
    ColumnFigures f;
    const std::string col = "CAST(" + Q::QuoteIdentifier(column) + " AS DOUBLE)";
    const std::string from = " FROM " + Q::QuoteIdentifier(table);
    const std::string q = exact ? "quantile_cont(" : "approx_quantile(";
    std::string sql = "SELECT count(" + col + ") AS n, avg(" + col + ") AS mean, stddev_samp(" + col + ") AS sd, var_samp(" + col +
                      ") AS var, min(" + col + ") AS lo, max(" + col + ") AS hi, sum(" + col + ") AS total, skewness(" + col +
                      ") AS skew, kurtosis(" + col + ") AS kurt";
    for (size_t i = 0; i < std::size(kPercentiles); ++i)
        sql += ", CAST(" + q + col + ", " + Num(kPercentiles[i]) + ") AS DOUBLE) AS p" + std::to_string(i);
    QueryResult r = Run(run, table, sql + from, {}, "Distribution");
    if (!r.ok) {
        f.error = ErrorOf(r);
        return f;
    }
    f.count = Count(r.table, "n");
    f.mean = Value(r.table, "mean");
    f.std = Value(r.table, "sd");
    f.variance = Value(r.table, "var");
    f.min = Value(r.table, "lo");
    f.max = Value(r.table, "hi");
    f.sum = Value(r.table, "total");
    f.skewness = Value(r.table, "skew");
    f.kurtosis = Value(r.table, "kurt");
    for (size_t i = 0; i < std::size(kPercentiles); ++i) f.percentiles.push_back(Value(r.table, "p" + std::to_string(i)));
    f.q1 = f.percentiles[2];
    f.median = f.percentiles[3];
    f.q3 = f.percentiles[4];
    f.iqr = f.q3 - f.q1;
    f.range = f.max - f.min;
    f.sem = f.count > 1 ? f.std / std::sqrt(static_cast<double>(f.count)) : NAN;
    f.cv = f.mean != 0 ? 100.0 * f.std / std::fabs(f.mean) : NAN;

    // The most frequent value (the smallest of a tie) and the whiskers.
    const double lo = f.q1 - 1.5 * f.iqr, hi = f.q3 + 1.5 * f.iqr;
    QueryResult m = Run(run, table,
                        "SELECT " + col + " AS v, count(*) AS n" + from + " WHERE " + Q::QuoteIdentifier(column) +
                            " IS NOT NULL GROUP BY 1 ORDER BY n DESC, v LIMIT 1",
                        {}, "Distribution");
    QueryResult w = Run(run, table,
                        "SELECT min(" + col + ") FILTER (WHERE " + col + " >= " + Num(lo) + ") AS lo, max(" + col + ") FILTER (WHERE " +
                            col + " <= " + Num(hi) + ") AS hi" + from,
                        {}, "Distribution");
    if (!m.ok || !w.ok) {
        f.error = ErrorOf(m.ok ? w : m);
        return f;
    }
    f.mode = Value(m.table, "v");
    f.mode_count = Count(m.table, "n");
    f.whisker_lo = Value(w.table, "lo");
    f.whisker_hi = Value(w.table, "hi");
    return f;
}

ColumnFit FitColumn(const std::string& table, const std::string& column, size_t non_null, const QueryRunner& run) {
    ColumnFit c;
    const std::string col = Q::QuoteIdentifier(column);
    c.sampled = non_null > kFitValues;
    std::string sql = "SELECT CAST(" + col + " AS DOUBLE) AS v FROM " + Q::QuoteIdentifier(table) + " WHERE " + col + " IS NOT NULL";
    if (c.sampled) sql = "SELECT v FROM (" + sql + ") USING SAMPLE reservoir(" + std::to_string(kFitValues) + " ROWS) REPEATABLE (42)";
    QueryResult r = Run(run, table, sql, {}, "Distribution fits");
    if (!r.ok) {
        c.error = ErrorOf(r);
        return c;
    }
    auto values = r.table ? r.table->GetColumnByName("v") : nullptr;
    for (int k = 0; values && k < values->num_chunks(); ++k) {
        auto chunk = std::static_pointer_cast<arrow::DoubleArray>(values->chunk(k));
        for (int64_t i = 0; i < chunk->length(); ++i)
            if (chunk->IsValid(i) && std::isfinite(chunk->Value(i))) c.values.push_back(chunk->Value(i));
    }
    std::sort(c.values.begin(), c.values.end());
    if (c.values.size() < 3) {
        c.error = "Too few values to fit a distribution.";
        return c;
    }
    c.fits = DataAnalyzer::FitAllDistributions(c.values);
    return c;
}

namespace {
double Param(const DistributionFitResult& fit, const char* name) {
    auto it = fit.parameters.find(name);
    return it == fit.parameters.end() ? NAN : it->second;
}
}  // namespace

double NormalQuantile(double p) {
    if (!(p > 0 && p < 1)) return NAN;
    // Acklam's rational approximation, then one Newton step on the CDF.
    static const double a[] = {-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239};
    static const double b[] = {-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572};
    static const double c[] = {-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783};
    static const double d[] = {0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416};
    const auto tail = [&](double q) {
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    };
    double x;
    if (p < 0.02425) x = tail(std::sqrt(-2 * std::log(p)));
    else if (p > 1 - 0.02425) x = -tail(std::sqrt(-2 * std::log(1 - p)));
    else {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2 * 3.14159265358979323846) * std::exp(x * x / 2);
    return x - u / (1 + x * u / 2);
}

double FitDensity(const DistributionFitResult& fit, double x) {
    constexpr double kTwoPi = 2 * 3.14159265358979323846;
    switch (fit.type) {
        case DistributionType::Normal: {
            const double mu = Param(fit, "mu"), s = Param(fit, "sigma");
            return std::exp(-0.5 * std::pow((x - mu) / s, 2)) / (s * std::sqrt(kTwoPi));
        }
        case DistributionType::LogNormal: {
            if (x <= 0) return 0;
            const double mu = Param(fit, "mu"), s = Param(fit, "sigma");
            return std::exp(-0.5 * std::pow((std::log(x) - mu) / s, 2)) / (x * s * std::sqrt(kTwoPi));
        }
        case DistributionType::Uniform: {
            const double a = Param(fit, "a"), b = Param(fit, "b");
            return x >= a && x <= b && b > a ? 1 / (b - a) : 0;
        }
        case DistributionType::Exponential: {
            const double l = Param(fit, "lambda");
            return x < 0 ? 0 : l * std::exp(-l * x);
        }
        default: return NAN;
    }
}

double FitQuantile(const DistributionFitResult& fit, double p) {
    switch (fit.type) {
        case DistributionType::Normal: return Param(fit, "mu") + Param(fit, "sigma") * NormalQuantile(p);
        case DistributionType::LogNormal: return std::exp(Param(fit, "mu") + Param(fit, "sigma") * NormalQuantile(p));
        case DistributionType::Uniform: return Param(fit, "a") + p * (Param(fit, "b") - Param(fit, "a"));
        case DistributionType::Exponential: return -std::log(1 - p) / Param(fit, "lambda");
        default: return NAN;
    }
}

std::vector<std::pair<double, double>> QqPoints(const ColumnFit& column, const DistributionFitResult& fit, size_t points) {
    std::vector<std::pair<double, double>> out;
    const size_t n = column.values.size();
    if (n == 0 || points == 0) return out;
    const size_t k = std::min(n, points);
    for (size_t i = 0; i < k; ++i) {
        const size_t rank = k == 1 ? 0 : i * (n - 1) / (k - 1);
        const double p = (static_cast<double>(rank) + 0.5) / static_cast<double>(n);
        out.emplace_back(FitQuantile(fit, p), column.values[rank]);
    }
    return out;
}

}  // namespace cyxwiz::explore
