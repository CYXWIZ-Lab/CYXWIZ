#include "data_explore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

}  // namespace cyxwiz::explore
