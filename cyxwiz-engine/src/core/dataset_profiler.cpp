#include "dataset_profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

using Q = SessionQueryEngine;

constexpr size_t kBatch = 40;     // columns per aggregate query
constexpr int kBins = 20;
constexpr size_t kTopValues = 10;
constexpr size_t kSampleRows = 200;

std::string Num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

ColumnFacts::Type TypeOf(const std::string& sql_type) {
    std::string t;
    for (char c : sql_type) t += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (t == "BOOLEAN") return ColumnFacts::Type::Boolean;
    if (t == "TINYINT" || t == "SMALLINT" || t == "INTEGER" || t == "BIGINT" || t == "HUGEINT" || t == "UTINYINT" ||
        t == "USMALLINT" || t == "UINTEGER" || t == "UBIGINT")
        return ColumnFacts::Type::Integer;
    if (t == "FLOAT" || t == "REAL" || t == "DOUBLE" || t.rfind("DECIMAL", 0) == 0) return ColumnFacts::Type::Float;
    if (t == "VARCHAR" || t.rfind("VARCHAR", 0) == 0) return ColumnFacts::Type::Text;
    if (t == "DATE" || t.rfind("TIMESTAMP", 0) == 0 || t == "TIME") return ColumnFacts::Type::Temporal;
    return ColumnFacts::Type::Other;
}

bool IsNumber(ColumnFacts::Type t) {
    return t == ColumnFacts::Type::Integer || t == ColumnFacts::Type::Float;
}

// A cell of a one-row (or any) result by column name.
std::shared_ptr<arrow::Scalar> Cell(const std::shared_ptr<arrow::Table>& t, const std::string& name, int64_t row = 0) {
    if (!t) return nullptr;
    auto col = t->GetColumnByName(name);
    if (!col || row >= col->length()) return nullptr;
    auto s = col->GetScalar(row);
    return s.ok() && (*s)->is_valid ? *s : nullptr;
}

double AsDouble(const std::shared_ptr<arrow::Scalar>& s) {
    if (!s) return NAN;
    auto d = s->CastTo(arrow::float64());
    return d.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*d)->value : NAN;
}

size_t AsCount(const std::shared_ptr<arrow::Scalar>& s) {
    const double d = AsDouble(s);
    return std::isfinite(d) && d > 0 ? static_cast<size_t>(std::llround(d)) : 0;
}

std::string AsText(const std::shared_ptr<arrow::Scalar>& s) {
    if (!s) return {};
    if (s->type->id() == arrow::Type::STRING) return std::static_pointer_cast<arrow::StringScalar>(s)->ToString();
    return s->ToString();
}

}  // namespace

std::vector<ColumnFacts> DatasetProfile::Facts() const {
    std::vector<ColumnFacts> out;
    for (const auto& c : columns) out.push_back(c.facts);
    return out;
}

const ProfiledColumn* DatasetProfile::Find(const std::string& name) const {
    for (const auto& c : columns)
        if (c.facts.name == name) return &c;
    return nullptr;
}

size_t DatasetProfile::MissingCells() const {
    size_t n = 0;
    for (const auto& c : columns) n += c.missing;
    return n;
}

DatasetProfile ProfileTable(const std::string& table, const QueryRunner& run, const ProfileOptions& options) {
    DatasetProfile p;
    p.table = table;
    const auto start = std::chrono::steady_clock::now();
    const std::string from = " FROM " + Q::QuoteIdentifier(table);
    const auto stop = [&] { return options.should_stop && options.should_stop(); };
    const auto progress = [&](float f, const std::string& what) {
        if (options.progress) options.progress(f, what);
    };
    const auto query = [&](const std::string& sql, std::vector<QueryParam> params = {}) {
        QueryRequest r;
        r.sql = sql;
        r.params = std::move(params);
        r.inputs = {table};
        r.label = "Profile";
        r.cache = true;  // the same data profiled again (Data Studio, a dashboard, next session) reads the saved answers
        QueryResult res = run(r);
        if (!res.ok && p.error.empty()) p.error = res.cancelled ? "Cancelled." : res.error;
        return res;
    };

    // 1. Columns and their types.
    progress(0.02f, "Reading the columns");
    QueryResult schema = query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
                               {QueryParam::Of(table)});
    if (!schema.ok) return p;
    for (int64_t r = 0; r < schema.table->num_rows(); ++r) {
        ProfiledColumn c;
        c.facts.name = AsText(Cell(schema.table, "column_name", r));
        c.sql_type = AsText(Cell(schema.table, "data_type", r));
        c.facts.type = TypeOf(c.sql_type);
        c.numeric = IsNumber(c.facts.type);
        p.columns.push_back(std::move(c));
    }
    if (p.columns.empty()) {
        p.error = "'" + table + "' has no columns.";
        return p;
    }
    const bool wide = p.columns.size() > kExactColumns;
    p.exact = !wide;

    // 2. Counts, distinct values, ranges, quartiles, text length (in batches of columns).
    for (size_t b = 0; b < p.columns.size(); b += kBatch) {
        if (stop()) {
            p.error = "Cancelled.";
            return p;
        }
        progress(0.05f + 0.45f * static_cast<float>(b) / static_cast<float>(p.columns.size()), "Counting values");
        std::string sql = "SELECT count(*) AS n";
        std::vector<QueryParam> params;
        const size_t end = std::min(p.columns.size(), b + kBatch);
        for (size_t i = b; i < end; ++i) {
            const auto& c = p.columns[i];
            const std::string col = Q::QuoteIdentifier(c.facts.name);
            const std::string k = "c" + std::to_string(i) + "_";
            sql += ", count(" + col + ") AS " + k + "nn";
            sql += wide ? ", approx_count_distinct(" + col + ") AS " + k + "d" : ", count(DISTINCT " + col + ") AS " + k + "d";
            auto mt = options.missing_text.find(c.facts.name);
            if (mt != options.missing_text.end() && !mt->second.empty()) {
                std::string in;
                for (const auto& text : mt->second) {
                    in += (in.empty() ? "?" : ", ?");
                    params.push_back(QueryParam::Of(text));
                }
                sql += ", count_if(CAST(" + col + " AS VARCHAR) IN (" + in + ")) AS " + k + "mt";
            }
            if (c.numeric) {
                const std::string q = wide ? "approx_quantile(" : "quantile_cont(";
                sql += ", CAST(min(" + col + ") AS DOUBLE) AS " + k + "min, CAST(max(" + col + ") AS DOUBLE) AS " + k + "max";
                sql += ", CAST(avg(" + col + ") AS DOUBLE) AS " + k + "mean, CAST(stddev_samp(" + col + ") AS DOUBLE) AS " + k + "std";
                sql += ", CAST(" + q + col + ", 0.25) AS DOUBLE) AS " + k + "q1, CAST(" + q + col + ", 0.5) AS DOUBLE) AS " + k + "q2";
                sql += ", CAST(" + q + col + ", 0.75) AS DOUBLE) AS " + k + "q3";
            } else if (c.facts.type == ColumnFacts::Type::Text) {
                sql += ", CAST(avg(length(" + col + ")) AS DOUBLE) AS " + k + "len";
                sql += ", min(" + col + ") AS " + k + "lo, max(" + col + ") AS " + k + "hi";
            } else if (c.facts.type == ColumnFacts::Type::Temporal) {
                sql += ", CAST(min(" + col + ") AS VARCHAR) AS " + k + "lo, CAST(max(" + col + ") AS VARCHAR) AS " + k + "hi";
            }
        }
        QueryResult r = query(sql + from, std::move(params));
        if (!r.ok) return p;
        p.rows = AsCount(Cell(r.table, "n"));
        for (size_t i = b; i < end; ++i) {
            auto& c = p.columns[i];
            const std::string k = "c" + std::to_string(i) + "_";
            const size_t nn = AsCount(Cell(r.table, k + "nn"));
            c.missing_text = AsCount(Cell(r.table, k + "mt"));
            c.missing = p.rows - std::min(p.rows, nn) + c.missing_text;
            c.facts.rows = p.rows;
            c.facts.non_null = nn - std::min(nn, c.missing_text);
            c.facts.distinct = AsCount(Cell(r.table, k + "d"));
            if (c.missing_text > 0 && c.facts.distinct > 0) {
                // The missing texts were counted as values.
                const auto& texts = options.missing_text.at(c.facts.name);
                c.facts.distinct -= std::min(c.facts.distinct, texts.size());
            }
            if (c.numeric) {
                c.min = AsDouble(Cell(r.table, k + "min"));
                c.max = AsDouble(Cell(r.table, k + "max"));
                c.mean = AsDouble(Cell(r.table, k + "mean"));
                c.std = AsDouble(Cell(r.table, k + "std"));
                c.q1 = AsDouble(Cell(r.table, k + "q1"));
                c.median = AsDouble(Cell(r.table, k + "q2"));
                c.q3 = AsDouble(Cell(r.table, k + "q3"));
            } else {
                c.facts.avg_length = AsDouble(Cell(r.table, k + "len"));
                if (!std::isfinite(c.facts.avg_length)) c.facts.avg_length = 0;
                c.min_text = AsText(Cell(r.table, k + "lo"));
                c.max_text = AsText(Cell(r.table, k + "hi"));
            }
        }
    }

    // 3. Text samples: do they read as dates or file paths?
    std::vector<size_t> texts;
    for (size_t i = 0; i < p.columns.size(); ++i)
        if (p.columns[i].facts.type == ColumnFacts::Type::Text) texts.push_back(i);
    for (size_t b = 0; b < texts.size() && b < kDetailColumns; b += kBatch) {
        if (stop()) {
            p.error = "Cancelled.";
            return p;
        }
        progress(0.55f, "Looking at text values");
        std::string sql = "SELECT ";
        const size_t end = std::min(texts.size(), b + kBatch);
        for (size_t j = b; j < end; ++j) sql += (j == b ? "" : ", ") + Q::QuoteIdentifier(p.columns[texts[j]].facts.name);
        QueryResult r = query(sql + from + " LIMIT " + std::to_string(kSampleRows));
        if (!r.ok) return p;
        for (size_t j = b; j < end; ++j) {
            auto& c = p.columns[texts[j]];
            auto col = r.table->GetColumnByName(c.facts.name);
            size_t seen = 0, dates = 0, paths = 0;
            for (int64_t row = 0; col && row < col->length(); ++row) {
                const std::string v = AsText(Cell(r.table, c.facts.name, row));
                if (v.empty()) continue;
                ++seen;
                if (LooksLikeDate(v)) ++dates;
                if (LooksLikeMediaPath(v)) ++paths;
            }
            c.facts.date_share = seen ? static_cast<double>(dates) / static_cast<double>(seen) : 0.0;
            c.facts.path_share = seen ? static_cast<double>(paths) / static_cast<double>(seen) : 0.0;
        }
    }

    // 4. Top values (text, booleans, numbers with few values) and histograms (numbers).
    const size_t detail = std::min(p.columns.size(), kDetailColumns);
    for (size_t i = 0; i < detail; ++i) {
        if (stop()) {
            p.error = "Cancelled.";
            return p;
        }
        progress(0.6f + 0.3f * static_cast<float>(i) / static_cast<float>(detail), "Values of " + p.columns[i].facts.name);
        auto& c = p.columns[i];
        const std::string col = Q::QuoteIdentifier(c.facts.name);
        if (!c.numeric || c.facts.distinct <= 50) {
            QueryResult r = query("SELECT CAST(" + col + " AS VARCHAR) AS v, count(*) AS n" + from + " WHERE " + col +
                                  " IS NOT NULL GROUP BY 1 ORDER BY n DESC, v LIMIT " + std::to_string(kTopValues));
            if (!r.ok) return p;
            for (int64_t row = 0; row < r.table->num_rows(); ++row)
                c.top.emplace_back(AsText(Cell(r.table, "v", row)), AsCount(Cell(r.table, "n", row)));
        }
        if (c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min) {
            const double width = (c.max - c.min) / kBins;
            QueryResult r = query("SELECT least(CAST(floor((CAST(" + col + " AS DOUBLE) - " + Num(c.min) + ") / " + Num(width) +
                                  ") AS BIGINT), " + std::to_string(kBins - 1) + ") AS b, count(*) AS n" + from + " WHERE " + col +
                                  " IS NOT NULL GROUP BY 1");
            if (!r.ok) return p;
            c.hist_counts.assign(kBins, 0);
            for (int b = 0; b <= kBins; ++b) c.hist_edges.push_back(c.min + width * b);
            for (int64_t row = 0; row < r.table->num_rows(); ++row) {
                const double b = AsDouble(Cell(r.table, "b", row));
                if (std::isfinite(b) && b >= 0 && b < kBins) c.hist_counts[static_cast<size_t>(b)] += AsCount(Cell(r.table, "n", row));
            }
        }
    }

    // Outliers: outside the 1.5 x IQR fences (one query per batch of numbers).
    std::vector<size_t> nums;
    for (size_t i = 0; i < detail; ++i)
        if (p.columns[i].numeric && std::isfinite(p.columns[i].q1) && std::isfinite(p.columns[i].q3)) nums.push_back(i);
    for (size_t b = 0; b < nums.size(); b += kBatch) {
        std::string sql = "SELECT 1 AS one";
        const size_t end = std::min(nums.size(), b + kBatch);
        for (size_t j = b; j < end; ++j) {
            const auto& c = p.columns[nums[j]];
            const double iqr = c.q3 - c.q1;
            const std::string col = Q::QuoteIdentifier(c.facts.name);
            sql += ", count_if(" + col + " < " + Num(c.q1 - 1.5 * iqr) + " OR " + col + " > " + Num(c.q3 + 1.5 * iqr) + ") AS o" +
                   std::to_string(nums[j]);
        }
        QueryResult r = query(sql + from);
        if (!r.ok) return p;
        for (size_t j = b; j < end; ++j) p.columns[nums[j]].outliers = AsCount(Cell(r.table, "o" + std::to_string(nums[j])));
    }

    // 5. Duplicate rows (narrow enough tables).
    if (p.columns.size() <= kDetailColumns && !stop()) {
        progress(0.92f, "Looking for duplicate rows");
        QueryResult r = query("SELECT count(*) AS n FROM (SELECT DISTINCT *" + from + ")");
        if (!r.ok) return p;
        const size_t distinct_rows = AsCount(Cell(r.table, "n"));
        p.duplicates_known = true;
        p.duplicate_rows = p.rows - std::min(p.rows, distinct_rows);
    }

    // 6. The strongest correlations among the first numbers.
    std::vector<size_t> corr_cols;
    for (size_t i = 0; i < p.columns.size() && corr_cols.size() < kCorrelationColumns; ++i)
        if (p.columns[i].numeric && p.columns[i].max > p.columns[i].min) corr_cols.push_back(i);
    if (corr_cols.size() >= 2 && !stop()) {
        progress(0.96f, "Correlations");
        std::string sql = "SELECT 1 AS one";
        for (size_t a = 0; a < corr_cols.size(); ++a)
            for (size_t b = a + 1; b < corr_cols.size(); ++b)
                sql += ", corr(CAST(" + Q::QuoteIdentifier(p.columns[corr_cols[a]].facts.name) + " AS DOUBLE), CAST(" +
                       Q::QuoteIdentifier(p.columns[corr_cols[b]].facts.name) + " AS DOUBLE)) AS r" + std::to_string(a) + "_" + std::to_string(b);
        QueryResult r = query(sql + from);
        if (!r.ok) return p;
        for (size_t a = 0; a < corr_cols.size(); ++a)
            for (size_t b = a + 1; b < corr_cols.size(); ++b) {
                const double v = AsDouble(Cell(r.table, "r" + std::to_string(a) + "_" + std::to_string(b)));
                if (std::isfinite(v))
                    p.correlations.emplace_back(p.columns[corr_cols[a]].facts.name, p.columns[corr_cols[b]].facts.name, v);
            }
        std::stable_sort(p.correlations.begin(), p.correlations.end(),
                         [](const auto& x, const auto& y) { return std::fabs(std::get<2>(x)) > std::fabs(std::get<2>(y)); });
        if (p.correlations.size() > 10) p.correlations.resize(10);
    }
    p.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    progress(1.0f, "Profiled");
    return p;
}

}  // namespace cyxwiz
