#include "explore_read.h"

#include "session_query_engine.h"

#include <cmath>

namespace cyxwiz::explore_read {

namespace {

using Q = SessionQueryEngine;

const std::vector<std::string>& TextsFor(const explore::MissingTexts& texts, const std::string& column) {
    static const std::vector<std::string> kNone;
    const auto it = texts.find(column);
    return it == texts.end() ? kNone : it->second;
}

}  // namespace

// SELECT expressions: a number column as DOUBLE (NULL when missing or not a
// number), a label column as VARCHAR (NULL when missing).
std::string NumberExpr(const std::string& column, const explore::MissingTexts& texts, std::vector<QueryParam>& params,
                       const std::string& as) {
    const std::string missing = explore::MissingCondition(column, TextsFor(texts, column), params);
    return "CASE WHEN " + missing + " THEN NULL ELSE TRY_CAST(" + Q::QuoteIdentifier(column) + " AS DOUBLE) END AS " +
           Q::QuoteIdentifier(as);
}
std::string LabelExpr(const std::string& column, const explore::MissingTexts& texts, std::vector<QueryParam>& params,
                      const std::string& as) {
    const std::string missing = explore::MissingCondition(column, TextsFor(texts, column), params);
    return "CASE WHEN " + missing + " THEN NULL ELSE CAST(" + Q::QuoteIdentifier(column) + " AS VARCHAR) END AS " +
           Q::QuoteIdentifier(as);
}

Read Select(const std::string& table, const std::vector<std::string>& exprs, std::vector<QueryParam> params,
            const QueryRunner& run, const char* label) {
    QueryRequest r;
    r.sql = "SELECT ";
    for (size_t i = 0; i < exprs.size(); ++i) r.sql += (i ? ", " : "") + exprs[i];
    r.sql += " FROM " + Q::QuoteIdentifier(table);
    r.params = std::move(params);
    r.inputs = {table};
    r.label = label;
    const QueryResult q = run(r);
    if (!q.ok || !q.table) return {nullptr, q.cancelled ? "Cancelled." : q.error.empty() ? "the query returned nothing" : q.error};
    return {q.table, {}};
}

std::vector<std::optional<double>> Numbers(const arrow::Table& t, const std::string& name) {
    std::vector<std::optional<double>> out;
    const auto col = t.GetColumnByName(name);
    if (!col) return out;
    out.reserve(static_cast<size_t>(col->length()));
    for (const auto& chunk : col->chunks()) {
        const auto& d = static_cast<const arrow::DoubleArray&>(*chunk);
        for (int64_t i = 0; i < d.length(); ++i) {
            if (d.IsNull(i) || !std::isfinite(d.Value(i))) out.emplace_back();
            else out.emplace_back(d.Value(i));
        }
    }
    return out;
}

std::vector<std::optional<std::string>> Labels(const arrow::Table& t, const std::string& name) {
    std::vector<std::optional<std::string>> out;
    const auto col = t.GetColumnByName(name);
    if (!col) return out;
    out.reserve(static_cast<size_t>(col->length()));
    for (const auto& chunk : col->chunks()) {
        const auto& s = static_cast<const arrow::StringArray&>(*chunk);
        for (int64_t i = 0; i < s.length(); ++i) {
            if (s.IsNull(i)) out.emplace_back();
            else out.emplace_back(s.GetString(i));
        }
    }
    return out;
}

}  // namespace cyxwiz::explore_read
