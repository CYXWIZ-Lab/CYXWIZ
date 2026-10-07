#include "dashboard_runtime.h"

#include "../plot/plot_prepare.h"
#include "stop_words.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>

namespace cyxwiz::dashboard {

namespace {

std::string Quote(const std::string& name) {
    std::string q = "\"";
    for (char c : name) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

bool NumberType(const std::string& type) {
    return type == "int" || type == "float";
}

std::string Where(const std::string& cond) {
    return cond.empty() ? std::string() : " WHERE " + cond;
}

// " AND word NOT IN ('a', ...)" for `word` (empty when common words are kept).
std::string StopWordFilter(const std::string& word, bool keep) {
    if (keep) return std::string();
    std::string in;
    for (const auto& w : EnglishStopWords()) {
        in += in.empty() ? "'" : ", '";
        for (char c : w) in += c == '\'' ? std::string("''") : std::string(1, c);
        in += "'";
    }
    return " AND " + word + " NOT IN (" + in + ")";
}

std::string MeasureSql(Measure m, const std::string& col) {
    switch (m) {
        case Measure::Count: return "CAST(count(*) AS DOUBLE)";
        case Measure::Sum: return "CAST(sum(" + col + ") AS DOUBLE)";
        case Measure::Mean: return "CAST(avg(" + col + ") AS DOUBLE)";
        case Measure::Median: return "CAST(median(" + col + ") AS DOUBLE)";
        case Measure::Min: return "CAST(min(" + col + ") AS DOUBLE)";
        case Measure::Max: return "CAST(max(" + col + ") AS DOUBLE)";
        case Measure::Distinct: return "CAST(count(DISTINCT " + col + ") AS DOUBLE)";
        case Measure::MissingPct: return "100.0 * count_if(" + col + " IS NULL) / greatest(count(*), 1)";
        case Measure::MedianWords: return "CAST(median(len(" + TokensSql(col) + ")) AS DOUBLE)";
        case Measure::EmptyTexts: return "CAST(count_if(" + col + " IS NULL OR trim(CAST(" + col + " AS VARCHAR)) = '') AS DOUBLE)";
        case Measure::Vocabulary: return "CAST(count(DISTINCT word) AS DOUBLE)";  // over the words (KpiQuery)
    }
    return "CAST(count(*) AS DOUBLE)";
}

}  // namespace

Binding CheckBinding(const WidgetSpec& w, const DatasetContract& contract, const std::map<std::string, std::string>& known_types) {
    Binding b;
    if (w.IsQuery()) return b;  // its columns come from the query (an error there says so)
    const bool text = w.IsText();
    for (const auto& field : w.Fields()) {
        if (contract.Find(field)) continue;
        b.state = Binding::State::FieldMissing;
        b.field = field;
        // A rename: exactly one column that is new since the dashboard last bound, of the same type.
        auto known = known_types.find(field);
        std::vector<std::string> candidates;
        if (known != known_types.end())
            for (const auto& c : contract.columns)
                if (!known_types.count(c.name) && c.type == known->second) candidates.push_back(c.name);
        if (candidates.size() == 1) b.rename_candidate = candidates.front();
        b.message = "Field '" + field + "' no longer exists" +
                    (b.rename_candidate.empty() ? std::string(": rebind or remove the widget.") : ": rebind to '" + b.rename_candidate + "'?");
        return b;
    }
    if (w.type != WidgetType::Plot || text) return b;  // a text widget's plot reads its query's columns
    // A number slot whose column is no longer a number (a type change upstream).
    const WidgetKind& kind = KindOf(w);
    const auto needs_number = [&](const std::string& field, FieldNeed need) {
        const ColumnContract* c = contract.Find(field);
        return c && need == FieldNeed::Number && !NumberType(c->type);
    };
    std::vector<std::string> fields;
    // A year-bucketed date X is a number (its year).
    if (!w.plot.x_column.empty() && w.bucket != "year" && needs_number(w.plot.x_column, kind.x_need)) fields.push_back(w.plot.x_column);
    for (const auto& y : w.plot.y_columns)
        if (needs_number(y, kind.y_need)) fields.push_back(y);
    if (!fields.empty()) {
        b.state = Binding::State::RoleMismatch;
        b.field = fields.front();
        const ColumnContract* c = contract.Find(b.field);
        b.message = "'" + b.field + "' is now " + (c ? c->type : std::string("text")) + ": a " + kind.label +
                    " needs numbers here. A Bar (counts per value) fits it.";
    }
    return b;
}

std::string InlineParams(const std::string& sql, const std::vector<QueryParam>& params) {
    const auto literal = [](const QueryParam& p) -> std::string {
        char buf[64];
        switch (p.type) {
            case QueryParam::Type::Null: return "NULL";
            case QueryParam::Type::Bool: return p.b ? "TRUE" : "FALSE";
            case QueryParam::Type::Int: return std::to_string(p.i);
            case QueryParam::Type::Double:
                if (!std::isfinite(p.d)) return "NULL";
                std::snprintf(buf, sizeof(buf), "%.17g", p.d);
                return buf;
            case QueryParam::Type::Text: {
                std::string q = "'";
                for (char c : p.s) q += c == '\'' ? std::string("''") : std::string(1, c);
                return q + "'";
            }
        }
        return "NULL";
    };
    // Replace each ? outside quoted identifiers and strings, in order.
    std::string out;
    size_t next = 0;
    char quote = 0;
    for (char c : sql) {
        if (quote) {
            if (c == quote) quote = 0;
            out += c;
        } else if (c == '"' || c == '\'') {
            quote = c;
            out += c;
        } else if (c == '?' && next < params.size()) {
            out += literal(params[next++]);
        } else {
            out += c;
        }
    }
    return out;
}

std::string FilteredRowsSql(const std::string& table, const FilterState& filters) {
    std::vector<QueryParam> params;
    const std::string where = filters.WhereFor(std::string(), params);
    return InlineParams("SELECT * FROM " + Quote(table) + Where(where), params);
}

QueryRequest TextWidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap,
                             const std::string& words_table) {
    QueryRequest r;
    r.label = "Dashboard text widget";
    r.cache = true;
    // The words split once when saved (every row with its words), else split here.
    const bool saved = !words_table.empty();
    const TokenColumns tc{w.text_field, "cyxwiz_words", "cyxwiz_word_count"};
    r.inputs = {saved ? words_table : table};
    const std::string col = Quote(w.text_field), tokens = saved ? SavedTokensSql("cyxwiz_words") : TokensSql(col);
    // The rows under the other widgets' filters (a reproducible sample beyond row_cap).
    std::string rows = "SELECT * FROM " + Quote(saved ? words_table : table) + Where(filters.WhereFor(w.id, r.params, saved ? &tc : nullptr));
    if (row_cap > 0) rows = "SELECT * FROM (" + rows + ") AS cyxwiz_rows USING SAMPLE reservoir(" + std::to_string(row_cap) + " ROWS) REPEATABLE (42)";
    const std::string with = "WITH cyxwiz_text AS (" + rows + ")";
    switch (w.text_view) {
        case TextView::None:
        case TextView::Length:
            // The longest 1% of texts in the last bin, so a few very long ones do not squash the rest.
            r.sql = with + ", l AS (SELECT " + (saved ? std::string("cyxwiz_word_count") : "len(" + tokens + ")") + " AS n FROM cyxwiz_text)" +
                    " SELECT CAST(least(n, (SELECT ceil(quantile_cont(n, 0.99)) FROM l)) AS DOUBLE) AS words FROM l";
            break;
        case TextView::Words:
            r.sql = with + ", w AS (SELECT unnest(" + tokens + ") AS word FROM cyxwiz_text) SELECT word, CAST(count(*) AS DOUBLE) AS count FROM w" +
                    " WHERE length(word) > 1" + StopWordFilter("word", w.keep_stop_words) + " GROUP BY word ORDER BY count DESC, word LIMIT 15";
            break;
        case TextView::Phrases:
            r.sql = with + ", t AS (SELECT " + tokens + " AS t FROM cyxwiz_text)" +
                    ", p AS (SELECT unnest(list_transform(range(1, len(t)), lambda i: [t[i], t[i + 1]])) AS pair FROM t)" +
                    " SELECT pair[1] || ' ' || pair[2] AS phrase, CAST(count(*) AS DOUBLE) AS count FROM p" +
                    " WHERE length(pair[1]) > 1 AND length(pair[2]) > 1" + StopWordFilter("pair[1]", w.keep_stop_words) +
                    StopWordFilter("pair[2]", w.keep_stop_words) + " GROUP BY phrase ORDER BY count DESC, phrase LIMIT 12";
            break;
        case TextView::WordsByClass: {
            // Each top word's share of each class's texts (a text counts once per word).
            const std::string cls = w.label_field.empty() ? std::string("'all'") : "CAST(" + Quote(w.label_field) + " AS VARCHAR)";
            r.sql = with + ", r AS (SELECT " + cls + " AS cls, list_distinct(" + tokens + ") AS t FROM cyxwiz_text)" +
                    ", top AS (SELECT word, row_number() OVER (ORDER BY count(*) DESC, word) AS rank FROM (SELECT unnest(t) AS word FROM r)" +
                    " WHERE length(word) > 1" + StopWordFilter("word", w.keep_stop_words) + " GROUP BY word ORDER BY count(*) DESC, word LIMIT 10)" +
                    ", n AS (SELECT cls, count(*) AS n FROM r GROUP BY cls)" + ", x AS (SELECT cls, unnest(t) AS word FROM r)" +
                    " SELECT x.cls AS class, x.word AS word, CAST(count(*) AS DOUBLE) / any_value(n.n) AS share" +
                    " FROM x JOIN top USING (word) JOIN n USING (cls) GROUP BY x.cls, x.word ORDER BY any_value(top.rank), x.cls";
            break;
        }
    }
    return r;
}

QueryRequest WidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap,
                         const std::string& words_table) {
    if (w.IsText()) return TextWidgetQuery(w, table, filters, row_cap, words_table);
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard widget";
    r.cache = true;
    if (w.IsQuery()) {
        // The query's table name stands for the rows under the other widgets' filters.
        const std::string rows = Quote(w.query_table.empty() ? table : w.query_table) + " AS (SELECT * FROM " + Quote(table) +
                                 Where(filters.WhereFor(w.id, r.params)) + ")";
        std::string body = w.query;
        while (!body.empty() && (body.back() == ';' || std::isspace(static_cast<unsigned char>(body.back())))) body.pop_back();
        // A keyword at `at` (any case), followed by a space; the position after it, or npos.
        const auto word_at = [&](size_t at, const std::string& word) -> size_t {
            at = body.find_first_not_of(" \t\r\n", at);
            if (at == std::string::npos || at + word.size() >= body.size()) return std::string::npos;
            for (size_t i = 0; i < word.size(); ++i)
                if (std::toupper(static_cast<unsigned char>(body[at + i])) != word[i]) return std::string::npos;
            return std::isspace(static_cast<unsigned char>(body[at + word.size()])) ? at + word.size() : std::string::npos;
        };
        if (const size_t after_with = word_at(0, "WITH"); after_with != std::string::npos) {
            // Its own WITH: ours goes first in the same list.
            const size_t after_recursive = word_at(after_with, "RECURSIVE");
            if (after_recursive != std::string::npos) r.sql = "WITH RECURSIVE " + rows + ", " + body.substr(after_recursive);
            else r.sql = "WITH " + rows + ", " + body.substr(after_with);
        } else {
            r.sql = "WITH " + rows + " " + body;
        }
        if (row_cap > 0) r.sql = "SELECT * FROM (" + r.sql + ") AS cyxwiz_rows USING SAMPLE reservoir(" + std::to_string(row_cap) + " ROWS) REPEATABLE (42)";
        return r;
    }
    std::vector<std::string> cols;
    if (w.type == WidgetType::Plot) cols = plot::ColumnsNeeded(w.plot);
    else cols = w.columns;
    std::string select;
    for (const auto& c : cols) {
        // A bucketed date X: its year as a number, under the column's own name.
        const bool year = w.type == WidgetType::Plot && w.bucket == "year" && c == w.plot.x_column;
        select += (select.empty() ? "" : ", ") +
                  (year ? "CAST(year(TRY_CAST(" + Quote(c) + " AS DATE)) AS DOUBLE) AS " + Quote(c) : Quote(c));
    }
    if (select.empty()) select = "*";
    const std::string cond = filters.WhereFor(w.id, r.params);
    r.sql = "SELECT " + select + " FROM " + Quote(table) + Where(cond);
    if (w.type == WidgetType::Table) r.sql += " LIMIT " + std::to_string(std::max(1, w.rows));
    // The sample is taken from the filtered rows (DuckDB samples a FROM before its WHERE).
    else if (row_cap > 0) r.sql = "SELECT * FROM (" + r.sql + ") AS cyxwiz_rows USING SAMPLE reservoir(" + std::to_string(row_cap) + " ROWS) REPEATABLE (42)";
    return r;
}

QueryRequest MissingQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, const std::vector<std::string>& columns,
                          const std::map<std::string, std::vector<std::string>>& missing_text) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard missing values";
    r.cache = true;
    std::string sql = "SELECT count(*) AS rows";
    for (size_t i = 0; i < columns.size(); ++i) {
        const std::string col = Quote(columns[i]);
        std::string cond = col + " IS NULL";
        auto mt = missing_text.find(columns[i]);
        if (mt != missing_text.end() && !mt->second.empty()) {
            std::string in;
            for (const auto& text : mt->second) {
                in += in.empty() ? "?" : ", ?";
                r.params.push_back(QueryParam::Of(text));
            }
            cond += " OR CAST(" + col + " AS VARCHAR) IN (" + in + ")";
        }
        sql += ", count_if(" + cond + ") AS m" + std::to_string(i);
    }
    const std::string where = filters.WhereFor(w.id, r.params);  // after the IN values: params in text order
    r.sql = sql + " FROM " + Quote(table) + Where(where);
    return r;
}

QueryRequest KpiQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, const std::string& words_table) {
    QueryRequest r;
    r.label = "Dashboard KPI";
    r.cache = true;
    // A text measure reads the words split once when they are saved.
    const bool text_measure = w.measure == Measure::MedianWords || w.measure == Measure::Vocabulary || w.measure == Measure::EmptyTexts;
    const bool saved = text_measure && !words_table.empty();
    const std::string from_table = saved ? words_table : table;
    r.inputs = {from_table};
    const TokenColumns tc{w.field, "cyxwiz_words", "cyxwiz_word_count"};
    const std::string col = w.field.empty() ? std::string("NULL") : Quote(w.field);
    const std::string m = saved && w.measure == Measure::MedianWords ? std::string("CAST(median(cyxwiz_word_count) AS DOUBLE)") : MeasureSql(w.measure, col);
    const std::string cond = filters.WhereFor(w.id, r.params, saved ? &tc : nullptr);
    // The vocabulary counts the distinct words of the rows.
    const auto from = [&](const std::string& where) {
        if (w.measure == Measure::Vocabulary)
            return "(SELECT unnest(" + (saved ? SavedTokensSql("cyxwiz_words") : TokensSql(col)) + ") AS word FROM " + Quote(from_table) + where + ")";
        return Quote(from_table) + where;
    };
    r.sql = "SELECT (SELECT " + m + " FROM " + from(Where(cond)) + ") AS value, (SELECT " + m + " FROM " + from(std::string()) + ") AS all_rows";
    return r;
}

QueryRequest StripQuery(const std::string& table, const FilterState& filters, const DatasetProfile& profile, const std::string& target,
                        bool target_numeric, const std::map<std::string, std::vector<std::string>>& missing_text) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard summary";
    r.cache = true;
    std::vector<QueryParam> cond_params;
    const std::string cond = filters.WhereFor("", cond_params);
    const auto with_cond = [&] {
        r.params.insert(r.params.end(), cond_params.begin(), cond_params.end());
        return Where(cond);
    };
    const std::string t = Quote(table);
    std::string sql = "SELECT (SELECT count(*) FROM " + t + with_cond() + ") AS rows_now, (SELECT count(*) FROM " + t + ") AS rows_all";
    if (!profile.columns.empty() && profile.columns.size() <= kDetailColumns) {
        // Missing cells: nulls plus the texts marked as missing (as the profile counts them).
        std::string missing;
        for (const auto& c : profile.columns) {
            const std::string col = Quote(c.facts.name);
            std::string missing_cond = col + " IS NULL";
            auto mt = missing_text.find(c.facts.name);
            if (mt != missing_text.end() && !mt->second.empty()) {
                std::string in;
                for (const auto& text : mt->second) {
                    in += in.empty() ? "?" : ", ?";
                    r.params.push_back(QueryParam::Of(text));
                }
                missing_cond += " OR CAST(" + col + " AS VARCHAR) IN (" + in + ")";
            }
            missing += (missing.empty() ? "" : " + ") + std::string("count_if(") + missing_cond + ")";
        }
        sql += ", (SELECT " + missing + " FROM " + t;
        sql += with_cond() + ") AS missing_now";
    }
    if (!target.empty()) {
        const std::string tc = Quote(target);
        const std::string agg = target_numeric ? "CAST(avg(" + tc + ") AS DOUBLE)" : "CAST(mode(" + tc + ") AS VARCHAR)";
        sql += ", (SELECT " + agg + " FROM " + t + with_cond() + ") AS target_now, (SELECT " + agg + " FROM " + t + ") AS target_all";
    }
    r.sql = sql;
    return r;
}

std::vector<WidgetSpec> AutomaticWidgets(const DatasetProfile& profile, const DatasetContract& contract, DashboardSpec& spec) {
    std::vector<WidgetSpec> out;
    int slot = 0, top = 0;  // `top`: grid rows above the cards (the text KPIs)
    const auto place = [&](WidgetSpec& w) {
        w.plot.legend = !w.plot.color_column.empty();  // one series needs no legend
        w.at = {(slot % 3) * 4, top + (slot / 3) * 3, 4, 3};
        ++slot;
        w.automatic = true;
        w.id = spec.NewId();
        out.push_back(w);
    };
    const auto profile_of = [&](const std::string& name) { return profile.Find(name); };
    const std::optional<std::string> target = contract.Target();
    bool target_numeric = false;
    if (target) {
        const ColumnContract* tc = contract.Find(*target);
        target_numeric = tc && NumberType(tc->type);
    }
    // A text column (TOFIX134 P3 text, board 19): its KPIs in a row on top.
    std::string text_column;
    for (const auto& c : contract.columns)
        if (c.role == ColumnRole::Text && !(target && c.name == *target)) {
            text_column = c.name;
            break;
        }
    if (!text_column.empty()) {
        int x = 0;
        for (Measure m : {Measure::MedianWords, Measure::Vocabulary, Measure::EmptyTexts}) {
            WidgetSpec k;
            k.type = WidgetType::Kpi;
            k.measure = m;
            k.field = text_column;
            k.title = MeasureLabel(m);
            k.at = {x, 0, 4, 1};  // a row of their own, wide enough for their names
            x += 4;
            k.automatic = true;
            k.id = spec.NewId();
            out.push_back(k);
        }
        top = 1;
    }
    if (target) {
        WidgetSpec w;
        w.type = WidgetType::Plot;
        w.plot.kind = target_numeric ? plot::Kind::Histogram : plot::Kind::Bar;
        w.plot.x_column = *target;
        w.title = *target + " (target)";
        place(w);
    }
    if (!text_column.empty()) {
        const auto text_widget = [&](TextView view) {
            WidgetSpec w;
            w.type = WidgetType::Plot;
            w.text_view = view;
            w.text_field = text_column;
            w.title = std::string(TextViewLabel(view)) + (view == TextView::Length ? " (words)" : std::string());
            switch (view) {
                case TextView::None:
                case TextView::Length:
                    w.plot.kind = plot::Kind::Histogram;
                    w.plot.x_column = "words";
                    w.plot.x_label = "words per text (the longest 1% in the last bin)";
                    break;
                case TextView::Words:
                case TextView::Phrases:
                    w.plot.kind = plot::Kind::Bar;
                    w.plot.x_column = view == TextView::Words ? "word" : "phrase";
                    w.plot.y_columns = {"count"};
                    w.plot.bar_horizontal = true;
                    break;
                case TextView::WordsByClass:
                    w.label_field = target && !target_numeric ? *target : std::string();
                    w.plot.kind = plot::Kind::Heatmap;
                    w.plot.x_column = "class";
                    w.plot.y_columns = {"word"};
                    w.plot.value_column = "share";
                    break;
            }
            place(w);
        };
        text_widget(TextView::Length);
        text_widget(TextView::Words);
        text_widget(TextView::Phrases);
        // By class when there is a class column with a readable number of values.
        if (target && !target_numeric) {
            const ProfiledColumn* tp = profile_of(*target);
            if (tp && tp->facts.distinct >= 2 && tp->facts.distinct <= 20) text_widget(TextView::WordsByClass);
        }
        WidgetSpec samples;
        samples.type = WidgetType::Table;
        // The short class first, so it shows beside the long text.
        if (target) samples.columns.push_back(*target);
        samples.columns.push_back(text_column);
        samples.rows = 50;
        samples.title = "Sample texts";
        place(samples);
    }
    // Categories with a readable number of values, then numbers.
    int categories = 0, numbers = 0;
    for (const auto& c : contract.columns) {
        if (target && c.name == *target) continue;
        const ProfiledColumn* p = profile_of(c.name);
        if (!p) continue;
        // Bars stay readable up to 30 values (more: the Profile tab lists the top ones).
        if (c.role == ColumnRole::Category && categories < 6 && p->facts.distinct >= 2 && p->facts.distinct <= 30) {
            WidgetSpec w;
            w.type = WidgetType::Plot;
            w.plot.kind = plot::Kind::Bar;
            w.plot.x_column = c.name;
            place(w);
            ++categories;
        }
    }
    std::vector<std::string> number_columns;
    for (const auto& c : contract.columns) {
        if (c.role != ColumnRole::Numeric && !(target && c.name == *target && target_numeric)) continue;
        const ProfiledColumn* p = profile_of(c.name);
        if (!p || !(p->max > p->min)) continue;
        number_columns.push_back(c.name);
        if (target && c.name == *target) continue;
        if (numbers < 6) {
            WidgetSpec w;
            w.type = WidgetType::Plot;
            w.plot.kind = plot::Kind::Histogram;
            w.plot.x_column = c.name;
            place(w);
            ++numbers;
        }
    }
    // Dates: rows per year.
    for (const auto& c : contract.columns) {
        if (c.role != ColumnRole::DateTime) continue;
        const ProfiledColumn* p = profile_of(c.name);
        if (!p) continue;
        const auto year_of = [](const std::string& s) { return s.size() >= 4 ? std::atoi(s.substr(0, 4).c_str()) : 0; };
        const int span = year_of(p->max_text) - year_of(p->min_text);
        if (span < 1) continue;
        WidgetSpec w;
        w.type = WidgetType::Plot;
        w.plot.kind = plot::Kind::Histogram;
        w.plot.x_column = c.name;
        w.plot.bins = std::clamp(span + 1, 2, 120);
        w.bucket = "year";
        w.title = c.name + " \xC2\xB7 rows per year";
        place(w);
        break;
    }
    if (number_columns.size() >= 3) {
        WidgetSpec w;
        w.type = WidgetType::Plot;
        w.plot.kind = plot::Kind::Matrix;
        w.plot.y_columns.assign(number_columns.begin(), number_columns.begin() + std::min<size_t>(8, number_columns.size()));
        w.title = "Correlation";
        place(w);
    }
    // Missing values, when there are any.
    if (profile.MissingCells() > 0) {
        WidgetSpec w;
        w.type = WidgetType::Missing;
        w.title = "Missing values";
        place(w);
    }
    // The target by its strongest feature.
    if (target) {
        std::string best;
        double best_r = 0;
        for (const auto& [a, b, r] : profile.correlations) {
            const std::string other = a == *target ? b : b == *target ? a : std::string();
            if (!other.empty() && std::fabs(r) > std::fabs(best_r)) {
                best = other;
                best_r = r;
            }
        }
        if (best.empty() && !number_columns.empty())
            for (const auto& n : number_columns)
                if (n != *target) {
                    best = n;
                    break;
                }
        if (!best.empty()) {
            WidgetSpec w;
            w.type = WidgetType::Plot;
            if (target_numeric) {
                w.plot.kind = plot::Kind::Scatter;
                w.plot.x_column = best;
                w.plot.y_columns = {*target};
            } else {
                w.plot.kind = plot::Kind::Box;
                w.plot.y_columns = {best};
                w.plot.color_column = *target;
            }
            w.title = *target + " by " + best;
            place(w);
        }
    }
    return out;
}

}  // namespace cyxwiz::dashboard
