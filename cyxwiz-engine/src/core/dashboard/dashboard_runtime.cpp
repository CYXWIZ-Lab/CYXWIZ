#include "dashboard_runtime.h"

#include "../plot/plot_prepare.h"

#include <algorithm>
#include <cmath>
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
    }
    return "CAST(count(*) AS DOUBLE)";
}

}  // namespace

Binding CheckBinding(const WidgetSpec& w, const DatasetContract& contract, const std::map<std::string, std::string>& known_types) {
    Binding b;
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
    if (w.type != WidgetType::Plot) return b;
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

QueryRequest WidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard widget";
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

QueryRequest KpiQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard KPI";
    const std::string m = MeasureSql(w.measure, w.field.empty() ? std::string("NULL") : Quote(w.field));
    const std::string cond = filters.WhereFor(w.id, r.params);
    r.sql = "SELECT (SELECT " + m + " FROM " + Quote(table) + Where(cond) + ") AS value, (SELECT " + m + " FROM " + Quote(table) + ") AS all_rows";
    return r;
}

QueryRequest StripQuery(const std::string& table, const FilterState& filters, const DatasetProfile& profile, const std::string& target,
                        bool target_numeric, const std::map<std::string, std::vector<std::string>>& missing_text) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard summary";
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
    int slot = 0;
    const auto place = [&](WidgetSpec& w) {
        w.plot.legend = !w.plot.color_column.empty();  // one series needs no legend
        w.at = {(slot % 3) * 4, (slot / 3) * 3, 4, 3};
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
        WidgetSpec w;
        w.type = WidgetType::Plot;
        w.plot.kind = target_numeric ? plot::Kind::Histogram : plot::Kind::Bar;
        w.plot.x_column = *target;
        w.title = *target + " (target)";
        place(w);
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
