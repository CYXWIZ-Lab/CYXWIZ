#include "dashboard_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>

namespace cyxwiz::dashboard {

using nlohmann::json;

namespace {

// "a""b" quoting (as SessionQueryEngine::QuoteIdentifier; the model stays free of DuckDB).
std::string Quote(const std::string& name) {
    std::string q = "\"";
    for (char c : name) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

std::string Short(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

const char* TypeId(WidgetType t) {
    switch (t) {
        case WidgetType::Kpi: return "kpi";
        case WidgetType::Table: return "table";
        case WidgetType::Plot: return "plot";
        case WidgetType::Missing: return "missing";
    }
    return "plot";
}

const char* OpId(FilterPredicate::Op op) {
    switch (op) {
        case FilterPredicate::Op::In: return "in";
        case FilterPredicate::Op::Range: return "range";
        case FilterPredicate::Op::IsNull: return "is_null";
        case FilterPredicate::Op::NotNull: return "not_null";
    }
    return "in";
}

}  // namespace

const char* MeasureId(Measure m) {
    switch (m) {
        case Measure::Count: return "count";
        case Measure::Sum: return "sum";
        case Measure::Mean: return "mean";
        case Measure::Median: return "median";
        case Measure::Min: return "min";
        case Measure::Max: return "max";
        case Measure::Distinct: return "distinct";
        case Measure::MissingPct: return "missing_pct";
    }
    return "count";
}

const char* MeasureLabel(Measure m) {
    switch (m) {
        case Measure::Count: return "Rows";
        case Measure::Sum: return "Sum";
        case Measure::Mean: return "Mean";
        case Measure::Median: return "Median";
        case Measure::Min: return "Min";
        case Measure::Max: return "Max";
        case Measure::Distinct: return "Distinct";
        case Measure::MissingPct: return "Missing";
    }
    return "Rows";
}

std::optional<Measure> MeasureFromId(const std::string& id) {
    for (Measure m : {Measure::Count, Measure::Sum, Measure::Mean, Measure::Median, Measure::Min, Measure::Max, Measure::Distinct,
                      Measure::MissingPct})
        if (id == MeasureId(m)) return m;
    return std::nullopt;
}

std::vector<std::string> WidgetSpec::Fields() const {
    std::vector<std::string> out;
    const auto add = [&](const std::string& f) {
        if (!f.empty() && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    };
    switch (type) {
        case WidgetType::Kpi: add(field); break;
        case WidgetType::Missing: break;
        case WidgetType::Table:
            for (const auto& c : columns) add(c);
            break;
        case WidgetType::Plot:
            if (IsQuery()) break;  // the query's own columns
            add(plot.x_column);
            for (const auto& y : plot.y_columns) add(y);
            add(plot.color_column);
            add(plot.value_column);
            add(plot.u_column);
            add(plot.v_column);
            for (const auto& s : plot.spread_columns) add(s);
            break;
    }
    return out;
}

bool WidgetSpec::RenameField(const std::string& from, const std::string& to) {
    bool hit = false;
    const auto swap = [&](std::string& f) {
        if (f == from) {
            f = to;
            hit = true;
        }
    };
    swap(field);
    for (auto& c : columns) swap(c);
    swap(plot.x_column);
    for (auto& y : plot.y_columns) swap(y);
    swap(plot.color_column);
    swap(plot.value_column);
    swap(plot.u_column);
    swap(plot.v_column);
    for (auto& s : plot.spread_columns) swap(s);
    return hit;
}

std::string FilterPredicate::Text() const {
    switch (op) {
        case Op::In: {
            std::string v;
            for (size_t i = 0; i < values.size(); ++i) v += (i ? ", " : "") + values[i];
            return field + (values.size() == 1 ? " = " : " in ") + v;
        }
        case Op::Range: return field + (bucket.empty() ? "" : " " + bucket) + " " + Short(lo) + " to " + Short(hi);
        case Op::IsNull: return field + " is missing";
        case Op::NotNull: return field + " is not missing";
    }
    return field;
}

void FilterState::Set(FilterPredicate p) {
    predicates.erase(std::remove_if(predicates.begin(), predicates.end(),
                                    [&](const FilterPredicate& q) { return q.source_widget == p.source_widget && q.field == p.field; }),
                     predicates.end());
    if (p.op == FilterPredicate::Op::In && p.values.empty()) return;  // an empty selection clears
    predicates.push_back(std::move(p));
}

void FilterState::ClearWidget(const std::string& widget_id) {
    predicates.erase(std::remove_if(predicates.begin(), predicates.end(),
                                    [&](const FilterPredicate& q) { return q.source_widget == widget_id; }),
                     predicates.end());
}

std::string FilterState::WhereFor(const std::string& widget_id, std::vector<QueryParam>& params) const {
    std::string where;
    for (const auto& p : predicates) {
        if (!widget_id.empty() && p.source_widget == widget_id) continue;  // a widget is not filtered by its own selection
        std::string cond;
        const std::string col = Quote(p.field);
        switch (p.op) {
            case FilterPredicate::Op::In: {
                std::string in;
                for (const auto& v : p.values) {
                    in += in.empty() ? "?" : ", ?";
                    params.push_back(QueryParam::Of(v));
                }
                cond = "CAST(" + col + " AS VARCHAR) IN (" + in + ")";
                break;
            }
            case FilterPredicate::Op::Range:
                cond = (p.bucket == "year" ? "year(TRY_CAST(" + col + " AS DATE))" : col) + " BETWEEN ? AND ?";
                params.push_back(QueryParam::Of(p.lo));
                params.push_back(QueryParam::Of(p.hi));
                break;
            case FilterPredicate::Op::IsNull: cond = col + " IS NULL"; break;
            case FilterPredicate::Op::NotNull: cond = col + " IS NOT NULL"; break;
        }
        where += (where.empty() ? "" : " AND ") + cond;
    }
    return where;
}

std::string FilterState::Text() const {
    std::string out;
    for (const auto& p : predicates) out += (out.empty() ? "" : " and ") + p.Text();
    return out;
}

WidgetSpec* DashboardSpec::Find(const std::string& id) {
    for (auto& w : widgets)
        if (w.id == id) return &w;
    return nullptr;
}

const WidgetSpec* DashboardSpec::Find(const std::string& id) const {
    for (const auto& w : widgets)
        if (w.id == id) return &w;
    return nullptr;
}

void DashboardSpec::RemoveAutomatic() {
    std::vector<std::string> gone;
    for (const auto& w : widgets)
        if (w.automatic) gone.push_back(w.id);
    widgets.erase(std::remove_if(widgets.begin(), widgets.end(), [](const WidgetSpec& w) { return w.automatic; }), widgets.end());
    for (const auto& id : gone) filters.ClearWidget(id);
}

std::vector<std::pair<std::string, std::string>> DashboardSummary(const DashboardSpec& spec) {
    std::vector<std::pair<std::string, std::string>> rows;
    size_t automatic = 0, query = 0;
    for (const auto& w : spec.widgets) {
        automatic += w.automatic ? 1 : 0;
        query += w.IsQuery() ? 1 : 0;
    }
    std::string widgets = std::to_string(spec.widgets.size()) + (spec.widgets.size() == 1 ? " widget" : " widgets");
    if (spec.widgets.empty()) widgets = spec.automatic_done ? "no widgets" : "automatic layout on the first open";
    else {
        std::string parts;
        if (automatic) parts += std::to_string(automatic) + " automatic";
        if (spec.widgets.size() > automatic) parts += (parts.empty() ? "" : ", ") + std::to_string(spec.widgets.size() - automatic) + " added";
        if (query) parts += (parts.empty() ? "" : ", ") + std::to_string(query) + " from queries";
        widgets += " (" + parts + ")";
    }
    rows.push_back({"Widgets", widgets});
    if (!spec.filters.Empty()) rows.push_back({"Filters", spec.filters.Text()});
    if (!spec.title.empty()) rows.push_back({"Title", spec.title});
    return rows;
}

std::string DashboardToJson(const DashboardSpec& spec) {
    json j;
    j["version"] = DashboardSpec::kVersion;
    j["title"] = spec.title;
    j["next_id"] = spec.next_id;
    j["known_types"] = spec.known_types;
    j["automatic_done"] = spec.automatic_done;
    j["widgets"] = json::array();
    for (const auto& w : spec.widgets) {
        json o;
        o["id"] = w.id;
        o["type"] = TypeId(w.type);
        o["title"] = w.title;
        o["at"] = {w.at.x, w.at.y, w.at.w, w.at.h};
        o["automatic"] = w.automatic;
        if (w.IsQuery()) {
            o["query"] = w.query;
            o["query_table"] = w.query_table;
        }
        if (!w.bucket.empty()) o["bucket"] = w.bucket;
        switch (w.type) {
            case WidgetType::Plot: o["plot"] = json::parse(plot::SpecToJson(w.plot)); break;
            case WidgetType::Kpi:
                o["measure"] = MeasureId(w.measure);
                o["field"] = w.field;
                break;
            case WidgetType::Table:
                o["columns"] = w.columns;
                o["rows"] = w.rows;
                break;
            case WidgetType::Missing: break;
        }
        j["widgets"].push_back(o);
    }
    j["filters"] = json::array();
    for (const auto& p : spec.filters.predicates) {
        json f;
        f["field"] = p.field;
        f["op"] = OpId(p.op);
        f["values"] = p.values;
        f["lo"] = p.lo;
        f["hi"] = p.hi;
        f["widget"] = p.source_widget;
        if (!p.bucket.empty()) f["bucket"] = p.bucket;
        j["filters"].push_back(f);
    }
    return j.dump();
}

bool DashboardFromJson(const std::string& text, DashboardSpec& spec, std::string* problem) {
    const auto fail = [&](const std::string& why) {
        if (problem) *problem = why;
        return false;
    };
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception& e) {
        return fail(std::string("not JSON: ") + e.what());
    }
    if (!j.is_object()) return fail("not a dashboard");
    const int version = j.value("version", 0);
    if (version != DashboardSpec::kVersion) return fail("dashboard version " + std::to_string(version) + " is not read by this Engine");
    DashboardSpec out;
    out.title = j.value("title", std::string());
    out.next_id = std::max(1, j.value("next_id", 1));
    if (j.contains("known_types") && j["known_types"].is_object())
        out.known_types = j["known_types"].get<std::map<std::string, std::string>>();
    if (j.contains("widgets") && j["widgets"].is_array())
        for (const auto& o : j["widgets"]) {
            WidgetSpec w;
            w.id = o.value("id", std::string());
            const std::string type = o.value("type", std::string("plot"));
            if (type == "kpi") w.type = WidgetType::Kpi;
            else if (type == "table") w.type = WidgetType::Table;
            else if (type == "plot") w.type = WidgetType::Plot;
            else if (type == "missing") w.type = WidgetType::Missing;
            else return fail("unknown widget type '" + type + "'");
            w.title = o.value("title", std::string());
            if (o.contains("at") && o["at"].is_array() && o["at"].size() == 4)
                w.at = {o["at"][0].get<int>(), o["at"][1].get<int>(), std::max(1, o["at"][2].get<int>()), std::max(1, o["at"][3].get<int>())};
            w.automatic = o.value("automatic", false);
            w.query = o.value("query", std::string());
            w.query_table = o.value("query_table", std::string());
            w.bucket = o.value("bucket", std::string());
            if (w.type == WidgetType::Plot) {
                std::string why;
                if (!o.contains("plot") || !plot::SpecFromJson(o["plot"].dump(), w.plot, &why)) return fail("widget " + w.id + ": " + why);
            } else if (w.type == WidgetType::Kpi) {
                auto m = MeasureFromId(o.value("measure", std::string("count")));
                if (!m) return fail("widget " + w.id + ": unknown measure");
                w.measure = *m;
                w.field = o.value("field", std::string());
            } else if (w.type == WidgetType::Table) {
                if (o.contains("columns") && o["columns"].is_array()) w.columns = o["columns"].get<std::vector<std::string>>();
                w.rows = std::clamp(o.value("rows", 20), 1, 10000);
            }
            if (w.id.empty()) w.id = out.NewId();
            out.widgets.push_back(std::move(w));
        }
    if (j.contains("filters") && j["filters"].is_array())
        for (const auto& f : j["filters"]) {
            FilterPredicate p;
            p.field = f.value("field", std::string());
            const std::string op = f.value("op", std::string("in"));
            if (op == "in") p.op = FilterPredicate::Op::In;
            else if (op == "range") p.op = FilterPredicate::Op::Range;
            else if (op == "is_null") p.op = FilterPredicate::Op::IsNull;
            else if (op == "not_null") p.op = FilterPredicate::Op::NotNull;
            else return fail("unknown filter '" + op + "'");
            if (f.contains("values") && f["values"].is_array()) p.values = f["values"].get<std::vector<std::string>>();
            p.lo = f.value("lo", 0.0);
            p.hi = f.value("hi", 0.0);
            p.source_widget = f.value("widget", std::string());
            p.bucket = f.value("bucket", std::string());
            if (!p.field.empty()) out.filters.predicates.push_back(std::move(p));
        }
    // Saved before the flag existed: a dashboard with widgets had its layout.
    out.automatic_done = j.value("automatic_done", !out.widgets.empty());
    spec = std::move(out);
    return true;
}

const std::vector<WidgetKind>& WidgetKinds() {
    static const std::vector<WidgetKind> kinds = [] {
        std::vector<WidgetKind> k;
        k.push_back({"kpi", "KPI", "Summary", WidgetType::Kpi, plot::Kind::Line, FieldNeed::Any, FieldNeed::Any});
        k.push_back({"table", "Table", "Summary", WidgetType::Table, plot::Kind::Line, FieldNeed::Any, FieldNeed::Any});
        k.push_back({"missing", "Missing values", "Summary", WidgetType::Missing, plot::Kind::Line, FieldNeed::Any, FieldNeed::Any});
        for (const auto& info : plot::Kinds()) {
            WidgetKind w;
            w.id = std::string("plot.") + info.id;
            w.label = info.label;
            w.group = plot::GroupLabel(info.group);
            w.type = WidgetType::Plot;
            w.plot_kind = info.kind;
            // What the slots accept (the Data panel's pickers say the same).
            using K = plot::Kind;
            const K kind = info.kind;
            const bool category_x = kind == K::Bar || kind == K::Pie || kind == K::ErrorBars || kind == K::Heatmap || kind == K::Confusion ||
                                    kind == K::Roc || kind == K::PrCurve || kind == K::Calibration || kind == K::Importance ||
                                    kind == K::MapRegions;
            w.x_need = category_x ? FieldNeed::Category : FieldNeed::Number;
            if (kind == K::Roc || kind == K::PrCurve || kind == K::Calibration || kind == K::MapRegions) w.x_need = FieldNeed::Any;
            const bool any_y = kind == K::Heatmap || kind == K::Confusion || kind == K::Sankey || kind == K::Treemap;
            w.y_need = any_y ? FieldNeed::Any : FieldNeed::Number;
            k.push_back(std::move(w));
        }
        return k;
    }();
    return kinds;
}

const WidgetKind* FindWidgetKind(const std::string& id) {
    for (const auto& k : WidgetKinds())
        if (k.id == id) return &k;
    return nullptr;
}

const WidgetKind& KindOf(const WidgetSpec& w) {
    if (w.type == WidgetType::Kpi) return WidgetKinds()[0];
    if (w.type == WidgetType::Table) return WidgetKinds()[1];
    if (w.type == WidgetType::Missing) return WidgetKinds()[2];
    for (const auto& k : WidgetKinds())
        if (k.type == WidgetType::Plot && k.plot_kind == w.plot.kind) return k;
    return WidgetKinds()[3];
}

bool RoleFits(ColumnRole role, FieldNeed need) {
    switch (need) {
        case FieldNeed::Any: return role != ColumnRole::Ignore;
        case FieldNeed::Number: return role == ColumnRole::Numeric || role == ColumnRole::Target || role == ColumnRole::Weight;
        case FieldNeed::Category:
            return role == ColumnRole::Category || role == ColumnRole::Target || role == ColumnRole::Text || role == ColumnRole::DateTime ||
                   role == ColumnRole::Id || role == ColumnRole::FilePath;
    }
    return true;
}

}  // namespace cyxwiz::dashboard
