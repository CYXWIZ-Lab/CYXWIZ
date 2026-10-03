// Dashboard model (TOFIX134 P3.6): spec JSON round trip and refusals, the
// shared filter (cross-filter rule, bound parameters), fields and renames,
// widget kinds over every plot kind.

#include "../src/core/dashboard/dashboard_model.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::dashboard;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    DashboardSpec d;
    d.title = "Spotify";
    WidgetSpec rows;
    rows.id = d.NewId();
    rows.type = WidgetType::Kpi;
    rows.measure = Measure::Count;
    rows.at = {0, 0, 2, 1};
    rows.automatic = true;
    d.widgets.push_back(rows);
    WidgetSpec bars;
    bars.id = d.NewId();
    bars.type = WidgetType::Plot;
    bars.plot.kind = plot::Kind::Bar;
    bars.plot.x_column = "album_type";
    bars.at = {0, 1, 4, 3};
    bars.automatic = true;
    d.widgets.push_back(bars);
    WidgetSpec hist;
    hist.id = d.NewId();
    hist.type = WidgetType::Plot;
    hist.plot.kind = plot::Kind::Histogram;
    hist.plot.x_column = "track_popularity";
    hist.plot.bins = 24;
    hist.title = "Popularity";
    d.widgets.push_back(hist);
    WidgetSpec table;
    table.id = d.NewId();
    table.type = WidgetType::Table;
    table.columns = {"track_name", "artist_name"};
    table.rows = 50;
    d.widgets.push_back(table);
    FilterPredicate single;
    single.field = "album_type";
    single.values = {"single"};
    single.source_widget = bars.id;
    d.filters.Set(single);
    FilterPredicate range;
    range.field = "track_popularity";
    range.op = FilterPredicate::Op::Range;
    range.lo = 20;
    range.hi = 40;
    range.source_widget = hist.id;
    d.filters.Set(range);

    // Round trip.
    DashboardSpec back;
    std::string problem;
    Check(DashboardFromJson(DashboardToJson(d), back, &problem), "round trip parses: " + problem);
    Check(back.title == "Spotify" && back.widgets.size() == 4 && back.next_id == 5, "widgets and ids");
    Check(back.widgets[0].type == WidgetType::Kpi && back.widgets[0].measure == Measure::Count && back.widgets[0].automatic &&
              back.widgets[0].at.w == 2, "KPI kept");
    Check(back.widgets[2].plot.kind == plot::Kind::Histogram && back.widgets[2].plot.bins == 24 && back.widgets[2].title == "Popularity",
          "plot widget keeps its plot spec");
    Check(back.widgets[3].columns.size() == 2 && back.widgets[3].rows == 50, "table kept");
    Check(back.filters.predicates.size() == 2 && back.filters.predicates[1].op == FilterPredicate::Op::Range &&
              back.filters.predicates[1].hi == 40, "filters kept");
    DashboardSpec untouched;
    Check(!DashboardFromJson("{\"version\":2}", untouched, &problem) && problem.find("version 2") != std::string::npos, "newer version refused");
    Check(!DashboardFromJson("nope", untouched, &problem), "not JSON refused");
    Check(!DashboardFromJson("{\"version\":1,\"widgets\":[{\"type\":\"gauge\"}]}", untouched, &problem) &&
              problem.find("gauge") != std::string::npos, "unknown widget type refused");

    // A query widget and the automatic-layout flag survive the round trip.
    {
        DashboardSpec q;
        WidgetSpec w;
        w.id = q.NewId();
        w.query = "SELECT album_type, count(*) AS n FROM \"Spotify\" GROUP BY 1";
        w.query_table = "Spotify";
        w.plot.kind = plot::Kind::Bar;
        q.widgets.push_back(w);
        DashboardSpec qb;
        Check(DashboardFromJson(DashboardToJson(q), qb) && qb.widgets[0].IsQuery() && qb.widgets[0].query == w.query &&
                  qb.widgets[0].query_table == "Spotify" && !qb.automatic_done, "query widget kept; layout still to build");
        q.automatic_done = true;
        Check(DashboardFromJson(DashboardToJson(q), qb) && qb.automatic_done, "layout built kept");
        Check(DashboardFromJson("{\"version\":1,\"widgets\":[{\"type\":\"kpi\"}]}", qb) && qb.automatic_done &&
                  DashboardFromJson("{\"version\":1}", qb) && !qb.automatic_done, "without the flag: built when it has widgets");
    }

    // The shared filter: a widget is drawn with the other widgets' filters.
    std::vector<QueryParam> params;
    const std::string for_table = d.filters.WhereFor(table.id, params);
    Check(for_table == "CAST(\"album_type\" AS VARCHAR) IN (?) AND \"track_popularity\" BETWEEN ? AND ?" && params.size() == 3 &&
              params[0].s == "single" && params[1].d == 20 && params[2].d == 40, "both filters for another widget, values bound: " + for_table);
    params.clear();
    const std::string for_bars = d.filters.WhereFor(bars.id, params);
    Check(for_bars == "\"track_popularity\" BETWEEN ? AND ?" && params.size() == 2, "the bar chart is not filtered by its own selection");
    Check(d.filters.Text() == "album_type = single and track_popularity 20 to 40", "filters in words: " + d.filters.Text());
    // Selecting again on the same field replaces; an empty selection clears it.
    single.values = {"album", "compilation"};
    d.filters.Set(single);
    Check(d.filters.predicates.size() == 2 && d.filters.predicates.back().values.size() == 2 && d.filters.predicates.back().Text() ==
              "album_type in album, compilation", "a new selection replaces the old one");
    single.values.clear();
    d.filters.Set(single);
    Check(d.filters.predicates.size() == 1, "an empty selection clears it");
    // A field with a quote in its name stays an identifier.
    FilterPredicate odd;
    odd.field = "a\"b";
    odd.op = FilterPredicate::Op::IsNull;
    FilterState fs;
    fs.Set(odd);
    params.clear();
    Check(fs.WhereFor("", params) == "\"a\"\"b\" IS NULL", "quoted identifier");

    // Fields and renames (rebinding after a column was renamed upstream).
    Check(hist.Fields() == std::vector<std::string>({"track_popularity"}) && table.Fields().size() == 2, "fields a widget names");
    WidgetSpec copy = bars;
    Check(copy.RenameField("album_type", "release_kind") && copy.plot.x_column == "release_kind" && !copy.RenameField("x", "y"), "rename");

    // Regenerate keeps hand-made widgets and drops the automatic ones (and their filters).
    d.filters.Set(FilterPredicate{"album_type", FilterPredicate::Op::In, {"single"}, 0, 0, bars.id});
    d.RemoveAutomatic();
    Check(d.widgets.size() == 2 && d.Find(hist.id) && !d.Find(bars.id) && d.filters.predicates.size() == 1 &&
              d.filters.predicates[0].source_widget == hist.id, "automatic widgets and their filters removed");

    // Widget kinds: KPI, Table and every plot kind.
    Check(WidgetKinds().size() == 3 + plot::Kinds().size(), "every plot kind is a widget kind (plus KPI, Table, Missing values)");
    const WidgetKind* h = FindWidgetKind("plot.histogram");
    Check(h && h->type == WidgetType::Plot && h->plot_kind == plot::Kind::Histogram && h->x_need == FieldNeed::Number, "histogram kind");
    Check(FindWidgetKind("plot.bar")->x_need == FieldNeed::Category && FindWidgetKind("plot.map_regions") &&
              FindWidgetKind("kpi")->group == "Summary" && !FindWidgetKind("plot.gauge"), "kinds found by id");
    Check(KindOf(hist).id == "plot.histogram" && KindOf(rows).id == "kpi", "kind of a widget");
    Check(RoleFits(ColumnRole::Numeric, FieldNeed::Number) && !RoleFits(ColumnRole::Category, FieldNeed::Number) &&
              RoleFits(ColumnRole::Category, FieldNeed::Category) && !RoleFits(ColumnRole::Ignore, FieldNeed::Any), "roles fit slots");
    Check(MeasureFromId("missing_pct") == Measure::MissingPct && std::string(MeasureLabel(Measure::Count)) == "Rows", "measures");

    std::cout << "dashboard model: JSON round trip and refusals, shared filter with bound values and the own-widget rule, "
                 "replace / clear selections, quoted fields, renames, regenerate, widget kinds over every plot kind, query widgets and the layout flag. OK\n";
    return 0;
}
