// Dashboard runtime (TOFIX134 P3.7): binding checks with rename candidates,
// widget / KPI / summary SQL run through the real engine, the automatic layout.

#include "../src/core/dashboard/dashboard_runtime.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::dashboard;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> Tracks() {
    arrow::StringBuilder id, type;
    arrow::Int64Builder pop, artist_pop;
    arrow::DoubleBuilder dur;
    for (int i = 0; i < 30; ++i) {
        (void)id.Append("t" + std::to_string(i));
        (void)type.Append(i % 3 == 0 ? "single" : i % 3 == 1 ? "album" : "compilation");
        (void)pop.Append(20 + i * 2);
        (void)artist_pop.Append(30 + i * 2 + (i % 4));
        if (i == 5) (void)dur.AppendNull();
        else (void)dur.Append(2.0 + (i % 7) * 0.4);
    }
    std::shared_ptr<arrow::Array> a, b, c, d, e;
    (void)id.Finish(&a);
    (void)type.Finish(&b);
    (void)pop.Finish(&c);
    (void)artist_pop.Finish(&d);
    (void)dur.Finish(&e);
    return arrow::Table::Make(arrow::schema({arrow::field("track_id", arrow::utf8()), arrow::field("album_type", arrow::utf8()),
                                             arrow::field("track_popularity", arrow::int64()), arrow::field("artist_popularity", arrow::int64()),
                                             arrow::field("duration", arrow::float64())}),
                              {a, b, c, d, e});
}

double D(const QueryResult& r, const std::string& col) {
    auto s = r.table->GetColumnByName(col)->GetScalar(0);
    auto x = (*s)->CastTo(arrow::float64());
    return std::static_pointer_cast<arrow::DoubleScalar>(*x)->value;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_dashboard_runtime";
    std::error_code ec;
    fs::remove_all(root, ec);
    {
        SessionQueryEngine engine((root / "attach").string());
        std::string error;
        Check(engine.AttachArrow("Spotify", 1, Tracks(), &error), "attach: " + error);
        const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
        DatasetProfile profile = ProfileTable("Spotify", run);
        Check(profile.ok(), "profile: " + profile.error);
        DatasetContract contract = BuildContract("k", profile.Facts(), {{"track_popularity", ColumnRole::Target}}, {});

        // The automatic layout.
        DashboardSpec spec;
        auto autos = AutomaticWidgets(profile, contract, spec);
        Check(!autos.empty() && autos[0].plot.kind == plot::Kind::Histogram && autos[0].plot.x_column == "track_popularity" &&
                  autos[0].automatic, "the target first (a number: histogram)");
        bool bar = false, matrix = false, scatter = false, id_card = false;
        for (const auto& w : autos) {
            bar = bar || (w.plot.kind == plot::Kind::Bar && w.plot.x_column == "album_type");
            matrix = matrix || w.plot.kind == plot::Kind::Matrix;
            scatter = scatter || (w.plot.kind == plot::Kind::Scatter && w.plot.y_columns == std::vector<std::string>({"track_popularity"}) &&
                                  w.plot.x_column == "artist_popularity");
            id_card = id_card || w.plot.x_column == "track_id";
        }
        Check(bar && matrix && scatter && !id_card, "category bar, correlation matrix, target by its strongest feature, no ID card");
        Check(autos[1].at.x == 4 && autos[3].at.y == 3, "three across");
        spec.widgets = autos;

        // Widget queries run through the engine, with another widget's filter.
        FilterPredicate single;
        single.field = "album_type";
        single.values = {"single"};
        single.source_widget = "bars";
        spec.filters.Set(single);
        WidgetSpec hist = autos[0];
        QueryRequest q = WidgetQuery(hist, "Spotify", spec.filters);
        Check(q.sql == "SELECT \"track_popularity\" FROM \"Spotify\" WHERE CAST(\"album_type\" AS VARCHAR) IN (?)" && q.params.size() == 1 &&
                  q.inputs == std::vector<std::string>({"Spotify"}), "histogram query: " + q.sql);
        QueryResult r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 10 && r.table->num_columns() == 1, "10 singles, only the column it needs: " + r.error);
        q = WidgetQuery(hist, "Spotify", spec.filters, 4);
        r = engine.Run(q);
        Check(r.ok && r.table->num_rows() == 4, "a reproducible sample of 4: " + r.error);
        WidgetSpec bars = autos[1];
        bars.id = "bars";
        r = engine.Run(WidgetQuery(bars, "Spotify", spec.filters));
        Check(r.ok && r.table->num_rows() == 30, "the bar chart is not filtered by its own selection");
        WidgetSpec table;
        table.id = "t";
        table.type = WidgetType::Table;
        table.columns = {"track_id"};
        table.rows = 5;
        r = engine.Run(WidgetQuery(table, "Spotify", spec.filters));
        Check(r.ok && r.table->num_rows() == 5, "table rows limited");

        // A KPI: filtered and all in one row.
        WidgetSpec kpi;
        kpi.id = "k";
        kpi.type = WidgetType::Kpi;
        kpi.measure = Measure::Mean;
        kpi.field = "track_popularity";
        r = engine.Run(KpiQuery(kpi, "Spotify", spec.filters));
        Check(r.ok && std::fabs(D(r, "value") - 47.0) < 1e-9 && std::fabs(D(r, "all_rows") - 49.0) < 1e-9, "KPI mean of singles 47, all 49: " + r.error);
        kpi.measure = Measure::MissingPct;
        kpi.field = "duration";
        r = engine.Run(KpiQuery(kpi, "Spotify", spec.filters));
        Check(r.ok && std::fabs(D(r, "all_rows") - 100.0 / 30) < 1e-9, "missing share");
        kpi.measure = Measure::Count;
        r = engine.Run(KpiQuery(kpi, "Spotify", spec.filters));
        Check(r.ok && D(r, "value") == 10 && D(r, "all_rows") == 30, "rows filtered and all");

        // The summary strip.
        r = engine.Run(StripQuery("Spotify", spec.filters, profile, "track_popularity", true));
        Check(r.ok && D(r, "rows_now") == 10 && D(r, "rows_all") == 30 && D(r, "missing_now") == 0 && std::fabs(D(r, "target_now") - 47.0) < 1e-9 &&
                  std::fabs(D(r, "target_all") - 49.0) < 1e-9, "summary strip: " + r.error);
        r = engine.Run(StripQuery("Spotify", spec.filters, profile, "album_type", false));
        Check(r.ok, "summary with a category target (most frequent value): " + r.error);

        // Bindings: fine, a missing field with a rename candidate, a type change.
        std::map<std::string, std::string> known;
        for (const auto& c : contract.columns) known[c.name] = c.type;
        Check(CheckBinding(hist, contract, known).state == Binding::State::Ok, "bound");
        WidgetSpec gone = hist;
        gone.plot.x_column = "popularity_old";
        known["popularity_old"] = "int";
        known.erase("artist_popularity");  // as if artist_popularity were new since the last bind
        Binding b = CheckBinding(gone, contract, known);
        Check(b.state == Binding::State::FieldMissing && b.field == "popularity_old" && b.rename_candidate == "artist_popularity" &&
                  b.message.find("rebind to 'artist_popularity'") != std::string::npos, "missing field with a rename candidate: " + b.message);
        WidgetSpec text_hist = hist;
        text_hist.plot.x_column = "album_type";
        b = CheckBinding(text_hist, contract, known);
        Check(b.state == Binding::State::RoleMismatch && b.field == "album_type", "a histogram on text says so: " + b.message);
    }
    fs::remove_all(root, ec);
    std::cout << "dashboard runtime: automatic layout, widget / sample / table / KPI / summary SQL through the engine with the "
                 "cross-filter rule, binding checks with rename candidates and type changes. OK\n";
    return 0;
}
