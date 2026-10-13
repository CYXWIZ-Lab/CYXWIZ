// Dashboard runtime (TOFIX134 P3.7): binding checks with rename candidates,
// widget / KPI / summary SQL run through the real engine, the automatic layout.

#include "../src/core/dashboard/dashboard_runtime.h"
#include "../src/core/dashboard/text_words.h"

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

        // Missing values per column (nulls and texts marked as missing), under the filters.
        WidgetSpec miss;
        miss.id = "m";
        miss.type = WidgetType::Missing;
        r = engine.Run(MissingQuery(miss, "Spotify", FilterState{}, {"duration", "album_type"}, {{"album_type", {"compilation"}}}));
        Check(r.ok && D(r, "rows") == 30 && D(r, "m0") == 1 && D(r, "m1") == 10, "missing: 1 null duration, 10 'compilation' as missing: " + r.error);
        r = engine.Run(MissingQuery(miss, "Spotify", spec.filters, {"duration"}, {}));
        Check(r.ok && D(r, "rows") == 10, "missing under the filters");

        // A date X counted per year, and its range filter on the year.
        {
            DatasetContract dated;
            dated.columns.push_back({"release", "date", ColumnRole::DateTime, RoleSource::Inferred, "dates", std::nullopt});
            WidgetSpec y2;
            y2.type = WidgetType::Plot;
            y2.plot.kind = plot::Kind::Histogram;
            y2.plot.x_column = "release";
            Check(CheckBinding(y2, dated, {}).state == Binding::State::RoleMismatch, "a histogram of a raw date needs numbers");
            y2.bucket = "year";
            Check(CheckBinding(y2, dated, {}).state == Binding::State::Ok, "a year-bucketed date is a number");
        }
        WidgetSpec years;
        years.id = "y";
        years.type = WidgetType::Plot;
        years.plot.kind = plot::Kind::Histogram;
        years.plot.x_column = "release";
        years.bucket = "year";
        QueryRequest yq = WidgetQuery(years, "T", FilterState{});
        Check(yq.sql == "SELECT CAST(year(TRY_CAST(\"release\" AS DATE)) AS DOUBLE) AS \"release\" FROM \"T\"", "year bucket: " + yq.sql);
        FilterState by_year;
        FilterPredicate yr;
        yr.field = "release";
        yr.op = FilterPredicate::Op::Range;
        yr.lo = 1990;
        yr.hi = 1991;
        yr.bucket = "year";
        yr.source_widget = "y";
        by_year.Set(yr);
        std::vector<QueryParam> yp;
        Check(by_year.WhereFor("other", yp) == "year(TRY_CAST(\"release\" AS DATE)) BETWEEN ? AND ?" && by_year.Text() == "release year 1990 to 1991",
              "year filter: " + by_year.Text());

        // SQL as text (View SQL, Open in Data Studio): values written in, and it runs.
        {
            std::vector<QueryParam> ps = {QueryParam::Of(std::string("it's")), QueryParam::Of(2.5), QueryParam::Of(int64_t(7))};
            Check(InlineParams("SELECT '?' AS \"a?\" WHERE x IN (?) AND y = ? AND z = ?", ps) ==
                      "SELECT '?' AS \"a?\" WHERE x IN ('it''s') AND y = 2.5 AND z = 7", "inline: quoted ? untouched, quotes doubled");
            const std::string rows_sql = FilteredRowsSql("Spotify", spec.filters);
            Check(rows_sql == "SELECT * FROM \"Spotify\" WHERE CAST(\"album_type\" AS VARCHAR) IN ('single')", "filtered rows: " + rows_sql);
            QueryRequest text;
            text.sql = rows_sql;
            r = engine.Run(text);
            Check(r.ok && r.table->num_rows() == 10, "the filtered rows SQL runs: " + r.error);
            QueryRequest kq = KpiQuery(kpi, "Spotify", spec.filters);
            text.sql = InlineParams(kq.sql, kq.params);
            r = engine.Run(text);
            Check(r.ok && D(r, "value") == 10, "a KPI's SQL as text runs the same: " + r.error);
        }

        // A query widget: its SQL reads the rows under the other widgets' filters.
        {
            WidgetSpec qw;
            qw.id = "q";
            qw.type = WidgetType::Plot;
            qw.plot.kind = plot::Kind::Bar;
            qw.plot.x_column = "album_type";
            qw.plot.y_columns = {"tracks"};
            qw.query = "SELECT album_type, count(*) AS tracks FROM Spotify GROUP BY album_type ORDER BY album_type;";
            qw.query_table = "Spotify";
            Check(qw.Fields().empty() && CheckBinding(qw, contract, {}).state == Binding::State::Ok, "a query widget's columns are its own");
            QueryRequest qq = WidgetQuery(qw, "Spotify", spec.filters);
            r = engine.Run(qq);
            Check(r.ok && r.table->num_rows() == 1 && D(r, "tracks") == 10, "query widget under the filter (10 singles): " + r.error + " / " + qq.sql);
            r = engine.Run(WidgetQuery(qw, "Spotify", FilterState{}));
            Check(r.ok && r.table->num_rows() == 3, "query widget without filters: 3 album types: " + r.error);
            qw.query = "with t as (select * from Spotify) select album_type, count(*) as tracks from t group by 1";
            r = engine.Run(WidgetQuery(qw, "Spotify", spec.filters));
            Check(r.ok && r.table->num_rows() == 1, "a query with its own WITH: " + r.error);
            qw.query = "DROP TABLE Spotify";
            r = engine.Run(WidgetQuery(qw, "Spotify", FilterState{}));
            Check(!r.ok, "a query widget cannot write");
        }

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

        // A text column (TOFIX134 P3 text): an unnamed index, statements, a class.
        {
            const std::vector<std::pair<std::string, std::string>> notes = {
                {"I feel tired and I cannot sleep at night", "Anxiety"},
                {"Cannot sleep again, the night is long", "Anxiety"},
                {"I feel tired of the noise and the people", "Depression"},
                {"Today was a good day with friends and sun", "Normal"},
                {"Sleep came early, a good night for once", "Normal"},
                {"", "Normal"},
                {"I feel tired, so tired, and the night will not end", "Depression"},
                {"We walked by the river and talked about the week ahead", "Normal"},
                {"The exam is tomorrow and I cannot stop worrying about it at night", "Anxiety"},
                {"Feel tired and empty most mornings", "Depression"},
            };
            arrow::Int64Builder idx;
            arrow::StringBuilder statement, status;
            for (size_t i = 0; i < notes.size(); ++i) {
                (void)idx.Append(static_cast<int64_t>(i));
                (void)statement.Append(notes[i].first);
                (void)status.Append(notes[i].second);
            }
            (void)statement.AppendNull();  // a missing text
            (void)status.Append("Normal");
            (void)idx.Append(static_cast<int64_t>(notes.size()));
            std::shared_ptr<arrow::Array> a, s, l;
            (void)idx.Finish(&a);
            (void)statement.Finish(&s);
            (void)status.Finish(&l);
            auto notes_table = arrow::Table::Make(
                arrow::schema({arrow::field("C0", arrow::int64()), arrow::field("statement", arrow::utf8()), arrow::field("status", arrow::utf8())}),
                {a, s, l});
            Check(engine.AttachArrow("Notes", 1, notes_table, &error), "attach notes: " + error);
            DatasetProfile tp = ProfileTable("Notes", run);
            Check(tp.ok(), "notes profile: " + tp.error);
            DatasetContract tc = BuildContract("n", tp.Facts(), {{"status", ColumnRole::Target}}, {});
            Check(tc.Find("C0") && tc.Find("C0")->role == ColumnRole::Id, "an unnamed index column is an ID");
            Check(tc.Find("statement") && tc.Find("statement")->role == ColumnRole::Text, "the statements are text");

            DashboardSpec ts;
            auto tw = AutomaticWidgets(tp, tc, ts);
            Check(tw.size() >= 3 && tw[0].type == WidgetType::Kpi && tw[0].measure == Measure::MedianWords && tw[1].measure == Measure::Vocabulary &&
                      tw[2].measure == Measure::EmptyTexts && tw[0].at.w == 4 && tw[2].at.x == 8 && tw[0].at.y == 0,
                  "the text KPIs first, a row of their own");
            const WidgetSpec* words = nullptr;
            const WidgetSpec* phrases = nullptr;
            const WidgetSpec* by_class = nullptr;
            const WidgetSpec* length = nullptr;
            const WidgetSpec* samples = nullptr;
            bool c0_card = false;
            for (const auto& w : tw) {
                if (w.text_view == TextView::Words) words = &w;
                if (w.text_view == TextView::Phrases) phrases = &w;
                if (w.text_view == TextView::WordsByClass) by_class = &w;
                if (w.text_view == TextView::Length) length = &w;
                if (w.type == WidgetType::Table) samples = &w;
                c0_card = c0_card || w.plot.x_column == "C0";
            }
            Check(words && phrases && by_class && length && samples && !c0_card, "length, words, phrases, words by class, samples; no index card");
            Check(words->plot.kind == plot::Kind::Bar && words->plot.bar_horizontal && words->plot.x_column == "word", "top words: horizontal bars");
            Check(by_class->label_field == "status" && by_class->plot.kind == plot::Kind::Heatmap, "words by the class column");
            Check(samples->columns == std::vector<std::string>({"status", "statement"}), "samples: the class, then the text");
            Check(length->at.y >= 1 && words->Fields() == std::vector<std::string>({"statement"}), "cards under the KPIs; a text widget names its text column");

            // Top words: common words left out, then kept.
            QueryResult tr = engine.Run(WidgetQuery(*words, "Notes", FilterState{}));
            Check(tr.ok && tr.table->num_rows() > 0, "top words: " + tr.error);
            auto word_at = [&](const QueryResult& q, int64_t row) {
                auto sc = q.table->column(0)->GetScalar(row);
                return (*sc)->ToString();
            };
            Check(word_at(tr, 0) == "night" || word_at(tr, 0) == "tired", "the top word is night or tired: " + word_at(tr, 0));
            bool has_the = false;
            for (int64_t i = 0; i < tr.table->num_rows(); ++i) has_the = has_the || word_at(tr, i) == "the";
            Check(!has_the, "stop words left out");
            WidgetSpec keep = *words;
            keep.keep_stop_words = true;
            tr = engine.Run(WidgetQuery(keep, "Notes", FilterState{}));
            has_the = false;
            for (int64_t i = 0; tr.ok && i < tr.table->num_rows(); ++i) has_the = has_the || word_at(tr, i) == "the";
            Check(tr.ok && has_the && (word_at(tr, 0) == "and" || word_at(tr, 0) == "the"), "kept: common words count (and, the: 7 each): " + tr.error);

            // Phrases and words by class.
            tr = engine.Run(WidgetQuery(*phrases, "Notes", FilterState{}));
            Check(tr.ok && word_at(tr, 0) == "feel tired" && D(tr, "count") == 4, "top phrase 'feel tired' x4: " + tr.error);
            tr = engine.Run(WidgetQuery(*by_class, "Notes", FilterState{}));
            Check(tr.ok && tr.table->num_columns() == 3 && tr.table->num_rows() > 0, "words by class: " + tr.error);

            // KPIs: median words, vocabulary, empty or missing texts.
            WidgetSpec k = tw[2];
            tr = engine.Run(KpiQuery(k, "Notes", FilterState{}));
            Check(tr.ok && D(tr, "value") == 2, "empty or missing texts: 2: " + tr.error);
            k = tw[0];
            tr = engine.Run(KpiQuery(k, "Notes", FilterState{}));
            Check(tr.ok && D(tr, "value") >= 8 && D(tr, "value") <= 10, "median words: " + tr.error);
            k = tw[1];
            tr = engine.Run(KpiQuery(k, "Notes", FilterState{}));
            Check(tr.ok && D(tr, "value") > 40 && D(tr, "value") < 80, "vocabulary: " + tr.error);

            // A word clicked: the texts that have it (whole words), and a length range.
            FilterState wf;
            FilterPredicate has;
            has.field = "statement";
            has.op = FilterPredicate::Op::Contains;
            has.values = {"sleep"};
            has.source_widget = words->id;
            wf.Set(has);
            Check(wf.Text() == "statement has 'sleep'", "the filter in words: " + wf.Text());
            tr = engine.Run(StripQuery("Notes", wf, tp, "", false));
            Check(tr.ok && D(tr, "rows_now") == 3, "texts with the word 'sleep': " + tr.error);
            has.values = {"feel tired"};
            wf.Set(has);
            tr = engine.Run(StripQuery("Notes", wf, tp, "", false));
            Check(tr.ok && D(tr, "rows_now") == 4, "texts with the phrase 'feel tired': " + tr.error);
            FilterPredicate longer;
            longer.field = "statement";
            longer.op = FilterPredicate::Op::Range;
            longer.bucket = "words";
            longer.lo = 10;
            longer.hi = 100;
            longer.source_widget = length->id;
            FilterState lf;
            lf.Set(longer);
            tr = engine.Run(StripQuery("Notes", lf, tp, "", false));
            Check(tr.ok && D(tr, "rows_now") == 3, "texts of 10 words or more: " + tr.error);

            // Saved and read back.
            ts.widgets = tw;
            ts.filters = wf;
            DashboardSpec back;
            std::string why;
            Check(DashboardFromJson(DashboardToJson(ts), back, &why), "text dashboard JSON: " + why);
            const WidgetSpec* bw = back.Find(words->id);
            Check(bw && bw->text_view == TextView::Words && bw->text_field == "statement" && bw->plot.bar_horizontal &&
                      back.filters.predicates.size() == 1 && back.filters.predicates[0].op == FilterPredicate::Op::Contains,
                  "text widgets and the word filter round trip");

            // The words split once and saved (owner 2026-10-05): split, then loaded; the
            // text widgets, KPIs and word filters read them and agree with splitting here.
            const fs::path words_dir = root / "words";
            const std::string key = WordsCacheKey("", "Notes", 1, "statement");
            WordsTable wt = EnsureWordsTable("Notes", "statement", key, words_dir, run);
            Check(wt.error.empty() && !wt.loaded && wt.rows == 11 && fs::exists(wt.path), "words split and saved: " + wt.error);
            wt = EnsureWordsTable("Notes", "statement", key, words_dir, run);
            Check(wt.error.empty() && wt.loaded && wt.rows == 11, "words loaded the second time");
            Check(WordsCacheKey("", "Notes", 2, "statement") != key, "new data, new key");
            Check(engine.AttachParquet(wt.name, 1, wt.path, wt.rows, &error), "attach the words: " + error);
            tr = engine.Run(TextWidgetQuery(*phrases, "Notes", FilterState{}, 0, wt.name));
            Check(tr.ok && word_at(tr, 0) == "feel tired" && D(tr, "count") == 4, "phrases from the saved words: " + tr.error);
            tr = engine.Run(TextWidgetQuery(*length, "Notes", FilterState{}, 0, wt.name));
            Check(tr.ok && tr.table->num_rows() == 11, "lengths from the saved words: " + tr.error);
            k = tw[1];
            QueryResult direct = engine.Run(KpiQuery(k, "Notes", FilterState{}));
            tr = engine.Run(KpiQuery(k, "Notes", FilterState{}, wt.name));
            Check(tr.ok && direct.ok && D(tr, "value") == D(direct, "value"), "vocabulary from the saved words: " + tr.error);
            k = tw[0];
            direct = engine.Run(KpiQuery(k, "Notes", FilterState{}));
            tr = engine.Run(KpiQuery(k, "Notes", FilterState{}, wt.name));
            Check(tr.ok && D(tr, "value") == D(direct, "value"), "median words from the saved words: " + tr.error);
            FilterState sleep;
            FilterPredicate sp;
            sp.field = "statement";
            sp.op = FilterPredicate::Op::Contains;
            sp.values = {"sleep"};
            sp.source_widget = "elsewhere";
            sleep.Set(sp);
            tr = engine.Run(TextWidgetQuery(*length, "Notes", sleep, 0, wt.name));
            Check(tr.ok && tr.table->num_rows() == 3, "a word filter over the saved words: " + tr.error);
        }
    }
    fs::remove_all(root, ec);
    std::cout << "dashboard runtime: automatic layout, widget / sample / table / KPI / summary SQL through the engine with the "
                 "cross-filter rule, binding checks with rename candidates and type changes, SQL as text with values written in, query widgets over the filtered rows, text widgets (words, phrases, by class, lengths, KPIs, word filters). OK\n";
    return 0;
}
