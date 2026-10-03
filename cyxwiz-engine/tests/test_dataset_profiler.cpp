// Dataset profiler (TOFIX134 P3.5): known numbers on a small table through
// the real query engine.

#include "../src/core/dataset_profiler.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> Tracks() {
    // 10 rows: id unique, genre with "N/A" twice and one null, popularity
    // 1..9 plus 100 (an outlier), duration, release date, explicit.
    arrow::StringBuilder id, genre, date;
    arrow::Int64Builder pop;
    arrow::DoubleBuilder dur;
    arrow::BooleanBuilder expl;
    const char* genres[] = {"pop", "pop", "rap", "N/A", "pop", "rock", "N/A", "rap", "pop", nullptr};
    const int64_t pops[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    for (int i = 0; i < 10; ++i) {
        (void)id.Append("t" + std::to_string(i));
        if (genres[i]) (void)genre.Append(genres[i]);
        else (void)genre.AppendNull();
        (void)pop.Append(pops[i]);
        (void)dur.Append(2.0 + i * 0.5);
        (void)date.Append("2024-0" + std::to_string(1 + i % 9) + "-15");
        (void)expl.Append(i % 3 == 0);
    }
    std::shared_ptr<arrow::Array> a, b, c, d, e, f;
    (void)id.Finish(&a);
    (void)genre.Finish(&b);
    (void)pop.Finish(&c);
    (void)dur.Finish(&d);
    (void)date.Finish(&e);
    (void)expl.Finish(&f);
    auto schema = arrow::schema({arrow::field("track_id", arrow::utf8()), arrow::field("genre", arrow::utf8()),
                                 arrow::field("popularity", arrow::int64()), arrow::field("duration", arrow::float64()),
                                 arrow::field("release_date", arrow::utf8()), arrow::field("explicit", arrow::boolean())});
    // Two identical rows make one duplicate.
    auto t = arrow::Table::Make(schema, {a, b, c, d, e, f});
    auto extra = t->Slice(0, 1);
    return *arrow::ConcatenateTables({t, extra});
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_profiler";
    std::error_code ec;
    fs::remove_all(root, ec);
    {
        SessionQueryEngine engine((root / "attach").string());
        std::string error;
        Check(engine.AttachArrow("Spotify", 1, Tracks(), &error), "attach: " + error);
        const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
        ProfileOptions options;
        options.missing_text["genre"] = {"N/A"};
        int progress_calls = 0;
        options.progress = [&](float, const std::string&) { ++progress_calls; };
        DatasetProfile p = ProfileTable("Spotify", run, options);
        Check(p.ok(), "profiled: " + p.error);
        Check(p.rows == 11 && p.columns.size() == 6 && p.exact, "11 rows, 6 columns, exact");
        Check(progress_calls > 3, "progress reported");

        const ProfiledColumn* id = p.Find("track_id");
        Check(id->facts.type == ColumnFacts::Type::Text && id->facts.distinct == 10 && id->missing == 0, "id: 10 distinct of 11 (one duplicate row)");
        const ProfiledColumn* genre = p.Find("genre");
        Check(genre->missing_text == 2 && genre->missing == 3 && genre->facts.non_null == 8, "genre: one null and two N/A are missing");
        Check(genre->facts.distinct == 3, "genre: pop, rap, rock (N/A not a value)");
        Check(!genre->top.empty() && genre->top[0].first == "pop" && genre->top[0].second == 5, "top value pop x5 (the duplicate row adds one)");
        const ProfiledColumn* pop = p.Find("popularity");
        Check(pop->numeric && pop->min == 1 && pop->max == 100 && std::fabs(pop->median - 5.0) < 1e-9, "popularity range and median");
        Check(pop->outliers == 1, "100 is the one outlier");
        size_t hist = 0;
        for (size_t n : pop->hist_counts) hist += n;
        Check(pop->hist_counts.size() == 20 && pop->hist_edges.size() == 21 && hist == 11 && pop->hist_counts.back() == 1, "histogram holds every row");
        const ProfiledColumn* date = p.Find("release_date");
        Check(date->facts.date_share == 1.0 && date->min_text == "2024-01-15" && date->max_text == "2024-09-15", "dates read as dates");
        Check(InferRole(date->facts) == ColumnRole::DateTime, "and infer as a date");
        const ProfiledColumn* expl = p.Find("explicit");
        Check(expl->facts.type == ColumnFacts::Type::Boolean && expl->facts.distinct == 2 && expl->top.size() == 2, "explicit: two values");
        Check(p.duplicates_known && p.duplicate_rows == 1, "one duplicate row");
        Check(!p.correlations.empty() && std::get<2>(p.correlations.front()) > 0.5, "popularity and duration move together");
        Check(p.MissingCells() == 3, "missing cells");
        Check(p.Facts().size() == 6, "facts for roles");

        // The profile feeds the contract.
        DatasetContract contract = BuildContract("k", p.Facts(), {{"popularity", ColumnRole::Target}}, {});
        Check(contract.Find("track_id")->role == ColumnRole::Id && contract.Find("genre")->role == ColumnRole::Category &&
                  contract.Find("popularity")->role == ColumnRole::Target, "contract from the profile");

        // Cancelled before it starts counting.
        ProfileOptions stopped;
        stopped.should_stop = [] { return true; };
        Check(ProfileTable("Spotify", run, stopped).error == "Cancelled.", "cancellation");
        Check(!ProfileTable("Nope", run).ok(), "an unknown table is an error");
    }
    fs::remove_all(root, ec);
    std::cout << "dataset profiler: counts, missing texts, distinct, quartiles, outliers, histogram, top values, dates, booleans, "
                 "duplicates, correlations, contract from the profile, cancellation. OK\n";
    return 0;
}
