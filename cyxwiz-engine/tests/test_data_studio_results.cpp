// Data Studio > Results' model (TOFIX148 job 7): the Berean SFT generation
// previews (four prompts from sft_v2 epoch 4), a small classifier test, and
// (TOFIX157 B2) a regression test and errors joined back to the data by row.

#include "../src/core/data_studio_results.h"
#include "../src/core/test_predictions_table.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using namespace cyxwiz::studio;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
void Same(const std::string& got, const std::string& want, const std::string& what) {
    Check(got == want, what + ": got \"" + got + "\", want \"" + want + "\"");
}
std::string Prompt(const std::string& question, const std::string& ref) {
    return "System: Answer using the supplied evidence.\nUser: <question>" + question +
           "</question>\n<evidence>\n<source id=\"x\"><reference>" + ref +
           "</reference><kind>scripture</kind><text>...</text></source>\n</evidence>\nBerean: ";
}
}  // namespace

int main() {
    // One sample: question, evidence, citation match, loops.
    {
        auto s = PreviewSampleFrom(Prompt("Is it really lying?", "Psalm 34:13"),
                                   "<explanation>Psalm 34:1 says to keep your heart.</explanation>", "max_tokens");
        Same(s.question, "Is it really lying?", "question from <question>");
        Same(s.evidence_ref, "Psalm 34:13", "evidence reference");
        Same(s.cited_ref, "Psalm 34:1", "cited reference");
        Check(s.has_ref && s.book && s.chapter && !s.verse, "right book and chapter, wrong verse");
        Check(!s.loops, "no loop");
        Same(StopWords(s.stop), "token limit", "stop words");

        auto loops = PreviewSampleFrom(Prompt("My dog?", "Deuteronomy 25:4"),
                                       "Deuteronomy 25:2 says to be a shepherd, and the shepherds and the shepherds and the shepherds and the", "eos");
        Check(loops.loops, "a repeated phrase loops");
        Check(loops.book && loops.chapter && !loops.verse, "Deuteronomy 25");

        auto numbered = PreviewSampleFrom(Prompt("Give thanks?", "1 Thessalonians 5:18"), "Romans 5:24 says ...", "eos");
        Check(numbered.has_ref && !numbered.book, "a numbered book parses; Romans is the wrong book");

        auto plain = PreviewSampleFrom("User: What is 2 + 2?", "4", "eos");
        Same(plain.question, "What is 2 + 2?", "question after User:");
        Check(!plain.has_ref && plain.evidence_ref.empty(), "no evidence reference");
    }

    // A preview file: summary and slices, as the mockup board.
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_results" / "sft_v2" / "train-1" / "generation_previews";
    fs::remove_all(dir.parent_path().parent_path());
    fs::create_directories(dir);
    nlohmann::json j = {{"run_id", "train-1"}, {"completed_epoch", 4}, {"sampling", "greedy"}, {"max_new_tokens", 128},
                        {"cancelled", false}, {"samples", nlohmann::json::array()}};
    const auto add = [&](const std::string& q, const std::string& ref, const std::string& out, const std::string& stop) {
        j["samples"].push_back({{"prompt", Prompt(q, ref)}, {"generated_text", out}, {"stop_reason", stop}});
    };
    add("Limited time offer?", "Psalm 34:13", "Psalm 34:2 and the lie and the lie and the lie and the lie", "max_tokens");
    add("Dog chained?", "Deuteronomy 25:4", "Deuteronomy 25:2 says", "max_tokens");
    add("Give thanks?", "1 Thessalonians 5:18", "Romans 5:24 says", "eos");
    add("Owe no one?", "Romans 13:8", "Romans 13:2 says", "eos");
    {
        std::ofstream f(dir / "epoch_4.json");
        f << j.dump(2);
        std::ofstream g(dir / "epoch_2.json");
        g << j.dump(2);
    }
    const auto runs = FindPreviewRuns((dir.parent_path().parent_path().parent_path()).string());
    Check(runs.size() == 1, "one run");
    Same(runs[0].run_id, "train-1", "run id");
    Same(runs[0].graph, "sft_v2", "graph folder");
    Check(runs[0].epochs == std::vector<int>({2, 4}), "epochs ascending");

    PreviewEpoch epoch;
    std::string error;
    Check(LoadPreviewEpoch(dir / "epoch_4.json", epoch, error), "load: " + error);
    Check(epoch.epoch == 4 && epoch.samples.size() == 4 && epoch.sampling == "greedy", "epoch read");
    const auto sum = SummarizePreview(epoch);
    Check(sum.right_book == 3 && sum.right_chapter == 3 && sum.right_verse == 0, "3 of 4 book and chapter, 0 verse (the board)");
    Check(sum.stopped_itself == 2 && sum.hit_limit == 2 && sum.loops == 1, "2 stopped, 2 hit the limit, 1 loops (the board)");
    const auto slices = PreviewSlices(epoch);
    Check(slices.size() == 3, "stop = token limit, stop = end, loops");
    Same(slices[0].name, "stop = token limit", "first slice");
    Check(slices[0].rows == 2 && slices[1].rows == 2 && slices[2].rows == 1, "slice rows");
    Check(!LoadPreviewEpoch(dir / "missing.json", epoch, error), "a missing file");
    fs::remove_all(dir.parent_path().parent_path().parent_path());
    Check(FindPreviewRuns((fs::temp_directory_path() / "cyxwiz_no_such_folder").string()).empty(), "no folder, no runs");

    // A classifier test: 3 classes, 6 rows.
    {
        arrow::Int64Builder t, p;
        arrow::DoubleBuilder c, s0, s1, s2;
        const int64_t truth[] = {0, 0, 1, 1, 2, 2};
        const int64_t pred[] = {0, 1, 1, 1, 2, 0};
        const double conf[] = {0.9, 0.8, 0.7, 0.6, 0.95, 0.99};
        for (int i = 0; i < 6; ++i) {
            (void)t.Append(truth[i]);
            (void)p.Append(pred[i]);
            (void)c.Append(conf[i]);
            (void)s0.Append(0.1);
            (void)s1.Append(0.1);
            (void)s2.Append(0.8);
        }
        std::shared_ptr<arrow::Array> ta, pa, ca, a0, a1, a2;
        (void)t.Finish(&ta);
        (void)p.Finish(&pa);
        (void)c.Finish(&ca);
        (void)s0.Finish(&a0);
        (void)s1.Finish(&a1);
        (void)s2.Finish(&a2);
        auto schema = arrow::schema({arrow::field("true_class", arrow::int64()), arrow::field("predicted_class", arrow::int64()),
                                     arrow::field("confidence", arrow::float64()), arrow::field("score_cat", arrow::float64()),
                                     arrow::field("score_dog", arrow::float64()), arrow::field("score_bird", arrow::float64())});
        auto table = arrow::Table::Make(schema, {ta, pa, ca, a0, a1, a2});
        const auto r = ClassifierFromPredictions(table);
        Check(r.ok, "classifier results: " + r.error);
        Check(r.rows == 6 && std::abs(r.accuracy - 4.0 / 6.0) < 1e-9, "accuracy 4 of 6");
        Check(r.classes == std::vector<std::string>({"cat", "dog", "bird"}), "class names from score columns");
        Check(r.confusion[0][1] == 1 && r.confusion[2][0] == 1 && r.confusion[1][1] == 2, "confusion");
        Check(std::abs(r.per_class[0].recall - 0.5) < 1e-9 && std::abs(r.per_class[1].precision - 2.0 / 3.0) < 1e-9, "per class");
        Check(r.worst.size() == 2 && r.worst[0].row == 6 && r.worst[0].truth == "bird" && r.worst[0].predicted == "cat",
              "the worst row is the most confident wrong one");
        Check(!ClassifierFromPredictions(nullptr).ok, "no table");
        Check(!RowErrorsFromPredictions(table).ok, "no row_id: the rows are not known");
    }

    // The same test with its rows recorded (TOFIX157 B2): errors joined back to
    // the tested table's columns, read in two pieces like a file's row groups.
    {
        const cyxwiz::TestPredictionsSource source{{10, 3, 7, 0, 5, 8}, "cities"};
        const auto table = cyxwiz::BuildTestPredictionsTable({0, 0, 1, 1, 2, 2}, {0, 1, 1, 1, 2, 0},
                                                             {0.9f, 0.8f, 0.7f, 0.6f, 0.95f, 0.99f}, {}, 0, {}, source);
        const auto r = ClassifierFromPredictions(table);
        Check(r.ok && r.worst.size() == 2 && r.worst[0].row == 6 && r.worst[0].dataset_row == 8,
              "the worst row names its dataset row");
        const auto errors = RowErrorsFromPredictions(table);
        Check(errors.ok && errors.classifier && errors.source_dataset == "cities", "row errors: " + errors.error);
        Check(errors.errors == std::vector<double>({0, 1, 0, 0, 0, 1}), "a classifier's error is 1 when wrong");

        // The tested table: city A/B/C by row % 3, age = row * 10.
        const auto piece = [](int first, int count, bool numbers) {
            std::shared_ptr<arrow::Array> a;
            if (numbers) {
                arrow::Int64Builder b;
                for (int i = first; i < first + count; ++i) (void)b.Append(i * 10);
                (void)b.Finish(&a);
            } else {
                arrow::StringBuilder b;
                for (int i = first; i < first + count; ++i) (void)b.Append(i % 3 == 0 ? "A" : i % 3 == 1 ? "B" : "C");
                (void)b.Finish(&a);
            }
            return std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{a});
        };
        std::string why;
        cyxwiz::studio::GatheredColumn city;
        cyxwiz::studio::ColumnGatherer by_city(errors.row_ids);
        Check(by_city.Add(piece(0, 6, false), why) && by_city.Add(piece(6, 6, false), why) &&
                  by_city.Finish(city, why),
              "gather city: " + why);
        Check(!city.numeric && city.labels == std::vector<std::string>({"B", "A", "B", "A", "C", "C"}), "city per test row");
        const auto by_value = SliceErrors(city, errors.errors);
        Check(!by_value.binned && by_value.slices.size() == 3 && std::abs(by_value.overall - 2.0 / 6.0) < 1e-9, "three cities");
        Check(by_value.slices[0].value == "A" && std::abs(by_value.slices[0].error - 0.5) < 1e-9 && by_value.slices[0].rows == 2 &&
                  by_value.slices[2].value == "B" && by_value.slices[2].error == 0.0,
              "highest error first");

        cyxwiz::studio::GatheredColumn age;
        cyxwiz::studio::ColumnGatherer by_age(errors.row_ids);
        Check(by_age.Add(piece(0, 6, true), why) && by_age.Add(piece(6, 6, true), why) && by_age.Finish(age, why),
              "gather age: " + why);
        const auto ranges = SliceErrors(age, errors.errors, 3, 2);
        Check(ranges.binned && ranges.slices.size() == 2, "six ages in two ranges");
        Check(ranges.slices.size() == 2 && ranges.slices[0].value == "0 to 50" && ranges.slices[1].value == "70 to 100" &&
                  std::abs(ranges.slices[0].error - 1.0 / 3.0) < 1e-9,
              "ranges of equal rows");

        cyxwiz::studio::GatheredColumn short_table;
        cyxwiz::studio::ColumnGatherer past_end(errors.row_ids);
        Check(past_end.Add(piece(0, 6, false), why) && !past_end.Finish(short_table, why) &&
                  why.find("changed since the test") != std::string::npos,
              "a table shorter than the test's rows is refused");
    }

    // A regression test.
    {
        const auto table = cyxwiz::BuildRegressionPredictionsTable({1, 2, 3}, {1.5f, 2, 1}, 1, {{4, 5, 6}, "houses"});
        Check(IsRegressionPredictions(table), "a regression table");
        const auto r = RegressionFromPredictions(table);
        Check(r.ok && r.rows == 3 && r.targets == 1, "regression results: " + r.error);
        Check(std::abs(r.mae - 2.5 / 3.0) < 1e-6 && std::abs(r.rmse - std::sqrt(4.25 / 3.0)) < 1e-6 &&
                  std::abs(r.bias + 1.5 / 3.0) < 1e-6,
              "MAE, RMSE and bias");
        Check(r.worst.size() == 3 && r.worst[0].row == 3 && r.worst[0].dataset_row == 6 &&
                  std::abs(r.worst[0].error + 2.0) < 1e-6,
              "the largest error first");
        const auto errors = RowErrorsFromPredictions(table);
        Check(errors.ok && !errors.classifier && errors.source_dataset == "houses" &&
                  errors.row_ids == std::vector<int64_t>({4, 5, 6}) && std::abs(errors.errors[2] - 2.0) < 1e-6,
              "regression row errors are |error|");
        Check(!IsRegressionPredictions(cyxwiz::BuildTestPredictionsTable({0}, {0}, {1.0f}, {}, 0, {})),
              "a classifier table is not regression");
    }

    std::cout << "test_data_studio_results: all checks passed\n";
    return 0;
}
