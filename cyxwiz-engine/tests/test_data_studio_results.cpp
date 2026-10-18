// Data Studio > Results' model (TOFIX157 B2): a test's errors joined back to
// the data by row, for a classifier and a regression. Generation previews are
// tested with their presentation model (test_training_generation_preview).
// The run's own figures read back from its table are checked in
// test_training_executor_arrow_parquet (one calculation, TOFIX157).

#include "../src/core/data_studio_results.h"
#include "../src/core/test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

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
}  // namespace

int main() {
    // A test without its rows: they cannot be joined back.
    Check(!RowErrorsFromPredictions(cyxwiz::BuildTestPredictionsTable({0, 1}, {0, 0}, {0.9f, 0.8f}, {}, 0, {})).ok,
          "no row_id: the rows are not known");
    Check(!RowErrorsFromPredictions(nullptr).ok, "no table");

    // A kept table read back through a Data Input can have narrower integers.
    {
        arrow::UInt8Builder t8, p8;
        arrow::Int32Builder ids;
        (void)t8.AppendValues(std::vector<uint8_t>{0, 1, 1});
        (void)p8.AppendValues(std::vector<uint8_t>{0, 0, 1});
        (void)ids.AppendValues(std::vector<int32_t>{4, 9, 2});
        std::shared_ptr<arrow::Array> ta, pa, ia;
        (void)t8.Finish(&ta);
        (void)p8.Finish(&pa);
        (void)ids.Finish(&ia);
        const auto narrow = arrow::Table::Make(
            arrow::schema({arrow::field("row_id", arrow::int32()), arrow::field("true_class", arrow::uint8()),
                           arrow::field("predicted_class", arrow::uint8())},
                          arrow::key_value_metadata({cyxwiz::kTestPredictionsSourceKey}, {"blobs"})),
            {ia, ta, pa});
        const auto errors = RowErrorsFromPredictions(narrow);
        Check(errors.ok && errors.row_ids == std::vector<int64_t>({4, 9, 2}) &&
                  errors.errors == std::vector<double>({0, 1, 0}),
              "narrow integer columns: " + errors.error);
    }

    // The same test with its rows recorded (TOFIX157 B2): errors joined back to
    // the tested table's columns, read in two pieces like a file's row groups.
    {
        const cyxwiz::TestPredictionsSource source{{10, 3, 7, 0, 5, 8}, "cities"};
        const auto table = cyxwiz::BuildTestPredictionsTable({0, 0, 1, 1, 2, 2}, {0, 1, 1, 1, 2, 0},
                                                             {0.9f, 0.8f, 0.7f, 0.6f, 0.95f, 0.99f}, {}, 0, {}, source);
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
        const auto errors = RowErrorsFromPredictions(table);
        Check(errors.ok && !errors.classifier && errors.source_dataset == "houses" &&
                  errors.row_ids == std::vector<int64_t>({4, 5, 6}) && std::abs(errors.errors[2] - 2.0) < 1e-6,
              "regression row errors are |error|");
    }

    std::cout << "test_data_studio_results: all checks passed\n";
    return 0;
}
