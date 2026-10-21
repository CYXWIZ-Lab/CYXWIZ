// A dataset's first rows as a Table Viewer table (TOFIX140, off the UI
// thread since TOFIX159): typed cells straight from Arrow, chunk by chunk
// with progress and a stop; row groups read each once.
#include "../src/core/dataset_table_slice.h"
#include "../src/data/data_table.h"

#include <arrow/api.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

// `rows` rows from `first`: id (int64), name (text, empty every 7th) and
// pixel (uint8).
std::shared_ptr<arrow::Table> Rows(int64_t first, int64_t rows) {
    arrow::Int64Builder ids;
    arrow::StringBuilder names;
    arrow::UInt8Builder pixels;
    for (int64_t i = first; i < first + rows; ++i) {
        (void)ids.Append(i);
        if (i % 7 == 0) (void)names.AppendNull();
        else (void)names.Append("row " + std::to_string(i));
        (void)pixels.Append(static_cast<uint8_t>(i % 256));
    }
    std::shared_ptr<arrow::Array> a, b, c;
    (void)ids.Finish(&a);
    (void)names.Finish(&b);
    (void)pixels.Finish(&c);
    auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8()),
                                 arrow::field("pixel", arrow::uint8())});
    return arrow::Table::Make(schema, {a, b, c});
}

}  // namespace

int main() {
    // 450 of a 500-row dataset, 200 rows per chunk: three chunks, progress after each.
    std::vector<float> seen;
    DatasetSlice slice = DatasetSliceFromArrow(Rows(0, 500), "ids", 450, 500, [&seen](float done, const std::string&) {
        seen.push_back(done);
        return true;
    }, 200);
    Check(slice.error.empty() && slice.table && slice.table->GetRowCount() == 450, "450 rows");
    Check(slice.total_rows == 500 && slice.source && slice.source->num_rows() == 450, "total kept, source sliced");
    Check(seen.size() == 3 && seen.back() == 1.0f, "progress after each chunk, ending at 1");
    Check(slice.table->GetName() == "ids" &&
              slice.table->GetHeaders() == std::vector<std::string>({"id", "name", "pixel"}),
          "named, with the columns");
    Check(std::get<int64_t>(slice.table->GetCell(449, 0)) == 449, "rows in order across chunks");
    Check(std::holds_alternative<std::string>(slice.table->GetCell(1, 1)), "text stays text");
    Check(std::holds_alternative<std::monostate>(slice.table->GetCell(0, 1)), "null stays empty");
    Check(std::get<int64_t>(slice.table->GetCell(300, 2)) == 300 % 256, "uint8 is an integer");

    // Asking for more rows than there are opens them all.
    slice = DatasetSliceFromArrow(Rows(0, 30), "small", 1000, 30);
    Check(slice.table && slice.table->GetRowCount() == 30, "all 30 rows");

    // A stop ends the read with no half table.
    slice = DatasetSliceFromArrow(Rows(0, 500), "stopped", 500, 500, [](float, const std::string&) { return false; },
                                  100);
    Check(slice.stopped && !slice.table && !slice.source && slice.error.empty(), "stopped: no table, no error");

    // Row groups: each read once, only as many as the rows need.
    std::vector<int> reads;
    std::string error;
    const auto group = [&reads](int g) {
        reads.push_back(g);
        return Rows(g * 100, 100);
    };
    auto joined = FirstRowsOfRowGroups(5, group, 250, {}, error);
    Check(joined && joined->num_rows() == 250 && error.empty(), "250 rows from row groups");
    Check(reads == std::vector<int>({0, 1, 2}), "three groups read, each once");
    slice = DatasetSliceFromArrow(joined, "groups", 250, 500);
    Check(slice.table && std::get<int64_t>(slice.table->GetCell(249, 0)) == 249, "rows in order across groups");

    reads.clear();
    Check(!FirstRowsOfRowGroups(5, [](int) { return std::shared_ptr<arrow::Table>(); }, 10, {}, error) &&
              error == "Row group 1 could not be read.",
          "a failed group is reported");
    error.clear();
    Check(!FirstRowsOfRowGroups(5, group, 250, [](float, const std::string&) { return false; }, error) &&
              error.empty() && reads.empty(),
          "a stop reads nothing more");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("dataset table slice: typed rows by chunk, row groups once, stops honoured. OK\n");
    return 0;
}
