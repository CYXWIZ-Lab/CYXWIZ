// A loaded dataset's first rows as a Table Viewer table (TOFIX140): typed
// cells from a data-service page, pages appended in order.
#include "../src/core/dataset_table_slice.h"
#include "../src/data/data_table.h"

#include <algorithm>
#include <cstdio>
#include <string>

using namespace cyxwiz;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

DataPreviewPage Page(std::vector<std::vector<std::string>> rows) {
    DataPreviewPage page;
    page.ok = true;
    page.schema = {{"year", "int64"}, {"month", "string"}, {"passengers", "double"}};
    page.rows = std::move(rows);
    page.rows_returned = static_cast<int64_t>(page.rows.size());
    return page;
}

}  // namespace

int main() {
    auto table = DataTableFromPreview(Page({{"1949", "Jan", "112"}, {"1949", "", "118.5"}}), "airline_polar");
    Check(table != nullptr, "table built");
    Check(table->GetName() == "airline_polar", "named after the dataset");
    Check(table->GetHeaders() == std::vector<std::string>({"year", "month", "passengers"}), "headers");
    Check(table->GetRowCount() == 2, "two rows");
    Check(std::holds_alternative<int64_t>(table->GetCell(0, 0)) && std::get<int64_t>(table->GetCell(0, 0)) == 1949,
          "integer column stays an integer");
    Check(std::holds_alternative<std::string>(table->GetCell(0, 1)), "text column stays text");
    Check(std::holds_alternative<std::monostate>(table->GetCell(1, 1)), "empty cell is missing");
    Check(std::holds_alternative<double>(table->GetCell(1, 2)) && std::get<double>(table->GetCell(1, 2)) == 118.5,
          "floating column is a double");

    // A second page appends to the same table.
    table = DataTableFromPreview(Page({{"1949", "Mar", "132"}}), "airline_polar", table);
    Check(table->GetRowCount() == 3, "second page appended");

    // A malformed number in a numeric column keeps its text.
    auto odd = DataTableFromPreview(Page({{"n/a", "Apr", "x"}}), "odd");
    Check(std::holds_alternative<std::string>(odd->GetCell(0, 0)), "unparsable integer kept as text");

    // Paging: 450 rows asked of a 500-row dataset served 200 at a time.
    int calls = 0;
    const auto reader = [&calls](const DataPreviewRequest& request) {
        ++calls;
        DataPreviewPage page;
        page.ok = true;
        page.total_rows = 500;
        page.schema = {{"id", "int64"}};
        const int64_t n = std::min<int64_t>({request.row_limit, 200, 500 - request.offset});
        for (int64_t i = 0; i < n; ++i) page.rows.push_back({std::to_string(request.offset + i)});
        page.rows_returned = n;
        page.offset = request.offset;
        page.next_offset = request.offset + n;
        page.has_next = page.next_offset < 500;
        return page;
    };
    DatasetSlice slice = ReadDatasetSlice(reader, "ids", 450);
    Check(slice.error.empty() && slice.table && slice.table->GetRowCount() == 450, "450 rows over pages");
    Check(calls == 3 && slice.total_rows == 500, "three pages, total known");
    Check(std::get<int64_t>(slice.table->GetCell(449, 0)) == 449, "rows in order");

    // A failed page is reported, not a half table.
    slice = ReadDatasetSlice([](const DataPreviewRequest&) {
        DataPreviewPage page;
        page.reason = "Dataset not found";
        return page;
    }, "gone", 10);
    Check(!slice.table && slice.error == "Dataset not found", "failure reported");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("dataset table slice: typed cells and appended pages as expected. OK\n");
    return 0;
}
