// Kept tests (TOFIX157, Keep as node): a test's per-row table saved once as a
// Parquet file in <project>/results/tests - the file name, the round trip with
// its metadata and kept name, never replacing a file, and the list newest first.
#include "../src/core/kept_tests.h"
#include "../src/core/test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

std::string Metadata(const std::shared_ptr<arrow::Table>& table, const char* key) {
    const auto metadata = table ? table->schema()->metadata() : nullptr;
    const int at = metadata ? metadata->FindKey(key) : -1;
    return at >= 0 ? metadata->value(at) : std::string("(none)");
}

}  // namespace

int main() {
    // The file name: the name in lower case, other characters as "_", the time.
    std::tm when{};
    when.tm_year = 2026 - 1900;
    when.tm_mon = 9;
    when.tm_mday = 17;
    when.tm_hour = 14;
    when.tm_min = 32;
    when.tm_isdst = -1;
    const std::time_t t = std::mktime(&when);
    Check(cyxwiz::KeptTestFileName("Blobs MLP test", t) == "blobs_mlp_test_20261017_1432.parquet",
          "file name: " + cyxwiz::KeptTestFileName("Blobs MLP test", t));
    Check(cyxwiz::KeptTestFileName("  ##  ", t) == "test_20261017_1432.parquet", "no letters: test");
    Check(cyxwiz::KeptTestFileName("Iris / v2: final!", t) == "iris_v2_final_20261017_1432.parquet", "punctuation");
    Check(cyxwiz::KeptTestsFolder("D:/p") == fs::path("D:/p") / "results" / "tests", "the folder");

    // Round trip: rows, row ids, the tested table, class names and the kept name.
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_kept_tests";
    fs::remove_all(root);
    const fs::path folder = cyxwiz::KeptTestsFolder(root);
    cyxwiz::TestPredictionsSource source{{106, 319, 410}, "metric_blobs"};
    source.loss = 0.21;
    const auto rows = cyxwiz::BuildTestPredictionsTable({1, 1, 0}, {0, 0, 1}, {0.9f, 0.8f, 0.7f}, {}, 0,
                                                        {"Class 0", "Class 1"}, source);
    std::string error;
    const fs::path first = folder / cyxwiz::KeptTestFileName("Blobs MLP test", t);
    Check(rows && cyxwiz::WriteKeptTest(*rows, "Blobs MLP test", first, error), "write: " + error);
    const auto back = cyxwiz::ReadKeptTest(first, error);
    Check(back && back->num_rows() == 3 && back->GetColumnByName("row_id"), "read back: " + error);
    Check(Metadata(back, cyxwiz::kKeptTestNameKey) == "Blobs MLP test", "the kept name");
    Check(Metadata(back, cyxwiz::kTestPredictionsSourceKey) == "metric_blobs", "the tested table");
    Check(Metadata(back, cyxwiz::kTestPredictionsClassNamesKey) == "Class 0\nClass 1", "the class names");
    Check(Metadata(rows, cyxwiz::kKeptTestNameKey) == "(none)", "the published table is not changed");

    // Never replaced.
    Check(!cyxwiz::WriteKeptTest(*rows, "again", first, error) && error.find("already exists") != std::string::npos,
          "an existing file is never replaced");

    // A second, newer one: the list is newest first, by kept name.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const fs::path second = folder / "second.parquet";
    Check(cyxwiz::WriteKeptTest(*back, "Retrained", second, error), "write a kept test again: " + error);
    fs::last_write_time(second, fs::last_write_time(first) + std::chrono::seconds(5));
    const auto list = cyxwiz::ListKeptTests(folder);
    Check(list.size() == 2 && list[0].name == "Retrained" && list[1].name == "Blobs MLP test",
          "newest first, by kept name");
    Check(Metadata(cyxwiz::ReadKeptTest(second, error), cyxwiz::kKeptTestNameKey) == "Retrained",
          "kept again: the new name only");
    Check(cyxwiz::ListKeptTests(root / "missing").empty(), "no folder: none");
    Check(!cyxwiz::ReadKeptTest(root / "missing.parquet", error) && !error.empty(), "a missing file says why");
    fs::remove_all(root);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kept tests: file name, round trip, never replaced, list. OK\n");
    return 0;
}
