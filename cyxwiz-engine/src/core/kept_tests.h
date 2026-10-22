#pragma once

// Kept tests (TOFIX157, "Keep as node"): a test run's per-row table
// (test_predictions) saved once as a Parquet file in the project, so it stays
// when the model is tested again. The graph reads it through a Data Input and
// a Test Evaluation node; Results > Test opens it like a run. The file
// keeps the table's metadata (tested table, class names, loss, time) and the
// name it was kept under.

#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// The schema metadata key holding the name a test was kept under.
inline constexpr const char* kKeptTestNameKey = "kept_name";

// <project>/results/tests.
std::filesystem::path KeptTestsFolder(const std::filesystem::path& project_root);

// "Blobs MLP test" at 2026-10-17 14:32 -> "blobs_mlp_test_20261017_1432.parquet":
// the name in lower case with runs of other characters as "_", then the time.
std::string KeptTestFileName(const std::string& name, std::time_t when);

// Writes `rows` with the name in its metadata; false with `error` when the
// folder or file cannot be written (an existing file is never replaced).
bool WriteKeptTest(const arrow::Table& rows, const std::string& name, const std::filesystem::path& path, std::string& error);

// Reads a kept test back (with its metadata); nullptr with `error`.
std::shared_ptr<arrow::Table> ReadKeptTest(const std::filesystem::path& path, std::string& error);

struct KeptTest {
    std::string name;  // the kept name, else the file's stem
    std::filesystem::path path;
    std::filesystem::file_time_type time{};
};
// The project's kept tests, newest first (only the names are read).
std::vector<KeptTest> ListKeptTests(const std::filesystem::path& folder);

}  // namespace cyxwiz
