// Preferences > Storage's model (TOFIX160 P7): the five caches of a project
// measured, one cleared by id (files in use stay), no project -> the temp
// cache only, item wording.
#include "../src/core/storage_caches.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

void Make(const fs::path& p, size_t bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << std::string(bytes, 'x');
}

const StorageCache* Find(const std::vector<StorageCache>& caches, const std::string& id) {
    for (const auto& c : caches)
        if (c.id == id) return &c;
    return nullptr;
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("cyxwiz_storage_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Make(root / "cache" / "ingestion" / "mnist_test_1.parquet", 300);
    Make(root / "cache" / "ingestion" / "held.parquet", 200);
    Make(root / "cache" / "query_results" / "a.parquet", 50);
    Make(root / "cache" / "query_results" / "b.parquet", 70);
    const auto in_use = [](const fs::path& p) { return p.filename() == "held.parquet"; };

    const auto caches = MeasureStorageCaches(root, in_use);
    Check(caches.size() == 5, "five caches with a project");
    const auto* ingestion = Find(caches, "ingestion");
    Check(ingestion && ingestion->bytes == 500 && ingestion->items == 2, "prepared inputs measured");
    Check(ingestion && ingestion->in_use == 1, "one prepared input is in use");
    Check(ingestion && ingestion->limit == "kept 30 days, at most 10 GB", "the rule's limit");
    const auto* queries = Find(caches, "queries");
    Check(queries && queries->bytes == 120 && queries->items == 2, "saved query answers measured");
    const auto* streamed = Find(caches, "streamed");
    Check(streamed && streamed->bytes == 0 && streamed->items == 0, "a missing folder measures empty");
    const auto* prepared = Find(caches, "prepared");
    Check(prepared && prepared->unit == "entry" && prepared->items == 0, "prepared datasets count entries");

    const auto cleared = ClearStorageCache("ingestion", root, in_use);
    Check(cleared.removed == 1 && cleared.freed_bytes == 300 && cleared.kept == 1, "clear keeps the file in use");
    Check(fs::exists(root / "cache" / "ingestion" / "held.parquet"), "the held file is still there");
    const auto q = ClearStorageCache("queries", root, in_use);
    Check(q.removed == 2 && q.freed_bytes == 120, "saved answers cleared");
    Check(ClearStorageCache("nope", root).removed == 0, "an unknown id clears nothing");

    const auto no_project = MeasureStorageCaches("");
    Check(no_project.size() == 1 && no_project[0].id == "temp", "without a project only the temp cache");

    Check(StorageItemsText(0, "file") == "no files", "no files");
    Check(StorageItemsText(1, "file") == "1 file", "1 file");
    Check(StorageItemsText(390, "file") == "390 files", "390 files");
    Check(StorageItemsText(0, "entry") == "no entries", "no entries");
    Check(StorageItemsText(2, "entry") == "2 entries", "2 entries");

    std::error_code ec;
    fs::remove_all(root, ec);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_storage_caches: all checks passed\n");
    return 0;
}
