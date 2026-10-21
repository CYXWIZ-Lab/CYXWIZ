// The prepared-data caches' rule (TOFIX160 P7): files older than the age
// go, then the oldest until the folder fits; a file in use stays; a day-old
// temp file goes; only the covered extension counts; Clear spares in-use files.
#include "../src/core/cache_prune.h"

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

// A file of `bytes` written `days_ago` days ago.
void Make(const fs::path& p, size_t bytes, int days_ago) {
    { std::ofstream f(p, std::ios::binary); f << std::string(bytes, 'x'); }
    fs::last_write_time(p, fs::file_time_type::clock::now() - std::chrono::hours(24) * days_ago);
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("cyxwiz_cache_prune_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);

    Make(dir / "old.parquet", 100, 40);       // past the age
    Make(dir / "older_held.parquet", 100, 50); // past the age, but in use
    Make(dir / "a.parquet", 300, 3);
    Make(dir / "b.parquet", 300, 2);
    Make(dir / "c.parquet", 300, 1);
    Make(dir / "notes.txt", 999, 60);          // another extension: not covered
    Make(dir / "x.parquet.tmp.1.2", 50, 2);    // a crash's leftover

    CacheFolder folder;
    folder.path = dir;
    folder.in_use = [&](const fs::path& p) { return p.filename() == "older_held.parquet"; };

    const auto before = MeasureCacheFolder(folder);
    Check(before.files == 5 && before.bytes == 1100, "measure: the covered files only (no temp, no .txt)");

    // Age 30 days, size cap 700 bytes: old goes (age), held stays, then a goes (oldest) to fit.
    const auto r = PruneCacheFolder(folder, 700, 30);
    Check(!fs::exists(dir / "old.parquet"), "a file past the age is removed");
    Check(fs::exists(dir / "older_held.parquet"), "a file in use stays");
    Check(r.in_use == 1, "the in-use file is counted");
    Check(!fs::exists(dir / "a.parquet"), "the oldest goes to fit the size");
    Check(fs::exists(dir / "b.parquet") && fs::exists(dir / "c.parquet"), "newer files stay");
    Check(fs::exists(dir / "notes.txt"), "another extension is never touched");
    Check(!fs::exists(dir / "x.parquet.tmp.1.2"), "a day-old temp file is removed");
    Check(r.removed == 2 && r.freed_bytes == 400, "two files, 400 bytes freed");
    Check(r.kept == 3 && r.kept_bytes == 700, "three kept, 700 bytes (the held file is over the cap)");

    // Under the limits: nothing goes.
    const auto again = PruneCacheFolder(folder, 10000, 30);
    Check(again.removed == 0 && again.kept == 3, "under the limits nothing is removed");

    // Clear: everything but the file in use.
    const auto cleared = ClearCacheFolder(folder);
    Check(cleared.removed == 2 && cleared.in_use == 1, "clear removes all but the file in use");
    Check(fs::exists(dir / "older_held.parquet") && !fs::exists(dir / "b.parquet"), "clear keeps the held file");

    // A missing folder is not an error.
    CacheFolder missing;
    missing.path = dir / "nope";
    Check(PruneCacheFolder(missing).removed == 0 && MeasureCacheFolder(missing).files == 0, "a missing folder is fine");

    // The project folders are listed with labels.
    const auto folders = PreparedDataCacheFolders(dir);
    Check(folders.size() == 3, "temp, ingestion and streamed folders");
    Check(folders[1].folder.path == dir / "cache" / "ingestion", "the ingestion folder");
    Check(folders[2].folder.path == dir / "cache" / "materialized" / "streamed", "the streamed folder");
    Check(PreparedDataCacheFolders("").size() == 1, "without a project only the temp cache");

    std::error_code ec;
    fs::remove_all(dir, ec);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_cache_prune: all checks passed\n");
    return 0;
}
