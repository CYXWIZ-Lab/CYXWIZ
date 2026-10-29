#pragma once

// Preferences > Storage (TOFIX160 P7, approved mockup
// D:/bench157/storage_mockup.html): every disk cache of prepared data with
// its size, its limit and a way to clear it. The three folders under the
// prune rule (core/cache_prune) plus the two caches with limits of their
// own: prepared datasets (materialization cache) and saved query answers.
// Measuring walks the folders, so callers run it off the UI thread. No ImGui.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct StorageCache {
    std::string id;      // temp, ingestion, streamed, prepared, queries
    std::string label;   // "Prepared inputs (this project)"
    std::string detail;  // what it holds
    std::string path;
    std::string limit;   // "kept 30 days, at most 10 GB"
    uint64_t bytes = 0;
    int items = 0;
    std::string unit = "file";  // what an item is ("file", "entry")
    int in_use = 0;             // files a loaded dataset reads (Clear keeps them)
};

// The caches, measured; without a project only this PC's temp cache.
std::vector<StorageCache> MeasureStorageCaches(
    const std::filesystem::path& project_root,
    const std::function<bool(const std::filesystem::path&)>& in_use = {});

struct StorageClearResult {
    int removed = 0;
    uint64_t freed_bytes = 0;
    int kept = 0;  // in use, or could not be removed (open elsewhere)
};

// Empties one cache (by id); files in use stay.
StorageClearResult ClearStorageCache(
    const std::string& id, const std::filesystem::path& project_root,
    const std::function<bool(const std::filesystem::path&)>& in_use = {});

// "1 file", "390 files", "no entries".
std::string StorageItemsText(int items, const std::string& unit);

}  // namespace cyxwiz
